"""Benchmark CPU/CUDA inference using isolated processes and persistent CSV results."""

import csv
import gc
import math
import multiprocessing as mp
import os
import platform
import statistics
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any, cast

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging as transformers_logging

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in two sentences what tokenization is in language models."
INFERENCE_MAX_NEW_TOKENS = 96
# Warm-up uses the same generation budget as the measured iterations.
WARMUP_MAX_NEW_TOKENS = INFERENCE_MAX_NEW_TOKENS
RSS_SAMPLING_INTERVAL_SECONDS = 0.01
NUMBER_BENCHMARK_ITERATIONS = 5
CSV_FILE = Path("results/cpu_gpu_baseline.csv")
SUMMARY_REPORT_WIDTH = 76
METRIC_LABEL_WIDTH = 32
PROMPT_FORMAT = "chat_template_system_user"
SYSTEM_MESSAGE = (
    "Answer precisely in exactly two short sentences. "
    "Define the concept without adding examples or unrelated claims."
)

# Definitions are also written to each CSV result so that exported numbers
# retain their meaning independently of this script and its console report.
INFERENCE_TIMING_SCOPE = (
    "Full model.generate() wall-clock time (prompt prefill + autoregressive "
    "decode; excludes model loading, tokenization, warm-up and text decoding)"
)
AGGREGATE_THROUGHPUT_DEFINITION = (
    "Total newly generated tokens / summed model.generate() wall times "
    "(including prompt prefill and autoregressive decode)"
)
# A generated answer is retained for inspection, but it is not a quality score.
QUALITY_EVALUATION_STATUS = "not_evaluated"
QUALITY_EVALUATION_SCOPE = (
    "Performance-only: no semantic equivalence, factual correctness or "
    "response-quality assessment across configurations"
)
PROCESS_RSS_LOAD_SCOPE = (
    "Net process RAM RSS change while loading tokenizer, model and libraries"
)
PROCESS_RSS_INFERENCE_PEAK_SCOPE = (
    "Highest sampled process RSS during measured model.generate() iterations "
    "(including start/end snapshots); sampling may miss short-lived spikes "
    "and adds slight monitoring overhead; excludes warm-up and model loading"
)
PYTORCH_CUDA_MEMORY_SCOPE = (
    "PyTorch CUDA caching allocator statistics during measured iterations "
    "(includes resident model allocations). Reserved memory includes blocks "
    "cached by PyTorch and doesn't mean exclusively active tensor memory. "
    "Reserved-after-inference is a snapshot, not a peak. Not total GPU VRAM "
    "usage: excludes CUDA driver/context and other libraries' allocations."
)

CSV_METRIC_COLUMNS = (
    "RunStartedAt",
    "RunStatus",
    "ErrorMessage",
    "ModelID",
    "Device",
    "DataType",
    "AverageInferenceTime",
    "MedianInferenceTime",
    "InferenceTimeStdDev",
    "AggregateTokenThroughput",
    "InferenceTimingScope",
    "AggregateThroughputDefinition",
    "QualityEvaluationStatus",
    "QualityEvaluationScope",
    "TotalModelResponseTokens",
    "TotalGeneratedTokens",
    "TokenLimitReachedIterations",
    "LastResponseStopReason",
    "PythonMemoryBeforeModelLoad",
    "PythonMemoryAfterModelLoad",
    "ProcessRssDeltaDuringLoad",
    "ProcessRssDeltaScope",
    "PythonMemoryAfterInference",
    "PeakProcessRssDuringInference",
    "ProcessRssInferencePeakScope",
    "RssSamplingIntervalMs",
    "PyTorchCudaPeakAllocatedMemory",
    "PyTorchCudaPeakReservedMemory",
    "PyTorchCudaReservedAfterInference",
    "PyTorchCudaMemoryScope",
    "CPUName",
    "GPUName",
    "PyTorchVersion",
    "TransformersVersion",
    "NumberBenchmarkIterations",
    "EncodedPromptLength",
    "InferenceMaxNewTokens",
    "WarmupMaxNewTokens",
    "DoSample",
    "PromptFormat",
    "SystemMessage",
    "PromptText",
    "LastGeneratedResponse",
)

MEMORY_METRIC_COLUMNS = frozenset(
    {
        "PythonMemoryBeforeModelLoad",
        "PythonMemoryAfterModelLoad",
        "ProcessRssDeltaDuringLoad",
        "PythonMemoryAfterInference",
        "PeakProcessRssDuringInference",
        "PyTorchCudaPeakAllocatedMemory",
        "PyTorchCudaPeakReservedMemory",
        "PyTorchCudaReservedAfterInference",
    }
)

TIME_METRIC_COLUMNS = frozenset(
    {"AverageInferenceTime", "MedianInferenceTime", "InferenceTimeStdDev"}
)
RATE_METRIC_COLUMNS = frozenset({"AggregateTokenThroughput"})
CSV_NUMERIC_METRIC_UNITS = {
    **dict.fromkeys(MEMORY_METRIC_COLUMNS, "MB"),
    **dict.fromkeys(TIME_METRIC_COLUMNS, "s"),
    **dict.fromkeys(RATE_METRIC_COLUMNS, "tokens/s"),
}
BENCHMARK_TIMEOUT_SECONDS = 600  # 10 min.


@dataclass
class BenchmarkOutput:
    """Results of a single device/data-type benchmark configuration.

    Attributes:
        metrics: Quantitative measurements and execution metadata.
        generated_text: Decoded response from the last measured iteration,
            retained for inspection rather than automated quality assessment.
    """

    metrics: dict[str, Any]
    generated_text: str


def get_process_rss_mb() -> float:
    """Return the resident set size (RSS) of the entire process in decimal MB.

    RSS includes all currently resident process memory, not only model weights.
    """
    return psutil.Process().memory_info().rss / 1000**2


def get_cpu_model_name() -> str:
    """Identify the host CPU using Linux cpuinfo or platform fallbacks.

    Returns:
        str: CPU model label, processor name, or machine architecture.
    """
    if platform.system() == "Linux":
        cpuinfo_path = Path("/proc/cpuinfo")

        if cpuinfo_path.is_file():
            for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()

    return platform.processor() or platform.machine()


def empty_cuda_cache(torch_device: torch.device) -> None:
    """Release unused PyTorch CUDA cache blocks before a configuration run.

    This doesn't free memory held by live tensors or all device allocations.
    For a CPU device, no action is performed.

    Args:
        torch_device: CPU or CUDA device selected for the benchmark.
    """
    if torch_device.type == "cuda":
        with torch.cuda.device(torch_device.index):
            torch.cuda.empty_cache()


def load_model_for_inference(
    torch_device: torch.device, torch_data_type: torch.dtype
) -> PreTrainedModel:
    """Load the pretrained causal language model onto the selected device.

    Configure parameter dtype, move the module to CPU/CUDA, and set evaluation
    mode for greedy decoding. Results may still vary across software/hardware.

    Args:
        torch_device: Target inference device.
        torch_data_type: Requested model weight and computation dtype.

    Returns:
        PreTrainedModel: Loaded causal language model in evaluation mode.
    """

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch_data_type)

    module = cast(torch.nn.Module, model)
    module.to(device=torch_device)
    module.eval()

    return model


def run_warmup_inference(model: Any, encoded_prompt: Any) -> None:
    """Perform an unmeasured generation with the benchmark token budget.

    This exercises the same maximum decode length as the measured runs,
    although EOS may terminate either generation earlier.
    """
    with torch.inference_mode():
        model.generate(
            **encoded_prompt,
            do_sample=False,
            max_new_tokens=WARMUP_MAX_NEW_TOKENS,
        )


def measure_generation_time(
    model: Any, encoded_prompt: Any, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Time the complete model.generate() call, including prefill and decode.

    The prompt has already been tokenized. CUDA is synchronized before and
    after the call so the elapsed wall time also covers queued GPU work.
    This doesn't isolate time to first token or per-token decode latency.

    Args:
        model (Any): Causal language model used for generation.
        encoded_prompt (Any): Tokenized prompt tensors on the selected device.
        device (torch.device): CPU or CUDA device used for inference.

    Returns:
        tuple[torch.Tensor, float]: Generated IDs and full-call time in seconds.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Prefill and autoregressive decode both happen inside generate().
    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_token_ids = model.generate(
            **encoded_prompt,
            do_sample=False,
            max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return (generated_token_ids, time.perf_counter() - start_time)


def get_generation_stop_reason(
    model: Any, generated_token_ids: torch.Tensor, input_token_count: int
) -> str:
    """Classify the final token as EOS, token-budget exhaustion, or other.

    Use the model's generation configuration, whose EOS ID can be an integer
    or a list. This diagnoses termination, not the correctness of the text.
    """
    new_tokens = generated_token_ids.shape[-1] - input_token_count
    if new_tokens <= 0:
        return "no_new_tokens"

    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_values: set[int] = set()
    elif isinstance(eos_ids, int):
        eos_values = {eos_ids}
    else:
        eos_values = set(eos_ids)

    final_id = int(generated_token_ids[0, -1].item())
    if final_id in eos_values:
        return "eos_token"
    if new_tokens >= INFERENCE_MAX_NEW_TOKENS:
        return "max_new_tokens"
    return "other"


def run_generation_benchmark(
    model: Any, encoded_prompt: Any, torch_device: torch.device
) -> tuple[torch.Tensor, list[float], int, float, float, int, str]:
    """Repeat full-call inference timing and count newly generated tokens.

    Each duration includes prompt prefill and token-by-token decode; tokenizer
    loading, prompt tokenization and warm-up occur outside these measurements.
    A background thread samples process RSS throughout the measured iterations.
    Its peak is an observed (sampled) peak, not an exact instantaneous maximum.

    Args:
        model (Any): Causal language model used for generation.
        encoded_prompt (Any): Tokenized prompt tensors.
        torch_device (torch.device): Device on which the model is running.

    Raises:
        RuntimeError: If no iteration produces generated token IDs.

    Returns:
        tuple: Last generated sequence, full-call durations, total new tokens,
            sampled peak process RSS, final process RSS (decimal MB), number
            of iterations reaching the token budget, and final stop reason.
    """
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)

    generation_durations: list[float] = []
    total_new_tokens = 0
    token_limit_reached_iterations = 0
    last_stop_reason = "unknown"
    input_token_count = encoded_prompt["input_ids"].shape[-1]
    last_generated_token_ids: torch.Tensor | None = None

    process = psutil.Process()
    stop_sampling = threading.Event()
    peak_process_rss_mb = process.memory_info().rss / 1000**2

    def sample_process_rss() -> None:
        """Poll resident process memory without including warm-up or loading."""
        nonlocal peak_process_rss_mb
        while not stop_sampling.wait(RSS_SAMPLING_INTERVAL_SECONDS):
            current_mb = process.memory_info().rss / 1000**2
            peak_process_rss_mb = max(peak_process_rss_mb, current_mb)

    sampler_thread = threading.Thread(
        target=sample_process_rss, name="inference-rss-sampler", daemon=True
    )
    sampler_thread.start()

    try:
        for _ in range(NUMBER_BENCHMARK_ITERATIONS):
            last_generated_token_ids, elapsed_time = measure_generation_time(
                model, encoded_prompt, torch_device
            )
            generation_durations.append(elapsed_time)
            total_new_tokens += last_generated_token_ids.shape[-1] - input_token_count
            last_stop_reason = get_generation_stop_reason(
                model, last_generated_token_ids, input_token_count
            )
            if last_stop_reason == "max_new_tokens":
                token_limit_reached_iterations += 1
    finally:
        # Always stop/join the sampler, including when inference fails.
        stop_sampling.set()
        sampler_thread.join()

    rss_after_inference_mb = process.memory_info().rss / 1000**2
    peak_process_rss_mb = max(peak_process_rss_mb, rss_after_inference_mb)

    if last_generated_token_ids is None:
        raise RuntimeError("No generation result was produced.")

    return (
        last_generated_token_ids,
        generation_durations,
        total_new_tokens,
        peak_process_rss_mb,
        rss_after_inference_mb,
        token_limit_reached_iterations,
        last_stop_reason,
    )


def get_cuda_memory_stats_mb(
    torch_device: torch.device,
) -> tuple[str, float | None, float | None, float | None]:
    """Read peak and final PyTorch CUDA caching-allocator usage in decimal MB.

    Peaks are reset after warm-up, before measured iterations. The final
    reserved value comes from torch.cuda.memory_reserved(): it includes cached
    blocks and is not synonymous with actively allocated tensor memory.
    These statistics exclude allocations outside PyTorch (for example the CUDA
    context) and therefore do not measure total device VRAM usage.

    Args:
        torch_device: Device for which allocator statistics are requested.

    Returns:
        tuple[str, float | None, float | None, float | None]: GPU name,
            peak allocated MB, peak reserved MB, and reserved MB at the end
            of inference. CPU configurations have an empty name and no values.
    """
    if torch_device.type != "cuda":
        return "", None, None, None

    return (
        torch.cuda.get_device_name(torch_device),
        torch.cuda.max_memory_allocated(torch_device) / 1000**2,
        torch.cuda.max_memory_reserved(torch_device) / 1000**2,
        torch.cuda.memory_reserved(torch_device) / 1000**2,
    )


def build_benchmark_result(
    data_type_name: str,
    torch_device: torch.device,
    input_token_count: int,
    last_generated_token_ids: torch.Tensor,
    generation_durations: list[float],
    total_generated_tokens: int,
    memory_before_load_mb: float,
    memory_after_load_mb: float,
    peak_process_rss_during_inference_mb: float,
    process_rss_after_inference_mb: float,
    token_limit_reached_iterations: int,
    last_response_stop_reason: str,
) -> dict[str, Any]:
    """Build metrics with explicit timing and process-RSS measurement scopes.

    Mean and median inference time include both prefill and decode. Aggregate
    throughput divides newly generated tokens by the sum of these full-call
    durations: it is NOT decode-only tokens per second. Time to first token
    and per-token decode latency aren't measured separately.

    No semantic equivalence, correctness or quality checks are performed.
    The decoded response from the last iteration is diagnostic only.

    The load-stage RAM metric is a net process-wide RSS delta, not the memory
    footprint of model parameters alone. It includes tokenizer and library
    allocations, and doesn't represent CUDA VRAM usage. The sampled RSS peak
    covers only measured inference, while the final RSS is a point-in-time
    snapshot. GPU metrics cover the PyTorch CUDA allocator, not total VRAM.

    Args:
        data_type_name: Name of the precision used (for example, float32).
        torch_device: Actual device used for inference.
        input_token_count: Number of encoded prompt tokens.
        last_generated_token_ids: Full generated sequence from the final run.
        generation_durations: Wall-clock seconds for each measured generation.
        total_generated_tokens: Sum of new tokens across all measured runs.
        memory_before_load_mb: Process RSS before tokenizer/model loading, MB.
        memory_after_load_mb: Process RSS after tokenizer/model loading, MB.
        peak_process_rss_during_inference_mb: Sampled process RSS peak, MB.
        process_rss_after_inference_mb: Process RSS at benchmark end, MB.
        token_limit_reached_iterations: Number of capped measured generations.
        last_response_stop_reason: Why the final generation stopped.

    Returns:
        dict[str, Any]: Collected measurements and explanatory metadata;
            measurements remain numeric and use the documented CSV units.
    """
    (
        gpu_name,
        peak_gpu_allocated_mb,
        peak_gpu_reserved_mb,
        reserved_gpu_after_inference_mb,
    ) = get_cuda_memory_stats_mb(torch_device)

    last_generation_token_count = last_generated_token_ids.shape[-1] - input_token_count
    total_generation_time = sum(generation_durations)

    return {
        "ModelID": MODEL_ID,
        "Device": torch_device.type,
        "DataType": data_type_name,
        "GPUName": gpu_name,
        "CPUName": get_cpu_model_name(),
        "PyTorchVersion": str(torch.__version__),
        "TransformersVersion": transformers.__version__,
        "EncodedPromptLength": input_token_count,
        "TotalModelResponseTokens": last_generation_token_count,
        # Both statistics describe entire generate() calls: prefill + decode.
        "AverageInferenceTime": statistics.mean(generation_durations),
        "MedianInferenceTime": statistics.median(generation_durations),
        "InferenceTimeStdDev": (
            statistics.stdev(generation_durations)
            if len(generation_durations) >= 2
            else 0.0
        ),
        # The numerator counts only NEW tokens, but the denominator includes
        # prefill as well as decode.
        "AggregateTokenThroughput": total_generated_tokens / total_generation_time,
        "InferenceTimingScope": INFERENCE_TIMING_SCOPE,
        "AggregateThroughputDefinition": AGGREGATE_THROUGHPUT_DEFINITION,
        "QualityEvaluationStatus": QUALITY_EVALUATION_STATUS,
        "QualityEvaluationScope": QUALITY_EVALUATION_SCOPE,
        "NumberBenchmarkIterations": NUMBER_BENCHMARK_ITERATIONS,
        "PythonMemoryBeforeModelLoad": memory_before_load_mb,
        "PythonMemoryAfterModelLoad": memory_after_load_mb,
        # Host RAM RSS change for the entire process during the load stage.
        # Includes tokenizer, model and libraries; NOT only model parameters.
        "ProcessRssDeltaDuringLoad": memory_after_load_mb - memory_before_load_mb,
        "ProcessRssDeltaScope": PROCESS_RSS_LOAD_SCOPE,
        # The post-inference snapshot doesn't describe the inference peak.
        "PythonMemoryAfterInference": process_rss_after_inference_mb,
        "PeakProcessRssDuringInference": peak_process_rss_during_inference_mb,
        "ProcessRssInferencePeakScope": PROCESS_RSS_INFERENCE_PEAK_SCOPE,
        "RssSamplingIntervalMs": int(RSS_SAMPLING_INTERVAL_SECONDS * 1000),
        # PyTorch allocator figures aren't the total occupied device VRAM.
        "PyTorchCudaPeakAllocatedMemory": peak_gpu_allocated_mb,
        "PyTorchCudaPeakReservedMemory": peak_gpu_reserved_mb,
        # Includes cache blocks reserved by PyTorch, not only active tensors.
        "PyTorchCudaReservedAfterInference": reserved_gpu_after_inference_mb,
        "PyTorchCudaMemoryScope": (
            PYTORCH_CUDA_MEMORY_SCOPE if torch_device.type == "cuda" else "not_applicable"
        ),
        "TotalGeneratedTokens": total_generated_tokens,
        "TokenLimitReachedIterations": token_limit_reached_iterations,
        "LastResponseStopReason": last_response_stop_reason,
        "InferenceMaxNewTokens": INFERENCE_MAX_NEW_TOKENS,
        "WarmupMaxNewTokens": WARMUP_MAX_NEW_TOKENS,
        "DoSample": False,
        "PromptFormat": PROMPT_FORMAT,
        "SystemMessage": SYSTEM_MESSAGE,
        "PromptText": PROMPT,
    }


def run_model_benchmark(
    torch_device: torch.device, torch_data_type: torch.dtype
) -> BenchmarkOutput:
    """Execute model loading, warm-up, timed inference, and metric collection.

    Tokenization (using the model's chat template), model loading, warm-up,
    and decoding occur outside measured model.generate() durations. RSS
    loading measurements cover the entire process; CUDA measurements cover
    only the PyTorch allocator.

    Args:
        torch_device: CPU or CUDA device to execute inference on.
        torch_data_type: PyTorch dtype for the loaded causal language model.

    Returns:
        BenchmarkOutput: Numeric metrics and the last decoded answer.
    """

    gc.collect()

    if torch_device.type == "cuda":
        empty_cuda_cache(torch_device)

    # Both snapshots are process-wide host RAM RSS measurements. Their net
    # difference includes tokenizer/model/library allocations and releases;
    # it is not the model's isolated parameter footprint (nor CUDA VRAM).
    process_rss_before_load = get_process_rss_mb()
    print("  Tokenizer/model load ... ", end="", flush=True)
    transformers_logging.disable_progress_bar()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = load_model_for_inference(torch_device, torch_data_type)

    finally:
        transformers_logging.enable_progress_bar()

    process_rss_after_load = get_process_rss_mb()
    print("OK", flush=True)

    # Qwen2.5-Instruct expects structured chat turns, not a bare completion
    # prompt.
    encoded_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": PROMPT},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(torch_device)
    input_token_count = encoded_prompt["input_ids"].shape[-1]

    print("  Warm-up ................ ", end="", flush=True)
    run_warmup_inference(model, encoded_prompt)
    print("OK", flush=True)

    print("  Benchmark .............. ", end="", flush=True)
    (
        last_generated_token_ids,
        generation_durations,
        total_generated_tokens,
        peak_process_rss_during_inference_mb,
        process_rss_after_inference_mb,
        token_limit_reached_iterations,
        last_response_stop_reason,
    ) = run_generation_benchmark(model, encoded_prompt, torch_device)

    print("OK", flush=True)

    benchmark_metrics = build_benchmark_result(
        str(torch_data_type).replace("torch.", ""),
        torch_device,
        input_token_count,
        last_generated_token_ids,
        generation_durations,
        total_generated_tokens,
        process_rss_before_load,
        process_rss_after_load,
        peak_process_rss_during_inference_mb,
        process_rss_after_inference_mb,
        token_limit_reached_iterations,
        last_response_stop_reason,
    )

    # Preserve only the last iteration's response for optional manual review.
    # It doesn't establish correctness or equivalence between data types.
    new_token_ids = last_generated_token_ids[0][input_token_count:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return BenchmarkOutput(metrics=benchmark_metrics, generated_text=generated_text)


def benchmark_worker(
    device_name: str,
    data_type_name: str,
    result_queue: Any,
) -> None:
    """Run one benchmark in a spawned child process and report its outcome.

    Sends a (status, payload) pair to the parent: the BenchmarkOutput on
    success or a formatted traceback on failure. Exceptions do not escape
    into the parent without being reported through the queue.

    Args:
        device_name: PyTorch device identifier, such as cpu or cuda:0.
        data_type_name: Precision identifier, either float32 or float16.
        result_queue: Multiprocessing queue shared with the parent process.
    """

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    try:
        benchmark_result = run_model_benchmark(
            torch.device(device_name),
            dtype_by_name[data_type_name],
        )

        result_queue.put(("success", benchmark_result))

    except Exception:  # noqa: BLE001
        # Forward unexpected worker exceptions to the parent process.
        result_queue.put(("error", traceback.format_exc()))


def run_benchmark_in_subprocess(
    device_name: str,
    data_type_name: str,
) -> BenchmarkOutput:
    """Execute an isolated benchmark worker with a ten-minute deadline.

    Use the spawn start method so CPU/GPU runs have independent processes.
    Read the worker's outcome and ensure it exits successfully. On every
    exit path, close the queue and terminate or kill any lingering worker.

    Args:
        device_name: PyTorch device identifier, such as cpu or cuda:0.
        data_type_name: Precision identifier, either float32 or float16.

    Raises:
        TimeoutError: Worker exceeded the configured execution deadline.
        RuntimeError: Worker failed, returned an unexpected status, exited
            without a result, or failed to exit cleanly after reporting.

    Returns:
        BenchmarkOutput: Measurements returned by the child process.
    """
    context = mp.get_context("spawn")
    result_queue = context.Queue()

    worker_process = context.Process(
        target=benchmark_worker,
        args=(device_name, data_type_name, result_queue),
    )

    try:
        worker_process.start()

        deadline = time.monotonic() + BENCHMARK_TIMEOUT_SECONDS

        while True:
            remaining_seconds = deadline - time.monotonic()

            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"Benchmark exceeded {BENCHMARK_TIMEOUT_SECONDS} seconds."
                )

            try:
                status, result = result_queue.get(timeout=min(0.5, remaining_seconds))
                break

            except Empty:
                if not worker_process.is_alive():
                    raise RuntimeError(
                        "Benchmark worker exited without returning a result "
                        f"(exit code: {worker_process.exitcode})."
                    )

        if status == "error":
            raise RuntimeError(f"Benchmark worker failed:\n{result}")

        if status != "success":
            raise RuntimeError(f"Unexpected worker status: {status!r}")

        worker_process.join(timeout=5)

        if worker_process.is_alive():
            raise RuntimeError("Benchmark worker did not exit after sending its result.")

        if worker_process.exitcode != 0:
            raise RuntimeError(
                f"Benchmark worker exited with code {worker_process.exitcode}."
            )

        return cast(BenchmarkOutput, result)

    finally:
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=5)

        if worker_process.is_alive():
            worker_process.kill()
            worker_process.join()

        result_queue.close()
        result_queue.join_thread()


def print_benchmark_header(
    configurations: list[tuple[str, str]], started_at: str
) -> None:
    """Print benchmark settings and the timestamp shared by all runs.

    Args:
        configurations: Pairs of device identifiers and dtype names.
        started_at: Timezone-aware ISO timestamp for this benchmark group.
    """

    print("\n" + "=" * SUMMARY_REPORT_WIDTH)
    print("INFERENCE BENCHMARK")
    print("=" * SUMMARY_REPORT_WIDTH)
    print(f"Started at      : {started_at}")
    print(f"Model           : {MODEL_ID}")
    print(f"Prompt format   : {PROMPT_FORMAT}")
    print(f"Iterations      : {NUMBER_BENCHMARK_ITERATIONS}")
    print(f"Max new tokens  : {INFERENCE_MAX_NEW_TOKENS}")
    print(f"Warm-up max new : {WARMUP_MAX_NEW_TOKENS}")
    print(f"Configurations  : {len(configurations)}")
    print("Compare only CSV rows with matching prompt format and token budget.")
    print("=" * SUMMARY_REPORT_WIDTH)


def print_metric(label: str, value: float | str, unit: str = "") -> None:
    """Print a human-readable metric without changing its stored value.

    Args:
        label: Description of the metric displayed in the console.
        value: Previously formatted numeric value or descriptive string.
        unit: Optional measurement unit for console output only.
    """
    formatted_value = f"{value}{f' {unit}' if unit else ''}"
    print(f"  {label:<{METRIC_LABEL_WIDTH}} : {formatted_value}")


def print_benchmark_metrics(metrics: dict[str, Any]) -> None:
    """Print generation, process RSS, and available CUDA allocator metrics.

    Floating-point values are rounded for readability only in the console;
    the CSV writer stores the original numbers without unit suffixes.

    Args:
        metrics: Measured values and device metadata from a configuration.
    """
    print("\n  PROMPT AND GENERATION")
    print_metric("Prompt tokens", metrics["EncodedPromptLength"])
    print_metric("Generated tokens (last run)", metrics["TotalModelResponseTokens"])
    print_metric("Generated tokens (all runs)", metrics["TotalGeneratedTokens"])
    print_metric("Last response stop reason", metrics["LastResponseStopReason"])
    print_metric(
        "Iterations reaching token limit",
        f"{metrics['TokenLimitReachedIterations']}/{metrics['NumberBenchmarkIterations']}",
    )
    if metrics["TokenLimitReachedIterations"]:
        print("  Warning: Some responses reached the generation token limit.")
    print("\n  PERFORMANCE")
    print_metric("Mean generate() time", f"{metrics['AverageInferenceTime']:.2f}", "s")
    print_metric("Median generate() time", f"{metrics['MedianInferenceTime']:.2f}", "s")
    print_metric("Generate() std. dev.", f"{metrics['InferenceTimeStdDev']:.3f}", "s")

    print_metric(
        "Aggregate full-call throughput",
        f"{metrics['AggregateTokenThroughput']:.2f}",
        "tokens/s",
    )

    print("  Note: Inference time covers full model.generate() (prefill + decode).")
    print("        Aggregate throughput = new tokens / total full-call time;")
    print(
        "        it is not decode-only speed. TTFT and per-token latency aren't measured."
    )

    print("\n  PROCESS MEMORY (RAM / RSS)")

    print_metric(
        "Before tokenizer/model load",
        f"{metrics['PythonMemoryBeforeModelLoad']:.2f}",
        "MB",
    )
    print_metric(
        "After tokenizer/model load",
        f"{metrics['PythonMemoryAfterModelLoad']:.2f}",
        "MB",
    )
    print_metric(
        "Net RSS delta during load",
        f"{metrics['ProcessRssDeltaDuringLoad']:.2f}",
        "MB",
    )
    print_metric(
        "After inference (snapshot)",
        f"{metrics['PythonMemoryAfterInference']:.2f}",
        "MB",
    )
    print_metric(
        "Peak RSS during inference (sampled)",
        f"{metrics['PeakProcessRssDuringInference']:.2f}",
        "MB",
    )
    print("  Note: RSS peak is sampled every 10 ms and can miss brief spikes;")
    print("        the after-inference RSS is only a point-in-time snapshot.")
    print("        Host RAM RSS delta includes tokenizer, model and libraries;")
    print("        it is a net process change, not isolated model-weight memory.")

    if metrics["Device"] == "cuda":
        print("\n  GPU MEMORY (PYTORCH ALLOCATOR)")

        print_metric("GPU", metrics["GPUName"])
        print_metric(
            "PyTorch peak allocated",
            f"{metrics['PyTorchCudaPeakAllocatedMemory']:.2f}",
            "MB",
        )
        print_metric(
            "PyTorch peak reserved",
            f"{metrics['PyTorchCudaPeakReservedMemory']:.2f}",
            "MB",
        )
        print_metric(
            "PyTorch reserved after infer.",
            f"{metrics['PyTorchCudaReservedAfterInference']:.2f}",
            "MB",
        )
        print("  Note: Reserved includes PyTorch cached blocks, not just active tensors.")
        print("        These are allocator metrics, NOT total VRAM used;")
        print("        CUDA driver/context and external allocations may be excluded.")

    print("\n" + "-" * SUMMARY_REPORT_WIDTH, flush=True)


def format_comparison_value(
    metrics: dict[str, Any], metric_key: str, is_gpu_metric: bool
) -> str:
    """
    Format a benchmark metric value for the final comparison table.

    Args:
        `metrics`: Dictionary containing the benchmark metrics and
            the device used for inference.
        `metric_key`: Dictionary key identifying the metric to format.
        `is_gpu_metric`: Whether the metric applies exclusively to
            CUDA devices.

    Returns:
        `str`: "N/A" for GPU-only metrics on non-CUDA devices.
            Otherwise, returns integers without decimal places
            and floating-point values with two decimal places.
    """
    if is_gpu_metric and metrics["Device"] != "cuda":
        return "N/A"

    value = metrics[metric_key]

    if isinstance(value, int):
        return str(value)

    return f"{value:.2f}"


def print_final_comparison(outputs: list[BenchmarkOutput]) -> None:
    """
    Print a side-by-side comparison of benchmark results.

    Displays full-call inference times (prefill + decode), aggregate token
    throughput, generated token counts, and process/device memory metrics.
    GPU-specific metrics are shown as "N/A" for non-CUDA devices.

    Args:
        outputs: List of benchmark results, each containing the
            metrics collected for a specific device and data type
            configuration.
    """
    column_width = 15
    label_width = 29

    metric_rows = (
        ("Mean generate() (s)", "AverageInferenceTime", False),
        ("Median generate() (s)", "MedianInferenceTime", False),
        ("Generate() std. dev. (s)", "InferenceTimeStdDev", False),
        ("Aggregate throughput (tok/s)", "AggregateTokenThroughput", False),
        ("Tokens (last run)", "TotalModelResponseTokens", False),
        ("Tokens (all runs)", "TotalGeneratedTokens", False),
        ("Token-limit hits", "TokenLimitReachedIterations", False),
        ("RSS delta during load (MB)", "ProcessRssDeltaDuringLoad", False),
        ("RAM after inference (MB)", "PythonMemoryAfterInference", False),
        ("RAM sampled peak (MB)", "PeakProcessRssDuringInference", False),
        ("Torch CUDA peak alloc (MB)", "PyTorchCudaPeakAllocatedMemory", True),
        ("Torch CUDA peak reserv (MB)", "PyTorchCudaPeakReservedMemory", True),
    )

    comparison_width = max(
        SUMMARY_REPORT_WIDTH,
        label_width + column_width * len(outputs),
    )

    print("\n" + "=" * comparison_width)
    print("FINAL COMPARISON")
    print("=" * comparison_width)

    headings = [
        f"{output.metrics['Device'].upper()} {output.metrics['DataType']}".replace(
            "float", "FP"
        )
        for output in outputs
    ]

    print(
        f"{'Metric':<{label_width}}"
        + "".join(f"{heading:>{column_width}}" for heading in headings)
    )

    print("-" * comparison_width)

    for label, metric_key, is_gpu_metric in metric_rows:
        values = [
            format_comparison_value(output.metrics, metric_key, is_gpu_metric)
            for output in outputs
        ]

        print(
            f"{label:<{label_width}}"
            + "".join(f"{value:>{column_width}}" for value in values)
        )

    print("=" * comparison_width)
    print("IMPORTANT")
    print("• Inference timings include prompt prefill and autoregressive decode.")
    print("• Aggregate throughput = new tokens / summed full model.generate() times.")
    print(
        "• Standard deviation is across measured generate() calls "
        f"(sample, n={NUMBER_BENCHMARK_ITERATIONS})."
    )
    print("    It is not decode-only speed; TTFT and per-token latency aren't measured.")
    print("• RSS sampled peak is not a guaranteed instantaneous maximum.")
    print("• CUDA reserved memory includes cached blocks, not only active tensors.")
    print("• CUDA metrics reflect the PyTorch allocator, NOT total VRAM usage;")
    print("• CUDA context/driver and external allocations may be excluded.")
    print(
        "• Host RAM RSS delta includes tokenizer/model/library loading, not only weights."
    )
    print("• PERFORMANCE ONLY: Response correctness, semantic equivalence and quality")
    print("across FP32/FP16 aren't assessed; no quality conclusion can be drawn.")


def print_model_responses(outputs: list[BenchmarkOutput]) -> None:
    """
    Print the generated responses from the last benchmark iteration.

    Displays the decoded response for each benchmark configuration,
    identifying the device and data type used during inference.
    Empty responses are explicitly indicated.

    Args:
        outputs: List of benchmark results containing the generated
            text and configuration metrics for each execution.
    """
    print("\n" + "=" * SUMMARY_REPORT_WIDTH)
    print("GENERATED RESPONSES (LAST ITERATION PER CONFIGURATION)")
    print("=" * SUMMARY_REPORT_WIDTH)
    print("Diagnostic samples for manual inspection, not a quality evaluation.")
    print("Text differences alone do not quantify semantic or factual quality.")

    for output in outputs:
        metrics = output.metrics
        print(f"\n[{metrics['Device'].upper()} | {metrics['DataType']}]")
        print(output.generated_text.strip() or "(Empty response)")

    print("\n" + "=" * SUMMARY_REPORT_WIDTH)


def normalize_csv_numeric_value(value: Any, unit: str) -> float | int | str:
    """Convert recognized numeric metrics to unit-free CSV cell values.

    Existing CSV files may contain values such as "2.35 s" or "512.00 MB".
    Accept either those historical representations or plain numbers, without
    rounding. Do not silently discard unexpected or non-finite historical
    values: preserve their original textual representation for investigation.

    Args:
        value: Original number, unit-suffixed string, or missing value.
        unit: Historical suffix associated with this particular CSV column.

    Returns:
        float | int | str: Unit-free numeric value when conversion succeeds;
            empty string for missing entries or original text if invalid.
    """
    if value is None or value == "":
        return ""

    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(value) else str(value)

    text = str(value).strip()
    if unit and text.endswith(unit):
        text = text[: -len(unit)].strip()

    try:
        numeric_value = float(text)
    except (TypeError, ValueError):
        return str(value)

    return numeric_value if math.isfinite(numeric_value) else str(value)


def format_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    """Prepare a row without losing metadata or embedding units in numbers.

    All column names are retained, including RunID and unknown legacy fields.
    Recognized time, memory, and throughput fields are normalized, so old
    unit-bearing rows and new results can be analyzed together in pandas.
    Integer counts and all non-metric fields keep their original values.

    Args:
        result: Benchmark result or previously saved historical CSV row.

    Returns:
        dict[str, Any]: A copy with unit-free numeric metric values.
    """
    return {
        name: (
            normalize_csv_numeric_value(value, CSV_NUMERIC_METRIC_UNITS[name])
            if name in CSV_NUMERIC_METRIC_UNITS
            else value
        )
        for name, value in result.items()
    }


def save_benchmark_metrics_to_csv(
    benchmark_results: list[dict[str, Any]],
) -> None:
    """
    Save benchmark results to CSV while preserving historical records.

    Preserve all existing CSV columns and metadata, including any RunID values.
    Recognized historical time, memory, and throughput columns are converted
    from unit-bearing text to unit-free numeric cells; unrecognized legacy
    columns and invalid historical values remain unchanged. New rows are
    appended, then a temporary file atomically replaces the previous CSV.

    Args:
        benchmark_results: Per-configuration success or failure dictionaries.
    """
    if not benchmark_results:
        return

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    previous_results: list[dict[str, Any]] = []
    existing_columns: list[str] = []

    if CSV_FILE.exists():
        with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_columns = list(reader.fieldnames or [])

            # Keep unknown historical columns and RunID; normalize only
            # recognized metric fields whose old values may carry units.
            previous_results = [format_csv_row(row) for row in reader]

        for previous in previous_results:
            if not any(previous.get(key) for key in ("RunStartedAt", "RunStatus")):
                previous["RunStatus"] = "legacy"

    formatted_results = [format_csv_row(row) for row in benchmark_results]

    all_seen_columns = dict.fromkeys(
        [
            *existing_columns,
            *(key for row in formatted_results for key in row),
        ]
    )

    columns = [
        *CSV_METRIC_COLUMNS,
        *(name for name in all_seen_columns if name not in CSV_METRIC_COLUMNS),
    ]

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=CSV_FILE.parent,
            prefix=f".{CSV_FILE.stem}-",
            suffix=".tmp",
            delete=False,
        ) as csv_file:
            temporary_path = Path(csv_file.name)
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows([*previous_results, *formatted_results])

        os.replace(temporary_path, CSV_FILE)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"\nResults saved to   : {CSV_FILE}")
    print(f"New records        : {len(benchmark_results)}")
    print(f"Historical records : {len(previous_results)}")


def run_inference_benchmark() -> int:
    """
    Execute each benchmark configuration and immediately persist its outcome.

    The CPU baseline and any available CUDA FP32/FP16 benchmarks execute
    in isolated subprocesses. A failure is recorded without discarding
    earlier results or preventing later configurations from running.
    CSV write errors are propagated instead of being mistaken for
    individual benchmark failures. Return 1 on any configuration failure
    only after all configurations have been attempted; otherwise return 0.
    """

    started_at = datetime.now().astimezone()
    run_started_at = started_at.isoformat(timespec="microseconds")
    configurations = [("cpu", "float32")]

    if torch.cuda.is_available():
        configurations.extend([("cuda:0", "float32"), ("cuda:0", "float16")])

    print_benchmark_header(configurations, run_started_at)

    outputs: list[BenchmarkOutput] = []
    failures: list[tuple[str, str, str]] = []

    for index, (device_name, data_type_name) in enumerate(configurations, start=1):
        friendly_device = device_name.split(":")[0].upper()
        print(
            f"\n[{index}/{len(configurations)}] {friendly_device} | {data_type_name}",
            flush=True,
        )
        try:
            output = run_benchmark_in_subprocess(device_name, data_type_name)
        except Exception as exc:  # noqa: BLE001
            error_message = " | ".join(
                line.strip() for line in str(exc).splitlines() if line.strip()
            ) or repr(exc)
            save_benchmark_metrics_to_csv(
                [
                    {
                        "RunStartedAt": run_started_at,
                        "RunStatus": "failed",
                        "ErrorMessage": error_message,
                        "ModelID": MODEL_ID,
                        "Device": device_name.split(":")[0],
                        "DataType": data_type_name,
                        "PromptFormat": PROMPT_FORMAT,
                        "SystemMessage": SYSTEM_MESSAGE,
                        "PromptText": PROMPT,
                        "InferenceMaxNewTokens": INFERENCE_MAX_NEW_TOKENS,
                    }
                ]
            )
            failures.append((device_name, data_type_name, error_message))
            print(f"  FAILED: {error_message}", flush=True)
            continue

        # Save each success before attempting the next configuration.
        save_benchmark_metrics_to_csv(
            [
                {
                    "RunStartedAt": run_started_at,
                    "RunStatus": "completed",
                    "ErrorMessage": "",
                    **output.metrics,
                    "LastGeneratedResponse": output.generated_text,
                }
            ]
        )
        outputs.append(output)
        print_benchmark_metrics(output.metrics)

    if len(configurations) == 1:
        print("CUDA is not available; only the CPU baseline was executed.")

    if outputs:
        print_final_comparison(outputs)
        print_model_responses(outputs)
    else:
        print("\nNo benchmark configurations completed successfully.")

    print(f"\nCompleted configurations : {len(outputs)}")
    print(f"Failed configurations    : {len(failures)}")

    if failures:
        print("\nFAILED CONFIGURATIONS")
        for device_name, data_type_name, error_message in failures:
            reason = error_message.split(" | ")[-1]
            print(f"  {device_name} | {data_type_name}: {reason}")

        print(
            "\nBenchmark finished with partial failures; "
            f"all outcomes were recorded in {CSV_FILE}."
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_inference_benchmark())
