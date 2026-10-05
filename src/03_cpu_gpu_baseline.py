import gc
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDENTIFIER = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in two sentences what tokenization is in language models."
WARMUP_MAX_NEW_TOKENS = 5
MAX_NEW_TOKENS = 50
CUDA_DEVICE_INDEX = 0
RESULTS_PATH = Path("results/cpu_gpu_baseline.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)

def prepare_cuda_memory_for_inference(execution_device):
    """Prepare CUDA memory before loading a new model configuration."""

    gc.collect()

    with torch.cuda.device(execution_device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(execution_device)


def pre_inference_warmup(inference_model, input_data):
    """Run a short warm-up generation before measurement."""

    print("Running warm-up...")

    with torch.inference_mode():
        generated_warmup_ids = inference_model.generate(**input_data, do_sample=False, max_new_tokens=WARMUP_MAX_NEW_TOKENS)

    del generated_warmup_ids


def execute_inference_with_timing(inference_model, input_data, execution_device):
    """Run the measured generation and return its output and elapsed time."""

    print("Running measured inference...")

    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)

    inference_start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = inference_model.generate(**input_data, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,)

    if execution_device.type == "cuda":
        torch.cuda.synchronize(execution_device)

    elapsed_time_seconds = (time.perf_counter() - inference_start_time)

    return generated_ids, elapsed_time_seconds


def collect_gpu_memory_metrics(execution_device):
    """Collect GPU identification and VRAM usage metrics."""

    gpu_device_name = ""
    vram_peak_allocated_megabytes = 0.0
    vram_reserved_megabytes = 0.0

    if execution_device.type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(execution_device)
        vram_peak_allocated_megabytes = (torch.cuda.max_memory_allocated(execution_device) / (1024 ** 2))
        vram_reserved_megabytes = (torch.cuda.memory_reserved(execution_device) / (1024 ** 2))

    return (gpu_device_name, vram_peak_allocated_megabytes, vram_reserved_megabytes)


def print_inference_results(inference_metrics, inference_result):
    """Print inference metrics and generated text."""

    print(f"\n{inference_metrics['device'].upper()} / " f"{inference_metrics['dtype'].upper()} results")
    print(f"Model: {inference_metrics['model']}")
    print(f"Device Type: {inference_metrics['device']}")
    print(f"Data Type: {inference_metrics['dtype']}")

    if inference_metrics["gpu_device_name"]:
        print(f"GPU Name: {inference_metrics['gpu_device_name']}")

    print(f"Input Tokens Count: " f"{inference_metrics['input_tokens']}")
    print(f"Generated Tokens Count: " f"{inference_metrics['output_tokens']}")
    print(f"Elapsed Time (seconds): " f"{inference_metrics['elapsed_seconds']:.4f}")
    print(f"Generated Tokens per Second: " f"{inference_metrics['tokens_per_second']:.2f}")

    print(f"RAM Usage (MB): " f"{inference_metrics['ram_rss_mb']:.2f}")

    if inference_metrics["device"] == "cuda":
        print(f"VRAM Peak Allocated (MB): " f"{inference_metrics['vram_peak_allocated_mb']:.2f}")
        print(f"VRAM Reserved (MB): " f"{inference_metrics['vram_reserved_mb']:.2f}")

    print(f"\nInference Result:\n{inference_result}")


def save_inference_results_to_csv(inference_results):
    """Save all inference results to a CSV file."""

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_dataframe = pd.DataFrame(inference_results)
    results_dataframe.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")


def run_inference(execution_device, data_type):
    """Run one inference baseline configuration."""

    inference_model = None
    input_data = None
    generated_ids = None

    try:
        if execution_device.type == "cuda":
            prepare_cuda_memory_for_inference(execution_device)

        data_type_name = str(data_type).replace("torch.","")
        print(f"\nLoading model on {execution_device} " f"with {data_type}...")
        inference_model = (AutoModelForCausalLM.from_pretrained(MODEL_IDENTIFIER, dtype=data_type))
        inference_model.to(execution_device)
        inference_model.eval()
        input_data = tokenizer(PROMPT, return_tensors="pt").to(execution_device)
        input_token_count = (input_data["input_ids"].shape[-1])

        print(f"Model device: " f"{next(inference_model.parameters()).device}")
        print(f"Model dtype: " f"{next(inference_model.parameters()).dtype}")
        print(f"Input tokens: " f"{input_token_count}")
        pre_inference_warmup(inference_model, input_data)

        (generated_ids, inference_duration_seconds) = execute_inference_with_timing(inference_model, input_data, execution_device)
        generated_token_count = (generated_ids.shape[-1] - input_token_count)
        generated_tokens_per_second = (generated_token_count / inference_duration_seconds)
        memory_usage_bytes = (psutil.Process().memory_info().rss)
        memory_usage_megabytes = (memory_usage_bytes / (1024 ** 2))

        (
            gpu_device_name,
            vram_peak_allocated_megabytes,
            vram_reserved_megabytes,
        ) = collect_gpu_memory_metrics(
            execution_device
        )

        inference_result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        inference_metrics = {
            "model": MODEL_IDENTIFIER,
            "device": execution_device.type,
            "dtype": data_type_name,
            "gpu_device_name": gpu_device_name,
            "input_tokens": input_token_count,
            "output_tokens": generated_token_count,
            "elapsed_seconds": inference_duration_seconds,
            "tokens_per_second": generated_tokens_per_second,
            "ram_rss_mb": memory_usage_megabytes,
            "vram_peak_allocated_mb": (vram_peak_allocated_megabytes),
            "vram_reserved_mb": (vram_reserved_megabytes),
        }

        print_inference_results(inference_metrics, inference_result)

        return inference_metrics

    finally:
        del generated_ids
        del input_data
        del inference_model
        gc.collect()

        if execution_device.type == "cuda":
            with torch.cuda.device(execution_device):
                torch.cuda.empty_cache()


def main():
    """Run the CPU FP32, CUDA FP32 and CUDA FP16 baselines."""

    inference_results = []
    cpu_device = torch.device("cpu")
    inference_results.append(run_inference(cpu_device, torch.float32))

    if torch.cuda.is_available():
        cuda_device = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
        inference_results.append(run_inference(cuda_device, torch.float32))
        inference_results.append(run_inference(cuda_device, torch.float16))
    else:
        print("\nCUDA is not available.")

    save_inference_results_to_csv(inference_results)


if __name__ == "__main__":
    main()