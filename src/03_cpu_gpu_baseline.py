import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in two sentences what tokenization is in language models."
MAX_NEW_TOKENS = 50
RESULTS_PATH = Path("results/cpu_gpu_baseline.csv")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def reset_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_inference(device, dtype):
    if device == "cuda":
        reset_cuda_memory()

    dtype_name = str(dtype).replace("torch.", "")

    print(f"\nLoading model on {device} with {dtype}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    model.to(device)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_tokens = inputs["input_ids"].shape[-1]

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Input tokens: {input_tokens}")

    print("Running warm-up...")

    with torch.inference_mode():
        warmup_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=5,
        )

    del warmup_ids

    if device == "cuda":
        torch.cuda.synchronize()

    print("Running measured inference...")

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_seconds = time.perf_counter() - start_time

    output_tokens = generated_ids.shape[-1] - input_tokens
    tokens_per_second = output_tokens / elapsed_seconds

    process = psutil.Process()
    ram_rss_mb = process.memory_info().rss / (1024 ** 2)

    gpu_name = ""
    vram_peak_allocated_mb = 0.0
    vram_reserved_mb = 0.0

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_peak_allocated_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
        vram_reserved_mb = (
            torch.cuda.memory_reserved() / (1024 ** 2)
        )

    decoded_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print(f"\n{device.upper()} / {dtype_name.upper()} results")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype_name}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Elapsed seconds: {elapsed_seconds:.4f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"RAM RSS MB: {ram_rss_mb:.2f}")
    print(f"VRAM peak allocated MB: {vram_peak_allocated_mb:.2f}")
    print(f"VRAM reserved MB: {vram_reserved_mb:.2f}")
    print(f"\nGenerated text:\n{decoded_text}")

    result = {
        "model": MODEL_NAME,
        "device": device,
        "dtype": dtype_name,
        "gpu_name": gpu_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "ram_rss_mb": ram_rss_mb,
        "vram_peak_allocated_mb": vram_peak_allocated_mb,
        "vram_reserved_mb": vram_reserved_mb,
    }

    del generated_ids
    del inputs
    del model

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


results = []

results.append(
    run_inference("cpu", torch.float32)
)

if torch.cuda.is_available():
    results.append(
        run_inference("cuda", torch.float32)
    )
    results.append(
        run_inference("cuda", torch.float16)
    )
else:
    print("\nCUDA is not available.")


df = pd.DataFrame(results)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    RESULTS_PATH,
    index=False,
)

print(f"\nResults saved to: {RESULTS_PATH}")