import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in two sentences what tokenization is in language models."
MAX_NEW_TOKENS = 50


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def run_inference(device, dtype):
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
        _ = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=5,
        )

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

    decoded_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print(f"\n{device.upper()} / {str(dtype).replace('torch.', '').upper()} results")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Dtype: {str(dtype).replace('torch.', '')}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Elapsed seconds: {elapsed_seconds:.4f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"RAM RSS MB: {ram_rss_mb:.2f}")
    print(f"\nGenerated text:\n{decoded_text}")


run_inference("cpu", torch.float32)

if torch.cuda.is_available():
    run_inference("cuda", torch.float32)
else:
    print("\nCUDA is not available.")