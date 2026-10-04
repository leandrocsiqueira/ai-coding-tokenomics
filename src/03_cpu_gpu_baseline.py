import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDENTIFIER = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain in two sentences what tokenization is in language models."
WARMUP_MAX_NEW_TOKENS = 5
MAX_NEW_TOKENS = 50


tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)


def pre_inference_warmup(inference_model, input_data, execution_device):
    """Run a short warm-up generation before measurement."""

    print("Running warm-up...")
    
    with torch.inference_mode():
        generated_warmup_ids = inference_model.generate(**input_data, do_sample=False, 
                                                        max_new_tokens=WARMUP_MAX_NEW_TOKENS)
    
    del generated_warmup_ids
    
    if execution_device == "cuda":
        torch.cuda.synchronize()


def execute_inference_with_timing(inference_model, input_data, device_type):
    """Run the measured generation and return its output and elapsed time."""
    
    print("Running measured inference...")
    if device_type == "cuda":
        torch.cuda.synchronize()

    inference_start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = inference_model.generate(**input_data, do_sample = False, max_new_tokens = MAX_NEW_TOKENS)
    
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time_seconds = time.perf_counter() - inference_start_time

    return generated_ids, elapsed_time_seconds


def print_inference_results(device_type, data_type, input_token_count, generated_token_count, 
                           elapsed_time_seconds, tokens_processed_per_second, ram_usage_megabytes, 
                           inference_result):
    """Print inference metrics and generated text."""

    print(f"\n{device_type.upper()} / {data_type.upper()} results")
    print(f"Model: {MODEL_IDENTIFIER}")
    print(f"Device Type: {device_type}")
    print(f"Data Type: {data_type}")
    print(f"Input Tokens Count: {input_token_count}")
    print(f"Generated Tokens Count: {generated_token_count}")
    print(f"Elapsed Time (seconds): {elapsed_time_seconds:.4f}")
    print(f"Generated Tokens per Second: {tokens_processed_per_second:.2f}")
    print(f"RAM Usage (MB): {ram_usage_megabytes:.2f}")
    print(f"\nInference Result:\n{inference_result}")


def run_inference(device_type, data_type):
    """Run one inference baseline configuration."""

    data_type_name = str(data_type).replace("torch.", "")
    print(f"\nLoading model on {device_type} with {data_type}...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_IDENTIFIER, dtype=data_type)
    model.to(device_type)
    model.eval()
    input_tensor = tokenizer(PROMPT, return_tensors="pt").to(device_type)
    input_token_count = input_tensor["input_ids"].shape[-1]

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Input tokens: {input_token_count}")

    pre_inference_warmup(model, input_tensor, device_type)
    generated_ids, inference_duration_seconds = execute_inference_with_timing(model, input_tensor, device_type)
    generated_token_count = generated_ids.shape[-1] - input_token_count
    tokens_processed_per_second = generated_token_count / inference_duration_seconds
    memory_usage_bytes = psutil.Process().memory_info().rss
    memory_usage_mb = memory_usage_bytes/ (1024 ** 2) # Convert bytes to megabytes
    inference_result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print_inference_results(device_type, data_type_name, input_token_count, generated_token_count, 
                            inference_duration_seconds, tokens_processed_per_second, memory_usage_mb, 
                            inference_result)

def main():
    """Run the CPU and CUDA FP32 baselines."""

    run_inference("cpu", torch.float32)
    if torch.cuda.is_available():
        run_inference("cuda", torch.float32)
    else:
        print("\nCUDA is not available.")

if __name__ == "__main__":
    main()