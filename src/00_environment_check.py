import sys

import torch
import transformers


print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA build: {torch.version.cuda}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory_bytes / (1024 ** 3)

    print(f"GPU name: {gpu_name}")
    print(f"GPU total memory: {total_memory_gb:.2f} GB")
else:
    print("GPU name: CUDA not available")
    print("GPU total memory: CUDA not available")