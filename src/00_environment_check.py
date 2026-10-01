import platform
import sys


def main() -> int:
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}", file=sys.stderr)
        return 1

    cuda_available = torch.cuda.is_available()

    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only build'}")

    if not cuda_available:
        print("GPU: no CUDA device detected")
        return 0

    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)
        total_memory_gib = properties.total_memory / (1024 ** 3)

        print(f"GPU {device_index} name: {properties.name}")
        print(f"GPU {device_index} total memory: {total_memory_gib:.2f} GiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())