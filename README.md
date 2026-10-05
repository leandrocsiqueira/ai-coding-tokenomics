# AI Coding Tokenomics

An incremental engineering study exploring tokenization, inference performance, hardware utilization, cost and benchmarking methodologies for AI-assisted software development.

## Current scope

The current version investigates how different tokenizers represent the same text and establishes a local inference baseline comparing CPU/FP32, GPU/FP32, and GPU/FP16 execution.

## Environment

The experiments were executed on Ubuntu 24.04.5 LTS with Python 3.12.3, PyTorch 2.14.0+cu132, Transformers 5.17.0, and an NVIDIA RTX A3000 12GB Laptop GPU.

PyTorch was installed separately using the official installation selector for the appropriate CUDA environment.

## How to run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch must be installed separately using the official PyTorch installation selector for the appropriate CUDA environment.

Run the environment validation:

```bash
python src/00_environment_check.py
```

Run the basic tokenizer experiment:

```bash
python src/01_tokenizer_basics.py
```

Run the tokenizer comparison:

```bash
python src/02_compare_tokenizers.py
```

Run the CPU/GPU inference baseline:

```bash
python src/03_cpu_gpu_baseline.py
```

## Results

Tokenizer comparison results are stored in:

```text
results/tokenizer_comparison.csv
```

CPU/GPU inference results are stored in:

```text
results/cpu_gpu_baseline.csv
```

The NVIDIA environment snapshot is stored in:

```text
results/nvidia-smi.txt
```

## Current findings

Qwen and SmolLM2 produced different token counts for the same inputs. Across the five test texts, Qwen generated 36 tokens while SmolLM2 generated 48.

In the current local inference baseline, CPU/FP32 reached approximately 17.91 tokens per second, while GPU/FP32 reached approximately 28.20 tokens per second.

GPU/FP16 reached approximately 27.19 tokens per second. Its throughput was similar to GPU/FP32 in this run, while peak allocated VRAM decreased from approximately 1897 MB to 961 MB.

These measurements represent a single baseline run under the current experimental conditions. They should not be interpreted as definitive performance benchmarks.

Detailed observations and limitations are documented in `docs/findings.md`.

## Roadmap

Future iterations will expand the study with additional benchmarking methodology, hardware utilization analysis, and cost-oriented experiments.