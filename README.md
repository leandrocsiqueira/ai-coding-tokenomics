# AI Coding Tokenomics

An incremental engineering study exploring tokenization, inference performance, hardware utilization, numerical precision, cost, and benchmarking methodologies for AI-assisted software development.

## Current Scope

The current version focuses on two experimental stages:

- **Tokenization:** Investigating how tokenizers represent text, generate token IDs, and produce different token counts for identical inputs.

- **CPU/GPU inference:** Establishing a local inference baseline using a pretrained causal language model, comparing CPU FP32, CUDA FP32, and CUDA FP16 execution in terms of generation time, aggregate token throughput, and memory usage.

The project is being developed incrementally, with reproducible experiments, recorded measurements, and documented findings.

## Environment

The experiments were executed using the following environment:

| Component | Specification |
|---|---|
| Operating system | Ubuntu 24.04.5 LTS |
| Python | 3.12.3 |
| CPU | Intel Core i7-12800H |
| GPU | NVIDIA RTX A3000 12GB Laptop GPU |
| GPU memory | 12 GiB (12,288 MiB) |
| NVIDIA driver | 595.91.07 |
| CUDA version | 13.2 |
| PyTorch | 2.14.0+cu132 |
| Transformers | 5.17.0 |

PyTorch was installed separately using the official installation selector for the appropriate CUDA environment.

## How to Run

Clone the repository and enter the project directory:

```bash
git clone https://github.com/leandrocsiqueira/ai-coding-tokenomics.git
cd ai-coding-tokenomics
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch separately using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/), selecting a compatible configuration for the target environment.

### Environment Validation

```bash
python src/00_environment_check.py
```

Checks the local environment and available hardware.

### Experiment 01: Tokenizer Basics

```bash
python src/01_tokenizer_basics.py
```

Investigates tokenization behavior using a pretrained tokenizer, including token IDs, token representations, token counts, and text reconstruction.

### Experiment 02: Tokenizer Comparison

```bash
python src/02_compare_tokenizers.py
```

Compares how different tokenizers process identical inputs and records differences in token counts and tokenization density.

Results are saved to:

```text
results/tokenizer_comparison.csv
```

### Experiment 03: CPU/GPU Inference Baseline

```bash
python src/03_cpu_gpu_baseline.py
```

Executes inference using `Qwen/Qwen2.5-0.5B-Instruct` across the following configurations, depending on CUDA availability:

| Configuration | Device | Data type |
|---|---|---|
| CPU FP32 | CPU | float32 |
| CUDA FP32 | GPU | float32 |
| CUDA FP16 | GPU | float16 |

Each configuration runs in an isolated spawned subprocess, with one unmeasured warm-up and five measured generation iterations.

The benchmark uses greedy decoding (`do_sample=False`) and a structured chat template. CUDA execution is synchronized before and after timing to account for asynchronous GPU operations.

Measurements include generation duration, aggregate token throughput, process RAM usage, and PyTorch CUDA allocator memory statistics.

Results are saved to:

```text
results/cpu_gpu_baseline.csv
```

The CSV preserves historical records and stores measurement metadata to support interpretation and subsequent analysis.

## Experimental Results

### Tokenization Comparison

Qwen and SmolLM2 produced different token counts when processing identical inputs.

Across five test texts, Qwen generated 36 tokens in total, compared with 48 tokens for SmolLM2.

For this sample, SmolLM2 used approximately 33% more tokens than Qwen.

These observations demonstrate differences in tokenization behavior for the tested inputs. They do not establish general tokenization efficiency or overall model performance.

### CPU/GPU Inference Baseline

The latest documented benchmark was executed on October 9, 2026, using `Qwen/Qwen2.5-0.5B-Instruct`.

Each configuration processed 44 input tokens and generated 36 new token IDs per measured iteration. All measured generations terminated with an EOS token before reaching the 96-token limit.

#### Inference Performance

| Configuration | Mean generate() time (s) | Median time (s) | Aggregate throughput (tokens/s) |
|---|---:|---:|---:|
| CPU FP32 | 1.6820 | 1.6003 | 21.40 |
| CUDA FP32 | 0.7801 | 0.7702 | 46.15 |
| CUDA FP16 | 0.7880 | 0.7838 | 45.69 |

CUDA FP32 achieved approximately **2.16x the aggregate throughput** of CPU FP32, reducing mean generation time by approximately 53.6%.

CUDA FP16 achieved similar throughput to CUDA FP32, with an observed difference of approximately 1%. This execution did not demonstrate a meaningful speed advantage from using FP16.

**Timing scope:** Inference times include the complete `model.generate()` call, covering prompt prefill and autoregressive decoding. Model loading, tokenization, warm-up, and final text decoding are excluded.

Aggregate throughput is calculated as total newly generated token IDs divided by the summed full-call generation time. It does not represent decode-only throughput.

#### GPU Memory Usage

| PyTorch CUDA metric | FP32 (MB) | FP16 (MB) |
|---|---:|---:|
| Peak allocated memory | 1989.76 | 1007.96 |
| Peak reserved memory | 2122.32 | 1063.26 |

FP16 reduced peak PyTorch-allocated GPU memory by approximately **49.3%** and peak reserved memory by approximately **49.9%**.

These measurements represent PyTorch CUDA allocator statistics, not total GPU VRAM usage. Reserved memory includes cached allocator blocks, and allocations outside PyTorch may not be captured.

#### Interpretation and Limitations

The current baseline demonstrates a substantial GPU performance advantage over CPU execution and a significant reduction in PyTorch-managed GPU memory when using FP16.

However, the findings are limited to one model, one prompt, batch size 1, and a single complete benchmark execution containing five measured iterations per configuration.

The experiment does not separately measure time to first token, inter-token latency, GPU utilization, or total device memory usage.

Generated responses are retained for inspection, but the benchmark does not evaluate factual correctness, instruction following, semantic equivalence, or response quality.

The results should therefore be interpreted as exploratory baseline measurements rather than definitive performance comparisons.

## Results and Documentation

The repository contains the following experimental artifacts:

| File | Description |
|---|---|
| `results/tokenizer_comparison.csv` | Tokenizer comparison measurements |
| `results/cpu_gpu_baseline.csv` | Inference performance, memory metrics, and execution metadata |
| `results/nvidia-smi.txt` | NVIDIA hardware and driver information |
| `docs/findings.md` | Detailed experimental observations, interpretation, and limitations |
| `environment-freeze.txt` | Recorded Python environment dependencies |

Detailed findings and open questions are documented in [docs/findings.md](docs/findings.md).

## Roadmap

The project will evolve incrementally through the following areas:

1. Tokenization experiments and tokenizer comparison.
2. CPU/GPU inference baseline and performance measurements.
3. Numerical precision and memory analysis.
4. Model comparison across different workloads.
5. AI-assisted coding benchmarks.
6. Cost and output-quality analysis.
7. Reproducible benchmarking and reporting.

Future experiments will focus on improving measurement reliability, evaluating additional model configurations, and connecting inference performance to computational cost and practical software development workloads.

## Project Status

**Current stage:** Tokenization experiments and local CPU/GPU inference baseline completed.

The repository remains under active development. Findings reflect the experimental conditions and available measurements and may be refined as the methodology evolves.
