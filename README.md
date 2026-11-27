# AI Coding Tokenomics

An incremental engineering study exploring tokenization, inference performance, hardware utilization, numerical precision, cost, and benchmarking methodologies for AI-assisted software development.

## 1. Current Scope

The current version of the project covers two experimental stages:

- **Tokenization:** Examining how tokenizers process text, generate token IDs, and produce different token counts for the same input.

- **CPU/GPU Inference:** Establishing a local inference baseline using a pretrained causal language model. The experiments compare CPU FP32, CUDA FP32, and CUDA FP16 configurations based on generation time, overall token throughput, and memory usage.

The repository also includes an extended inference experiment and study notebooks covering Python and Pandas, tokenizer behavior, text preprocessing, and investigation of the inference baseline. The inference study includes an ongoing capacity and infrastructure-cost case study.

The project is being developed step by step, with experiments designed to be reproducible, measurements recorded, and findings documented.

## 2. Environment

The hardware and system details below are recorded references from the project, rather than a live inventory of the current installation. The [NVIDIA-SMI snapshot](results/nvidia-smi.txt), dated September 22, 2026, records the GPU capacity, driver, and CUDA version reported by that tool. The other details are documented in the [environment notes](docs/findings.md#historical-environment-snapshot) and the inference study notebook. These records were collected at different times and do not establish the environment of every benchmark run.

| Component        | Specification                    |
| ---------------- | -------------------------------- |
| Operating system | Ubuntu 24.04.5 LTS               |
| Python           | 3.12.3                           |
| CPU              | Intel Core i7-12800H             |
| GPU              | NVIDIA RTX A3000 12GB Laptop GPU |
| GPU memory       | 12 GiB (12,288 MiB)              |
| NVIDIA driver    | 595.91.07                        |
| CUDA reported by NVIDIA-SMI | 13.2                   |


### 2.1. Installation Requirements and Recorded Package Versions

The dependency files serve different purposes:

| File | Purpose |
| --- | --- |
| [requirements.txt](requirements.txt) | Installation specification for the scripts: pinned PyTorch, Transformers, psutil, and Pandas, with an additional PyTorch CUDA package index. |
| [requirements-dev.txt](requirements-dev.txt) | Includes the script requirements and pins JupyterLab, ipykernel, Matplotlib, and NumPy for notebook development. |
| [environment-freeze.txt](environment-freeze.txt) | Preserved package snapshot, including transitive dependencies. It is a record of the environment when captured, not a live inventory or the installation specification used in section 3.1. |

| Package | Installation requirements | Preserved package snapshot | October 9 benchmark record |
| --- | --- | --- | --- |
| PyTorch | `2.15.0+cu132` | `2.15.0` | `2.14.0+cu132` |
| Transformers | `5.22.2` | `5.22.2` | `5.17.0` |
| NumPy | `2.5.5` in development requirements | `2.5.4` | Not recorded in the benchmark CSV |

The requirements and snapshot differ: NumPy has different pinned versions, and the snapshot omits the PyTorch `+cu132` build identifier. Use the requirements files for the documented installation; the preserved snapshot is a separate reference and does not imply an identical environment.

The October 9 versions come from the [historical benchmark CSV](https://github.com/leandrocsiqueira/ai-coding-tokenomics/blob/9d12a8b312d48bf5daa1c3ee05f157b670aae341/results/cpu_gpu_baseline.csv). Updating requirements does not demonstrate that those measurements were reproduced. The simple-baseline CSV does not record dependency versions, so the historical environment cannot be assigned to it from that file.


## 3. How to Run The Project

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

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 3.1. Installing Dependencies

The installation specification pins `torch==2.15.0+cu132` in `requirements.txt`. This file also configures the [official CUDA 13.2 package index](https://download.pytorch.org/whl/cu132) for that build. Installing from the requirements therefore includes PyTorch; no separate installation is necessary for this setup.

To run the Python scripts, install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Alternatively, to work with the project's Jupyter notebooks, install the development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

The development requirements include all core project dependencies, along with JupyterLab, ipykernel, Matplotlib, and NumPy for the notebooks and their plots. Installing both requirements files separately is therefore unnecessary.

To start JupyterLab, run:

```bash
jupyter lab
```

The pinned CUDA-specific PyTorch build targets compatible NVIDIA GPU environments. To use another PyTorch distribution, including a CPU-only build, adapt the dependency specification following the [official installation instructions](https://pytorch.org/get-started/locally/). Applying the unchanged requirements afterward would still require the pinned CUDA build.

### 3.2. Environment Snapshot

Use `environment-freeze.txt` to inspect the preserved package versions, including development tools and transitive dependencies. A fresh installation should follow section 3.1; the snapshot and requirements are not interchangeable, as shown in section 2.1.

The snapshot does not include hardware, the operating system, NVIDIA drivers, or all system-level dependencies. Its `torch==2.15.0` entry also does not identify the `+cu132` build required by the installation specification. Reproducing a particular benchmark requires its own environment metadata, the appropriate package distribution and indexes, and compatible system dependencies. The snapshot alone does not establish reproducibility of the historical results.


### 3.3. Environment Validation

```bash
python src/00_environment_check.py
```

Checks the local environment and available hardware.


## 4. Experiments

### 4.1. Tokenization Basics

```bash
python src/01_tokenizer_basics.py
```

This script explores how tokenization works in language models using the tokenizer from [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). It processes different text samples, examines how they're converted into tokens and numerical identifiers (token IDs), and decodes them to reconstruct the original text. It also records character and token counts, calculates the average number of characters per token, and checks whether the original text is preserved after decoding. These steps help show how different text characteristics affect the way the tokenizer represents input.

### 4.2. Tokenizer Comparison

```bash
python src/02_compare_tokenizers.py
```

This script compares how different tokenizers process the same input texts. It uses the tokenizers from [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) to measure character counts, token counts, and the average number of characters per token. The results are displayed in a table and saved to a CSV file (`results/tokenizer_comparison.csv`), making it easier to identify differences in text segmentation and understand how the choice of tokenizer affects the number of tokens needed to represent the same content.


### 4.3. CPU and GPU Inference Benchmark

```bash
python src/03_cpu_gpu_baseline.py
```

This script establishes an initial inference performance baseline for [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) by comparing three execution configurations: CPU FP32, CUDA FP32, and CUDA FP16. For each available configuration, it loads the model, prepares the input, runs a warmup generation, and measures a separate generation run. The script records inference time, the number of generated tokens, effective generated-token throughput (tokens per second), process RAM usage, and PyTorch-managed GPU memory metrics. Results are displayed in the terminal and saved to a CSV file (`results/cpu_gpu_baseline.csv`), providing a basis for comparing inference performance and memory usage across hardware configurations and numerical precisions.

### 4.4. Extended CPU/GPU Inference Experiment

Run the [extended inference experiment](experiments/cpu_gpu_complete_test.py) from the repository root:

```bash
python experiments/cpu_gpu_complete_test.py
```

This experiment uses Qwen/Qwen2.5-0.5B-Instruct to compare CPU FP32 and, when CUDA is available, CUDA FP32 and CUDA FP16. Each configuration runs in a separate subprocess and uses a system-and-user chat template, one unmeasured warmup generation of up to 96 new tokens, and five measured generations with the same 96-token limit. Generation uses `do_sample=False` and can stop before reaching the limit.

The experiment records mean, median, and standard deviation of generation time, aggregate generated-token throughput, process RSS and PyTorch CUDA memory metrics, environment and protocol metadata, execution status, and the last generated response. Timings cover the complete `model.generate()` call, including prompt prefill and autoregressive decoding. Generated responses are retained for inspection; the experiment does not automatically evaluate response quality. Its protocol differs from the simple baseline described in section 4.3.

**Results file - current implementation:** Both inference scripts write to `results/cpu_gpu_baseline.csv`. The extended experiment preserves existing records and adds new rows and columns as needed. The simple baseline replaces the CSV, so running it afterward removes the accumulated records from that file. Preserve a copy before running either experiment if the existing results need to be retained. Improving result preservation and traceability in the simple baseline is planned in section 7 of the ongoing 01c inference study.

### 4.5. Study and Exploration Notebooks

After installing `requirements-dev.txt`, start `jupyter lab` from the repository root and open the relevant notebook:

| Notebook | Focus |
| --- | --- |
| [Python and Pandas Review](notebooks/01_python_pandas_review.ipynb) | Python and Pandas fundamentals used in the project, memory units, DataFrame indexes, and CSV export/import. |
| [Tokenizer Exploration](notebooks/02_tokenizer_exploration.ipynb) | Qwen tokenizer behavior, token IDs, internal token representations, decoding, offsets, and effects of input formatting. |
| [Token Counts and Text Preprocessing](experiments/token_counts_and_preprocessing.ipynb) | Comparison of Qwen, SmolLM2, mBERT, and mT5 tokenizers across text formats and preprocessing conditions, including checks on text preservation. This notebook counts tokens without running model inference. |
| [Inference Baseline Investigation - 01c](notebooks/03_estudo_inferencia_01c.ipynb) | Portuguese-language investigation of execution flow, generation settings, and measurement methodology, with an ongoing capacity and infrastructure-cost case study. |

**Language convention:** The 01c inference study notebook and commit messages for changes specific to that study are written in Portuguese. This README and the other notebooks listed above are written in English.

In the published 01c notebook, sections 1–3 have been developed, while sections 4–8 remain marked `TO-DO`. Its reference baseline is revision [`3a1be99`](https://github.com/leandrocsiqueira/ai-coding-tokenomics/tree/3a1be993041ed75f77f26f23b0cacf0aff507d30). The remaining activities cover model loading, interpretation of RAM and GPU memory, throughput and speedup calculations, result preservation and traceability, and the client case. The supporting [client case](notebooks/01c_Inferencia/docs/CASO_CLIENTE_INFRAESTRUTURA_01c.md) proposes measuring 24 synthetic samples and projecting the processing time and infrastructure cost for 10,000 clients; a final recommendation is still pending in the published notebook.


## 5. Experimental Results

**Status: Exploratory results with documented provenance.**

The project includes recorded tokenization and CPU/GPU inference measurements. Their sources, measurement protocols, observations, and limitations are documented in [`docs/findings.md`](docs/findings.md). The summaries below describe the available records; they do not establish general conclusions across models, workloads, or hardware.

The [`results/`](results/) directory contains the tokenizer comparison and simple-baseline CSVs. The October 9 inference record is preserved in the historical revision linked below. These records do not demonstrate that the measurements have been reproduced with the current dependency versions.

### 5.1. Inference Protocols and Result Provenance

The inference records represent two protocols using Qwen/Qwen2.5-0.5B-Instruct:

| Aspect | Simple baseline | Historical experiment - October 9, 2026 |
| --- | --- | --- |
| Prompt format | Direct tokenization | System-and-user chat template |
| Recorded input tokens | 13 | 44 |
| Warmup limit (new tokens) | 5 | 96 |
| Measured generations per configuration | 1 | 5 |
| Measured generation limit (new tokens) | 50 | 96 |
| Process isolation | Configurations share one process | Separate subprocess per configuration |
| CPU FP32 throughput (tokens/s) | 17.91 | 21.40 |
| CUDA FP32 throughput (tokens/s) | 28.20 | 46.15 |
| CUDA FP32 / CPU FP32 throughput ratio | Approximately **1.57x** | Approximately **2.16x** |

The simple-baseline figures come from the [CSV in the current tree](results/cpu_gpu_baseline.csv). The October 9 figures come from the [historical CSV at revision `9d12a8b`](https://github.com/leandrocsiqueira/ai-coding-tokenomics/blob/9d12a8b312d48bf5daa1c3ee05f157b670aae341/results/cpu_gpu_baseline.csv). The expanded implementation is maintained in `experiments/cpu_gpu_complete_test.py`; the historical figures do not represent a new execution of that script with the current dependencies.

Each ratio compares CUDA FP32 with CPU FP32 within its own protocol, using the recorded throughput values before rounding. The difference between 1.57x and 2.16x does not establish a performance improvement between benchmark versions: prompt format, generation limits, repetition count, and process isolation changed. Throughput covers the complete measured `model.generate()` call, including prompt prefill and autoregressive decoding.

These are exploratory results: the simple baseline contains one measured generation per configuration, and the historical record contains one benchmark session with five measured generations per configuration. See the [result provenance and limitations](docs/findings.md#result-provenance) for the detailed analysis.

### 5.2. Selected Findings and Limits

- **Tokenizer comparison:** Across the five inputs in the [comparison CSV](results/tokenizer_comparison.csv), Qwen produced 36 token IDs and SmolLM2 produced 48. This observation applies to these inputs and tokenizers; it does not establish a universal efficiency ranking or the cost of a billed API request.
- **Numerical precision:** In the October 9 experiment, CUDA FP16 reduced peak memory allocated by the PyTorch CUDA allocator by approximately 49% relative to CUDA FP32. Throughput was 45.69 tokens/s for FP16 and 46.15 tokens/s for FP32, so this execution did not demonstrate a meaningful speed advantage from FP16. The memory metric does not represent total device VRAM usage.
- **Response quality:** The inference experiments measure performance and memory without a dedicated evaluation of correctness or response quality. Those aspects remain to be evaluated before drawing conclusions about practical output quality.

The [detailed findings](docs/findings.md) distinguish observations from hypotheses and identify further measurements needed to test their reliability, including independent benchmark sessions and additional workloads.


## 6. Roadmap

The project will evolve incrementally through the following areas:

1. Tokenization experiments and tokenizer comparison.
2. CPU/GPU inference baseline and performance measurements.
3. Numerical precision and memory analysis.
4. Model comparison across different workloads.
5. AI-assisted coding benchmarks.
6. Cost and output-quality analysis.
7. Reproducible benchmarking and reporting.

Future experiments will focus on improving measurement reliability, evaluating additional model configurations, and connecting inference performance to computational cost and practical software development workloads.

## 7. Project Status

**Current stage:** Tokenization and CPU/GPU inference experiments implemented, with exploratory measurements, documented provenance and limitations, and an ongoing inference study covering capacity and infrastructure cost.

The repository remains under active development. Existing findings reflect the experimental conditions and available measurements and may be refined as the methodology evolves.
