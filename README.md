# AI Coding Tokenomics

An incremental engineering study exploring tokenization, inference performance, hardware utilization, numerical precision, cost, and benchmarking methodologies for AI-assisted software development.

## 1. Current Scope

The current version of the project covers two experimental stages:

- **Tokenization:** Examining how tokenizers process text, generate token IDs, and produce different token counts for the same input.

- **CPU/GPU Inference:** Establishing a local inference baseline using a pretrained causal language model. The experiments compare CPU FP32, CUDA FP32, and CUDA FP16 configurations based on generation time, overall token throughput, and memory usage.

The repository also includes an extended inference experiment and study notebooks covering Python and Pandas, tokenizer behavior, text preprocessing, and investigation of the inference baseline. The inference study includes an ongoing capacity and infrastructure-cost case study.

The project is being developed step by step, with experiments designed to be reproducible, measurements recorded, and findings documented.

## 2. Environment

The experiments were conducted using the hardware and operating system listed below. The Python package versions match the current environment snapshot saved in `environment-freeze.txt`.

| Component        | Specification                    |
| ---------------- | -------------------------------- |
| Operating system | Ubuntu 24.04.5 LTS               |
| Python           | 3.12.3                           |
| CPU              | Intel Core i7-12800H             |
| GPU              | NVIDIA RTX A3000 12GB Laptop GPU |
| GPU memory       | 12 GiB (12,288 MiB)              |
| NVIDIA driver    | 595.91.07                        |
| CUDA version     | 13.2                             |
| PyTorch          | 2.15.0+cu132                     |
| Transformers     | 5.22.2                           |


The project uses three files to manage and document its Python dependencies:

| File                     | Description                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `requirements.txt`       | Lists the direct dependencies needed to run the project.                                           |
| `requirements-dev.txt`   | Includes the project dependencies and additional tools used for notebook development.              |
| `environment-freeze.txt` | Records all installed Python packages and their exact versions, including transitive dependencies. |


PyTorch can be installed separately by following the [official installation instructions](https://pytorch.org/get-started/locally/) and selecting a build compatible with the target CPU or CUDA setup.


**Note:** The current environment snapshot lists PyTorch 2.15.0+cu132 and Transformers 5.22.2, replacing the previously documented versions of 2.14.0+cu132 and 5.17.0, respectively. The freeze records PyTorch as torch==2.15.0, while the installed build reports the additional +cu132 identifier. This update doesn't mean that earlier benchmark results were reproduced using the newer versions.


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

The project requires PyTorch 2.15.0 with CUDA 13.2 support, as specified in `requirements.txt`. This file also configures the [official CUDA 13.2 package index](https://download.pytorch.org/whl/cu132), where the pinned `torch==2.15.0+cu132` build is hosted. PyPI alone does not provide this CUDA-specific version, so the additional index is required to avoid a `No matching distribution found` error. No separate PyTorch installation is necessary.

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

The pinned CUDA-specific PyTorch build targets compatible NVIDIA GPU environments. CPU-only environments may require a different PyTorch distribution and a corresponding adjustment to the dependency specification.

### 3.2. Environment Snapshot

The `environment-freeze.txt` file records the installed Python packages and their versions, including development tools and transitive dependencies.

To install the recorded package versions on a compatible system, run:

```bash
python -m pip install -r environment-freeze.txt
```

This file provides a reference for the software environment and helps support reproducibility. However, it doesn't include details about the operating system, NVIDIA drivers, hardware, or all system-level dependencies.

The snapshot also doesn't retain the `+cu132` PyTorch build identifier. To reproduce the CUDA environment, the corresponding PyTorch build must be obtained or verified separately.


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

**Results file — current implementation:** Both inference scripts write to `results/cpu_gpu_baseline.csv`. The extended experiment preserves existing records and adds new rows and columns as needed. The simple baseline replaces the CSV, so running it afterward removes the accumulated records from that file. Preserve a copy before running either experiment if the existing results need to be retained. Improving result preservation and traceability in the simple baseline is planned in section 7 of the ongoing 01c inference study.

### 4.5. Study and Exploration Notebooks

After installing `requirements-dev.txt`, start `jupyter lab` from the repository root and open the relevant notebook:

| Notebook | Focus |
| --- | --- |
| [Python and Pandas Review](notebooks/01_python_pandas_review.ipynb) | Python and Pandas fundamentals used in the project, memory units, DataFrame indexes, and CSV export/import. |
| [Tokenizer Exploration](notebooks/02_tokenizer_exploration.ipynb) | Qwen tokenizer behavior, token IDs, internal token representations, decoding, offsets, and effects of input formatting. |
| [Token Counts and Text Preprocessing](experiments/token_counts_and_preprocessing.ipynb) | Comparison of Qwen, SmolLM2, mBERT, and mT5 tokenizers across text formats and preprocessing conditions, including checks on text preservation. This notebook counts tokens without running model inference. |
| [Inference Baseline Investigation — 01c](notebooks/03_estudo_inferencia_01c.ipynb) | Portuguese-language investigation of execution flow, generation settings, and measurement methodology, with an ongoing capacity and infrastructure-cost case study. |

**Language convention:** The 01c inference study notebook and commit messages for changes specific to that study are written in Portuguese. This README and the other notebooks listed above are written in English.

In the published 01c notebook, sections 1–3 have been developed, while sections 4–8 remain marked `TO-DO`. Its reference baseline is revision [`3a1be99`](https://github.com/leandrocsiqueira/ai-coding-tokenomics/tree/3a1be993041ed75f77f26f23b0cacf0aff507d30). The remaining activities cover model loading, interpretation of RAM and GPU memory, throughput and speedup calculations, result preservation and traceability, and the client case. The supporting [client case](notebooks/01c_Inferencia/docs/CASO_CLIENTE_INFRAESTRUTURA_01c.md) proposes measuring 24 synthetic samples and projecting the processing time and infrastructure cost for 10,000 clients; a final recommendation is still pending in the published notebook.


## 5. Experimental Results

**Status: Pending Review**

The project includes experimental results from tokenization and CPU/GPU inference benchmarks. However, these results and their associated analyses have not yet been fully reviewed against the corresponding script versions and execution methodologies.

Once the experimental records have been verified, this section will consolidate the quantitative comparisons, observations, and conclusions.

Existing results are available in the [`results/`](results/) directory. Additional findings are documented in [`docs/findings.md`](docs/findings.md) and should be considered preliminary until reviewed.


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

**Current stage:** Initial tokenization and CPU/GPU inference experiments implemented, with experimental results pending review and consolidation.

The repository remains under active development. Existing findings reflect the experimental conditions and available measurements and may be refined as the methodology evolves.
