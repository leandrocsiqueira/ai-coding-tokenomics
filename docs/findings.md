## Environment

- Operating system: Ubuntu 24.04.5 LTS
- Python: 3.12.3
- PyTorch: 2.14.0+cu132
- Transformers: 5.17.0
- CUDA available in PyTorch: True
- GPU: NVIDIA RTX A3000 12GB Laptop GPU
- Total VRAM: 11.63 GB
- NVIDIA driver: 595.91.07
- CUDA reported by nvidia-smi: 13.2
- PyTorch CUDA build: 13.2

## Tokenization findings

A token does not always correspond to a complete word. For example, "tokenização" was represented by 2 tokens, while "João Paulo Peçanha Navarro" required 9 tokens.

Spaces, punctuation, and accented characters affected tokenization. In "Olá, meu nome é Leandro.", punctuation and Portuguese characters appeared as separate or encoded token fragments, resulting in 9 tokens for 24 characters.

Portuguese words can be split into multiple tokens. "tokenização" was split into 2 tokens, and "Inteligência" was represented by multiple token fragments.

The code sample was tokenized around meaningful code fragments such as "def", "calculate", "_total", "_cost", "_tokens", punctuation, and identifiers. The 54-character code sample resulted in 10 tokens.

The rocket emoji was represented by its own encoded token representation. Although the token representation was not human-readable in the terminal, decoding reconstructed the original emoji correctly. The complete text "🚀 Inteligência Artificial!" used 6 tokens.

## Tokenizer comparison findings

The same text does not necessarily use the same number of tokens with different tokenizers. For example, "Olá, meu nome é Leandro." required 9 tokens with Qwen and 11 with SmolLM2. The code sample required 10 tokens with Qwen and 15 with SmolLM2.

Across the five test texts, Qwen generated 36 tokens in total, while SmolLM2 generated 48.

The largest difference occurred in the code sample, where Qwen generated 10 tokens and SmolLM2 generated 15.

An interesting result was "João Paulo Peçanha Navarro", which required exactly 9 tokens with both tokenizers, even though the other texts produced different token counts.

Tokenization can affect cost because processing more tokens may require more computation and, in token-priced services, can increase usage. However, token count alone is not enough to compare the final cost of different models because pricing per token may also differ.

## CPU vs GPU

The model fit entirely on the NVIDIA RTX A3000 12GB Laptop GPU.

CPU/FP32 generated 50 tokens in 2.56 seconds, reaching approximately 19.53 tokens per second. GPU/FP32 generated the same 50 tokens in 1.07 seconds, reaching approximately 46.63 tokens per second. In this run, GPU/FP32 was about 2.39 times faster than CPU/FP32.

GPU/FP16 generated 50 tokens in 1.09 seconds, reaching approximately 46.02 tokens per second. In this short benchmark, FP16 did not provide a meaningful throughput improvement over FP32.

The main difference between GPU/FP32 and GPU/FP16 was memory usage. Peak allocated VRAM decreased from approximately 1897 MB in FP32 to 961 MB in FP16, a reduction of about 49%.

The model was confirmed to be running on CUDA by checking `next(model.parameters()).device`, which reported `cuda:0`.

If the model did not fit in GPU memory, I would investigate a smaller model, FP16 or BF16, quantization, or CPU/GPU offload.
