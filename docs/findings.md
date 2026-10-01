## Environment
- **Operating system:** `Ubuntu 24.04.5 LTS`
- **Python:** `3.12.3`
- **PyTorch:** `2.14.0+cu132`
- **Transformers:** `5.17.0`
- **CUDA available in PyTorch:** `True`
- **GPU:** `NVIDIA RTX A3000 12GB Laptop GPU`
- **Total VRAM:** `11.63 GB`
- **NVIDIA driver:** `595.91.07`
- **CUDA (as reported by `nvidia-smi`):** `13.2`
- **PyTorch CUDA build**: `13.2`
---

## Tokenization Findings

- **Density:** The relationship between character count and token count varied significantly across the tested inputs. For example, `Olá, meu nome é Leandro.` contained 24 characters and produced 9 tokens (2.67 chars/token), while `🚀 Inteligência Artificial!` had two more characters but produced only 6 tokens (4.33 chars/token). This shows that character count alone doesn't determine token count, as different text patterns can result in substantially different tokenization densities.

- **Token representation vs decoded text:** The experiment highlighted the difference between a tokenizer's internal token representation and the final decoded text. For example, token strings associated with accented text appeared in forms like `Ã¡`, `Ã£o`, and `Ãªncia`. Even so, `decode()` correctly reconstructed the original text. This shows that the output of `convert_ids_to_tokens()` should not necessarily be interpreted as human-readable text. `convert_ids_to_tokens()` is useful for inspecting the tokenizer's internal token pieces, while `decode()` should be used when reconstructing text from token IDs.

- **Python code tokenization:** The Python code sample was tokenized relatively compactly. The 54-character expression resulted in just 10 tokens, including `def`, `Ġcalculate`, `_total`, `_cost`, `_tokens`, and `):`. This suggests that the tokenizer's vocabulary contains representations for several common code fragments and identifier substrings, allowing this particular code sample to be represented with relatively few tokens.

- **Implications for token-based cost:** Texts with similar character counts can still result in different numbers of tokens, which can affect the cost of APIs that charge based on token usage. However, this experiment doesn't show that Portuguese is necessarily more expensive to process than English or code. To make a fair comparison, you'd need equivalent texts in each language, along with the pricing rules of the specific service being evaluated.

---
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
