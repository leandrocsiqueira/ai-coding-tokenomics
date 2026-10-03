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
## Tokenization Comparison Findings

**Different tokenizers, different token counts:** Different tokenizers can produce different token counts for the exact same text. For example, `Olá, meu nome é Leandro.` was split into 9 tokens by Qwen and 11 by SmolLM2, while `tokenização` was split into 2 and 3 tokens, respectively. This shows that tokenization depends on the model: there's no universal token count for a given piece of text.

**Overall tokenization density:** Across the five test inputs, Qwen generated 36 tokens in total, compared with 48 for SmolLM2. For this particular sample, SmolLM2 therefore used about 33% more tokens than Qwen. This doesn't mean Qwen is always more token-efficient, but it does show that its tokenizer represented these specific inputs more compactly.

**Code tokenization:** The Python code sample showed one of the largest absolute differences between the two tokenizers. The 54-character input took 10 tokens with Qwen and 15 with SmolLM2. That works out to 5.40 characters per token for Qwen versus 3.60 for SmolLM2, meaning Qwen represented this particular code sample using fewer token pieces.

**Special characters and accented text:** The input `🚀 Inteligência Artificial!` produced 6 tokens with Qwen and 10 with SmolLM2, the largest relative difference in the experiment. SmolLM2 used about 67% more tokens for this input. This suggests that differences in tokenizer vocabulary and segmentation can become especially noticeable when accented text and special characters appear together, although more examples would be needed to determine exactly which parts of the input caused the difference.

**Equal token counts don't necessarily mean equal tokenization:** `João Paulo Peçanha Navarro` produced exactly 9 tokens with both tokenizers. However, matching token counts don't necessarily mean the text was split at the same positions or into the same token pieces. We'd need to inspect the actual token representations before concluding that the two tokenizers handled this input in the same way.

**Implications for token-based cost:** Different tokenizers can represent the same input using different numbers of tokens, which can affect computational workload and token-based API usage. Still, fewer tokens don't automatically mean a model or service is cheaper. The final cost also depends on factors such as the provider's input and output token pricing, the model architecture, and the underlying inference infrastructure.

## CPU vs GPU

The model fit entirely on the NVIDIA RTX A3000 12GB Laptop GPU.

CPU/FP32 generated 50 tokens in 2.56 seconds, reaching approximately 19.53 tokens per second. GPU/FP32 generated the same 50 tokens in 1.07 seconds, reaching approximately 46.63 tokens per second. In this run, GPU/FP32 was about 2.39 times faster than CPU/FP32.

GPU/FP16 generated 50 tokens in 1.09 seconds, reaching approximately 46.02 tokens per second. In this short benchmark, FP16 did not provide a meaningful throughput improvement over FP32.

The main difference between GPU/FP32 and GPU/FP16 was memory usage. Peak allocated VRAM decreased from approximately 1897 MB in FP32 to 961 MB in FP16, a reduction of about 49%.

The model was confirmed to be running on CUDA by checking `next(model.parameters()).device`, which reported `cuda:0`.

If the model did not fit in GPU memory, I would investigate a smaller model, FP16 or BF16, quantization, or CPU/GPU offload.
