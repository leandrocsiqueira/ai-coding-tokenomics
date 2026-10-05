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

## CPU/GPU Inference Findings

**GPU provided a real but moderate speedup:** CUDA FP32 reached 28.20 tokens/s, compared with 17.91 tokens/s on CPU FP32, which corresponds to a speedup of about 1.57x. Total generation time dropped from 2.7921 seconds to 1.7730 seconds, a reduction of about 36.5%. This's a meaningful improvement, but not a dramatic one. With a 0.5B-parameter model, batch size 1, and autoregressive generation, the GPU's likely running below full utilization, so execution overhead and memory access costs make up a relatively large share of the total runtime.

**FP16 didn't provide a measurable speed advantage over FP32:** CUDA FP16 reached 27.19 tokens/s, compared with 28.20 tokens/s for CUDA FP32, making FP16 about 3.6% slower in this run. This doesn't mean FP16 is inherently slower. With such a small model, batch size 1, and short autoregressive generation, raw arithmetic throughput may not be the main bottleneck. Kernel launch overhead, low GPU occupancy, memory access, and the sequential nature of token generation can reduce the speed advantage normally associated with lower numerical precision.

**The clearest FP16 benefit was lower VRAM usage:** Peak allocated VRAM dropped from 1897.18 MB in FP32 to 961.24 MB in FP16, a reduction of about 49.3%. Reserved VRAM also fell from 2024.00 MB to 1014.00 MB, or about 49.9%. This's consistent with the expected storage difference between 32-bit and 16-bit floating-point values. Peak allocated memory's the more direct measure of tensor memory usage, while reserved memory also includes memory held by PyTorch's CUDA caching allocator.

**The generated text was identical across all three configurations:** CPU FP32, CUDA FP32, and CUDA FP16 produced the same decoded output for this prompt. Since generation used greedy decoding with `do_sample=False`, the numerical differences between FP32 and FP16 weren't large enough to change the generated text in this particular run.

**Higher process RAM usage doesn't demonstrate a memory leak:** RSS increased from 2749.77 MB on CPU to 2870.36 MB on CUDA FP32 and 3324.46 MB on CUDA FP16. However, these values were measured while the current model, inputs, and generated tensors were still alive. Cleanup with `del`, `gc.collect()`, and `torch.cuda.empty_cache()` happens afterward in the finally block. CUDA initialization and persistent library resources may also remain in memory after the first GPU run. A proper memory-retention test would need to measure RSS again after cleanup for each configuration.

**A single run isn't enough to establish a small performance difference:** Each configuration was measured only once, and the GPU runs lasted less than two seconds. At that scale, normal variation from OS scheduling, GPU clock behavior, thermal conditions, and other runtime effects could easily explain a difference of a few percent. The current results therefore support the conclusion that no meaningful FP32-versus-FP16 speed difference was observed, rather than the stronger claim that FP16 is slower.

**The short warm-up may also affect measurement stability:** The benchmark performs only a 5-token warm-up before measuring a 50-token generation run. This's enough to avoid measuring a completely cold first execution, but it may not be enough to fully stabilize GPU clocks, kernel behavior, or other runtime effects. Repeating the benchmark several times after a longer warm-up would produce a more reliable comparison.

**The workload's too small to generalize about FP16 performance:** The experiment uses a 0.5B-parameter model, batch size 1, and only 50 generated tokens. Under these conditions, GPU utilization may remain relatively low. Larger models, larger batch sizes, and different sequence lengths could produce a very different FP32-versus-FP16 relationship. More tests would be needed before drawing broader conclusions about precision-related performance.

**The token counts make the three runs directly comparable:** All three configurations processed the same 13 input tokens and generated exactly 50 new tokens. The script calculates generated tokens as the final sequence length minus the input length and then uses that value to calculate tokens per second. This's what keeps the amount of measured generation work constant across the configurations. Although `max_new_tokens=50` defines only an upper limit, all three runs actually reached that limit.

**Batching would mainly target throughput rather than single-request latency:** Larger batch sizes and continuous batching can improve GPU utilization by processing more work in parallel. This's especially useful when the goal is higher aggregate throughput across multiple requests.