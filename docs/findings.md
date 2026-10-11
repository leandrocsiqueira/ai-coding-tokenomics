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

### Experimental setup and methodology

This experiment establishes a local inference baseline using `Qwen/Qwen2.5-0.5B-Instruct` across three execution configurations: CPU FP32, CUDA FP32, and CUDA FP16.

The benchmark was executed on an Intel Core i7-12800H processor and an NVIDIA RTX A3000 12GB Laptop GPU. Each configuration runs in a separate spawned subprocess to reduce interference from previously loaded models and persistent process allocations.

The model uses greedy decoding (`do_sample=False`) with a structured system-and-user chat template. Each configuration performs one unmeasured warm-up followed by five measured inference iterations.

The encoded prompt contains 44 tokens. Although the generation limit is 96 new tokens, every measured iteration generated 36 new tokens and terminated with an EOS token. No iteration reached the configured generation limit.

The following observations are based on the completed benchmark recorded on October 9, 2026.

### Inference performance

| Configuration | Mean time (s) | Median time (s) | Std. dev. (s) | Throughput (tokens/s) |
|---|---:|---:|---:|---:|
| CPU FP32 | 1.6820 | 1.6003 | 0.1317 | 21.40 |
| CUDA FP32 | 0.7801 | 0.7702 | 0.0148 | 46.15 |
| CUDA FP16 | 0.7880 | 0.7838 | 0.0098 | 45.69 |

**GPU acceleration:** CUDA FP32 achieved approximately 2.16 times the aggregate token throughput of CPU FP32. The mean inference time decreased by approximately 53.6%.

This demonstrates a clear performance advantage for GPU execution under the tested conditions. However, the result applies to this particular model, prompt, hardware configuration, and inference workload. It should not be generalized to other model sizes or execution environments.

**FP32 versus FP16 performance:** CUDA FP16 achieved 45.69 tokens/s, compared with 46.15 tokens/s for CUDA FP32. The observed throughput difference was approximately 1%, with a small advantage for FP32.

This experiment therefore did not demonstrate a meaningful inference-speed advantage from switching to FP16.

However, this doesn't establish that FP32 is inherently faster. The benchmark doesn't measure GPU occupancy, memory bandwidth utilization, kernel execution times, or other hardware-level characteristics needed to explain the observed difference.

### Timing metrics

The `AverageInferenceTime` and `MedianInferenceTime` measurements represent the complete wall-clock duration of each `model.generate()` call, including prompt prefill and autoregressive decoding.

The benchmark excludes model loading, tokenization, warm-up, and final text decoding from these timings.

`AggregateTokenThroughput` is calculated by dividing the total number of newly generated tokens by the sum of the measured `model.generate()` durations.

Consequently, this metric represents an effective generated-token rate over the complete inference call. It should not be interpreted as decode-only throughput.

The experiment also doesn't separately measure time to first token (TTFT) or inter-token latency.

### GPU memory and numerical precision

| PyTorch CUDA metric | FP32 (MB) | FP16 (MB) |
|---|---:|---:|
| Peak allocated memory | 1989.76 | 1007.96 |
| Peak reserved memory | 2122.32 | 1063.26 |
| Reserved memory after inference | 2122.32 | 1063.26 |

**FP16 substantially reduced GPU memory requirements:** Peak allocated memory decreased by approximately 49.3%, while peak reserved memory decreased by approximately 49.9%.

This is consistent with the expected reduction in storage requirements when using 16-bit rather than 32-bit floating-point values.

Unlike the small difference in inference speed, the reduction in PyTorch CUDA allocator memory was substantial.

However, these measurements must be interpreted carefully. Allocated memory represents allocations tracked by PyTorch, while reserved memory also includes cached blocks maintained by its CUDA caching allocator.

These values include resident model allocations but do not represent total GPU VRAM usage. CUDA driver resources, execution context overhead, and allocations managed outside the PyTorch allocator may not be included.

The reserved-after-inference metric is a snapshot taken before worker termination, not a measurement of memory retained after complete cleanup.

All memory figures in this experiment are expressed in decimal megabytes (MB).

### System RAM and process memory

The experiment also measures process-wide resident set size (RSS), which represents physical system RAM occupied by the benchmark process.

| Configuration | RSS load delta (MB) | Sampled inference RSS peak (MB) |
|---|---:|---:|
| CPU FP32 | 2099.87 | 2882.11 |
| CUDA FP32 | 378.72 | 1744.31 |
| CUDA FP16 | 890.28 | 2431.06 |

The RSS load delta is the difference between process memory before and after loading the tokenizer and model. It includes process-wide allocations and releases associated with model loading and supporting libraries, rather than exclusively measuring model weights.

Interestingly, CUDA FP16 exhibited higher host-process RSS than CUDA FP32 despite using substantially less PyTorch-managed GPU memory.

This demonstrates that system RAM and GPU memory are distinct measurements and should not be interpreted interchangeably. The current experiment doesn't isolate the allocations responsible for this difference.

The inference RSS peak is sampled approximately every 10 milliseconds. Therefore, short-lived memory spikes may not be captured, and the sampling mechanism introduces some monitoring overhead.

The benchmark also doesn't establish whether memory is retained after cleanup or whether a memory leak exists.

### Generated tokens and response comparison

All three configurations generated 36 new token IDs per measured iteration, producing 180 generated token IDs across the five iterations for each configuration.

These counts include the EOS token used to terminate generation, even though special tokens are omitted from the final decoded response.

The same decoded response was observed in the final measured iteration of CPU FP32, CUDA FP32, and CUDA FP16.

However, identical decoded text doesn't establish numerical equivalence between the execution configurations. FP16 changes floating-point representation and may affect intermediate calculations even when the final generated tokens remain unchanged.

An additional observation is that the generated response contained only one sentence, despite the system instruction requesting exactly two short sentences.

This illustrates an important limitation: the benchmark records generated responses for inspection but doesn't automatically evaluate instruction following, factual correctness, semantic equivalence, or response quality.

Therefore, the current experiment supports performance and memory observations, but not conclusions about relative output quality.

### Measurement reliability and limitations

The benchmark improves measurement consistency by using separate processes, an unmeasured warm-up, CUDA synchronization, repeated generation calls, and explicit measurement definitions.

Nevertheless, the available results represent a single complete benchmark execution containing five measured iterations per configuration.

The reported standard deviations describe variation between generation calls within that execution. They do not measure variation across independently repeated benchmark sessions.

The configurations are also executed in a fixed order, so changes in GPU clocks, thermal conditions, CPU scheduling, or other system activity cannot be fully excluded.

Although the measured workload is comparable across configurations, it contains only one prompt, batch size 1, and a relatively short generated response.

These limitations are particularly important when interpreting the approximately 1% throughput difference between CUDA FP32 and CUDA FP16.

### Conclusions

The experiment establishes three main observations.

1. **GPU execution provided a substantial performance advantage over CPU execution** for this model and workload, with CUDA FP32 achieving approximately 2.16 times the aggregate throughput of CPU FP32.

2. **FP16 reduced PyTorch-managed GPU memory consumption by approximately half**, without producing a meaningful throughput improvement over FP32 in this execution.

3. **Identical generated responses do not establish equivalent numerical behavior or response quality**, because the experiment doesn't implement dedicated correctness or quality evaluation.

These results form a useful local inference baseline, but they should be treated as exploratory measurements rather than definitive hardware or numerical-precision benchmarks.

## Open questions

A future experiment should repeat the complete benchmark across multiple independent executions and preserve individual iteration timings to characterize runtime variability more reliably.

It would also be useful to measure time to first token, inter-token latency, GPU utilization, power consumption, and total device memory usage separately from PyTorch allocator statistics.

Additional experiments should investigate larger models, different input and output lengths, and multiple batch sizes to determine how workload characteristics influence GPU acceleration and the FP32-versus-FP16 relationship.

Finally, performance benchmarking should eventually be complemented by explicit output-quality evaluation, particularly when comparing numerical precision, different models, and coding-oriented workloads.
