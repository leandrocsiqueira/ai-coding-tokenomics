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
