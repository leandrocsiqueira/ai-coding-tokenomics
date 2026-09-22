from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]

TEXTS = [
    "Olá, meu nome é Leandro.",
    "tokenização",
    "João Paulo Peçanha Navarro",
    "def calculate_total_cost(input_tokens, output_tokens):",
    "🚀 Inteligência Artificial!",
]

RESULTS_PATH = Path("results/tokenizer_comparison.csv")

rows = []

for model_name in MODELS:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for text in TEXTS:
        token_ids = tokenizer.encode(text)

        characters = len(text)
        tokens = len(token_ids)

        rows.append(
            {
                "model": model_name,
                "text": text,
                "characters": characters,
                "tokens": tokens,
                "characters_per_token": characters / tokens,
            }
        )

df = pd.DataFrame(rows)

print(df.to_string(index=False))

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_PATH, index=False)

print(f"\nResults saved to: {RESULTS_PATH}")