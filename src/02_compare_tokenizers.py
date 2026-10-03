from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODEL_IDENTIFIERS = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",]

EXAMPLE_TEXTS = ["Olá, meu nome é Leandro.", "tokenização", "João Paulo Peçanha Navarro", 
                 "def calculate_total_cost(input_tokens, output_tokens):", "🚀 Inteligência Artificial!",]

RESULTS_PATH = Path("results/tokenizer_comparison.csv")


def collect_tokenization_data(model_identifiers, texts):
    """Collect tokenization data for the given models and texts."""
    tokenization_results = []

    for model_identifier in model_identifiers:
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)

        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False,)
            num_characters = len(text)
            num_tokens = len(token_ids)

            tokenization_results.append(
                {
                    "model": model_identifier,
                    "text": text,
                    "characters": num_characters,
                    "tokens": num_tokens,
                    "characters_per_token": (num_characters / num_tokens if num_tokens > 0 else float("nan")),
                }
            )

    return pd.DataFrame(tokenization_results)


def save_results_to_csv(df, path):
    """Save the tokenization results to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    tokenization_results_df = collect_tokenization_data(MODEL_IDENTIFIERS, EXAMPLE_TEXTS,)
    
    print(tokenization_results_df.to_string(index=False))

    save_results_to_csv(
        tokenization_results_df,
        RESULTS_PATH,
    )