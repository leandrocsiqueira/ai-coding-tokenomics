from transformers import AutoTokenizer


MODEL_IDENTIFIER = "Qwen/Qwen2.5-0.5B-Instruct"

EXAMPLE_TEXTS = ["Olá, meu nome é Leandro.", "tokenização", "João Paulo Peçanha Navarro", 
                 "def calculate_total_cost(input_tokens, output_tokens):", "🚀 Inteligência Artificial!",]


def inspect_tokenization(model_identifier, texts):
    """Inspect how the given tokenizer processes each input text."""
    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        decoded_text = tokenizer.decode(token_ids)

        num_characters = len(text)
        num_tokens = len(token_ids)
        characters_per_token = (num_characters / num_tokens if num_tokens > 0 else float("nan"))
        round_trip_preserved = decoded_text == text

        print("=" * 60)
        print(f"Original text: {text}")
        print(f"Character count: {num_characters}")
        print(f"Token ID list: {token_ids}")
        print(f"Token list: {tokens}")
        print(f"Decoded token pieces: {decoded_tokens}")
        print(f"Token count: {num_tokens}")
        print(f"Characters per token: {characters_per_token:.2f}")
        print(f"Decoded text: {decoded_text}")
        print(f"Round trip preserved: {round_trip_preserved}")


if __name__ == "__main__":
    inspect_tokenization(MODEL_IDENTIFIER, EXAMPLE_TEXTS)