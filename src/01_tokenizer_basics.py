from transformers import AutoTokenizer

TEXTS = [
    "Olá, meu nome é Leandro.",
    "tokenização",
    "João Paulo Peçanha Navarro",
    "def calculate_total_cost(input_tokens, output_tokens):",
    "🚀 Inteligência Artificial!",
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

for text in TEXTS:
    # 1. Convert text to tokens
    tokens = tokenizer.tokenize(text)

    # 2. Convert text to token IDs
    tokenized_ids = tokenizer.encode(text, add_special_tokens=False)

    # 3. Reconstruct the text from token IDs
    decoded_text = tokenizer.decode(tokenized_ids, clean_up_tokenization_spaces=False)

    # 4. Calculate the average number of characters per token
    average_characters_per_token = (len(text) / len(tokenized_ids) if tokenized_ids else float("nan"))

    print(
        f"Original Text..: {text!r}\n"
        f"Total Characters..: {len(text)}\n"
        f"Total Tokens..: {len(tokenized_ids)}\n"
        f"\n"
        f"Tokenization:\n"
        f"  • Tokens..: {tokens}\n"
        f"  • Token IDs..: {tokenized_ids}\n"
        f"  • Average Characters per Token..: {average_characters_per_token:.2f}\n"
        f"  • Decoded Text..: {decoded_text!r}\n"
        f"  • Individually Decoded Tokens..: {[tokenizer.decode([token_id]) for token_id in tokenized_ids]}\n"
        f"  • Was the original text preserved during decoding?..: {decoded_text == text}\n"
        f"\n"
    )
