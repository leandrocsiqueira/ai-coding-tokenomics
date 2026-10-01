from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXTS = ["Olá, meu nome é Leandro.", "tokenização", "João Paulo Peçanha Navarro", 
         "def calculate_total_cost(input_tokens, output_tokens):", "🚀 Inteligência Artificial!",]

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

for text in TEXTS:
    # Disable special tokens to inspect only the tokenization of the input text.
    TOKEN_IDS = TOKENIZER.encode(
        text,
        add_special_tokens=False,
    )
    # Show the tokenizer's internal token representation.
    TOKENS = TOKENIZER.convert_ids_to_tokens(TOKEN_IDS)
    decoded_tokens = [TOKENIZER.decode([token_id]) for token_id in TOKEN_IDS]
    decoded_text = TOKENIZER.decode(TOKEN_IDS)
    character_count = len(text)
    token_count = len(TOKEN_IDS)
    characters_per_token = character_count / token_count
    # Check whether encoding and decoding preserve the original text.
    round_trip_preserved = decoded_text == text

    print("=" * 60)
    print(f"Original text: {text}")
    print(f"Character count: {character_count}")
    print(f"Token ID list: {TOKEN_IDS}")
    print(f"Token list: {TOKENS}")
    print(f"Decoded token pieces: {decoded_tokens}")
    print(f"Token count: {token_count}")
    print(f"Characters per token: {characters_per_token:.2f}")
    print(f"Decoded text: {decoded_text}")
    print(f"Round trip preserved: {round_trip_preserved}")