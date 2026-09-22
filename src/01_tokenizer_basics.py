from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXTS = [
    "Olá, meu nome é Leandro.",
    "tokenização",
    "João Paulo Peçanha Navarro",
    "def calculate_total_cost(input_tokens, output_tokens):",
    "🚀 Inteligência Artificial!",
]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for text in TEXTS:
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded_text = tokenizer.decode(token_ids)

    print("=" * 60)
    print(f"Original text: {text}")
    print(f"Character count: {len(text)}")
    print(f"Token ID list: {token_ids}")
    print(f"Token list: {tokens}")
    print(f"Token count: {len(token_ids)}")
    print(f"Decoded text: {decoded_text}")