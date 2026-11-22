"""Base simples para ler o lote diário de consumo de cartão."""

import json
from pathlib import Path


def carregar_dataset(caminho):
    """Leia o JSON em UTF-8 e devolva seu conteúdo, incluindo as amostras."""
    with Path(caminho).open(encoding="utf-8") as arquivo:
        dataset = json.load(arquivo)
    return dataset
