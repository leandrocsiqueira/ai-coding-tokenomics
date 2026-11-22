# Guia do lote diário de consumo de cartão

O arquivo [consumo_cartao_credito_01c.json](dataset/consumo_cartao_credito_01c.json) contém **24 amostras sintéticas**, uma por cliente. São a amostra fornecida para medir o [caso do cliente](CASO_CLIENTE_INFRAESTRUTURA_01c.md), cuja base de produção tem **10 mil clientes por execução diária**. Execute somente essas 24 amostras e projete o tempo do lote completo.

Cada amostra contém de 36 a 60 compras acumuladas de 1 a 18 de outubro de 2026. O lote é reprocessado a cada 24 horas para atualizar os resumos; não significa que cada cliente realizou todas essas compras em um único dia.

## Campos

- `dataset_id`, `version`, `frequency_hours` e `reference_date`: identificação e contexto do lote.
- `production_client_count`: os 10 mil clientes da base de produção; eles não estão materializados neste arquivo.
- `samples`: lista das 24 amostras.
- `sample_id` e `customer_id`: identificadores fictícios para relacionar entrada, saída e medição.
- `transactions`: compras, com data, estabelecimento, categoria e `amount_cents`, em centavos de real.
- `total_cents` e `totals_by_category_cents`: totais calculados a partir das compras, fornecidos como dados de apoio ao resumo.
- `prompt`: **entrada completa para o modelo**, com instrução, período, compras e valores apurados. Use esse texto diretamente, preservando seu conteúdo.

A instrução pede duas frases, com até 50 palavras, informando o total, as duas maiores categorias e a maior compra. Os totais já estão na entrada para que o modelo descreva o consumo a partir dos dados disponíveis.

## Loader

O [loader fornecido](loader_dataset.py) usa apenas a biblioteca padrão do Python. `carregar_dataset(caminho)` lê o JSON em UTF-8 e devolve um dicionário. Percorra `dataset["samples"]` e use o campo `prompt` de cada amostra na inferência.

**Use o loader como base para escrever sua versão. Na apresentação, você deve saber explicar o que ele recebe, faz e devolve.** Preserve `sample_id` nos resultados para relacionar cada resumo à sua entrada.

Os gastos estão em reais; os preços de infraestrutura estão em dólares. São valores de naturezas diferentes e não precisam ser convertidos para este exercício.
