# Caso do cliente para resumos diários de consumo de cartão

Uma instituição financeira precisa atualizar, **a cada 24 horas**, o resumo do consumo de cartão de **10 mil clientes**. Ela forneceu uma amostra de **24 clientes** para você medir o processamento. Cada entrada reúne as compras acumuladas no mês; o modelo deve resumir total gasto, duas maiores categorias e maior compra.

A cada execução diária, o cliente **inicia uma máquina, carrega o modelo, processa o lote, salva os resumos e desliga a máquina**. O lote completo precisa terminar em **até 24 horas**, antes da próxima execução. O objetivo é escolher a solução com **menor custo total de computação que atenda a esse prazo**. Primeiro avalie uma única máquina por alternativa.

## Meça apenas as 24 amostras

- Use Qwen2.5-0.5B-Instruct nas três configurações: CPU FP32, GPU FP32 e GPU FP16. Uma execução completa por configuração é suficiente nesta rodada.
- Processe as mesmas 24 amostras, uma entrada por vez. Elas são a amostra do benchmark; o batch de produção terá 10 mil entradas. Não é necessário executar esse volume no notebook.
- Tokenize diretamente o campo `prompt` de cada amostra. Use `do_sample=False` e `max_new_tokens=128`, sem forçar uma quantidade mínima de saída. Mantenha os demais parâmetros iguais e use quatro threads na CPU.
- Carregue modelo e tokenizer uma vez por configuração. Faça warmup de até 5 tokens antes do lote e libere os recursos ao trocar de configuração.
- Meça a preparação, do carregamento do tokenizer e do modelo até o fim do warmup, e o processamento completo, da leitura do lote até o salvamento dos resumos. Mantenha os intervalos separados e sem sobreposição. Sincronize a GPU nas medições.
- Aproveite as métricas do baseline para salvar, por amostra, tokens de entrada e saída, tempo de geração e tokens/s. Marque quais respostas atingiram o limite de 128 tokens.

Confira uma amostra antes da rodada completa e estime a duração. Se a previsão das três configurações ultrapassar vinte minutos no seu notebook, registre a estimativa e deixe a rodada completa para a revisão. Instalação e download devem estar concluídos antes desse teste local.

## Projete o lote de produção

O fator de escala é `10.000 / 24` (aproximadamente 416,67). Use o tempo de processamento das 24 amostras para estimar o lote completo:

`tempo_processamento_10_mil = tempo_processamento_24 × (10.000 / 24)`

Adote como premissa que as amostras representam o perfil dos demais clientes e que o tempo cresce proporcionalmente. Compare o tempo total projetado, incluindo inicialização e preparação, com **86.400 segundos, ou 24 horas**. Calcule também quantos clientes caberiam nesse período, descontando inicialização e preparação antes de aplicar a taxa de processamento medida.

## Calcule o custo do lote completo

| Referência AWS | Configuração | Preço por hora |
| --- | --- | ---: |
| `c6i.xlarge`, 4 vCPUs | CPU FP32 | US$ 0,170 |
| `g4dn.xlarge`, GPU NVIDIA T4 | GPU FP32 | US$ 0,526 |
| A mesma `g4dn.xlarge` | GPU FP16 | US$ 0,526 |

Preços Linux On-Demand em `us-east-1`, referência de **19/10/2026**: [valores preservados](../dataset/referencias_aws_2026-10-19.json) e [regras de cobrança EC2](https://aws.amazon.com/ec2/pricing/on-demand/).

Para o exercício, considere uma imagem com bibliotecas e arquivos do modelo disponíveis em disco. Adote **60 segundos adicionais para inicialização e encerramento da máquina**, iguais nas três alternativas. Esse valor é uma premissa da simulação, não uma medição AWS.

Some esses 60 segundos à preparação medida e ao processamento projetado:

`tempo_total = 60 + tempo_preparacao + tempo_processamento_10_mil`

Inicialização, carregamento e warmup ocorrem uma única vez por lote completo: não multiplique essas etapas pelo fator de escala. Arredonde o tempo total para o próximo segundo para calcular a cobrança.

`custo do lote completo = preço por hora × segundos cobrados / 3600`

Se o lote terminar antes de 24 horas, a máquina é desligada ao final. Se ultrapassar esse prazo, a alternativa não atende à demanda diária: o custo calculado é o de concluir um lote, não o de uma operação diária viável. O cálculo cobre computação; armazenamento, rede e impostos ficam fora desta estimativa. Para a simulação, use os tempos locais como referência das instâncias; essa equivalência ainda precisaria ser validada na AWS. Não é necessário contratar uma máquina.

Confira os resumos dos mesmos três clientes nas três configurações: total, duas maiores categorias e maior compra devem corresponder à entrada. Se uma resposta atingir 128 tokens, veja se terminou ou foi cortada. Registre os problemas encontrados antes de recomendar uma opção como mais barata para gerar resumos utilizáveis.

Responda às três perguntas da atividade 8: viabilidade em 24 horas, custo de CPU ou GPU em FP32 e valor da mudança para FP16. A conclusão deve combinar **prazo de até 24 horas e custo por lote de 10 mil resumos utilizáveis**. Se nenhuma alternativa atender, indique essa limitação; não é necessário projetar uma arquitetura com várias máquinas nesta atividade.
