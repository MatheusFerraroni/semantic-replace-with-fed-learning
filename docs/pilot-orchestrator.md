# Orquestrador retomável do piloto B0/F0/F1

## Escopo implementado

O comando `federated_leakage.run_pilot` executa exclusivamente o piloto de
desenvolvimento não privado fixado em `configs/main-v3.yaml`:

- seed `101`, agenda `F0-F1` e `k=1`;
- AdamW `3e-5` nos dez clientes-vítima e no auxiliar;
- uma auditoria B0 compartilhada;
- 20 rodadas F0, recarga do Tucano 2 pinado e 20 rodadas F1;
- auditoria greedy de 20 alvos após todas as rodadas;
- auditorias adicionais de 1, 5 e 200 alvos em B0 e na rodada 20;
- utilidade held-out em B0, F0 rodada 20 e F1 rodada 20.

Isso totaliza 40 rodadas, 44.000 conversas de treinamento, 11.000 passos locais,
12.992 gerações de extração e 1.500 avaliações de conversa de utilidade. F2-F5,
a varredura `k=1..10` e a campanha principal não fazem parte deste comando.

## Gate da calibração v4

Antes do modelo ou de qualquer escrita, o runner lê estritamente o marcador
`outputs/runs/memorization-calibration-greedy-lr-seed-101-v4/completed.json` e
seu manifesto seguro. O resultado aceito é fixado por SHA-256 em `main-v3` e
exige:

- baseline reprovado;
- `calibrated=true`;
- primeiro braço aprovado `lr-000030`;
- 100/100 pares distintivos exatos e 20/20 canários expostos nesse braço;
- hashes compatíveis de configuração, dataset, preflight, modelo, agendas e
  proveniência.

O checkpoint treinado da calibração não é carregado. O piloto começa sempre do
baseline `Polygl0t/Tucano2-0.6B-Base` pinado, e o fingerprint desse baseline deve
coincidir com o auditado na calibração.

## Utilidade sintética

O fluxo `utility` produz em memória 100 perfis disjuntos e 500 conversas: quatro
protegidas e uma geral por perfil, todas com perda integral. Colisões de nome,
CPF, RG, telefone, e-mail e endereço são rejeitadas contra vítimas, toda a
agenda auxiliar e canários. Datas e horários permanecem repetíveis conforme o
protocolo.

As 500 conversas são tokenizadas uma vez e reutilizadas nos três checkpoints.
O avaliador não gera texto, não calcula gradientes e não altera o modelo. Ele
relata somente:

- perda causal média por conversa;
- NLL ponderada por token;
- perplexidade;
- contagens, tempo e pico de memória;
- deltas de F0/F1 contra B0.

Não há limiar automático. Texto, tokens, valores e entidades de utilidade não
são persistidos; somente resultados agregados e hashes entram na transação.

## Determinismo CUDA e preflight

Antes de qualquer processo Python com CUDA, exporte exatamente:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

O Python falha se a variável estiver ausente ou divergente e não a corrige. O
launcher também fixa o cache offline. Para validar gate, dados, colisões, modelo,
tokenização, agendas de auditoria e utilidade sem escrever:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v3.yaml \
  --device cuda \
  --preflight-only
```

O preflight deve confirmar 1.000 conversas-vítima, 4.000 auxiliares e 500 de
utilidade, mas não cria datasets, auditorias ou checkpoints.

## Execução e retomada

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v3.yaml \
  --device cuda
```

O `run_id` padrão é `pilot-greedy-lr-000030-seed-101-k01-v3`. `--fresh` exige
que esse diretório ainda não exista; sem ele, uma execução compatível retoma o
último estado confirmado. Também são aceitos `--cache-dir`,
`--model-artifact-dir`, `--output-root` e `--run-id`.

B0 é compartilhado. Na rodada 1, F0 e F1 partem do mesmo baseline. Da rodada 2
em diante, cada trajetória parte do próprio estado final anterior; dados,
pesos, seeds, ordem e agenda continuam pareados, mas os estados globais não
precisam coincidir.

## Execução Slurm em uma L40S

Use o launcher promovido, sempre da raiz limpa do repositório:

```bash
sbatch scripts/run_pilot_lr_000030_l40s.sbatch preflight
sbatch scripts/run_pilot_lr_000030_l40s.sbatch start
sbatch scripts/run_pilot_lr_000030_l40s.sbatch resume
```

- `preflight` acrescenta `--preflight-only` e não publica artefatos científicos;
- `start` acrescenta `--fresh` e recusa o run existente;
- `resume` exige o mesmo run existente e nunca usa `--fresh`.

O launcher reserva uma L40S, oito CPUs, 64 GiB e 24 horas, usa
`.venv/bin/python`, exporta o ambiente offline e determinístico, chama o processo
por `srun` e propaga seu código de saída. A dependência `singleton` serializa
jobs com o mesmo nome, mas nunca devem ser submetidos dois jobs para o mesmo
`run_id`.

O launcher anterior `scripts/run_pilot_l40s.sbatch`, `main-v1.yaml`,
`main-v2.yaml` e seus runs permanecem históricos e não são retomados pela v3.
Não há requeue automático. Após `TIMEOUT`, verifique `sacct` e os logs e submeta
manualmente `resume`.

## Transação de B0 e das rodadas

B0 só é confirmado depois das quatro auditorias, da utilidade e do marcador
compatível. Cada rodada federada só é confirmada depois de:

1. treinamento dos 11 clientes e aplicação atômica do FedAvg;
2. auditorias greedy e verificação do fingerprint;
3. avaliação de utilidade, somente na rodada 20;
4. publicação do checkpoint `safetensors`;
5. commit da rodada e atualização do estado retomável.

Resultados completos compatíveis são relidos. Resíduos parciais não são
promovidos; a etapa incompleta é reproduzida. Os checkpoints permanentes ficam
nas rodadas 1, 10 e 20, com um único checkpoint móvel nas demais. Eles não
contêm otimizador, deltas, tokens, textos ou registros protegidos.

## Árvore de saída

```text
outputs/
├── datasets/pilot-greedy-lr-000030-seed-101-k01-v3-dataset-v4/clients/
│   ├── victim/<client_id>/conversations.jsonl
│   └── auxiliary/F0-F1/<presentation>/round-N/conversations.jsonl
└── runs/pilot-greedy-lr-000030-seed-101-k01-v3/
    ├── run_manifest.json
    ├── baseline/
    │   ├── completed.json
    │   └── evaluator/
    │       ├── private/
    │       ├── summaries/
    │       └── utility/summary.json
    ├── trajectories/
    │   ├── F0-k01/
    │   └── F1-k01/
    ├── paired/round-N.json
    └── completed.json
```

Os contratos ativos são `pilot-execution/v3`, `federated-trajectory/v3` e
`federated-checkpoint/v3`. Datasets, pesos, auditorias e toda a árvore `outputs/`
permanecem fora do Git.

## Aceitação científica

O marcador final reúne extração e utilidade, mas não congela automaticamente a
receita. A promoção para a campanha principal exige revisão humana conjunta de
privacidade, utilidade, estabilidade, custo, memória e limitações desta única
seed de desenvolvimento.
