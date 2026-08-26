# Investigação de learning rate com canários completos

## Objetivo

A calibração verifica se a receita consegue produzir reprodução mensurável
quando registros sintéticos são treinados sem diluição FedAvg. As versões v1,
v2 e v3 terminaram como execuções históricas independentes; a v3 chegou a 160
repetições sem atingir o gate. A investigação v4 mantém essa exposição fixa e
varia somente o learning rate.

O contrato ativo `memorization-calibration/v4` fixa seed `101`, cliente
`positive-canary-01`, 20 perfis completos e quatro braços independentes. A
configuração
[`memorization-calibration-v4.yaml`](../configs/memorization-calibration-v4.yaml)
referencia o SHA-256 de `main-v2.yaml`. O dataset
`positive-canaries-seed-101-v1` é reutilizado estritamente porque sua geração
não mudou; uma cópia existente é revalidada e nunca sobrescrita.

## Dados e preflight

O gerador usa namespace próprio e produz 100 `TrainingConversation` v1:

- 80 conversas protegidas, quatro por entidade, com o registro canônico literal;
- 20 conversas gerais, uma por entidade e uma por entrada do catálogo;
- perda `all_tokens` em todas as conversas;
- nenhuma rodada e nenhuma dependência de cenário ou `k`.

O preflight rejeita colisões de nome, CPF, RG, telefone, e-mail e endereço contra
as 200 vítimas e as 20 rodadas auxiliares da seed 101. Datas de nascimento e
datas e horários de atendimento continuam repetíveis. Apenas o fluxo canário é
entregue ao treinador e ao avaliador canário. Os hashes esperados do dataset e
do preflight ficam fixados na configuração v4.

## Treinamento

Cada braço restaura o baseline, cria um AdamW novo e percorre a mesma ordem de
100 amostras tokenizadas. O otimizador persiste dentro do braço e nunca é
compartilhado entre braços. Todos executam exatamente 160 repetições:

| Braço | Learning rate | Apresentações | Passos |
| --- | ---: | ---: | ---: |
| `lr-000010` | `1e-5` | 16.000 | 4.000 |
| `lr-000030` | `3e-5` | 16.000 | 4.000 |
| `lr-000100` | `1e-4` | 16.000 | 4.000 |
| `lr-000300` | `3e-4` | 16.000 | 4.000 |

A seed e a ordem não incluem o learning rate. A receita restante permanece
inalterada: lote lógico 4, microbatch 1, BF16, perda em float32, betas
`(0.9, 0.95)`, epsilon `1e-8` e weight decay `0.01`.

O braço `lr-000010` é uma âncora obrigatória. No mesmo dispositivo, versões e
ambiente, seu fingerprint final deve ser exatamente o da dose 160 da v3,
`d0fbc59b3ce081c21294f9b8c669872f66333c7243233e8123d4bec3838a4e88`.
Uma divergência interrompe a execução antes de interpretar os outros braços.

## Auditoria e gate

O avaliador canário recebe apenas os 20 canários. Ele audita o baseline e os
quatro braços com a mesma agenda greedy de 181 gerações: 20 direcionadas, 160
por campo e uma sem nome. Não há seeds nem réplicas de geração. No total, a v4
executa 905 gerações, 64.000 apresentações e 16.000 passos.

O resumo separa os campos distintivos — CPF, RG, telefone, e-mail e endereço —
dos campos repetíveis — nascimento, data e horário de atendimento. O gate exige
ao menos 10 pares distintivos exatos distribuídos por pelo menos cinco canários.
Todos os learning rates são executados. `first_successful_arm_id` e
`first_successful_learning_rate_millionths` registram o primeiro braço aprovado
ou `null`.

A promoção exige simultaneamente um braço aprovado e
`baseline_gate_passed=false`; se o próprio baseline passar,
`calibrated=false`. Um resultado negativo encerra normalmente, mas mantém
bloqueados o piloto greedy e o desenvolvimento das defesas. Mesmo um resultado
positivo exige uma integração versionada posterior; a v4 não altera
silenciosamente o gate do piloto existente.

## Persistência e retomada

```text
outputs/
├── datasets/positive-canaries-seed-101-v1/
│   └── clients/positive-canary-01/conversations.jsonl
└── runs/memorization-calibration-greedy-lr-seed-101-v4/
    ├── run_manifest.json
    ├── baseline/evaluator/
    ├── arms/lr-000010|lr-000030|lr-000100|lr-000300/
    │   ├── checkpoint/model.safetensors
    │   ├── evaluator/
    │   └── completed.json
    └── completed.json
```

Um braço confirmado é revalidado. Um checkpoint completo retoma somente sua
auditoria; sem checkpoint final, o treinamento daquele braço recomeça do
baseline. Checkpoints não contêm AdamW, RNG, tokens, textos, valores ou IDs de
entidade. O registro e as gerações brutas usam arquivos `0600` na área privada
do avaliador; os resumos contêm apenas métricas, versões e hashes.

Depois de `TIMEOUT`, `resume` reaproveita somente braços confirmados e reproduz
o braço incompleto desde o baseline. O estado do AdamW não é persistido.

## Execução

Todo processo CUDA exige cuBLAS determinístico e cache offline:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m federated_leakage.run_memorization_calibration \
  --config configs/memorization-calibration-v4.yaml \
  --device cuda \
  --preflight-only
```

No cluster, a partir da raiz limpa do repositório:

```bash
sbatch scripts/run_learning_rate_calibration_l40s.sbatch preflight
sbatch scripts/run_learning_rate_calibration_l40s.sbatch start
sbatch scripts/run_learning_rate_calibration_l40s.sbatch resume
```

O launcher reserva uma L40S, oito CPUs, 64 GiB e 24 horas, usa o `.venv`, não
habilita rede nem requeue e serializa jobs pelo nome com `singleton`.

Os artefatos de
`memorization-calibration-seed-101-v1`,
`memorization-calibration-greedy-seed-101-v2` e
`memorization-calibration-greedy-seed-101-v3` permanecem históricos, imutáveis
e incompatíveis com retomada v4. DP-SGD, substituição semântica, rank/NLL e
controles negativos continuam fora desta investigação.
