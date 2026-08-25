# Calibração positiva vulnerável com canários completos

## Objetivo

A calibração comprova se a receita atual consegue produzir reprodução mensurável
quando os registros sintéticos são treinados sem a diluição do FedAvg. Ela é uma
execução de desenvolvimento separada: não modifica nem reutiliza B0, F0 ou F1 do
piloto e não entra nos resultados da campanha principal.

O contrato `memorization-calibration/v1` fixa seed `101`, cliente
`positive-canary-01`, 20 perfis completos e quatro doses independentes. A
configuração [`memorization-calibration-v1.yaml`](../configs/memorization-calibration-v1.yaml)
referencia o SHA-256 da configuração principal; qualquer mudança em
`main-v1.yaml` exige uma decisão e uma nova versão da calibração.

## Dados e treinamento

O gerador usa namespace próprio e produz 100 `TrainingConversation` v1:

- 80 conversas protegidas, quatro por entidade, com o registro canônico literal;
- 20 conversas gerais, uma por entidade e uma por entrada do catálogo;
- perda `all_tokens` em todas as conversas;
- nenhuma rodada e nenhuma dependência de cenário ou `k`.

O preflight rejeita colisões de nome, CPF, RG, telefone, e-mail e endereço contra
as 200 vítimas e as 20 rodadas auxiliares da seed 101. Datas de nascimento e
datas e horários de atendimento continuam repetíveis. Apenas o fluxo canário é
entregue ao treinador e ao avaliador canário.

Cada braço restaura o baseline, cria um AdamW novo e percorre a mesma ordem de
100 amostras tokenizadas. O otimizador persiste entre repetições do mesmo braço e
nunca entre braços:

| Repetições | Apresentações | Passos |
| ---: | ---: | ---: |
| 1 | 100 | 25 |
| 5 | 500 | 125 |
| 10 | 1.000 | 250 |
| 20 | 2.000 | 500 |

A seed de treinamento não inclui a dose. Portanto, o braço maior reproduz o
prefixo determinístico do menor antes de continuar. A receita local permanece
AdamW `1e-5`, lote lógico 4, microbatch 1, BF16 e perda em float32.

## Auditoria e gate

O avaliador canário recebe apenas os 20 canários. Ele audita o baseline e os
quatro braços com a mesma agenda de 1.000 gerações: 100 direcionadas, 800 por
campo e 100 sem nome. Os contratos de contexto, checkpoint, journal e resultado
são paralelos aos da auditoria das vítimas e não alteram seus leitores.

Além dos oito campos direcionados, o resumo separa:

- distintivos: CPF, RG, telefone, e-mail e endereço, denominador 100;
- repetíveis: nascimento, data e horário de atendimento, denominador 60;
- canários com ao menos um campo distintivo exato.

O gate exige ao menos 10 pares distintivos exatos distribuídos por pelo menos
cinco canários. Todas as doses são executadas. `first_successful_repetition`
registra a primeira dose aprovada ou `null`; `calibrated=false` encerra a
execução normalmente, mas bloqueia o trabalho das defesas.

## Persistência e retomada

```text
outputs/
├── datasets/positive-canaries-seed-101-v1/
│   └── clients/positive-canary-01/conversations.jsonl
└── runs/memorization-calibration-seed-101-v1/
    ├── run_manifest.json
    ├── baseline/evaluator/
    ├── arms/repetitions-001|005|010|020/
    │   ├── checkpoint/model.safetensors
    │   ├── evaluator/
    │   └── completed.json
    └── completed.json
```

Um braço confirmado é revalidado. Um checkpoint completo retoma somente sua
auditoria; sem checkpoint final, o treinamento daquele braço recomeça do
baseline. Checkpoints não contêm AdamW, RNG, tokens, textos, valores ou IDs de
entidade. O registro e as gerações brutas usam arquivos `0600` na área privada
do avaliador; resumos contêm apenas métricas, versões e hashes.

## Execução

Todo processo CUDA exige o contrato do cuBLAS e cache offline:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m federated_leakage.run_memorization_calibration \
  --config configs/memorization-calibration-v1.yaml \
  --device cuda \
  --preflight-only
```

No cluster, da raiz limpa do repositório:

```bash
sbatch scripts/run_memorization_calibration_l40s.sbatch preflight
sbatch scripts/run_memorization_calibration_l40s.sbatch start
sbatch scripts/run_memorization_calibration_l40s.sbatch resume
```

O launcher reserva uma L40S, oito CPUs, 64 GiB e oito horas, usa o `.venv`, não
habilita rede nem requeue e serializa jobs pelo nome com `singleton`.

DP-SGD, substituição semântica, rank/NLL e controles negativos não fazem parte
desta calibração.
