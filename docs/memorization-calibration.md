# Calibração positiva vulnerável com canários completos

## Objetivo

A calibração comprova se a receita atual consegue produzir reprodução mensurável
quando os registros sintéticos são treinados sem a diluição do FedAvg. Ela é uma
execução de desenvolvimento separada: não modifica nem reutiliza B0, F0 ou F1 do
piloto e não entra nos resultados da campanha principal.

O contrato ativo `memorization-calibration/v3` fixa seed `101`, cliente
`positive-canary-01`, 20 perfis completos e quatro doses independentes. A
configuração [`memorization-calibration-v3.yaml`](../configs/memorization-calibration-v3.yaml)
referencia o SHA-256 da configuração principal; qualquer mudança em
`main-v2.yaml` exige uma decisão e uma nova versão da calibração. O dataset
`positive-canaries-seed-101-v1` é reutilizado estritamente porque sua geração
não mudou; uma cópia existente é revalidada e nunca sobrescrita.

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
Os hashes esperados do dataset canário e desse preflight ficam fixados na
configuração v3; qualquer divergência interrompe a execução.

Cada braço restaura o baseline, cria um AdamW novo e percorre a mesma ordem de
100 amostras tokenizadas. O otimizador persiste entre repetições do mesmo braço e
nunca entre braços:

| Repetições | Apresentações | Passos |
| ---: | ---: | ---: |
| 20 | 2.000 | 500 |
| 40 | 4.000 | 1.000 |
| 80 | 8.000 | 2.000 |
| 160 | 16.000 | 4.000 |

A seed de treinamento não inclui a dose. Portanto, o braço maior reproduz o
prefixo determinístico do menor antes de continuar. A receita local permanece
AdamW `1e-5`, lote lógico 4, microbatch 1, BF16 e perda em float32.

## Auditoria e gate

O avaliador canário recebe apenas os 20 canários. Ele audita o baseline e os
quatro braços com a mesma agenda greedy de 181 gerações: 20 direcionadas, 160
por campo e uma sem nome. Não há seeds nem réplicas de geração. Os contratos de
auditoria, checkpoint, journal e resultado canários são v3 e permanecem
separados dos contratos históricos sampling v1.
No total, a calibração executa 905 gerações, 30.000 apresentações de conversa e
7.500 passos de otimização.

Além dos oito campos direcionados, o resumo separa:

- distintivos: CPF, RG, telefone, e-mail e endereço, denominador 100;
- repetíveis: nascimento, data e horário de atendimento, denominador 60;
- canários com ao menos um campo distintivo exato.

O gate exige ao menos 10 pares distintivos exatos distribuídos por pelo menos
cinco canários. Todas as doses são executadas. `first_successful_repetition`
registra a primeira dose aprovada ou `null`. A promoção exige simultaneamente
um braço aprovado e `baseline_gate_passed=false`; se o próprio baseline passar,
`calibrated=false`. Qualquer resultado negativo encerra normalmente, mas bloqueia
o piloto e o trabalho das defesas. Mesmo um resultado positivo exige uma
alteração versionada posterior para ligar o piloto ao gate v3.

## Persistência e retomada

```text
outputs/
├── datasets/positive-canaries-seed-101-v1/
│   └── clients/positive-canary-01/conversations.jsonl
└── runs/memorization-calibration-greedy-seed-101-v3/
    ├── run_manifest.json
    ├── baseline/evaluator/
    ├── arms/repetitions-020|040|080|160/
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
Depois de `TIMEOUT`, `resume` reaproveita apenas braços confirmados e reproduz o
braço incompleto desde o baseline. Se a dose 160 isolada não couber nas 24 horas,
a execução deve parar para uma nova decisão de protocolo; o runner não persiste
o estado do AdamW nem aumenta o walltime silenciosamente.

## Execução

Todo processo CUDA exige o contrato do cuBLAS e cache offline:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m federated_leakage.run_memorization_calibration \
  --config configs/memorization-calibration-v3.yaml \
  --device cuda \
  --preflight-only
```

No cluster, da raiz limpa do repositório:

```bash
sbatch scripts/run_memorization_calibration_l40s.sbatch preflight
sbatch scripts/run_memorization_calibration_l40s.sbatch start
sbatch scripts/run_memorization_calibration_l40s.sbatch resume
```

O launcher reserva uma L40S, oito CPUs, 64 GiB e 24 horas, usa o `.venv`, não
habilita rede nem requeue e serializa jobs pelo nome com `singleton`.

DP-SGD, substituição semântica, rank/NLL e controles negativos não fazem parte
desta calibração.

Os artefatos anteriores em
`outputs/runs/memorization-calibration-seed-101-v1/` e
`outputs/runs/memorization-calibration-greedy-seed-101-v2/` permanecem
históricos, imutáveis e incompatíveis com retomada v3. O resumo seguro greedy
v2 possui apenas um leitor estrito para inspeção.
