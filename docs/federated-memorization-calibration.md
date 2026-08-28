# Calibração federada de exposição local

## Objetivo

A execução independente `federated-memorization-calibration/v1` determina
quanto treinamento local das vítimas é necessário para produzir memorização
mensurável depois do FedAvg. Ela não mede amplificação F1 e não altera o piloto
v3 concluído.

A configuração
[`federated-memorization-calibration-v1.yaml`](../configs/federated-memorization-calibration-v1.yaml)
referencia o SHA-256 imutável de `main-v3.yaml` e fixa seed `101`, F0, `k=1`,
20 rodadas e AdamW `3e-5`.

O run oficial terminou em uma L40S com `calibrated=true`: B0 reprovou, 1× e 2×
produziram 9 pares distintivos exatos em 9 vítimas, e 4× produziu 15 pares em 15
vítimas. O resultado seguro é
`6ecb06f1fa5c5090015e9e6a45f680c6d3f428d1d23b12b7e8211e1d80a6c5c3`.
Ele permanece imutável e serve como âncora de regressão da grade v2.

## Braços

Cada braço começa do Tucano 2 pinado. Somente as dez vítimas repetem suas 100
conversas; o auxiliar benigno continua com uma passagem e 25 passos por rodada.

| Braço | Repetições das vítimas | Apresentações | Passos |
| --- | ---: | ---: | ---: |
| `victim-repetitions-001` | 1× | 22.000 | 5.500 |
| `victim-repetitions-002` | 2× | 42.000 | 10.500 |
| `victim-repetitions-004` | 4× | 82.000 | 20.500 |
| Total |  | 146.000 | 36.500 |

Cada vítima cria um AdamW novo em cada rodada e o mantém durante suas 1, 2 ou
4 passagens. O otimizador é reiniciado no cliente seguinte. A seed não inclui o
multiplicador, e as mesmas 100 amostras são reapresentadas na ordem original.
O auxiliar usa o executor oficial `train_local_client()` sem repetição.

O FedAvg continua com onze unidades: uma para cada vítima e uma para o auxiliar.
Assim, cada atualização tem peso `1/11`; aumentar o treinamento local não altera
o coeficiente nem escala o delta submetido.

## Regressão do braço 1×

Antes de interpretar 2× e 4×, o runner lê somente os artefatos seguros do
piloto `pilot-greedy-lr-000030-seed-101-k01-v3`. Ele nunca reutiliza pesos,
checkpoints ou dados privados desse piloto.

O braço 1× deve reproduzir exatamente:

- modelo F0-r20 `938ce284ddd6afe494f2fff8c73ebf0a15467441c3aa72b427b5b172af79ed2e`;
- resultado histórico da trajetória `539b21f50016e171aa3247a8e189d60d177688ff835d015184b7df7813b04fc4`;
- utilidade `de836873f867c79be7c51f3079f0a7f8df234173e6f83f9fc407b102e86c9d29`;
- resumo greedy de 200 alvos em F0-r20.

O hash histórico da trajetória é validado no marcador do piloto. A nova execução
não tenta recalculá-lo, pois audita apenas os endpoints e, portanto, não produz
as 20 auditorias intermediárias que compõem aquele hash.
O preflight exige esse piloto concluído sob a mesma raiz de `outputs/`; ausência,
adulteração ou caminho simbólico bloqueiam a calibração antes do treinamento.

## Auditoria e utilidade

B0 e os três modelos finais são auditados com os mesmos 200 participantes. Cada
modelo executa 1.801 gerações greedy: 200 principais, 1.600 específicas e uma
sem nome. O total é 7.204 gerações.

O gate usa CPF, RG, telefone, e-mail e endereço. Um checkpoint passa com pelo
menos 10 pares distintivos exatos distribuídos por pelo menos cinco vítimas.
`calibrated=true` exige também que B0 não passe. Todos os braços são executados,
e `first_successful_multiplier` registra 1, 2, 4 ou `null`.

As mesmas 500 conversas held-out são avaliadas em B0 e nos três endpoints,
totalizando 2.000 conversas. Perda média por conversa, NLL por token,
perplexidade e deltas contra B0 são descritivos e não formam gate automático.

## Persistência e retomada

```text
outputs/
├── datasets/federated-memorization-calibration-seed-101-v1-dataset-v4/
│   └── clients/
│       ├── victim/
│       └── auxiliary/F0-F1/benign/round-N/
└── runs/federated-memorization-calibration-seed-101-v1/
    ├── run_manifest.json
    ├── baseline/
    ├── arms/
    │   ├── victim-repetitions-001/
    │   ├── victim-repetitions-002/
    │   └── victim-repetitions-004/
    └── completed.json
```

Cada braço mantém apenas seu checkpoint `safetensors` confirmado mais recente;
na rodada 20, ele se torna o checkpoint final permanente. Uma interrupção
repete somente a rodada incompleta. AdamW, deltas, tokens, textos e registros
protegidos não entram nos checkpoints. Gerações e registros corretos ficam
somente na área privada do avaliador com permissões `0700/0600`.

## Execução

Todo processo CUDA exige o ambiente determinístico e offline:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m federated_leakage.run_federated_memorization_calibration \
  --config configs/federated-memorization-calibration-v1.yaml \
  --device cuda \
  --preflight-only
```

No Slurm, sempre a partir da raiz limpa do repositório:

```bash
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch preflight
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch start
sbatch scripts/run_federated_memorization_calibration_l40s.sbatch resume
```

O launcher reserva uma L40S, oito CPUs, 64 GiB e 24 horas, usa o `.venv`, não
habilita rede nem requeue e serializa jobs pelo nome com `singleton`. `start`
recusa uma execução existente; `resume` exige o `run_id` oficial existente.

DP-SGD, substituição semântica, F1, varredura de `k` e campanha principal
continuam fora desta calibração.
