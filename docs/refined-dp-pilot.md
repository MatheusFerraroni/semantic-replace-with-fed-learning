# Piloto refinado com DP-AdamW

Versão: `refined-defense-pilot/v1`.

Este piloto começa sempre do artefato Fórum/Tec
`ae3238fde6675942cac5`. Checkpoints upstream, da grade, da substituição ou da
calibração não são aceitos como baseline.

## Matriz fixada

Cada seed (`101` e `361506353`) executa oito trajetórias independentes de 20
rodadas, com `k=1`:

| Trajetória | Vítimas | Auxiliar |
| --- | --- | --- |
| F0 / F1 | AdamW `1e-4`, 4 passagens, 100 passos | benigno / adversário, AdamW `3e-5`, 25 passos |
| F2 / F3, ε 3 | DP-AdamW, sigma `2,81`, 100 passos | benigno / adversário, não privado |
| F2 / F3, ε 8 | DP-AdamW, sigma `1,36`, 100 passos | benigno / adversário, não privado |
| F4 / F5 | substituição rotativa, AdamW `1e-4`, 4 passagens | benigno / adversário, não privado |

O DP usa cada uma das 100 conversas do cliente como unidade, Poisson `q=0,04`,
clipping flat `1,0`, lote físico máximo 1, Opacus `1.6.0`, RDP accountant e
`delta=1e-5`. O accountant é independente por cliente e persiste por 2.000
passos. O ε agregado reportado é o máximo dos dez clientes disjuntos.

F2/F3 do mesmo orçamento reconstroem a mesma agenda Poisson e o mesmo ruído.
Cenário e `k` não entram na derivação. A implementação não publica loss, normas
ou clipping rate de vítimas.

## Gates

O runner executa B0 e F0/F1 antes das defesas. Em cada seed, F0 e F1 precisam
atingir ao menos 50 pares distintivos, 25 vítimas e dois tipos distintivos, com
B0 abaixo desse gate. As duas seeds precisam produzir gates válidos antes de
F2-F5.

Se uma seed terminar primeiro, ela retorna
`awaiting-peer-vulnerability-gate` sem iniciar as defesas. Depois que a outra
seed publicar seu gate, submeta a primeira novamente com `resume`. Um gate
vulnerável insuficiente produz `inconclusive.json` e bloqueia F2-F5.

Para cada ε, F2 é comparado a F0 e F3 a F1. A classificação por seed exige
redução de pelo menos 90% dos pares originais exatos e zero perfil original
completo. O resumo de duas seeds classifica o orçamento como `approved`,
`unstable`, `insufficient` ou `inconclusive`. Substituição e utilidade são
relatadas no mesmo resultado.

## Preparação e smokes

No headnode, prepare o ambiente e o artefato:

```bash
python -m pip install -e '.[model,dp]'

python -m federated_leakage.prepare_refined_artifact \
  --config configs/main-v5.yaml \
  --archive /caminho/absoluto/ae3238fde6675942cac5.zip \
  --output-root artifacts/models
```

Na L40S, com cache upstream já disponível e ambiente offline:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m federated_leakage.smoke_private_training \
  --config configs/main-v5.yaml \
  --model-artifact-dir "$PWD/artifacts/models/ae3238fde6675942cac5" \
  --cache-dir artifacts/huggingface --device cuda --epsilon 3 --steps 1

python -m federated_leakage.smoke_private_training \
  --config configs/main-v5.yaml \
  --model-artifact-dir "$PWD/artifacts/models/ae3238fde6675942cac5" \
  --cache-dir artifacts/huggingface --device cuda --epsilon 3 --steps 100
```

Ambos restauram o modelo e mostram `escrita: nao`. O modo `hooks` é obrigatório;
OOM ou incompatibilidade interrompem a execução sem fallback.

O export declara `tokenizer_class=TokenizersBackend`, uma classe do runtime do
produtor. O consumidor não instancia essa classe: valida o `tokenizer.json`
bruto contra o snapshot upstream pinado e somente então usa o fast tokenizer
upstream equivalente. Divergência de backend, vocabulário ou probes interrompe o
smoke antes do treinamento privado.

O `config.json` também usa o dialeto do Transformers 5.14.1. Após validar o
objeto completo, o loader traduz em memória `dtype` e `rope_parameters` para os
atributos equivalentes do runtime 4.53.2, preservando FP32 declarado e
`rope_theta=50000`; nenhuma outra configuração recebe fallback.

## Slurm e retomada

Execute primeiro os preflights:

```bash
sbatch --job-name=refined-defense-s101-v1 \
  scripts/run_refined_defense_pilot_l40s.sbatch preflight 101
sbatch --job-name=refined-defense-s361506353-v1 \
  scripts/run_refined_defense_pilot_l40s.sbatch preflight 361506353
```

Depois use `start` para cada seed. Após `TIMEOUT`, confira `sacct` e os logs e
submeta apenas a seed afetada com `resume`. Nunca execute dois jobs simultâneos
para o mesmo run ID. `singleton` impede duplicatas com o mesmo job name.

Cada rodada é confirmada somente depois de treinamento, FedAvg, auditoria,
checkpoint e marcador. Rodadas incompletas são repetidas. Checkpoints permanentes
ficam nas rodadas 1, 10 e 20; nas demais há somente um checkpoint móvel. Pesos
usam `safetensors`; accountants são persistidos, mas otimizadores, gradientes,
tokens e conversas não.

Ao final das duas seeds:

```bash
python -m federated_leakage.summarize_refined_defense_pilot \
  --output-root outputs
```

Os totais esperados são 16 trajetórias, 320 rodadas, 328.000 passos, 122.086
gerações greedy e 9.000 avaliações de utilidade. A contagem privada de
apresentações não é fixada: o resultado registra as seleções Poisson realizadas.

## Réplica RTX PRO 6000 Blackwell

A L40S permanece a referência. A RTX PRO 6000 é uma réplica operacional
independente com exatamente os mesmos arquivos `main-v5.yaml` e
`refined-defense-pilot-v1.yaml`. Ela não usa a `.venv` da L40S e não compartilha
diretórios de run.

No headnode, fora de uma alocação Slurm:

```bash
cd /caminho/do/repositorio
scripts/prepare_rtxpro6000_cu128_env.sh
```

O ambiente `.venv-rtxpro6000-cu128/` fixa PyTorch `2.7.1+cu128`. O preparador
usa staging, recusa sobrescrita e, quando o ambiente já existe, apenas o
revalida. O perfil `execution-runtime-profile/v1` exige uma única
`NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`, compute capability
12.0, ao menos 90 GiB, BF16 e `sm_120` presente no build do PyTorch.

Execute os smokes de 1 e 100 passos dentro de uma alocação RTX usando o novo
Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

.venv-rtxpro6000-cu128/bin/python \
  -m federated_leakage.validate_rtxpro6000_runtime

for STEPS in 1 100; do
  .venv-rtxpro6000-cu128/bin/python \
    -m federated_leakage.smoke_private_training \
    --config configs/main-v5.yaml \
    --model-artifact-dir "$PWD/artifacts/models/ae3238fde6675942cac5" \
    --cache-dir artifacts/huggingface --device cuda --seed 101 \
    --epsilon 3 --steps "$STEPS"
done
```

Os hashes das agendas Poisson e de ruído devem ser os mesmos já observados na
L40S. Depois submeta os preflights:

```bash
sbatch --job-name=refined-defense-rtxpro6000-s101-v1 \
  scripts/run_refined_defense_pilot_rtxpro6000.sbatch preflight 101
sbatch --job-name=refined-defense-rtxpro6000-s361506353-v1 \
  scripts/run_refined_defense_pilot_rtxpro6000.sbatch preflight 361506353
```

Substitua `preflight` por `start` somente após ambos passarem. Use `resume` após
timeout ou liberação do gate par. O launcher valida o runtime antes de carregar
o modelo e usa exclusivamente:

```text
outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1/
├── runtime_manifest.json
└── runs/
```

O preflight não cria esse diretório. No `start`, o manifesto é publicado de
forma atômica; duas seeds concorrentes somente o aceitam se o conteúdo for
idêntico. Uma retomada rejeita mudança de driver, versões, arquiteturas CUDA,
GPU ou variáveis de reprodutibilidade. Cache e artefato do modelo são somente
leitura e podem ser compartilhados com a L40S; journals, checkpoints,
auditorias e resultados não podem.

Após as duas seeds, gere primeiro o resumo próprio da RTX e depois a comparação:

```bash
.venv-rtxpro6000-cu128/bin/python \
  -m federated_leakage.summarize_refined_defense_pilot \
  --output-root outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1

.venv-rtxpro6000-cu128/bin/python \
  -m federated_leakage.summarize_refined_runtime_replication \
  --output-root outputs
```

A comparação registra resultados por hardware e classifica as conclusões como
`consistent` ou `runtime_sensitive`. Ela nunca combina médias nem promove a
réplica sobre a referência.
