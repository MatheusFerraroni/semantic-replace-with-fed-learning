# Grade federada de intensidade v2

## Objetivo

A grade mede quanta exposição federada produz vazamento intenso e se o efeito
se repete em duas seeds. Ela é uma calibração de desenvolvimento F0, não a
campanha principal e não inclui defesas.

Os dois runs foram concluídos e a revisão humana selecionou `1e-4 / 4×` como
condição vulnerável do piloto de substituição semântica. Os artefatos da grade
permanecem imutáveis e são validados pelo novo runner; seus pesos não são
reutilizados.

| LR das vítimas | Repetições | Seeds |
| ---: | ---: | --- |
| `3e-5` | `4×`, `8×`, `16×` | `101`, `361506353` |
| `1e-4` | `4×`, `8×`, `16×` | `101`, `361506353` |

Todos os 12 braços começam do Tucano 2 pinado. Cada vítima mantém um AdamW
durante suas repetições e o descarta ao trocar de cliente ou rodada. O auxiliar
benigno usa sempre a receita oficial `3e-5`, uma passagem, 25 passos e peso
FedAvg `1/11`. LR e multiplicador não alteram a seed nem a ordem das amostras.

Por seed são 120 rodadas, 1.132.000 apresentações, 283.000 passos, 12.607
gerações greedy e 3.500 avaliações de utilidade. Os totais conjuntos são o
dobro. O preflight de qualquer seed reconstrói ambas e valida colisões globais.

## Gate e resumo conjunto

Um braço passa em uma seed quando B0 reprova e o endpoint tem ao menos 50 pares
distintivos exatos, 25 vítimas com algum campo distintivo exato e acertos em
dois tipos entre CPF, RG, telefone, e-mail e endereço. O resumo combinado usa:

- `robust`: passa nas duas seeds;
- `unstable`: passa em somente uma;
- `insufficient`: não passa.

`first_robust_arm` prioriza menor learning rate e, dentro dele, menor
multiplicador. O resultado não promove automaticamente uma receita; métricas de
utilidade, mínimo, máximo e diferença entre seeds exigem revisão humana.

## Execução L40S

Use a raiz limpa do repositório e job names distintos. O `singleton` do Slurm
serializa apenas submissões com o mesmo nome, portanto as duas seeds podem rodar
em paralelo:

```bash
sbatch --job-name=federated-grid-s101-v2 \
  scripts/run_federated_memorization_grid_l40s.sbatch preflight 101

sbatch --job-name=federated-grid-s361506353-v2 \
  scripts/run_federated_memorization_grid_l40s.sbatch preflight 361506353
```

Depois de confirmar ambos os preflights, troque `preflight` por `start`. Se um
job atingir o limite de 24 horas, confira `sacct` e os logs e submeta `resume`
com o mesmo job name e seed. Não há requeue automático; a rodada incompleta é
repetida a partir do último checkpoint confirmado.

Após os dois `completed.json`, gere o resumo idempotente:

```bash
python -m federated_leakage.summarize_federated_memorization_grid \
  --config configs/federated-memorization-grid-v2.yaml
```

Os artefatos ficam em `outputs/runs/federated-memorization-grid-seed-<seed>-v2/`
e permanecem fora do Git. A seed 101 exige regressão exata do braço `4×/3e-5`
contra a calibração v1 concluída.
