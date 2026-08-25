# Orquestrador retomável do piloto B0/F0/F1

## Escopo implementado

Este runner só é liberado depois de concluir e aprovar a calibração greedy v2.
A ordem operacional é calibração preflight → calibração completa → revisão do
gate → piloto preflight → piloto completo. Um gate negativo interrompe o fluxo.

O comando `federated_leakage.run_pilot` executa exclusivamente o piloto de
desenvolvimento não privado fixado em `configs/main-v2.yaml`:

- seed `101` e agenda auxiliar `F0-F1`;
- `k=1`, com peso `1/11` por vítima e `1/11` para o auxiliar;
- uma auditoria B0 compartilhada;
- 20 rodadas F0 seguidas de recarga do baseline e 20 rodadas F1;
- auditoria de 20 alvos após todas as rodadas;
- auditorias adicionais de 1, 5 e 200 alvos em B0 e na rodada 20.

Isso totaliza 40 rodadas federadas, 44.000 conversas processadas, 11.000 passos
locais e 12.992 gerações de auditoria greedy. F2-F5, a varredura `k=1..10` e a campanha
principal de 405 execuções não fazem parte deste comando.

B0 usa 2.038 gerações: 181 para o orçamento de referência e 10, 46 e 1.801
para as sensibilidades. Cada trajetória usa 5.477: 20 auditorias de 181, mais
10, 46 e 1.801 na rodada 20. Orçamentos permanecem em artefatos separados.

## Contrato de determinismo CUDA

Antes de iniciar qualquer processo Python que use CUDA determinístico, o
ambiente deve conter exatamente:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Esse valor integra a receita em `reproducibility.cuda_cublas_workspace_config`.
O piloto, o treinamento local e o avaliador validam a variável antes de usar o
modelo ou o RNG. Valor ausente ou divergente encerra a execução; o Python não
define, substitui nem corrige o ambiente. CPU e MPS não exigem essa variável.

## Preflight

O preflight completo ocorre antes de qualquer escrita. Ele gera em memória os
dez datasets das vítimas e os 40 lotes auxiliares pareados, valida colisões e
agendas e descarta os lotes auxiliares. Depois carrega o modelo estritamente
offline, tokeniza as vítimas uma vez e verifica os quatro orçamentos com o
tokenizador real.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v2.yaml \
  --device cuda \
  --preflight-only
```

`--preflight-only` não cria datasets, diretórios da execução, auditorias ou
checkpoints. Antes dele, o runner exige o marcador concluído da calibração
`memorization-calibration-greedy-seed-101-v2`: algum braço deve ter passado o
gate e o baseline não. O resultado, manifesto, dataset canário, preflight de
colisões, estratégia, baseline e proveniência devem coincidir com a configuração
v2. O dispositivo solicitado deve existir; não há fallback
para CPU.

## Execução e retomada

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m federated_leakage.run_pilot \
  --config configs/main-v2.yaml \
  --device cuda
```

O `run_id` padrão é `pilot-greedy-seed-101-k01-v2`. Uma segunda chamada compatível retoma
o último estado confirmado; se `completed.json` já existir, todos os artefatos
necessários são relidos e revalidados sem repetir treinamento ou gerações.
Ambas as trajetórias registram o fingerprint do mesmo baseline e o hash agregado
das quatro auditorias B0 compartilhadas. Na rodada 1, F0 e F1 partem desse mesmo
baseline. A partir da rodada 2, cada cenário parte exclusivamente de seu próprio
modelo final anterior; a validação pareada compara dados, pesos, seeds, agenda e
proveniência, mas não exige que os modelos correntes dos dois braços coincidam.
`--fresh` recusa qualquer diretório de execução existente. Os argumentos
operacionais opcionais são:

- `--cache-dir`: cache local do snapshot Hugging Face pinado;
- `--model-artifact-dir`: artefato local absoluto compatível com o contrato v1;
- `--output-root`: raiz que conterá `datasets/` e `runs/`;
- `--run-id`: identificador seguro sem separadores de caminho.

O terminal publica apenas progresso, contagens, métricas agregadas, hashes e o
destino. Nomes, textos, tokens, valores protegidos, entidades, deltas e pesos não
são impressos.

## Execução Slurm em uma L40S

O launcher `scripts/run_pilot_l40s.sbatch` é a interface versionada para o
cluster. Ele deve ser submetido a partir da raiz do repositório, usa
`.venv/bin/python` e fixa uma task, uma L40S, oito CPUs, 64 GiB de RAM e 24 horas.
Duas GPUs não são usadas por um mesmo processo.

O primeiro argumento é obrigatório e aceita somente:

- `preflight`: executa a CLI com `--preflight-only` e não publica datasets,
  checkpoints nem auditorias;
- `start`: acrescenta `--fresh` e recusa o diretório da execução desse `run_id`
  quando ele já existe;
- `resume`: exige o diretório do mesmo `run_id`, não usa `--fresh`, revalida os
  artefatos e continua do último checkpoint confirmado.

Execute somente um modo conforme a etapa. Para validar sem escrever, submeta:

```bash
sbatch scripts/run_pilot_l40s.sbatch preflight
```

Depois da revisão explícita do preflight, inicie o piloto com:

```bash
sbatch scripts/run_pilot_l40s.sbatch start
```

Somente para uma execução oficial já existente, retome com:

```bash
sbatch scripts/run_pilot_l40s.sbatch resume
```

O launcher fixa a configuração `configs/main-v2.yaml`, o cache
`artifacts/huggingface`, a raiz `outputs/` e o `run_id`
`pilot-greedy-seed-101-k01-v2`. Antes de chamar `srun`, ele exporta
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false` e
`PYTHONUNBUFFERED=1`. Ele recusa execução fora do Slurm, sem GPU, com Python
diferente de 3.12, dependências quebradas ou alterações rastreadas no worktree.

A diretiva de dependência `singleton` serializa jobs com o mesmo nome. Ela não
substitui a identidade científica: nunca devem existir duas submissões para o
mesmo `run_id`. Os logs `slurm-%x-%j.out` e `slurm-%x-%j.err` são gravados na
raiz do repositório, contêm apenas contexto técnico e progresso seguro e ficam
fora do Git.

Não há requeue automático. Se o Slurm encerrar o job com `TIMEOUT`, consulte o
estado e os logs antes de retomar:

```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
sbatch scripts/run_pilot_l40s.sbatch resume
```

O modo `resume` reaproveita o último checkpoint confirmado. Resíduos de uma
rodada não confirmada são revalidados e, se não puderem concluir a transação,
essa rodada é reproduzida deterministicamente.

## Transação de uma rodada

Cada trajetória materializa apenas a sua apresentação auxiliar da rodada atual.
Os 11 clientes começam do mesmo snapshot global, e cada delta é consumido
imediatamente pelo acumulador FedAvg. Uma rodada só é confirmada depois de:

1. treinamento local e aplicação atômica do FedAvg;
2. auditoria e validação do fingerprint do modelo;
3. publicação atômica do checkpoint `safetensors`;
4. publicação do commit da rodada e atualização de `state.json`.

Na F1, o resultado FedAvg e as auditorias também são comparados com a mesma
rodada F0 antes da confirmação. A rodada 1 deve começar do baseline compartilhado;
nas demais, o estado inicial de cada braço deve ser exatamente o estado final da
rodada anterior daquele cenário. Essa continuidade é revalidada ao carregar um
prefixo confirmado e ao recuperar um checkpoint. Uma falha anterior ao commit
restaura o snapshot da rodada. Um checkpoint completo deixado antes do commit
pode concluir a transação após revalidação; qualquer candidato incompatível é
descartado e a rodada é reproduzida. Auditorias já concluídas só são reutilizadas
quando o fingerprint do modelo e todos os metadados esperados coincidem.

Os checkpoints permanentes são as rodadas 1, 10 e 20. Nas demais rodadas existe
somente um checkpoint móvel de retomada. Cada checkpoint contém o modelo BF16 em
`model.safetensors`, resultado seguro da rodada, metadados/hashes e estados RNG
de CPU e do dispositivo. Ele não contém otimizadores, deltas, tokens, textos ou
registros protegidos.

## Árvore de saída

```text
outputs/
├── datasets/pilot-greedy-seed-101-k01-v2-dataset-v4/clients/
│   ├── victim/<client_id>/conversations.jsonl
│   └── auxiliary/F0-F1/<presentation>/round-N/conversations.jsonl
└── runs/pilot-greedy-seed-101-k01-v2/
    ├── run_manifest.json
    ├── baseline/
    │   ├── completed.json
    │   └── evaluator/
    ├── trajectories/
    │   ├── F0-k01/
    │   │   ├── rounds/
    │   │   ├── checkpoints/
    │   │   └── evaluator/
    │   └── F1-k01/
    │       ├── rounds/
    │       ├── checkpoints/
    │       └── evaluator/
    ├── paired/round-N.json
    └── completed.json
```

Os JSONL de conversas são restritos ao cliente e permanecem fora do Git. As
gerações e o registro correto ficam somente em `evaluator/private/`; os resumos
seguros ficam em `evaluator/summaries/`. Checkpoints, caches, datasets e toda a
árvore de execução também permanecem fora do controle de versão.

O piloto e seus checkpoints usam `pilot-execution/v2`,
`federated-trajectory/v2` e `federated-checkpoint/v2`. A execução sampling v1 em
`outputs/runs/pilot-seed-101-k01/` permanece somente como histórico e nunca é
retomada ou combinada com o piloto greedy.

## Aceitação científica

Concluir o comando cria um marcador revisável, mas não congela automaticamente
a receita da campanha principal. Antes de promover a configuração, é obrigatória
a revisão humana das métricas, estabilidade, custo, memória, saídas privadas e
limitações do piloto.
