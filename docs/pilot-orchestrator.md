# Orquestrador retomável do piloto B0/F0/F1

## Escopo implementado

O comando `federated_leakage.run_pilot` executa exclusivamente o piloto de
desenvolvimento não privado fixado em `configs/main-v1.yaml`:

- seed `101` e agenda auxiliar `F0-F1`;
- `k=1`, com peso `1/11` por vítima e `1/11` para o auxiliar;
- uma auditoria B0 compartilhada;
- 20 rodadas F0 seguidas de recarga do baseline e 20 rodadas F1;
- auditoria de 20 alvos após todas as rodadas;
- auditorias adicionais de 1, 5 e 200 alvos em B0 e na rodada 20.

Isso totaliza 40 rodadas federadas, 44.000 conversas processadas, 11.000 passos
locais e 69.710 gerações de auditoria. F2-F5, a varredura `k=1..10` e a campanha
principal de 405 execuções não fazem parte deste comando.

## Preflight

O preflight completo ocorre antes de qualquer escrita. Ele gera em memória os
dez datasets das vítimas e os 40 lotes auxiliares pareados, valida colisões e
agendas e descarta os lotes auxiliares. Depois carrega o modelo estritamente
offline, tokeniza as vítimas uma vez e verifica os quatro orçamentos com o
tokenizador real.

```bash
HF_HUB_OFFLINE=1 python -m federated_leakage.run_pilot \
  --config configs/main-v1.yaml \
  --device cuda \
  --preflight-only
```

`--preflight-only` não cria datasets, diretórios da execução, auditorias ou
checkpoints. O dispositivo solicitado deve existir; não há fallback para CPU.

## Execução e retomada

```bash
HF_HUB_OFFLINE=1 python -m federated_leakage.run_pilot \
  --config configs/main-v1.yaml \
  --device cuda
```

O `run_id` padrão é `pilot-seed-101-k01`. Uma segunda chamada compatível retoma
o último estado confirmado; se `completed.json` já existir, todos os artefatos
necessários são relidos e revalidados sem repetir treinamento ou gerações.
Ambas as trajetórias registram o fingerprint do mesmo baseline e o hash agregado
das quatro auditorias B0 compartilhadas.
`--fresh` recusa qualquer diretório de execução existente. Os argumentos
operacionais opcionais são:

- `--cache-dir`: cache local do snapshot Hugging Face pinado;
- `--model-artifact-dir`: artefato local absoluto compatível com o contrato v1;
- `--output-root`: raiz que conterá `datasets/` e `runs/`;
- `--run-id`: identificador seguro sem separadores de caminho.

O terminal publica apenas progresso, contagens, métricas agregadas, hashes e o
destino. Nomes, textos, tokens, valores protegidos, entidades, deltas e pesos não
são impressos.

## Transação de uma rodada

Cada trajetória materializa apenas a sua apresentação auxiliar da rodada atual.
Os 11 clientes começam do mesmo snapshot global, e cada delta é consumido
imediatamente pelo acumulador FedAvg. Uma rodada só é confirmada depois de:

1. treinamento local e aplicação atômica do FedAvg;
2. auditoria e validação do fingerprint do modelo;
3. publicação atômica do checkpoint `safetensors`;
4. publicação do commit da rodada e atualização de `state.json`.

Na F1, o resultado FedAvg e as auditorias também são comparados com a mesma
rodada F0 antes da confirmação. Uma falha anterior ao commit restaura o snapshot
da rodada. Um checkpoint completo deixado antes do commit pode concluir a
transação após revalidação; qualquer candidato incompatível é descartado e a
rodada é reproduzida.

Os checkpoints permanentes são as rodadas 1, 10 e 20. Nas demais rodadas existe
somente um checkpoint móvel de retomada. Cada checkpoint contém o modelo BF16 em
`model.safetensors`, resultado seguro da rodada, metadados/hashes e estados RNG
de CPU e do dispositivo. Ele não contém otimizadores, deltas, tokens, textos ou
registros protegidos.

## Árvore de saída

```text
outputs/
├── datasets/pilot-seed-101-k01-dataset-v4/clients/
│   ├── victim/<client_id>/conversations.jsonl
│   └── auxiliary/F0-F1/<presentation>/round-N/conversations.jsonl
└── runs/pilot-seed-101-k01/
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

## Aceitação científica

Concluir o comando cria um marcador revisável, mas não congela automaticamente
a receita da campanha principal. Antes de promover a configuração, é obrigatória
a revisão humana das métricas, estabilidade, custo, memória, saídas privadas e
limitações do piloto.
