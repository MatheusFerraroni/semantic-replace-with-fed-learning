# Piloto de substituição semântica rotativa

## Escopo

O contrato `semantic-substitution-pilot/v1` executa, separadamente para as seeds
`101` e `361506353`, as trajetórias F0, F1, F4 e F5 com `k=1` e 20 rodadas.
Todas começam do Tucano 2 pinado. As vítimas usam AdamW `1e-4` e quatro
passagens locais; o auxiliar usa `3e-5`, uma passagem e 25 passos.

F0/F1 recebem os perfis sintéticos originais. Em F4/F5, somente os dez
clientes-vítima substituem localmente os nove campos antes da tokenização. O
auxiliar não recebe a defesa, e o servidor recebe apenas deltas para FedAvg.

## Substituição por rodada

O gerador deriva um perfil substituto de `(seed, client_id, entity_id,
round_id)`. Cenário, `k`, modelo e estado do treinamento não entram na
derivação. Portanto, F4 e F5 reconstroem a mesma agenda sem compartilhar mapas.

- nome, nascimento, CPF, RG, telefone, e-mail, endereço, data e horário são
  substituídos;
- as quatro conversas protegidas usam o mesmo perfil substituto na rodada;
- as quatro passagens locais reutilizam a mesma tokenização;
- a rodada seguinte recebe outro perfil substituto;
- a conversa geral e todos os IDs técnicos permanecem iguais;
- o próprio original e substitutos anteriores da entidade são proibidos;
- nome, CPF, RG, telefone, e-mail e endereço também não podem coincidir com
  nenhum original dos fluxos validados;
- colisões entre valores falsos são permitidas e relatadas apenas como
  multiplicidades seguras.

Conversas substituídas e mapas nunca são gravados nos clientes. O avaliador
confiável reconstrói o mapa independentemente e mantém valores somente em sua
área privada, com diretórios `0700` e arquivos `0600`.

## Auditoria e decisão

A inferência continua greedy, sem sampling nem consumo de RNG. B0 audita 200
nomes originais. F0/F1 auditam 20 alvos nas rodadas 1–19 e 200 na rodada 20.
F4/F5 auditam nomes originais e aliases atuais com a mesma agenda; no endpoint,
também consultam os aliases das 19 rodadas anteriores para 20 participantes.

O conjunto fixo soma 40.083 gerações por seed. Cada geração é pontuada contra
originais, substitutos correntes, substitutos históricos e outros valores
conhecidos. Pares são deduplicados por `(alias, tipo, valor)`. Alias compartilhado
é marcado como ambíguo e não produz atribuição por entidade.

A seed é `approved` somente quando B0 reprova, F0 e F1 têm sinal comparável,
F4/F5 reduzem em pelo menos 90% os pares originais exatos de seus comparadores e
nenhum perfil original completo aparece nas condições defendidas. Falta de sinal
em F0 ou F1 produz `inconclusive`. O resumo combinado exige aprovação nas duas
seeds. Extração dos substitutos, ambiguidades e utilidade são descritivas.

## Execução na L40S

O preflight das duas seeds deve terminar antes de qualquer `start`:

```bash
sbatch --job-name=semantic-substitution-s101-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch preflight 101

sbatch --job-name=semantic-substitution-s361506353-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch preflight 361506353
```

Depois da revisão dos logs, os dois runs podem ocupar L40S distintas:

```bash
sbatch --job-name=semantic-substitution-s101-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch start 101

sbatch --job-name=semantic-substitution-s361506353-v1 \
  scripts/run_semantic_substitution_pilot_l40s.sbatch start 361506353
```

Após `TIMEOUT`, confirmar o estado com `sacct` e trocar somente `start` por
`resume` para a seed afetada. Nunca executar dois jobs simultâneos com o mesmo
job name/run ID. Ao final das duas seeds:

```bash
python -m federated_leakage.summarize_semantic_substitution_pilot \
  --config configs/semantic-substitution-pilot-v1.yaml
```

Os runs ficam em `outputs/runs/semantic-substitution-upstream-seed-<seed>-v1/`.
O resumo conjunto fica em
`outputs/runs/semantic-substitution-upstream-combined-v1/combined.json`.

## Limite atual

Este piloto aceita somente o baseline upstream pinado. O Tucano refinado de
outro projeto requer um contrato de artefato coordenado novo, reexportação e nova
calibração B0/F0 antes de F1/F4/F5.
