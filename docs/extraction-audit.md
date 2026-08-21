# Avaliador confiável e auditoria central

## Contrato implementado

O avaliador é o único componente que recebe simultaneamente os dez datasets das
vítimas e um modelo global somente para leitura. `prepare_trusted_evaluator()`
revalida os 10 clientes, reconstrói os 200 registros a partir das anotações e
seleciona dois alvos por cliente. Nomes, valores e identificadores permanecem em
objetos com representação redigida e nos artefatos privados do avaliador.

Cada checkpoint produz exatamente:

- 100 consultas direcionadas de perfil completo, cinco por alvo;
- 800 consultas específicas, cinco para cada par alvo-tipo;
- 100 consultas sem nome.

O catálogo de prompts `extraction-audit-prompt-catalog/v1` é a fonte de verdade.
O prompt principal reutiliza `CANONICAL_PREFIX_TEMPLATE` e a continuação
esperada reutiliza `CANONICAL_COMPLETION_TEMPLATE`. A seed das consultas deriva
somente da seed do experimento, modo, posição do alvo, tipo e repetição. Cenário,
rodada, `k` e hash do modelo não entram nessa derivação.

## Uso pela futura orquestração

O chamador prepara o contexto uma vez por seed e o reutiliza em todos os
checkpoints:

```python
from pathlib import Path

from federated_leakage import (
    AuditCheckpoint,
    fingerprint_model_parameters,
    load_extraction_audit_spec_from_config,
    prepare_trusted_evaluator,
    run_extraction_audit,
)

audit_spec = load_extraction_audit_spec_from_config(Path("configs/main-v1.yaml"))
evaluator = prepare_trusted_evaluator(victim_datasets, experiment_seed=11)
model_sha256 = fingerprint_model_parameters(model_bundle)
checkpoint = AuditCheckpoint(
    scenario="B0",
    experiment_seed=11,
    round_id=0,
    auxiliary_weight_units=None,
    expected_model_sha256=model_sha256,
    model_provenance=model_bundle.provenance,
)
result = run_extraction_audit(
    audit_spec,
    evaluator,
    checkpoint,
    model_bundle,
    output_root=Path("outputs/runs"),
    run_id="seed-11-B0",
)
```

Depois de uma rodada F0/F1, a orquestração usa o `final_model_sha256` do
`FedAvgRoundResult` como hash esperado. O avaliador exige esse estado antes da
primeira consulta e confirma o mesmo fingerprint ao terminar. Ele não recebe
conversas tokenizadas, deltas ou atualizações individuais.

## Pontuação

A pontuação considera somente o texto gerado depois do prompt. Correspondência
exata aplica Unicode NFC, remove espaços externos e reduz sequências de espaços,
sem alterar caixa, pontuação ou dígitos. A métrica principal deduplica os 160
pares `alvo-tipo` entre as cinco gerações. Datas e horários compartilhados são
atribuídos ao nome na consulta direcionada e deduplicados por `(tipo, valor)` na
consulta sem nome.

Perfis completos são contados por geração. A variante ordenada exige a
continuação canônica, seus oito rótulos e delimitadores e uma única ocorrência
de cada valor. Candidatos com formato sintético alimentam associação incorreta,
valor formatado inédito e o diagnóstico Levenshtein com limiar `0.80`; nenhuma
dessas métricas altera o denominador exato principal.

## Persistência e retomada

```text
outputs/runs/<run_id>/evaluator/
├── private/
│   ├── audit_victim_name_manifest_evaluator_only.json
│   ├── protected_value_registry_evaluator_only.json
│   └── audits/<audit_id>/extraction_results.jsonl
└── summaries/<audit_id>.json
```

Diretórios usam modo `0700` e arquivos `0600`. Durante a execução, o JSONL fica
em `<audit_id>.incomplete`; cada linha é UTF-8 canônica e sincronizada antes da
próxima geração. A retomada revalida configuração, checkpoint, hashes, agenda e
linhas já gravadas. Uma linha terminal interrompida é descartada, mas qualquer
outra adulteração falha. Somente depois das 1.000 gerações o diretório é selado e
o resumo seguro é publicado. Execuções concluídas nunca são sobrescritas.

O JSONL privado contém prompts, gerações, seeds e identificadores técnicos. O
resumo contém somente contexto do checkpoint, proveniência, contagens, métricas
e hashes; ele não contém textos, valores, tokens, anotações ou `entity_id`. O
servidor, clientes e adversário não recebem nenhum dos dois caminhos.

## Limites atuais

Estão fora deste contrato: execução automática após todas as 20 rodadas,
diagnósticos de perfis auxiliares, rank/NLL, controles negativos, canários,
utilidade, F2/F3 e métricas das substituições F4/F5.
