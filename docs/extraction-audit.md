# Avaliador confiável e auditoria central

## Contrato implementado

O avaliador é o único componente que recebe simultaneamente os dez datasets das
vítimas e um modelo global somente para leitura. `prepare_trusted_evaluator()`
revalida os 10 clientes, reconstrói os 200 registros a partir das anotações e
constrói uma agenda mestre determinística e aninhada. O orçamento de 20 alvos
seleciona exatamente dois por cliente; também são aceitos 1, 5 e 200 alvos.
Nomes, valores e identificadores permanecem em objetos com representação
redigida e nos artefatos privados do avaliador.

Para `n` alvos, cada checkpoint produz `9n + 1` gerações:

- `n` consultas direcionadas de perfil completo;
- `8n` consultas específicas, uma para cada par alvo-tipo;
- uma consulta sem nome.

Assim, os orçamentos de 1, 5, 20 e 200 alvos produzem respectivamente 10, 46,
181 e 1.801 gerações.

Cada chamada usa exatamente `do_sample=False`, `num_beams=1`,
`num_return_sequences=1`, `repetition_penalty=1.0` e `use_cache=True`.
`temperature`, `top_p` e `top_k` não pertencem à configuração v2 nem são
enviados ao modelo. Greedy escolhe o argmax condicional em cada passo; isso não
equivale a buscar a sequência completa de maior probabilidade.
Se o snapshot trouxer defaults de amostragem em `generation_config.json`, o
avaliador os neutraliza apenas durante `generate()` e restaura o objeto ao sair,
inclusive em caso de falha.

O catálogo de prompts `extraction-audit-prompt-catalog/v1` é a fonte de verdade.
O prompt principal reutiliza `CANONICAL_PREFIX_TEMPLATE` e a continuação
esperada reutiliza `CANONICAL_COMPLETION_TEMPLATE`. A seleção aninhada dos alvos
continua derivada somente da seed do experimento. A geração é greedy token a
token e não recebe seed, não chama `manual_seed` e não consome RNG. Cenário,
rodada, `k`, orçamento e hash do modelo não alteram prompts compartilhados.

## Uso pelo orquestrador

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

audit_spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
evaluator = prepare_trusted_evaluator(
    victim_datasets,
    experiment_seed=101,
    target_count=20,
)
model_sha256 = fingerprint_model_parameters(model_bundle)
checkpoint = AuditCheckpoint(
    scenario="B0",
    experiment_seed=101,
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
    run_id="pilot-greedy-seed-101-B0-v2",
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
pares `alvo-tipo` na única geração direcionada. Datas e horários compartilhados são
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
│   ├── protected_value_registry_evaluator_only.json
│   ├── target_manifests/targets-<NNN>.json
│   └── audits/<audit_id>/extraction_results.jsonl
└── summaries/<audit_id>.json
```

Diretórios usam modo `0700` e arquivos `0600`. Durante a execução, o JSONL fica
em `<audit_id>.incomplete`; cada linha é UTF-8 canônica e sincronizada antes da
próxima geração. A retomada revalida configuração, checkpoint, hashes, agenda e
linhas já gravadas. Uma linha terminal interrompida é descartada, mas qualquer
outra adulteração falha. Somente depois de todas as gerações do orçamento o
diretório é selado e o resumo seguro é publicado. Uma auditoria concluída é
relida e revalidada quando a mesma identidade é retomada; nunca é sobrescrita.
O `audit_id` inclui `targets-001`, `targets-005`, `targets-020` ou `targets-200`.

O JSONL privado contém prompts, gerações e identificadores técnicos, mas não
contém seeds nem índices de réplica. O
resumo contém somente contexto do checkpoint, proveniência, contagens, métricas
e hashes; ele não contém textos, valores, tokens, anotações ou `entity_id`. O
servidor, clientes e adversário não recebem nenhum dos dois caminhos.

## Limites atuais

Os contratos ativos são `extraction-audit/v2`,
`extraction-audit-record/v2`, `extraction-audit-result/v3` e
`extraction-audit-journal/v3`, com
`decoding_strategy=tokenwise_greedy_argmax/v1` e `rng_used=false`. Os leitores
v1 existem apenas para inspeção histórica e nunca habilitam geração ou retomada.

O piloto B0/F0/F1 invoca automaticamente a auditoria depois de cada uma das
20 rodadas. Continuam fora deste contrato diagnósticos de perfis auxiliares,
rank/NLL, controles negativos, utilidade, F2/F3 e métricas das substituições
F4/F5. O controle positivo canário possui executor paralelo próprio, descrito em
`memorization-calibration.md`, sem alterar os contratos B0/F0/F1 desta página.
