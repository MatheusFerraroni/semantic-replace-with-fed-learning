# Semantic Replace with Federated Learning

Federated privacy research with Tucano 2 0.6B.

This project evaluates whether federated training can expose controlled,
entirely synthetic secrets in Brazilian Portuguese conversations. It also
compares structural amplification by an auxiliary malicious client with
DP-SGD and semantic substitution defenses.

A negative result is valid. The experiment must not assume that leakage will
occur.

## Current state

This directory currently contains the research protocol and the model handoff
contract only. Training code, executable configurations, generated data and
tests have not been implemented yet.

The complete experimental specification is in [the protocol](docs/protocol.md).
The accepted model interface is defined by [the model artifact
contract](docs/model-artifact-contract.md).

## Default model

The first implementation must start from the published upstream checkpoint:

```yaml
kind: huggingface
model_id: Polygl0t/Tucano2-0.6B-Base
revision: dad97dc864a8f9a1d240fb9351d098f3af9511d7
sequence_length: 1024
```

The original tokenizer, vocabulary and special tokens are immutable. The
model's native context window is 4,096 tokens; experiments use sequences of
1,024 tokens. Federated training uses FedAvg over all model parameters.

Runs made from this checkpoint are labeled `upstream_baseline`. A compatible
locally refined Hugging Face artifact may later replace it through the model
configuration. When that happens, all B0 and F0-F5 scenarios must restart from
the new frozen artifact and be rerun; upstream federated weights and results
are not final results and must not be reused.

## Boundary with model refinement

This project owns only:

- generation of synthetic profiles, conversations, canaries and controls;
- isolated client datasets and full-model federated training;
- the structural-amplification attack and the evaluated defenses;
- extraction auditing, privacy metrics and utility metrics.

It must not contain real datasets, mounted-corpus paths, corpus parsers,
continual-pretraining code or model weights. A refined model enters only as an
external, immutable artifact that satisfies the documented contract.

## Non-negotiable privacy boundary

- Leakage targets are synthetic canaries only; never use real personal data.
- Each victim owns a disjoint local dataset and unique private values.
- The auxiliary malicious client sees only the global model, public schema and
  its own synthetic data.
- Only the auditor can read the victim secret registry and defense replacement
  mappings.
- Model checkpoints, generated data, secrets, caches and run outputs remain
  outside Git.
