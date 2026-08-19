# AGENTS.md

## Goal

Build a small, reproducible research implementation that evaluates leakage of
controlled synthetic secrets during full-parameter federated training of
Tucano 2 0.6B.

## Project boundary

- Use synthetic profiles, conversations, secrets and canaries only.
- Never add real datasets, mounted-data paths, corpus ingestion or continual
  pretraining to this project.
- Accept the upstream model or a compatible external Hugging Face artifact
  only through the documented model artifact contract.
- Do not commit datasets, model weights, secret registries, replacement maps,
  caches, checkpoints or generated run outputs.

## Experimental invariants

- Default to `Polygl0t/Tucano2-0.6B-Base` at revision
  `dad97dc864a8f9a1d240fb9351d098f3af9511d7` and label those runs
  `upstream_baseline`.
- Keep the original tokenizer, vocabulary and special tokens unchanged.
- Use a sequence length of 1,024 and train all model parameters in every
  federated condition.
- Every federated scenario has 10 victim clients plus one auxiliary slot.
- Use a benign auxiliary client in F0, F2 and F4, and the matched malicious
  variant in F1, F3 and F5.
- Match the two auxiliary variants on profiles, sample count, local epochs and
  FedAvg weight. Keep update scaling at `1.0`.
- Apply DP-SGD or semantic substitution only to the 10 victim clients.
- Keep victim datasets disjoint. The malicious client must never access victim
  datasets, victim secrets, local updates, auditor files or replacement maps.
- Keep the auditor separate from all training clients.
- Restart and rerun every scenario when the initial model artifact changes.

## Implementation rules

- Keep the implementation simple; prefer plain Python and small reusable
  modules.
- Do not add frameworks, services, databases, dashboards or abstractions unless
  required by the protocol.
- Before adding a dependency, verify that the existing stack cannot meet the
  requirement.
- Make runs reproducible with fixed seeds and versioned configuration files.
- Fail closed when isolation, annotation or secret-overlap checks fail.
- Save metrics separately from model checkpoints and avoid unnecessary
  checkpoints or large intermediate files.
- Prefer resumable scripts for long-running generation, training and auditing.
- Do not silently change experimental assumptions. Record changes in the
  protocol, README or run configuration.
- Report privacy and utility together, including negative and inconclusive
  outcomes.

Optimize for research clarity and reproducibility, not production
infrastructure.
