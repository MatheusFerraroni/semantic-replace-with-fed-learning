"""Smoke offline de um ou cem passos DP-AdamW sem persistir pesos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .dp_contracts import PrivateTrainingError, load_dp_accounting_spec_from_config
from .federated_round import prepare_victim_training_inputs
from .model_contracts import DEFAULT_MODEL_CACHE, ModelArtifactError, ModelConfigurationError, ModelDependencyError, ModelLoadError
from .model_loading import load_model_bundle, load_model_spec_from_config
from .private_training import diagnose_private_local_training
from .synthetic_profiles import VictimDatasetGenerator
from .training_contracts import load_local_training_spec_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa diagnóstico DP de 1 ou 100 passos em uma vítima."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-artifact-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--seed", type=int, choices=(101, 361506353), default=101)
    parser.add_argument("--epsilon", type=float, choices=(3.0, 8.0), default=3.0)
    parser.add_argument("--steps", type=int, choices=(1, 100), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.model_artifact_dir.is_absolute():
        print("erro: --model-artifact-dir deve ser absoluto", file=sys.stderr)
        return 2
    try:
        bundle = load_model_bundle(
            load_model_spec_from_config(arguments.config),
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
        )
        victims = VictimDatasetGenerator(arguments.seed).generate()
        prepared = prepare_victim_training_inputs(victims, bundle)
        result = diagnose_private_local_training(
            prepared.client_samples[0],
            bundle,
            load_local_training_spec_from_config(arguments.config),
            load_dp_accounting_spec_from_config(arguments.config),
            seed=arguments.seed,
            target_epsilon=arguments.epsilon,
            optimizer_steps=arguments.steps,
        )
    except (
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
        PrivateTrainingError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada no smoke privado", file=sys.stderr)
        return 1
    print("status: smoke privado validado")
    print(f"passos: {result.optimizer_steps}")
    print(f"selecoes_poisson: {result.sampled_conversation_count}")
    print(f"epsilon_parcial: {result.realized_epsilon}")
    print(f"modelo_alterado_durante_smoke: {'sim' if result.model_changed else 'nao'}")
    print(f"modelo_restaurado: {'sim' if result.model_restored else 'nao'}")
    print(f"agenda_poisson_sha256: {result.sample_schedule_sha256}")
    print(f"agenda_ruido_sha256: {result.noise_schedule_sha256}")
    print("escrita: nao")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
