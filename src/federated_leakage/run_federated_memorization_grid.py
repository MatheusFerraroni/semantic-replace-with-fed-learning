"""CLI offline da grade federada de intensidade v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .federated_grid_calibration import run_federated_memorization_grid
from .federated_grid_contracts import (
    EXPERIMENT_SEEDS,
    FederatedGridError,
    FederatedGridPreflightResult,
    FederatedGridSeedResult,
    default_run_id,
    load_federated_grid_spec_from_config,
)
from .model_contracts import DEFAULT_MODEL_CACHE
from .model_loading import ModelArtifactError, ModelConfigurationError, ModelDependencyError, ModelLoadError
from .reproducibility import ReproducibilityEnvironmentError, validate_cuda_reproducibility_environment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa um dos dois runs da grade federada v2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser


def _progress(payload: Mapping[str, object]) -> None:
    print("progresso: " + json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True), flush=True)


def _summary(result: FederatedGridSeedResult | FederatedGridPreflightResult, output_root: Path) -> None:
    if isinstance(result, FederatedGridPreflightResult):
        print("status: preflight da grade federada validado")
        print(f"seed_selecionada: {result.selected_seed}")
        print(f"seeds_validadas: {','.join(str(value) for value in result.validated_seeds)}")
        print(f"conversas_vitima: {result.victim_conversation_count}")
        print(f"conversas_auxiliares: {result.auxiliary_conversation_count}")
        print(f"conversas_utilidade: {result.utility_conversation_count}")
        print(f"colisoes_cruzadas_sha256: {result.cross_seed_collision_preflight_sha256}")
        print(f"modelo_sha256: {result.model_state_sha256}")
        print("escrita: nao")
        return
    print("status: grade federada da seed concluída")
    print(f"run_id: {result.run_id}")
    print(f"seed: {result.experiment_seed}")
    print(f"bracos: {len(result.arms)}")
    print(f"rodadas_federadas: {result.total_federated_rounds}")
    print(f"apresentacoes_conversas: {result.total_conversation_presentations}")
    print(f"passos_otimizador: {result.total_optimizer_steps}")
    print(f"geracoes_auditoria: {result.total_audit_generations}")
    print(f"conversas_utilidade: {result.total_utility_conversations}")
    print(f"baseline_atingiu_gate: {'sim' if result.baseline_gate_passed else 'nao'}")
    print(f"algum_braco_atingiu_gate: {'sim' if result.any_arm_passed else 'nao'}")
    print(f"primeiro_braco_bem_sucedido: {result.first_successful_arm}")
    print(f"resultado_sha256: {result.result_sha256}")
    print(f"saida: {output_root / 'runs' / result.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    run_id = arguments.run_id or default_run_id(arguments.seed)
    try:
        validate_cuda_reproducibility_environment(arguments.device)
        spec = load_federated_grid_spec_from_config(arguments.config)
        result = run_federated_memorization_grid(
            spec,
            seed=arguments.seed,
            config_path=arguments.config,
            output_root=arguments.output_root,
            run_id=run_id,
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_progress,
        )
    except (FileExistsError, FederatedGridError, ModelArtifactError, ModelConfigurationError, ModelDependencyError, ModelLoadError, ReproducibilityEnvironmentError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada durante a grade federada", file=sys.stderr)
        return 1
    _summary(result, arguments.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
