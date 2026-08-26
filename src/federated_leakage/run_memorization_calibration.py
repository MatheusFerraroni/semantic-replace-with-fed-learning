"""CLI offline da calibração vulnerável com canários completos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .calibration_contracts import (
    MemorizationCalibrationError,
    MemorizationCalibrationPreflightResult,
    MemorizationCalibrationResult,
    load_memorization_calibration_spec_from_config,
)
from .memorization_calibration import run_memorization_calibration
from .model_contracts import DEFAULT_MODEL_CACHE
from .model_loading import (
    ModelArtifactError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
)
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa ou retoma a calibração positiva vulnerável fixada."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser


def _progress(payload: Mapping[str, object]) -> None:
    print(
        "progresso: "
        + json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _summary(
    result: MemorizationCalibrationPreflightResult | MemorizationCalibrationResult,
    output_root: Path,
) -> None:
    if isinstance(result, MemorizationCalibrationPreflightResult):
        print("status: preflight da calibração validado")
        print(f"seed: {result.experiment_seed}")
        print(f"perfis_canario: {result.canary_profile_count}")
        print(f"conversas_canario: {result.canary_conversation_count}")
        print(f"rodadas_auxiliares_preflight: {result.auxiliary_round_count}")
        print(f"modelo_sha256: {result.model_state_sha256}")
        print(f"dataset_sha256: {result.canary_dataset_sha256}")
        print("escrita: nao")
        return
    print("status: calibração concluída")
    print(f"run_id: {result.run_id}")
    print(f"seed: {result.experiment_seed}")
    print(f"bracos: {len(result.arms)}")
    print(f"apresentacoes_conversas: {result.total_conversation_presentations}")
    print(f"passos_otimizador: {result.total_optimizer_steps}")
    print(f"geracoes_auditoria: {result.total_audit_generations}")
    print(
        "baseline_atingiu_gate: "
        f"{'sim' if result.baseline_gate_passed else 'nao'}"
    )
    print(f"calibrated: {'sim' if result.calibrated else 'nao'}")
    print(f"primeiro_braco_bem_sucedido: {result.first_successful_arm_id}")
    print(
        "primeiro_learning_rate_bem_sucedido_milionesimos: "
        f"{result.first_successful_learning_rate_millionths}"
    )
    print(f"resultado_sha256: {result.result_sha256}")
    print(f"saida: {Path(output_root) / 'runs' / result.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    try:
        validate_cuda_reproducibility_environment(arguments.device)
        spec = load_memorization_calibration_spec_from_config(arguments.config)
        result = run_memorization_calibration(
            spec,
            output_root=arguments.output_root,
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
            run_id=arguments.run_id,
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_progress,
        )
    except (
        FileExistsError,
        MemorizationCalibrationError,
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
        ReproducibilityEnvironmentError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada durante a calibração", file=sys.stderr)
        return 1
    _summary(result, arguments.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
