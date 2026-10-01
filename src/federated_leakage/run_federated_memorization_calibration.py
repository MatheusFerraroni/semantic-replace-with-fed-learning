"""CLI offline da calibração federada de exposição local."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .federated_exposure_calibration import (
    run_federated_memorization_calibration,
)
from .federated_exposure_contracts import (
    DEFAULT_RUN_ID,
    FederatedExposureError,
    FederatedExposurePreflightResult,
    FederatedMemorizationCalibrationResult,
    load_federated_exposure_spec_from_config,
)
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
        description=(
            "Executa ou retoma a calibração F0 de exposição local das vítimas."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        required=True,
        help="Dispositivo exato, sem fallback automático.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser


def _print_progress(payload: Mapping[str, object]) -> None:
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


def _print_summary(
    result: FederatedMemorizationCalibrationResult
    | FederatedExposurePreflightResult,
    *,
    output_root: Path,
) -> None:
    if isinstance(result, FederatedExposurePreflightResult):
        print("status: preflight da calibração federada validado")
        print(f"seed: {result.experiment_seed}")
        print(f"clientes_vitima: {result.victim_client_count}")
        print(f"conversas_vitima: {result.victim_conversation_count}")
        print(f"rodadas_auxiliares: {result.auxiliary_round_count}")
        print(f"conversas_auxiliares: {result.auxiliary_conversation_count}")
        print(f"perfis_utilidade: {result.utility_profile_count}")
        print(f"conversas_utilidade: {result.utility_conversation_count}")
        print(f"vitimas_sha256: {result.victim_dataset_sha256}")
        print(f"agenda_f0_sha256: {result.benign_schedule_sha256}")
        print(f"utilidade_sha256: {result.utility_dataset_sha256}")
        print(f"modelo_sha256: {result.model_state_sha256}")
        print("escrita: nao")
        return
    print("status: calibração federada concluída")
    print(f"run_id: {result.run_id}")
    print(f"seed: {result.experiment_seed}")
    print(f"bracos: {len(result.arms)}")
    print(f"rodadas_federadas: {result.total_federated_rounds}")
    print(
        "apresentacoes_conversas: "
        f"{result.total_conversation_presentations}"
    )
    print(f"passos_otimizador: {result.total_optimizer_steps}")
    print(f"geracoes_auditoria: {result.total_audit_generations}")
    print(f"conversas_utilidade: {result.total_utility_conversations}")
    print(f"baseline_atingiu_gate: {'sim' if result.baseline_gate_passed else 'nao'}")
    print(f"calibrated: {'sim' if result.calibrated else 'nao'}")
    print(f"primeiro_multiplicador_bem_sucedido: {result.first_successful_multiplier}")
    print(f"resultado_sha256: {result.result_sha256}")
    print(f"saida: {Path(output_root) / 'runs' / result.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    try:
        validate_cuda_reproducibility_environment(arguments.device)
        spec = load_federated_exposure_spec_from_config(arguments.config)
        result = run_federated_memorization_calibration(
            spec,
            config_path=arguments.config,
            output_root=arguments.output_root,
            run_id=arguments.run_id,
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_print_progress,
        )
    except (
        FileExistsError,
        FederatedExposureError,
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
        ReproducibilityEnvironmentError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada durante a calibração federada", file=sys.stderr)
        return 1
    _print_summary(result, output_root=arguments.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
