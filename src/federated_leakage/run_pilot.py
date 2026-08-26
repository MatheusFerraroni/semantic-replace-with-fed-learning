"""CLI offline do piloto pareado B0/F0/F1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .execution_contracts import (
    PILOT_DEFAULT_RUN_ID,
    PilotExecutionError,
    PilotExecutionResult,
    PilotPreflightResult,
    build_pilot_run_identity,
    load_pilot_execution_spec_from_config,
)
from .calibration_gate import load_completed_calibration_gate
from .model_contracts import DEFAULT_MODEL_CACHE
from .model_loading import (
    ModelArtifactError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
)
from .pilot_execution import run_paired_pilot
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)


DEFAULT_OUTPUT_ROOT = Path("outputs")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa ou retoma offline o piloto pareado B0/F0/F1 fixado pelo protocolo."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuração principal versionada do experimento.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        required=True,
        help="Dispositivo exato, sem fallback automático.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="Cache local do snapshot Hugging Face pinado.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        help="Diretório absoluto do artefato local quando configurado.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Raiz contendo datasets/ e runs/.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Identificador seguro opcional; o padrão é "
            f"{PILOT_DEFAULT_RUN_ID}."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Valida dados, modelo, tokenização e auditoria sem escrever ou treinar.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recusa qualquer execução existente em vez de retomá-la.",
    )
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
    result: PilotExecutionResult | PilotPreflightResult,
    *,
    output_root: Path,
    preflight_only: bool,
) -> None:
    if isinstance(result, PilotPreflightResult):
        print("status: preflight validado")
        print(f"seed: {result.experiment_seed}")
        print(f"clientes_vitima: {result.victim_client_count}")
        print(f"conversas_vitima: {result.victim_conversation_count}")
        print(f"rodadas_auxiliares: {result.auxiliary_round_count}")
        print(f"conversas_auxiliares: {result.auxiliary_conversation_count}")
        print(f"perfis_utilidade: {result.utility_profile_count}")
        print(f"conversas_utilidade: {result.utility_conversation_count}")
        print(f"utilidade_sha256: {result.utility_dataset_sha256}")
        print(f"modelo_sha256: {result.model_state_sha256}")
        print(f"agenda_pareada_sha256: {result.paired_schedule_sha256}")
        print(f"escrita: {'nao' if preflight_only else 'sim'}")
        return
    print("status: piloto concluido")
    print(f"run_id: {result.identity.run_id}")
    print(f"seed: {result.identity.experiment_seed}")
    print(f"k: {result.identity.auxiliary_weight_units}")
    print(f"rodadas_federadas: {result.total_federated_rounds}")
    print(f"conversas_processadas: {result.total_conversation_count}")
    print(f"passos_otimizador: {result.total_optimizer_steps}")
    print(f"geracoes_auditoria: {result.total_audit_generations}")
    print("conversas_utilidade_avaliadas: 1500")
    for comparison in result.utility_comparisons:
        print(
            f"utilidade_{comparison.scenario}_delta_perplexidade: "
            f"{comparison.perplexity_delta}"
        )
    print(f"pareamento_sha256: {result.paired_results_sha256}")
    print(f"saida: {Path(output_root) / 'runs' / result.identity.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    try:
        validate_cuda_reproducibility_environment(arguments.device)
        spec = load_pilot_execution_spec_from_config(arguments.config)
        calibration_gate = load_completed_calibration_gate(
            arguments.output_root, spec
        )
        identity = build_pilot_run_identity(
            spec,
            run_id=arguments.run_id,
            calibration_result_sha256=calibration_gate.result_sha256,
            calibration_manifest_sha256=calibration_gate.manifest_sha256,
        )
        result = run_paired_pilot(
            spec,
            identity,
            config_path=arguments.config,
            output_root=arguments.output_root,
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_print_progress,
        )
    except (
        FileExistsError,
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
        PilotExecutionError,
        ReproducibilityEnvironmentError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada durante o piloto", file=sys.stderr)
        return 1
    _print_summary(
        result,
        output_root=arguments.output_root,
        preflight_only=arguments.preflight_only,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
