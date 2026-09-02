"""Wrapper operacional isolado do piloto refinado na RTX PRO 6000."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .dp_contracts import PrivateTrainingError
from .model_contracts import DEFAULT_MODEL_CACHE, ModelArtifactError, ModelConfigurationError, ModelDependencyError, ModelLoadError
from .refined_pilot import run_refined_defense_pilot
from .refined_pilot_contracts import (
    EXPERIMENT_SEEDS,
    RefinedGatePendingResult,
    RefinedPilotError,
    RefinedPilotResult,
    RefinedPreflightResult,
    default_run_id,
    load_refined_pilot_spec_from_config,
)
from .runtime_profile import (
    ExecutionRuntimeError,
    capture_execution_runtime,
    load_execution_runtime_spec,
    publish_runtime_manifest,
    runtime_output_root,
    validate_runtime_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa a réplica Blackwell isolada.")
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser


def _progress(payload: Mapping[str, object]) -> None:
    print(
        "progresso: "
        + json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _summary(value: object, scientific_root: Path, runtime_sha256: str) -> None:
    print("perfil_runtime: rtxpro6000-blackwell-cu128-v1")
    print(f"runtime_sha256: {runtime_sha256}")
    if isinstance(value, RefinedPreflightResult):
        print("status: preflight da réplica RTX validado")
        print(f"seed: {value.seed}")
        print(f"modelo_sha256: {value.baseline_model_sha256}")
        print(f"conversas_vitima: {value.victim_conversation_count}")
        print(f"conversas_auxiliares: {value.auxiliary_conversation_count}")
        print(f"conversas_utilidade: {value.utility_conversation_count}")
        print("accountant: validado")
        print("escrita: nao")
        return
    if isinstance(value, RefinedGatePendingResult):
        print("status: fase vulnerável da réplica RTX concluída")
        print(f"run_id: {value.run_id}")
        print(f"seed: {value.seed}")
        print(f"fase: {value.phase}")
        print("proximo_passo: resume após conferir os gates das duas seeds RTX")
        return
    if not isinstance(value, RefinedPilotResult):
        raise RefinedPilotError("resultado da réplica possui tipo inválido")
    print("status: réplica RTX do piloto refinado concluída")
    print(f"run_id: {value.run_id}")
    print(f"seed: {value.seed}")
    print(f"status_defesas: {value.defense.status}")
    print(f"rodadas_federadas: {value.total_federated_rounds}")
    print(f"passos_otimizador: {value.total_optimizer_steps}")
    print(f"selecoes_poisson: {value.private_sampled_conversation_count}")
    print(f"geracoes_auditoria: {value.total_audit_generations}")
    print(f"conversas_utilidade: {value.total_utility_conversations}")
    print(f"resultado_sha256: {value.result_sha256}")
    print(f"saida: {scientific_root / 'runs' / value.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    if not arguments.model_artifact_dir.is_absolute():
        parser.error("--model-artifact-dir deve ser absoluto")
    try:
        runtime_spec = load_execution_runtime_spec(arguments.runtime_config)
        runtime = capture_execution_runtime(runtime_spec)
        scientific_root = runtime_output_root(arguments.output_root, runtime_spec)
        if arguments.preflight_only:
            if scientific_root.exists():
                # Preflight nunca publica nem exige o manifesto, mas um namespace
                # preexistente não pode ser um link ou arquivo.
                if scientific_root.is_symlink() or not scientific_root.is_dir():
                    raise ExecutionRuntimeError("namespace operacional existente é inválido")
        elif arguments.fresh:
            publish_runtime_manifest(arguments.output_root, runtime_spec, runtime)
        else:
            validate_runtime_manifest(
                scientific_root / "runtime_manifest.json", runtime_spec, runtime
            )
        spec = load_refined_pilot_spec_from_config(runtime_spec.scientific_config_path)
        result = run_refined_defense_pilot(
            spec,
            seed=arguments.seed,
            config_path=runtime_spec.scientific_config_path,
            output_root=scientific_root,
            run_id=default_run_id(arguments.seed),
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device="cuda",
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_progress,
        )
    except (
        ExecutionRuntimeError,
        FileExistsError,
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
        PrivateTrainingError,
        RefinedPilotError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada na réplica RTX", file=sys.stderr)
        return 1
    _summary(result, scientific_root, runtime.runtime_sha256)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
