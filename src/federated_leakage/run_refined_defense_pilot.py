"""CLI offline do piloto F0-F5 com Fórum/Tec, DP e substituição."""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa uma seed do piloto refinado com DP-AdamW."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--model-artifact-dir", type=Path, required=True)
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


def _summary(value, output_root: Path) -> None:
    if isinstance(value, RefinedPreflightResult):
        print("status: preflight do piloto refinado validado")
        print(f"seed: {value.seed}")
        print(f"modelo_sha256: {value.baseline_model_sha256}")
        print(f"conversas_vitima: {value.victim_conversation_count}")
        print(f"conversas_auxiliares: {value.auxiliary_conversation_count}")
        print(f"conversas_utilidade: {value.utility_conversation_count}")
        print("accountant: validado")
        print("escrita: nao")
        return
    if isinstance(value, RefinedGatePendingResult):
        print("status: fase vulnerável concluída")
        print(f"run_id: {value.run_id}")
        print(f"seed: {value.seed}")
        print(f"fase: {value.phase}")
        print("proximo_passo: resume após conferir os gates das duas seeds")
        return
    if not isinstance(value, RefinedPilotResult):
        raise RefinedPilotError("resultado da CLI possui tipo inválido")
    print("status: piloto refinado concluído")
    print(f"run_id: {value.run_id}")
    print(f"seed: {value.seed}")
    print(f"status_defesas: {value.defense.status}")
    print(f"rodadas_federadas: {value.total_federated_rounds}")
    print(f"passos_otimizador: {value.total_optimizer_steps}")
    print(f"selecoes_poisson: {value.private_sampled_conversation_count}")
    print(f"geracoes_auditoria: {value.total_audit_generations}")
    print(f"conversas_utilidade: {value.total_utility_conversations}")
    print(f"resultado_sha256: {value.result_sha256}")
    print(f"saida: {output_root / 'runs' / value.run_id}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.preflight_only and arguments.fresh:
        parser.error("--fresh não se aplica a --preflight-only")
    if not arguments.model_artifact_dir.is_absolute():
        parser.error("--model-artifact-dir deve ser absoluto")
    try:
        spec = load_refined_pilot_spec_from_config(arguments.config)
        result = run_refined_defense_pilot(
            spec,
            seed=arguments.seed,
            config_path=arguments.config,
            output_root=arguments.output_root,
            run_id=arguments.run_id or default_run_id(arguments.seed),
            cache_dir=arguments.cache_dir,
            model_artifact_dir=arguments.model_artifact_dir,
            device=arguments.device,
            preflight_only=arguments.preflight_only,
            fresh=arguments.fresh,
            progress_callback=_progress,
        )
    except (
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
        print("erro: falha inesperada no piloto refinado", file=sys.stderr)
        return 1
    _summary(result, arguments.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
