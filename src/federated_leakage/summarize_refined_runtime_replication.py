"""Comparação segura entre a referência L40S e a réplica Blackwell."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .refined_pilot_contracts import (
    EXPERIMENT_SEEDS,
    REFINED_COMBINED_SCHEMA_VERSION,
    RefinedPilotError,
    default_run_id,
    refined_pilot_result_from_payload,
    safe_result_sha256,
)
from .runtime_profile import (
    RTX_PROFILE_ID,
    load_execution_runtime_spec,
    runtime_output_root,
)
from .semantic_pilot_storage import canonical_json_bytes, read_safe_json, write_idempotent


RUNTIME_COMPARISON_SCHEMA_VERSION = "refined-runtime-comparison/v1"
RUNTIME_COMPARISON_ID = "refined-defense-l40s-vs-rtxpro6000-cu128-v1"


def _epsilon_statuses(defense: Mapping[str, Any]) -> dict[str, str]:
    entries = defense.get("epsilon_statuses")
    if not isinstance(entries, (list, tuple)) or len(entries) != 2:
        raise RefinedPilotError("status DP do resultado é inválido")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            raise RefinedPilotError("status DP do resultado é inválido")
        result[f"{float(entry[0]):.1f}"] = str(entry[1])
    if set(result) != {"3.0", "8.0"}:
        raise RefinedPilotError("orçamentos DP do resultado divergem")
    return result


def _validate_combined(root: Path) -> dict[str, Any]:
    combined = read_safe_json(
        root / "runs" / "refined-defense-forum-tech-combined-v1" / "combined.json"
    )
    expected_keys = {
        "schema_version", "source_result_sha256_by_seed", "dp_status_by_epsilon",
        "substitution_status", "overall_status", "require_both_seeds",
        "total_trajectories", "total_federated_rounds", "total_optimizer_steps",
        "total_non_private_conversation_presentations",
        "total_private_sampled_conversation_count", "total_audit_generations",
        "total_utility_conversations", "result_sha256",
    }
    unsigned = {key: value for key, value in combined.items() if key != "result_sha256"}
    if (
        set(combined) != expected_keys
        or combined.get("schema_version") != REFINED_COMBINED_SCHEMA_VERSION
        or combined.get("require_both_seeds") is not True
        or combined.get("total_trajectories") != 16
        or combined.get("total_federated_rounds") != 320
        or combined.get("total_optimizer_steps") != 328_000
        or combined.get("total_non_private_conversation_presentations") != 656_000
        or combined.get("total_audit_generations") != 122_086
        or combined.get("total_utility_conversations") != 9_000
        or combined.get("result_sha256")
        != safe_result_sha256(unsigned, b"refined-defense-combined-result/v1")
    ):
        raise RefinedPilotError("resumo combinado refinado é inválido")
    return combined


def _load_hardware_results(root: Path) -> dict[str, Any]:
    combined = _validate_combined(root)
    sources: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    expected_source_hashes = combined["source_result_sha256_by_seed"]
    if not isinstance(expected_source_hashes, dict) or set(expected_source_hashes) != {
        str(seed) for seed in EXPERIMENT_SEEDS
    }:
        raise RefinedPilotError("fontes do resumo combinado são inválidas")
    for seed in EXPERIMENT_SEEDS:
        run_root = root / "runs" / default_run_id(seed)
        result = refined_pilot_result_from_payload(read_safe_json(run_root / "completed.json"))
        manifest = read_safe_json(run_root / "run_manifest.json")
        if (
            result.seed != seed
            or result.result_sha256 != expected_source_hashes[str(seed)]
            or manifest.get("experiment_seed") != seed
            or manifest.get("run_id") != default_run_id(seed)
            or manifest.get("baseline_model_sha256") != result.baseline_model_sha256
        ):
            raise RefinedPilotError("fonte de hardware refinada diverge")
        sources[str(seed)] = result.as_safe_dict()
        manifests[str(seed)] = manifest
    first, second = (manifests[str(seed)] for seed in EXPERIMENT_SEEDS)
    if (
        first.get("config_sha256") != second.get("config_sha256")
        or first.get("main_config_sha256") != second.get("main_config_sha256")
        or first.get("baseline_model_sha256") != second.get("baseline_model_sha256")
        or first.get("model_provenance") != second.get("model_provenance")
        or first.get("scenario_order") != second.get("scenario_order")
    ):
        raise RefinedPilotError("runs do mesmo hardware não compartilham identidade científica")
    return {"combined": combined, "sources": sources, "identity": first}


def _validate_replica_runtime(output_root: Path, profile_config: Path) -> tuple[Path, dict[str, Any]]:
    spec = load_execution_runtime_spec(profile_config)
    replica_root = runtime_output_root(output_root, spec)
    manifest = read_safe_json(replica_root / "runtime_manifest.json")
    if set(manifest) != {"schema_version", "profile_id", "output_namespace", "runtime"}:
        raise RefinedPilotError("manifesto operacional RTX possui campos inválidos")
    runtime = manifest.get("runtime")
    if (
        manifest.get("schema_version") != "execution-runtime-manifest/v1"
        or manifest.get("profile_id") != RTX_PROFILE_ID
        or manifest.get("output_namespace") != spec.output_namespace
        or not isinstance(runtime, dict)
    ):
        raise RefinedPilotError("manifesto operacional RTX é incompatível")
    digest = runtime.get("runtime_sha256")
    unsigned = {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    expected = hashlib.sha256(
        b"execution-runtime-fingerprint/v1\0" + canonical_json_bytes(unsigned)
    ).hexdigest()
    if (
        digest != expected
        or runtime.get("profile_id") != spec.profile_id
        or runtime.get("profile_config_sha256") != spec.profile_config_sha256
        or runtime.get("scientific_config_sha256") != spec.scientific_config_sha256
        or runtime.get("main_config_sha256") != spec.main_config_sha256
        or runtime.get("torch_version") != spec.torch_version
        or runtime.get("torch_cuda_version") != spec.torch_cuda_version
        or spec.required_cuda_arch not in runtime.get("cuda_architectures", [])
        or runtime.get("gpu_name") != spec.gpu_name
        or runtime.get("compute_capability") != list(spec.compute_capability)
    ):
        raise RefinedPilotError("fingerprint operacional RTX é inválido")
    return replica_root, manifest


def _safe_hardware_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    combined = value["combined"]
    seeds = {}
    for seed, result in value["sources"].items():
        defense = result["defense"]
        seeds[seed] = {
            "result_sha256": result["result_sha256"],
            "baseline_model_sha256": result["baseline_model_sha256"],
            "overall_status": defense["status"],
            "dp_status_by_epsilon": _epsilon_statuses(defense),
            "substitution_status": defense["substitution_status"],
            "total_federated_rounds": result["total_federated_rounds"],
            "total_optimizer_steps": result["total_optimizer_steps"],
            "total_audit_generations": result["total_audit_generations"],
            "total_utility_conversations": result["total_utility_conversations"],
        }
    return {
        "combined_result_sha256": combined["result_sha256"],
        "overall_status": combined["overall_status"],
        "dp_status_by_epsilon": combined["dp_status_by_epsilon"],
        "substitution_status": combined["substitution_status"],
        "seeds": seeds,
    }


def build_refined_runtime_comparison(
    output_root: Path = Path("outputs"),
    profile_config: Path = Path("configs/refined-runtime-rtxpro6000-cu128-v1.yaml"),
) -> dict[str, Any]:
    root = Path(output_root)
    if ".." in root.parts or root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RefinedPilotError("raiz da comparação é inválida")
    replica_root, runtime_manifest = _validate_replica_runtime(root, Path(profile_config))
    reference = _load_hardware_results(root)
    replica = _load_hardware_results(replica_root)
    reference_identity = reference["identity"]
    replica_identity = replica["identity"]
    for key in (
        "config_sha256", "main_config_sha256", "baseline_model_sha256",
        "model_provenance", "scenario_order",
    ):
        if reference_identity.get(key) != replica_identity.get(key):
            raise RefinedPilotError("identidade científica diverge entre hardwares")
    safe_reference = _safe_hardware_summary(reference)
    safe_replica = _safe_hardware_summary(replica)
    comparable = []
    for seed in map(str, EXPERIMENT_SEEDS):
        left = safe_reference["seeds"][seed]
        right = safe_replica["seeds"][seed]
        comparable.append(
            (
                left["overall_status"], left["dp_status_by_epsilon"], left["substitution_status"]
            )
            == (
                right["overall_status"], right["dp_status_by_epsilon"], right["substitution_status"]
            )
        )
    comparable.append(
        (
            safe_reference["overall_status"], safe_reference["dp_status_by_epsilon"],
            safe_reference["substitution_status"],
        )
        == (
            safe_replica["overall_status"], safe_replica["dp_status_by_epsilon"],
            safe_replica["substitution_status"],
        )
    )
    unsigned = {
        "schema_version": RUNTIME_COMPARISON_SCHEMA_VERSION,
        "comparison_id": RUNTIME_COMPARISON_ID,
        "classification": "consistent" if all(comparable) else "runtime_sensitive",
        "scientific_identity": {
            "config_sha256": reference_identity["config_sha256"],
            "main_config_sha256": reference_identity["main_config_sha256"],
            "baseline_model_sha256": reference_identity["baseline_model_sha256"],
            "experiment_seeds": list(EXPERIMENT_SEEDS),
        },
        "hardware_results": {
            "l40s_reference": safe_reference,
            "rtxpro6000_blackwell_cu128_replica": safe_replica,
        },
        "runtime_fingerprint_sha256": runtime_manifest["runtime"]["runtime_sha256"],
        "metrics_combined_across_hardware": False,
    }
    result = {
        **unsigned,
        "result_sha256": safe_result_sha256(unsigned, b"refined-runtime-comparison/v1"),
    }
    target = root / "runtime-comparisons" / RUNTIME_COMPARISON_ID
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink() or not target.is_dir():
        raise RefinedPilotError("diretório da comparação é inválido")
    os.chmod(target, 0o700)
    write_idempotent(target / "comparison.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara L40S e RTX PRO 6000 sem combinar métricas.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/refined-runtime-rtxpro6000-cu128-v1.yaml"),
    )
    arguments = parser.parse_args(argv)
    try:
        result = build_refined_runtime_comparison(arguments.output_root, arguments.runtime_config)
    except (FileNotFoundError, RefinedPilotError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada na comparação entre hardwares", file=sys.stderr)
        return 1
    print("status: comparação entre hardwares criada")
    print(f"classificacao: {result['classification']}")
    print("metricas_combinadas: nao")
    print(f"resultado_sha256: {result['result_sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
