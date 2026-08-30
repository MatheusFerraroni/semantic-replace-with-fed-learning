"""Resumo seguro e idempotente das duas seeds da substituição semântica."""

from __future__ import annotations

from pathlib import Path

from .semantic_pilot_contracts import (
    EXPERIMENT_SEEDS,
    SemanticCombinedResult,
    SemanticPilotError,
    SemanticPilotSpec,
    safe_result_sha256,
    semantic_combined_from_payload,
    semantic_pilot_result_from_payload,
    validate_semantic_pilot_spec,
)
from .semantic_pilot_storage import read_safe_json, write_idempotent


COMBINED_RUN_ID = "semantic-substitution-upstream-combined-v1"


def build_semantic_substitution_combined_result(
    spec: SemanticPilotSpec,
    *,
    output_root: Path = Path("outputs"),
) -> SemanticCombinedResult:
    resolved = validate_semantic_pilot_spec(spec)
    results = tuple(
        semantic_pilot_result_from_payload(
            read_safe_json(
                Path(output_root)
                / "runs"
                / resolved.run_id_for_seed(seed)
                / "completed.json"
            )
        )
        for seed in EXPERIMENT_SEEDS
    )
    statuses = tuple((value.experiment_seed, value.gate.status) for value in results)
    if all(status == "approved" for _, status in statuses):
        combined_status = "approved"
    elif any(status == "inconclusive" for _, status in statuses):
        combined_status = "inconclusive"
    else:
        combined_status = "failed"
    unsigned = {
        "schema_version": "semantic-substitution-combined/v1",
        "source_result_sha256_by_seed": {
            str(value.experiment_seed): value.result_sha256 for value in results
        },
        "status_by_seed": {str(seed): status for seed, status in statuses},
        "combined_status": combined_status,
        "require_both_seeds": True,
        "total_trajectories": sum(len(value.trajectories) for value in results),
        "total_federated_rounds": sum(value.total_federated_rounds for value in results),
        "total_conversation_presentations": sum(
            value.total_conversation_presentations for value in results
        ),
        "total_optimizer_steps": sum(value.total_optimizer_steps for value in results),
        "total_audit_generations": sum(value.total_audit_generations for value in results),
        "total_utility_conversations": sum(
            value.total_utility_conversations for value in results
        ),
    }
    result = semantic_combined_from_payload(
        {
            **unsigned,
            "result_sha256": safe_result_sha256(
                unsigned, b"semantic-substitution-combined-result/v1"
            ),
        }
    )
    destination = Path(output_root) / "runs" / COMBINED_RUN_ID
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise SemanticPilotError("destino do resumo combinado é inválido")
    write_idempotent(destination / "combined.json", result.as_safe_dict())
    return result


__all__ = [
    "COMBINED_RUN_ID",
    "build_semantic_substitution_combined_result",
]
