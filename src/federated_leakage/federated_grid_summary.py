"""Resumo conjunto seguro dos dois runs independentes da grade v2."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .federated_grid_contracts import (
    EXPERIMENT_SEEDS,
    FederatedGridCombinedResult,
    FederatedGridError,
    FederatedGridSpec,
    GridArmClassification,
    grid_seed_result_from_payload,
    safe_result_sha256,
    validate_federated_grid_spec,
)
from .federated_grid_storage import read_safe_json, write_idempotent


def build_federated_grid_combined_result(
    spec: FederatedGridSpec,
    *,
    output_root: Path = Path("outputs"),
) -> FederatedGridCombinedResult:
    resolved = validate_federated_grid_spec(spec)
    results = tuple(
        grid_seed_result_from_payload(
            read_safe_json(Path(output_root) / "runs" / resolved.run_id_for_seed(seed) / "completed.json"),
            resolved,
        )
        for seed in EXPERIMENT_SEEDS
    )
    if tuple(value.experiment_seed for value in results) != EXPERIMENT_SEEDS:
        raise FederatedGridError("ordem das seeds concluídas diverge")
    classifications = []
    for index, arm_spec in enumerate(resolved.arms):
        arm_results = tuple(value.arms[index] for value in results)
        if any(value.arm_id != arm_spec.arm_id for value in arm_results):
            raise FederatedGridError("ordem dos braços concluídos diverge")
        passed = tuple(
            result.experiment_seed
            for result, arm_result in zip(results, arm_results)
            if not result.baseline_gate_passed and arm_result.audit.gate_passed
        )
        classification = "robust" if len(passed) == 2 else "unstable" if len(passed) == 1 else "insufficient"
        pairs = tuple(value.audit.distinctive_exact_pair_count for value in arm_results)
        entities = tuple(value.audit.distinctive_exposed_entity_count for value in arm_results)
        perplexities = tuple(value.utility.perplexity for value in arm_results)
        classifications.append(
            GridArmClassification(
                arm_id=arm_spec.arm_id,
                classification=classification,
                passed_seeds=passed,
                distinctive_exact_pair_min=min(pairs),
                distinctive_exact_pair_max=max(pairs),
                distinctive_exact_pair_difference=abs(pairs[1] - pairs[0]),
                distinctive_entity_min=min(entities),
                distinctive_entity_max=max(entities),
                utility_perplexity_min=min(perplexities),
                utility_perplexity_max=max(perplexities),
                utility_perplexity_difference=abs(perplexities[1] - perplexities[0]),
            )
        )
    first_robust = next((value.arm_id for value in classifications if value.classification == "robust"), None)
    unsigned = {
        "schema_version": "federated-memorization-grid-combined/v2",
        "source_result_sha256_by_seed": {str(value.experiment_seed): value.result_sha256 for value in results},
        "classifications": [asdict(value) for value in classifications],
        "first_robust_arm": first_robust,
        "human_review_required": True,
        "total_arms": sum(len(value.arms) for value in results),
        "total_federated_rounds": sum(value.total_federated_rounds for value in results),
        "total_conversation_presentations": sum(value.total_conversation_presentations for value in results),
        "total_optimizer_steps": sum(value.total_optimizer_steps for value in results),
        "total_audit_generations": sum(value.total_audit_generations for value in results),
        "total_utility_conversations": sum(value.total_utility_conversations for value in results),
    }
    result = FederatedGridCombinedResult(
        source_result_sha256_by_seed=tuple((value.experiment_seed, value.result_sha256) for value in results),
        classifications=tuple(classifications),
        first_robust_arm=first_robust,
        human_review_required=True,
        total_arms=unsigned["total_arms"],
        total_federated_rounds=unsigned["total_federated_rounds"],
        total_conversation_presentations=unsigned["total_conversation_presentations"],
        total_optimizer_steps=unsigned["total_optimizer_steps"],
        total_audit_generations=unsigned["total_audit_generations"],
        total_utility_conversations=unsigned["total_utility_conversations"],
        result_sha256=safe_result_sha256(unsigned, b"federated-memorization-grid-combined/v2"),
    )
    if (
        (result.total_arms, result.total_federated_rounds, result.total_conversation_presentations, result.total_optimizer_steps, result.total_audit_generations, result.total_utility_conversations)
        != resolved.expected_combined
    ):
        raise FederatedGridError("totais combinados da grade divergem")
    destination = Path(output_root) / "runs" / "federated-memorization-grid-v2"
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise FederatedGridError("destino do resumo conjunto é inválido")
    write_idempotent(destination / "combined.json", result.as_safe_dict())
    return result


__all__ = ["build_federated_grid_combined_result"]
