"""Leitura estrita e somente leitura dos resumos amostrados v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .audit_contracts import ExtractionAuditError


_DERIVED_KEYS = frozenset(
    {
        "targeted_exact_pair_recall",
        "targeted_partial_pair_recall",
        "targeted_complete_generation_rate",
        "targeted_ordered_complete_generation_rate",
        "targeted_any_field_profile_exposure_rate",
        "targeted_misassociation_rate",
        "targeted_unseen_synthetic_value_rate",
    }
)
_LEGACY_RESULT_KEYS = frozenset(
    {
        "scenario",
        "experiment_seed",
        "round_id",
        "auxiliary_weight_units",
        "generation_count",
        "primary_generation_count",
        "field_specific_generation_count",
        "untargeted_generation_count",
        "target_count",
        "targeted_exact_pair_count",
        "targeted_exact_pair_denominator",
        "targeted_partial_pair_count",
        "targeted_complete_generation_count",
        "targeted_ordered_complete_generation_count",
        "targeted_exposed_profile_count",
        "total_exact_reproductions",
        "targeted_misassociation_count",
        "targeted_known_association_count",
        "targeted_unseen_formatted_count",
        "field_metrics",
        "field_specific_exact_pair_count",
        "field_specific_partial_pair_count",
        "untargeted_exact_hit_count",
        "untargeted_unique_value_count",
        "untargeted_victim_name_count",
        "untargeted_exposed_profile_count",
        "registry_sha256",
        "target_schedule_sha256",
        "prompt_catalog_sha256",
        "generation_schedule_sha256",
        "generation_records_sha256",
        "model_state_sha256",
        "model_provenance",
        "schema_version",
        "audit_schema_version",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "generated_text",
        "entity_id",
        "field_values",
        "annotations",
        "input_ids",
        "labels",
        "tokens",
    }
)
_LEGACY_CALIBRATION_KEYS = frozenset(
    {
        "schema_version",
        "experiment_seed",
        "run_id",
        "dataset_id",
        "baseline_model_sha256",
        "arms",
        "audits",
        "total_conversation_presentations",
        "total_optimizer_steps",
        "total_audit_generations",
        "calibrated",
        "first_successful_repetition",
        "result_sha256",
    }
)
_LEGACY_PILOT_KEYS = frozenset(
    {
        "schema_version",
        "identity",
        "baseline_model_sha256",
        "baseline_audit_sha256",
        "baseline_audit_count",
        "trajectories",
        "total_federated_rounds",
        "total_conversation_count",
        "total_optimizer_steps",
        "total_audit_generations",
        "paired_results_sha256",
        "completed",
    }
)
_LEGACY_PILOT_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_id",
        "experiment_seed",
        "auxiliary_weight_units",
        "schedule_id",
        "config_sha256",
    }
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _load(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise ExtractionAuditError("resumo legado está ausente")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExtractionAuditError("resumo legado contém chave duplicada")
            result[key] = value
        return result

    try:
        raw = resolved.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except ExtractionAuditError:
        raise
    except Exception as error:
        raise ExtractionAuditError("resumo legado contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise ExtractionAuditError("resumo legado não é canônico")
    return value


def _reject_private(value: object) -> None:
    if isinstance(value, Mapping):
        if any(key in _FORBIDDEN_KEYS for key in value):
            raise ExtractionAuditError("resumo legado contém conteúdo privado")
        for item in value.values():
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def read_legacy_extraction_audit_summary(path: Path) -> Mapping[str, Any]:
    """Valida um resumo sampling v1 sem habilitar nova geração ou retomada."""

    value = _load(path)
    _reject_private(value)
    target_count = value.get("target_count")
    if type(target_count) is not int or target_count not in {1, 5, 20, 200}:
        raise ExtractionAuditError("orçamento do resumo legado é inválido")
    primary = 5 * target_count
    field_specific = 40 * target_count
    if (
        set(value) != _LEGACY_RESULT_KEYS | _DERIVED_KEYS
        or value.get("schema_version") != "extraction-audit-result/v2"
        or value.get("audit_schema_version") != "extraction-audit/v1"
        or value.get("primary_generation_count") != primary
        or value.get("field_specific_generation_count") != field_specific
        or value.get("untargeted_generation_count") != 100
        or value.get("generation_count") != primary + field_specific + 100
        or value.get("targeted_exact_pair_denominator") != 8 * target_count
        or not _is_sha256(value.get("model_state_sha256"))
        or not isinstance(value.get("field_metrics"), list)
        or len(value["field_metrics"]) != 8
    ):
        raise ExtractionAuditError("resumo legado diverge do protocolo sampling v1")
    return value


def read_legacy_memorization_calibration_summary(path: Path) -> Mapping[str, Any]:
    """Valida o marcador final da calibração sampling v1 somente para inspeção."""

    value = _load(path)
    _reject_private(value)
    arms = value.get("arms")
    audits = value.get("audits")
    if (
        set(value) != _LEGACY_CALIBRATION_KEYS
        or value.get("schema_version") != "memorization-calibration/v1"
        or value.get("run_id") != "memorization-calibration-seed-101-v1"
        or value.get("experiment_seed") != 101
        or value.get("dataset_id") != "positive-canaries-seed-101-v1"
        or not _is_sha256(value.get("baseline_model_sha256"))
        or not _is_sha256(value.get("result_sha256"))
        or value.get("total_conversation_presentations") != 3_600
        or value.get("total_optimizer_steps") != 900
        or value.get("total_audit_generations") != 5_000
        or not isinstance(arms, list)
        or len(arms) != 4
        or not isinstance(audits, list)
        or len(audits) != 5
        or tuple(
            item.get("repetitions") if isinstance(item, Mapping) else None
            for item in arms
        )
        != (1, 5, 10, 20)
        or tuple(
            item.get("repetitions") if isinstance(item, Mapping) else None
            for item in audits
        )
        != (0, 1, 5, 10, 20)
        or any(
            item.get("schema_version") != "positive-canary-audit-result/v1"
            or item.get("generation_count") != 1_000
            for item in audits
        )
    ):
        raise ExtractionAuditError("calibração legada diverge do protocolo sampling v1")
    return value


def read_legacy_pilot_summary(path: Path) -> Mapping[str, Any]:
    """Valida o marcador final do piloto sampling v1 sem permitir retomada."""

    value = _load(path)
    _reject_private(value)
    identity = value.get("identity")
    trajectories = value.get("trajectories")
    if (
        set(value) != _LEGACY_PILOT_KEYS
        or value.get("schema_version") != "pilot-execution/v1"
        or not isinstance(identity, Mapping)
        or set(identity) != _LEGACY_PILOT_IDENTITY_KEYS
        or identity.get("schema_version") != "pilot-execution/v1"
        or identity.get("run_id") != "pilot-seed-101-k01"
        or not isinstance(identity.get("dataset_id"), str)
        or identity.get("experiment_seed") != 101
        or identity.get("auxiliary_weight_units") != 1
        or identity.get("schedule_id") != "F0-F1"
        or not _is_sha256(identity.get("config_sha256"))
        or not _is_sha256(value.get("baseline_model_sha256"))
        or not _is_sha256(value.get("baseline_audit_sha256"))
        or not _is_sha256(value.get("paired_results_sha256"))
        or value.get("baseline_audit_count") != 4
        or not isinstance(trajectories, list)
        or len(trajectories) != 2
        or tuple(
            item.get("scenario") if isinstance(item, Mapping) else None
            for item in trajectories
        )
        != ("F0", "F1")
        or any(
            item.get("schema_version") != "federated-trajectory/v1"
            for item in trajectories
        )
        or value.get("total_federated_rounds") != 40
        or value.get("total_conversation_count") != 44_000
        or value.get("total_optimizer_steps") != 11_000
        or value.get("total_audit_generations") != 69_710
        or value.get("completed") is not True
    ):
        raise ExtractionAuditError("piloto legado diverge do protocolo sampling v1")
    return value


__all__ = [
    "read_legacy_extraction_audit_summary",
    "read_legacy_memorization_calibration_summary",
    "read_legacy_pilot_summary",
]
