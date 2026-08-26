"""Validação estrita do gate greedy antes do piloto v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

from .execution_contracts import PilotExecutionError, PilotExecutionSpec
from .calibration_contracts import (
    CALIBRATION_CLIENT_ID,
    CALIBRATION_DATASET_ID,
    CanaryFieldMetric,
    MemorizationCalibrationArmResult,
    PositiveCanaryAuditResult,
    validate_memorization_calibration_arm_result,
    validate_positive_canary_audit_result,
)
from .model_contracts import ModelProvenance


_TOP_LEVEL_KEYS = frozenset(
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
        "baseline_gate_passed",
        "calibrated",
        "first_successful_repetition",
        "result_sha256",
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
_REPETITIONS = (0, 1, 5, 10, 20)


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationGate:
    run_id: str
    result_sha256: str
    manifest_sha256: str
    canary_dataset_sha256: str
    collision_preflight_sha256: str
    baseline_model_sha256: str
    model_provenance: ModelProvenance
    audit_model_sha256: Tuple[str, ...]
    decoding_strategy: str


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


def _load_json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PilotExecutionError("gate da calibração contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("gate da calibração contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise PilotExecutionError("gate da calibração não usa JSON canônico")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _reject_private_keys(value: object) -> None:
    if isinstance(value, Mapping):
        if any(key in _FORBIDDEN_KEYS for key in value):
            raise PilotExecutionError("gate da calibração contém conteúdo privado")
        for item in value.values():
            _reject_private_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_keys(item)


def _provenance(value: object) -> ModelProvenance:
    expected = {item.name for item in fields(ModelProvenance)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PilotExecutionError("proveniência do gate da calibração é inválida")
    try:
        return ModelProvenance(**value)
    except Exception as error:
        raise PilotExecutionError(
            "proveniência do gate da calibração é incompatível"
        ) from error


def _arm_result(value: object) -> MemorizationCalibrationArmResult:
    expected = {item.name for item in fields(MemorizationCalibrationArmResult)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PilotExecutionError("braço do gate da calibração é inválido")
    try:
        payload = dict(value)
        payload["model_provenance"] = _provenance(payload["model_provenance"])
        result = MemorizationCalibrationArmResult(**payload)
        return validate_memorization_calibration_arm_result(
            result,
            allowed_repetitions=(1, 5, 10, 20),
            expected_schema_version="memorization-calibration-arm/v1",
        )
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("braço do gate da calibração diverge") from error


def _audit_result(value: object) -> PositiveCanaryAuditResult:
    expected = {item.name for item in fields(PositiveCanaryAuditResult)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PilotExecutionError("auditoria do gate da calibração é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = _provenance(payload["model_provenance"])
        metric_keys = {item.name for item in fields(CanaryFieldMetric)}
        metrics = payload.get("field_metrics")
        if not isinstance(metrics, list) or any(
            not isinstance(metric, Mapping) or set(metric) != metric_keys
            for metric in metrics
        ):
            raise PilotExecutionError("métricas do gate da calibração são inválidas")
        payload["field_metrics"] = tuple(CanaryFieldMetric(**metric) for metric in metrics)
        result = PositiveCanaryAuditResult(**payload)
        return validate_positive_canary_audit_result(
            result,
            allowed_repetitions=(1, 5, 10, 20),
            expected_schema_version="positive-canary-audit-result/v2",
        )
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("auditoria do gate da calibração diverge") from error


def load_completed_calibration_gate(
    output_root: Path,
    spec: PilotExecutionSpec,
) -> CalibrationGate:
    """Aceita somente uma calibração greedy concluída e cientificamente válida."""

    root = Path(output_root)
    runs_root = root / "runs"
    run_root = root / "runs" / spec.calibration_run_id
    manifest_path = run_root / "run_manifest.json"
    completed = run_root / "completed.json"
    if (
        root.is_symlink()
        or runs_root.is_symlink()
        or run_root.is_symlink()
        or not run_root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or completed.is_symlink()
        or not completed.is_file()
    ):
        raise PilotExecutionError("calibração greedy concluída está ausente")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = _load_json(manifest_raw)
        payload = _load_json(completed.read_bytes())
    except OSError as error:
        raise PilotExecutionError("gate da calibração não pode ser lido") from error
    if set(payload) != _TOP_LEVEL_KEYS:
        raise PilotExecutionError("marcador da calibração possui chaves inválidas")
    _reject_private_keys(payload)
    _reject_private_keys(manifest)
    expected_manifest_keys = {
        "schema_version",
        "run_id",
        "experiment_seed",
        "dataset_id",
        "client_id",
        "repetitions",
        "main_config_sha256",
        "canary_dataset_sha256",
        "collision_preflight_sha256",
        "model_provenance",
        "decoding_strategy",
        "rng_used",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != spec.calibration_schema_version
        or manifest.get("run_id") != spec.calibration_run_id
        or manifest.get("experiment_seed") != spec.experiment_seed
        or manifest.get("dataset_id") != CALIBRATION_DATASET_ID
        or manifest.get("client_id") != CALIBRATION_CLIENT_ID
        or manifest.get("repetitions") != [1, 5, 10, 20]
        or manifest.get("main_config_sha256") != spec.config_sha256
        or manifest.get("canary_dataset_sha256")
        != spec.calibration_canary_dataset_sha256
        or manifest.get("collision_preflight_sha256")
        != spec.calibration_collision_preflight_sha256
        or manifest.get("decoding_strategy") != spec.calibration_decoding_strategy
        or manifest.get("rng_used") is not False
    ):
        raise PilotExecutionError("manifesto da calibração greedy diverge")
    manifest_provenance = _provenance(manifest.get("model_provenance"))
    safe_without_hash = dict(payload)
    result_sha256 = safe_without_hash.pop("result_sha256")
    expected_hash = hashlib.sha256(
        b"memorization-calibration-result/v2\0" + _canonical(safe_without_hash)
    ).hexdigest()
    arms = payload.get("arms")
    audits = payload.get("audits")
    if (
        payload.get("schema_version") != spec.calibration_schema_version
        or payload.get("run_id") != spec.calibration_run_id
        or payload.get("dataset_id") != CALIBRATION_DATASET_ID
        or payload.get("experiment_seed") != spec.experiment_seed
        or payload.get("total_conversation_presentations") != 3_600
        or payload.get("total_optimizer_steps") != 900
        or payload.get("total_audit_generations") != 905
        or payload.get("baseline_gate_passed") is not False
        or payload.get("calibrated") is not True
        or payload.get("first_successful_repetition") not in {1, 5, 10, 20}
        or not _is_sha256(payload.get("baseline_model_sha256"))
        or result_sha256 != expected_hash
        or not isinstance(arms, list)
        or len(arms) != 4
        or not isinstance(audits, list)
        or len(audits) != 5
    ):
        raise PilotExecutionError("calibração greedy não liberou o piloto")
    arm_results = tuple(_arm_result(item) for item in arms)
    if tuple(item.repetitions for item in arm_results) != (1, 5, 10, 20):
        raise PilotExecutionError("braços do gate da calibração divergem")
    audit_hashes = []
    provenance = None
    audit_results = tuple(_audit_result(item) for item in audits)
    for expected_repetitions, audit in zip(_REPETITIONS, audit_results):
        current_provenance = audit.model_provenance
        if provenance is None:
            provenance = current_provenance
        if (
            current_provenance != provenance
            or audit.repetitions != expected_repetitions
            or audit.decoding_strategy != spec.calibration_decoding_strategy
        ):
            raise PilotExecutionError("auditoria do gate da calibração diverge")
        audit_hashes.append(audit.model_state_sha256)
    if (
        provenance is None
        or provenance != manifest_provenance
        or audit_hashes[0] != payload["baseline_model_sha256"]
    ):
        raise PilotExecutionError("baseline do gate da calibração diverge")
    if any(
        arm.model_provenance != provenance
        or arm.initial_model_sha256 != payload["baseline_model_sha256"]
        or arm.final_model_sha256 != audit.model_state_sha256
        or not all(
            math.isfinite(metric)
            for metric in (
                arm.mean_loss,
                arm.first_step_loss,
                arm.last_step_loss,
                arm.mean_gradient_norm,
                arm.max_gradient_norm,
            )
        )
        for arm, audit in zip(arm_results, audit_results[1:])
    ):
        raise PilotExecutionError("modelos dos braços da calibração divergem")
    registry_hash = audit_results[0].registry_sha256
    target_hash = audit_results[0].target_schedule_sha256
    generation_hash = audit_results[0].generation_schedule_sha256
    if any(
        audit.registry_sha256 != registry_hash
        or audit.target_schedule_sha256 != target_hash
        or audit.generation_schedule_sha256 != generation_hash
        for audit in audit_results[1:]
    ):
        raise PilotExecutionError("agenda das auditorias da calibração diverge")
    successful = tuple(
        audit.repetitions
        for audit in audit_results[1:]
        if audit.calibrated_at_checkpoint
    )
    if (
        not successful
        or payload["first_successful_repetition"] != min(successful)
        or audit_results[0].calibrated_at_checkpoint
    ):
        raise PilotExecutionError("critério do gate da calibração diverge")
    return CalibrationGate(
        run_id=spec.calibration_run_id,
        result_sha256=result_sha256,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        canary_dataset_sha256=manifest["canary_dataset_sha256"],
        collision_preflight_sha256=manifest["collision_preflight_sha256"],
        baseline_model_sha256=payload["baseline_model_sha256"],
        model_provenance=provenance,
        audit_model_sha256=tuple(audit_hashes),
        decoding_strategy=spec.calibration_decoding_strategy,
    )


__all__ = ["CalibrationGate", "load_completed_calibration_gate"]
