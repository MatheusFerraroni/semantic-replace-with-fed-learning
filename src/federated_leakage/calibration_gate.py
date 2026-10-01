"""Validação estrita do gate de learning rate v4 antes do piloto v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

from .calibration_contracts import (
    CALIBRATION_CLIENT_ID,
    CALIBRATION_DATASET_ID,
    CALIBRATION_FIXED_REPETITIONS,
    CALIBRATION_LEARNING_RATE_MILLIONTHS,
    EXPECTED_ANCHOR_MODEL_SHA256,
    CanaryFieldMetric,
    MemorizationCalibrationArmResult,
    PositiveCanaryAuditResult,
    learning_rate_arm_id,
    validate_memorization_calibration_arm_result,
    validate_positive_canary_audit_result,
)
from .execution_contracts import PilotExecutionError, PilotExecutionSpec
from .model_contracts import ModelProvenance


_FORBIDDEN_KEYS = frozenset(
    {"prompt", "generated_text", "entity_id", "field_values", "annotations", "input_ids", "labels", "tokens"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationGate:
    run_id: str
    result_sha256: str
    manifest_sha256: str
    canary_dataset_sha256: str
    collision_preflight_sha256: str
    baseline_model_sha256: str
    selected_arm_id: str
    selected_learning_rate_millionths: int
    model_provenance: ModelProvenance
    audit_model_sha256: Tuple[str, ...]
    decoding_strategy: str


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


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
        raise PilotExecutionError("proveniência do gate é incompatível") from error


def _arm_result(value: object) -> MemorizationCalibrationArmResult:
    expected = {item.name for item in fields(MemorizationCalibrationArmResult)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PilotExecutionError("braço do gate da calibração é inválido")
    try:
        payload = dict(value)
        payload["model_provenance"] = _provenance(payload["model_provenance"])
        return validate_memorization_calibration_arm_result(
            MemorizationCalibrationArmResult(**payload),
            allowed_repetitions=(CALIBRATION_FIXED_REPETITIONS,),
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
        raw_metrics = payload["field_metrics"]
        if not isinstance(raw_metrics, list) or any(
            not isinstance(metric, Mapping) or set(metric) != metric_keys
            for metric in raw_metrics
        ):
            raise PilotExecutionError("métricas do gate são inválidas")
        payload["field_metrics"] = tuple(CanaryFieldMetric(**metric) for metric in raw_metrics)
        return validate_positive_canary_audit_result(
            PositiveCanaryAuditResult(**payload),
            allowed_repetitions=(CALIBRATION_FIXED_REPETITIONS,),
        )
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("auditoria do gate da calibração diverge") from error


def load_completed_calibration_gate(output_root: Path, spec: PilotExecutionSpec) -> CalibrationGate:
    """Aceita somente o resultado científico v4 fixado pela receita do piloto."""

    root = Path(output_root)
    run_root = root / "runs" / spec.calibration_run_id
    manifest_path = run_root / "run_manifest.json"
    completed_path = run_root / "completed.json"
    if (
        root.is_symlink() or (root / "runs").is_symlink() or run_root.is_symlink()
        or not run_root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file()
        or completed_path.is_symlink() or not completed_path.is_file()
    ):
        raise PilotExecutionError("calibração de learning rate concluída está ausente")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = _load_json(manifest_raw)
        payload = _load_json(completed_path.read_bytes())
    except OSError as error:
        raise PilotExecutionError("gate da calibração não pode ser lido") from error
    _reject_private_keys(manifest)
    _reject_private_keys(payload)

    manifest_keys = {
        "schema_version", "run_id", "experiment_seed", "dataset_id", "client_id",
        "fixed_repetitions", "learning_rate_arms", "expected_anchor_model_sha256",
        "main_config_sha256", "canary_dataset_sha256", "collision_preflight_sha256",
        "model_provenance", "decoding_strategy", "rng_used",
    }
    expected_arms = [
        {"arm_id": learning_rate_arm_id(value), "learning_rate_millionths": value}
        for value in CALIBRATION_LEARNING_RATE_MILLIONTHS
    ]
    if (
        set(manifest) != manifest_keys
        or manifest.get("schema_version") != spec.calibration_schema_version
        or manifest.get("run_id") != spec.calibration_run_id
        or manifest.get("experiment_seed") != spec.experiment_seed
        or manifest.get("dataset_id") != CALIBRATION_DATASET_ID
        or manifest.get("client_id") != CALIBRATION_CLIENT_ID
        or manifest.get("fixed_repetitions") != CALIBRATION_FIXED_REPETITIONS
        or manifest.get("learning_rate_arms") != expected_arms
        or manifest.get("expected_anchor_model_sha256")
        != EXPECTED_ANCHOR_MODEL_SHA256
        or manifest.get("main_config_sha256") != spec.calibration_main_config_sha256
        or manifest.get("canary_dataset_sha256") != spec.calibration_canary_dataset_sha256
        or manifest.get("collision_preflight_sha256") != spec.calibration_collision_preflight_sha256
        or manifest.get("decoding_strategy") != spec.calibration_decoding_strategy
        or manifest.get("rng_used") is not False
    ):
        raise PilotExecutionError("manifesto da calibração v4 diverge")
    manifest_provenance = _provenance(manifest["model_provenance"])

    result_keys = {
        "schema_version", "experiment_seed", "run_id", "dataset_id",
        "baseline_model_sha256", "arms", "audits", "total_conversation_presentations",
        "total_optimizer_steps", "total_audit_generations", "baseline_gate_passed",
        "calibrated", "first_successful_arm_id", "first_successful_learning_rate_millionths",
        "result_sha256",
    }
    if set(payload) != result_keys:
        raise PilotExecutionError("marcador da calibração v4 possui chaves inválidas")
    without_hash = dict(payload)
    result_sha256 = without_hash.pop("result_sha256")
    expected_result_hash = hashlib.sha256(
        b"memorization-calibration-result/v4\0" + _canonical(without_hash)
    ).hexdigest()
    raw_arms = payload.get("arms")
    raw_audits = payload.get("audits")
    if (
        payload.get("schema_version") != spec.calibration_schema_version
        or payload.get("experiment_seed") != spec.experiment_seed
        or payload.get("run_id") != spec.calibration_run_id
        or payload.get("dataset_id") != CALIBRATION_DATASET_ID
        or payload.get("baseline_model_sha256") != spec.calibration_baseline_model_sha256
        or payload.get("total_conversation_presentations") != 64_000
        or payload.get("total_optimizer_steps") != 16_000
        or payload.get("total_audit_generations") != 905
        or payload.get("baseline_gate_passed") is not False
        or payload.get("calibrated") is not True
        or payload.get("first_successful_arm_id") != spec.calibration_selected_arm_id
        or payload.get("first_successful_learning_rate_millionths") != spec.calibration_selected_learning_rate_millionths
        or result_sha256 != expected_result_hash
        or result_sha256 != spec.calibration_result_sha256
        or not isinstance(raw_arms, list) or len(raw_arms) != 4
        or not isinstance(raw_audits, list) or len(raw_audits) != 5
    ):
        raise PilotExecutionError("calibração v4 não liberou o piloto")

    arms = tuple(_arm_result(item) for item in raw_arms)
    audits = tuple(_audit_result(item) for item in raw_audits)
    if tuple(item.learning_rate_millionths for item in arms) != CALIBRATION_LEARNING_RATE_MILLIONTHS:
        raise PilotExecutionError("ordem dos braços da calibração v4 diverge")
    if (
        audits[0].repetitions != 0 or audits[0].calibrated_at_checkpoint
        or tuple(item.learning_rate_millionths for item in audits[1:]) != CALIBRATION_LEARNING_RATE_MILLIONTHS
    ):
        raise PilotExecutionError("agenda das auditorias da calibração v4 diverge")
    selected_index = CALIBRATION_LEARNING_RATE_MILLIONTHS.index(
        spec.calibration_selected_learning_rate_millionths
    )
    selected_audit = audits[selected_index + 1]
    if (
        selected_audit.arm_id != spec.calibration_selected_arm_id
        or selected_audit.distinctive_exact_pair_count != 100
        or selected_audit.distinctive_exact_pair_denominator != 100
        or selected_audit.distinctive_exposed_entity_count != 20
        or not selected_audit.calibrated_at_checkpoint
    ):
        raise PilotExecutionError("braço promovido da calibração v4 diverge")

    registry = audits[0].registry_sha256
    targets = audits[0].target_schedule_sha256
    schedule = audits[0].generation_schedule_sha256
    if any(
        audit.model_provenance != manifest_provenance
        or audit.registry_sha256 != registry
        or audit.target_schedule_sha256 != targets
        or audit.generation_schedule_sha256 != schedule
        or audit.decoding_strategy != spec.calibration_decoding_strategy
        or audit.rng_used
        for audit in audits
    ):
        raise PilotExecutionError("auditorias do gate v4 são incompatíveis")
    if audits[0].model_state_sha256 != payload["baseline_model_sha256"]:
        raise PilotExecutionError("baseline do gate v4 diverge")
    if any(
        arm.model_provenance != manifest_provenance
        or arm.initial_model_sha256 != payload["baseline_model_sha256"]
        or arm.final_model_sha256 != audit.model_state_sha256
        or arm.arm_id != audit.arm_id
        or arm.learning_rate_millionths != audit.learning_rate_millionths
        for arm, audit in zip(arms, audits[1:])
    ):
        raise PilotExecutionError("modelos dos braços do gate v4 divergem")

    return CalibrationGate(
        run_id=spec.calibration_run_id,
        result_sha256=result_sha256,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        canary_dataset_sha256=manifest["canary_dataset_sha256"],
        collision_preflight_sha256=manifest["collision_preflight_sha256"],
        baseline_model_sha256=payload["baseline_model_sha256"],
        selected_arm_id=spec.calibration_selected_arm_id,
        selected_learning_rate_millionths=spec.calibration_selected_learning_rate_millionths,
        model_provenance=manifest_provenance,
        audit_model_sha256=tuple(item.model_state_sha256 for item in audits),
        decoding_strategy=spec.calibration_decoding_strategy,
    )


__all__ = ["CalibrationGate", "load_completed_calibration_gate"]
