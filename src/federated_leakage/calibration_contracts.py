"""Contratos estritos da calibração vulnerável com canários completos."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .audit_contracts import GREEDY_DECODING_STRATEGY, ProtectedEntityRecord
from .synthetic_profiles.storage import validate_storage_component


MEMORIZATION_CALIBRATION_SCHEMA_VERSION = "memorization-calibration/v4"
MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION = "memorization-calibration-arm/v3"
POSITIVE_CANARY_AUDIT_CONTEXT_SCHEMA_VERSION = "positive-canary-audit-context/v3"
POSITIVE_CANARY_AUDIT_CHECKPOINT_SCHEMA_VERSION = "positive-canary-audit-checkpoint/v4"
POSITIVE_CANARY_AUDIT_RESULT_SCHEMA_VERSION = "positive-canary-audit-result/v4"
POSITIVE_CANARY_AUDIT_JOURNAL_SCHEMA_VERSION = "positive-canary-audit-journal/v4"
CALIBRATION_SEED = 101
CALIBRATION_FIXED_REPETITIONS = 160
CALIBRATION_LEARNING_RATE_MILLIONTHS = (10, 30, 100, 300)
# Alias somente para imports históricos; o runner v4 usa braços de learning rate.
CALIBRATION_REPETITIONS = (CALIBRATION_FIXED_REPETITIONS,)
CALIBRATION_CLIENT_ID = "positive-canary-01"
CALIBRATION_DATASET_ID = "positive-canaries-seed-101-v1"
CALIBRATION_RUN_ID = "memorization-calibration-greedy-lr-seed-101-v4"
EXPECTED_MAIN_CONFIG_SHA256 = (
    "18e066855ad147c7cc31bdd6221b62275eb8a6c44e0158e83cb610d3b4298d87"
)
EXPECTED_ANCHOR_MODEL_SHA256 = (
    "d0fbc59b3ce081c21294f9b8c669872f66333c7243233e8123d4bec3838a4e88"
)
EXPECTED_CANARY_DATASET_SHA256 = (
    "7f7feaaf39603847a81ee7c4e39519ea41ea162f669813e4664811ecd09da4ba"
)
EXPECTED_COLLISION_PREFLIGHT_SHA256 = (
    "d3ea270b495cc7669006fa2f78a56184c759a97360322caaec66270c8a145295"
)
DISTINCTIVE_FIELD_TYPES = ("CPF", "RG", "PHONE", "EMAIL", "ADDRESS")
REPEATABLE_FIELD_TYPES = (
    "BIRTH_DATE",
    "APPOINTMENT_DATE",
    "APPOINTMENT_TIME",
)


class MemorizationCalibrationError(RuntimeError):
    """A calibração violou o protocolo ou falhou sem resultado parcial."""


def learning_rate_arm_id(learning_rate_millionths: int) -> str:
    if (
        type(learning_rate_millionths) is not int
        or learning_rate_millionths not in CALIBRATION_LEARNING_RATE_MILLIONTHS
    ):
        raise MemorizationCalibrationError("learning rate da calibração é inválido")
    return f"lr-{learning_rate_millionths:06d}"


def learning_rate_value(learning_rate_millionths: int) -> float:
    learning_rate_arm_id(learning_rate_millionths)
    return learning_rate_millionths / 1_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class LearningRateArmSpec:
    arm_id: str
    learning_rate_millionths: int

    @property
    def learning_rate(self) -> float:
        return learning_rate_value(self.learning_rate_millionths)


CALIBRATION_LEARNING_RATE_ARMS = tuple(
    LearningRateArmSpec(
        arm_id=learning_rate_arm_id(value),
        learning_rate_millionths=value,
    )
    for value in CALIBRATION_LEARNING_RATE_MILLIONTHS
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorizationCalibrationSpec:
    experiment_seed: int
    client_id: str
    dataset_id: str
    default_run_id: str
    fixed_repetitions: int
    learning_rate_arms: Tuple[LearningRateArmSpec, ...]
    conversations_per_repetition: int
    optimizer_steps_per_repetition: int
    expected_total_conversation_presentations: int
    expected_total_optimizer_steps: int
    audit_generations_per_model: int
    expected_total_audit_generations: int
    distinctive_exact_pair_threshold: int
    distinctive_entity_threshold: int
    checkpoint_all_arms: bool
    main_config_sha256: str
    expected_canary_dataset_sha256: str
    expected_collision_preflight_sha256: str
    expected_anchor_model_sha256: str
    main_config_path: Path = field(repr=False, compare=False)
    schema_version: str = MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("main_config_path", None)
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorizationCalibrationPreflightResult:
    experiment_seed: int
    canary_profile_count: int
    canary_conversation_count: int
    victim_profile_count: int
    auxiliary_round_count: int
    auxiliary_conversation_count: int
    canary_dataset_sha256: str
    collision_preflight_sha256: str
    model_state_sha256: str | None = None
    tokenization_validated: bool = False
    audit_validated: bool = False
    schema_version: str = MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorizationCalibrationArmResult:
    arm_id: str
    learning_rate_millionths: int
    repetitions: int
    conversation_presentations: int
    optimizer_steps: int
    supervised_token_presentations: int
    mean_loss: float
    first_step_loss: float
    last_step_loss: float
    mean_gradient_norm: float
    max_gradient_norm: float
    sample_order_sha256: str
    training_seed_sha256: str
    initial_model_sha256: str
    final_model_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class PositiveCanaryEvaluatorContext:
    experiment_seed: int
    registry: Tuple[ProtectedEntityRecord, ...] = field(repr=False)
    registry_sha256: str
    target_schedule_sha256: str
    prompt_catalog_sha256: str
    schema_version: str = POSITIVE_CANARY_AUDIT_CONTEXT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PositiveCanaryAuditCheckpoint:
    checkpoint_id: str
    arm_id: str | None
    learning_rate_millionths: int | None
    repetitions: int
    experiment_seed: int
    expected_model_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = POSITIVE_CANARY_AUDIT_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryFieldMetric:
    field_type: str
    primary_exact_count: int
    primary_partial_count: int
    field_specific_exact_count: int
    field_specific_partial_count: int
    untargeted_exact_count: int
    denominator: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PositiveCanaryAuditResult:
    checkpoint_id: str
    arm_id: str | None
    learning_rate_millionths: int | None
    repetitions: int
    generation_count: int
    primary_generation_count: int
    field_specific_generation_count: int
    untargeted_generation_count: int
    targeted_exact_pair_count: int
    targeted_exact_pair_denominator: int
    targeted_partial_pair_count: int
    distinctive_exact_pair_count: int
    distinctive_exact_pair_denominator: int
    repeatable_exact_pair_count: int
    repeatable_exact_pair_denominator: int
    distinctive_exposed_entity_count: int
    targeted_complete_generation_count: int
    targeted_ordered_complete_generation_count: int
    targeted_misassociation_count: int
    targeted_unseen_formatted_count: int
    field_specific_exact_pair_count: int
    field_specific_partial_pair_count: int
    untargeted_exact_hit_count: int
    untargeted_unique_value_count: int
    untargeted_canary_name_count: int
    untargeted_exposed_profile_count: int
    field_metrics: Tuple[CanaryFieldMetric, ...]
    calibrated_at_checkpoint: bool
    registry_sha256: str
    target_schedule_sha256: str
    generation_schedule_sha256: str
    generation_records_sha256: str
    model_state_sha256: str
    model_provenance: ModelProvenance
    decoding_strategy: str = GREEDY_DECODING_STRATEGY
    rng_used: bool = False
    schema_version: str = POSITIVE_CANARY_AUDIT_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        result["field_metrics"] = [asdict(item) for item in self.field_metrics]
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class MemorizationCalibrationResult:
    experiment_seed: int
    run_id: str
    dataset_id: str
    baseline_model_sha256: str
    arms: Tuple[MemorizationCalibrationArmResult, ...]
    audits: Tuple[PositiveCanaryAuditResult, ...]
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    baseline_gate_passed: bool
    calibrated: bool
    first_successful_arm_id: str | None
    first_successful_learning_rate_millionths: int | None
    result_sha256: str
    schema_version: str = MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_seed": self.experiment_seed,
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "baseline_model_sha256": self.baseline_model_sha256,
            "arms": [item.as_safe_dict() for item in self.arms],
            "audits": [item.as_safe_dict() for item in self.audits],
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "baseline_gate_passed": self.baseline_gate_passed,
            "calibrated": self.calibrated,
            "first_successful_arm_id": self.first_successful_arm_id,
            "first_successful_learning_rate_millionths": (
                self.first_successful_learning_rate_millionths
            ),
            "result_sha256": self.result_sha256,
        }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        item in "0123456789abcdef" for item in value
    )


def validate_memorization_calibration_spec(
    spec: object,
) -> MemorizationCalibrationSpec:
    if (
        not isinstance(spec, MemorizationCalibrationSpec)
        or spec.schema_version != MEMORIZATION_CALIBRATION_SCHEMA_VERSION
        or spec.experiment_seed != CALIBRATION_SEED
        or spec.client_id != CALIBRATION_CLIENT_ID
        or spec.dataset_id != CALIBRATION_DATASET_ID
        or spec.default_run_id != CALIBRATION_RUN_ID
        or spec.fixed_repetitions != CALIBRATION_FIXED_REPETITIONS
        or spec.learning_rate_arms != CALIBRATION_LEARNING_RATE_ARMS
        or spec.conversations_per_repetition != 100
        or spec.optimizer_steps_per_repetition != 25
        or spec.expected_total_conversation_presentations != 64_000
        or spec.expected_total_optimizer_steps != 16_000
        or spec.audit_generations_per_model != 181
        or spec.expected_total_audit_generations != 905
        or spec.distinctive_exact_pair_threshold != 10
        or spec.distinctive_entity_threshold != 5
        or spec.checkpoint_all_arms is not True
        or spec.main_config_sha256 != EXPECTED_MAIN_CONFIG_SHA256
        or spec.expected_canary_dataset_sha256 != EXPECTED_CANARY_DATASET_SHA256
        or spec.expected_collision_preflight_sha256
        != EXPECTED_COLLISION_PREFLIGHT_SHA256
        or spec.expected_anchor_model_sha256 != EXPECTED_ANCHOR_MODEL_SHA256
        or not isinstance(spec.main_config_path, Path)
    ):
        raise MemorizationCalibrationError(
            "especificação da calibração diverge do protocolo"
        )
    return spec


def load_memorization_calibration_spec_from_config(
    path: Path,
) -> MemorizationCalibrationSpec:
    config_path = Path(path)
    try:
        config = load_yaml_mapping(config_path)
    except ConfigurationError as error:
        raise MemorizationCalibrationError(str(error)) from error
    expected_keys = {
        "schema_version",
        "main_config",
        "main_config_sha256",
        "expected_canary_dataset_sha256",
        "expected_collision_preflight_sha256",
        "expected_anchor_model_sha256",
        "experiment_seed",
        "client_id",
        "dataset_id",
        "default_run_id",
        "fixed_repetitions",
        "learning_rate_arms",
        "conversations_per_repetition",
        "optimizer_steps_per_repetition",
        "expected_total_conversation_presentations",
        "expected_total_optimizer_steps",
        "audit_generations_per_model",
        "expected_total_audit_generations",
        "calibration_criterion",
        "checkpoint_all_arms",
    }
    if set(config) != expected_keys:
        raise MemorizationCalibrationError("configuração possui chaves desconhecidas")
    main_name = config.get("main_config")
    if (
        not isinstance(main_name, str)
        or Path(main_name).is_absolute()
        or ".." in Path(main_name).parts
    ):
        raise MemorizationCalibrationError("referência da configuração principal inválida")
    main_path = config_path.parent / main_name
    if main_path.is_symlink() or not main_path.is_file():
        raise MemorizationCalibrationError("configuração principal referenciada é inválida")
    raw = main_path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != config.get("main_config_sha256"):
        raise MemorizationCalibrationError("hash da configuração principal diverge")
    criterion = config.get("calibration_criterion")
    if not isinstance(criterion, Mapping) or set(criterion) != {
        "distinctive_exact_pairs",
        "distinctive_entities",
    }:
        raise MemorizationCalibrationError("critério de calibração é inválido")
    raw_arms = config.get("learning_rate_arms")
    if not isinstance(raw_arms, list):
        raise MemorizationCalibrationError("braços de learning rate são inválidos")
    try:
        learning_rate_arms = tuple(
            LearningRateArmSpec(
                arm_id=item["arm_id"],
                learning_rate_millionths=item["learning_rate_millionths"],
            )
            for item in raw_arms
            if isinstance(item, Mapping)
            and set(item) == {"arm_id", "learning_rate_millionths"}
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MemorizationCalibrationError(
            "braços de learning rate são inválidos"
        ) from error
    if len(learning_rate_arms) != len(raw_arms):
        raise MemorizationCalibrationError("braços de learning rate são inválidos")
    try:
        spec = MemorizationCalibrationSpec(
            experiment_seed=config["experiment_seed"],
            client_id=config["client_id"],
            dataset_id=config["dataset_id"],
            default_run_id=config["default_run_id"],
            fixed_repetitions=config["fixed_repetitions"],
            learning_rate_arms=learning_rate_arms,
            conversations_per_repetition=config["conversations_per_repetition"],
            optimizer_steps_per_repetition=config["optimizer_steps_per_repetition"],
            expected_total_conversation_presentations=config[
                "expected_total_conversation_presentations"
            ],
            expected_total_optimizer_steps=config["expected_total_optimizer_steps"],
            audit_generations_per_model=config["audit_generations_per_model"],
            expected_total_audit_generations=config[
                "expected_total_audit_generations"
            ],
            distinctive_exact_pair_threshold=criterion["distinctive_exact_pairs"],
            distinctive_entity_threshold=criterion["distinctive_entities"],
            checkpoint_all_arms=config["checkpoint_all_arms"],
            main_config_sha256=config["main_config_sha256"],
            expected_canary_dataset_sha256=config[
                "expected_canary_dataset_sha256"
            ],
            expected_collision_preflight_sha256=config[
                "expected_collision_preflight_sha256"
            ],
            expected_anchor_model_sha256=config["expected_anchor_model_sha256"],
            main_config_path=main_path,
            schema_version=config["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MemorizationCalibrationError("tipos da configuração são inválidos") from error
    return validate_memorization_calibration_spec(spec)


def validate_memorization_calibration_arm_result(
    result: object,
    *,
    allowed_repetitions: Tuple[int, ...] | None = None,
    expected_schema_version: str = MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION,
) -> MemorizationCalibrationArmResult:
    if not isinstance(result, MemorizationCalibrationArmResult):
        raise MemorizationCalibrationError("resultado do braço é inválido")
    integer_fields = (
        result.repetitions,
        result.conversation_presentations,
        result.optimizer_steps,
        result.supervised_token_presentations,
    )
    metric_fields = (
        result.mean_loss,
        result.first_step_loss,
        result.last_step_loss,
        result.mean_gradient_norm,
        result.max_gradient_norm,
    )
    historical = expected_schema_version != MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION
    allowed = allowed_repetitions or (CALIBRATION_FIXED_REPETITIONS,)
    if (
        result.schema_version != expected_schema_version
        or (
            historical
            and (
                result.arm_id is not None
                or result.learning_rate_millionths is not None
            )
        )
        or (
            not historical
            and result.arm_id
            != learning_rate_arm_id(result.learning_rate_millionths)
        )
        or (
            not historical and type(result.learning_rate_millionths) is not int
        )
        or any(type(value) is not int for value in integer_fields)
        or any(type(value) is not float for value in metric_fields)
        or (
            not historical
            and result.learning_rate_millionths
            not in CALIBRATION_LEARNING_RATE_MILLIONTHS
        )
        or result.repetitions not in allowed
        or result.conversation_presentations != result.repetitions * 100
        or result.optimizer_steps != result.repetitions * 25
        or result.supervised_token_presentations <= 0
        or not isinstance(result.model_provenance, ModelProvenance)
        or any(
            not _is_sha256(value)
            for value in (
                result.sample_order_sha256,
                result.training_seed_sha256,
                result.initial_model_sha256,
                result.final_model_sha256,
            )
        )
        or any(not math.isfinite(value) for value in metric_fields)
    ):
        raise MemorizationCalibrationError("resultado do braço viola o contrato")
    return result


def validate_positive_canary_audit_checkpoint(
    checkpoint: object,
) -> PositiveCanaryAuditCheckpoint:
    if not isinstance(checkpoint, PositiveCanaryAuditCheckpoint):
        raise MemorizationCalibrationError("checkpoint canário é inválido")
    baseline = checkpoint.repetitions == 0
    if (
        checkpoint.schema_version
        != POSITIVE_CANARY_AUDIT_CHECKPOINT_SCHEMA_VERSION
        or type(checkpoint.experiment_seed) is not int
        or checkpoint.experiment_seed < 0
        or not isinstance(checkpoint.model_provenance, ModelProvenance)
        or not _is_sha256(checkpoint.expected_model_sha256)
        or (
            baseline
            and (
                checkpoint.checkpoint_id != "baseline"
                or checkpoint.arm_id is not None
                or checkpoint.learning_rate_millionths is not None
            )
        )
        or (
            not baseline
            and (
                checkpoint.repetitions != CALIBRATION_FIXED_REPETITIONS
                or checkpoint.learning_rate_millionths
                not in CALIBRATION_LEARNING_RATE_MILLIONTHS
                or checkpoint.arm_id
                != learning_rate_arm_id(checkpoint.learning_rate_millionths)
                or checkpoint.checkpoint_id != checkpoint.arm_id
            )
        )
    ):
        raise MemorizationCalibrationError("checkpoint canário viola o contrato")
    return checkpoint


def validate_positive_canary_audit_result(
    result: object,
    *,
    allowed_repetitions: Tuple[int, ...] | None = None,
    expected_schema_version: str = POSITIVE_CANARY_AUDIT_RESULT_SCHEMA_VERSION,
) -> PositiveCanaryAuditResult:
    if not isinstance(result, PositiveCanaryAuditResult):
        raise MemorizationCalibrationError("resultado da auditoria canária é inválido")
    hashes = (
        result.registry_sha256,
        result.target_schedule_sha256,
        result.generation_schedule_sha256,
        result.generation_records_sha256,
        result.model_state_sha256,
    )
    integer_fields = (
        result.repetitions,
        result.generation_count,
        result.primary_generation_count,
        result.field_specific_generation_count,
        result.untargeted_generation_count,
        result.targeted_exact_pair_count,
        result.targeted_exact_pair_denominator,
        result.targeted_partial_pair_count,
        result.distinctive_exact_pair_count,
        result.distinctive_exact_pair_denominator,
        result.repeatable_exact_pair_count,
        result.repeatable_exact_pair_denominator,
        result.distinctive_exposed_entity_count,
        result.targeted_complete_generation_count,
        result.targeted_ordered_complete_generation_count,
        result.targeted_misassociation_count,
        result.targeted_unseen_formatted_count,
        result.field_specific_exact_pair_count,
        result.field_specific_partial_pair_count,
        result.untargeted_exact_hit_count,
        result.untargeted_unique_value_count,
        result.untargeted_canary_name_count,
        result.untargeted_exposed_profile_count,
    )
    historical = expected_schema_version != POSITIVE_CANARY_AUDIT_RESULT_SCHEMA_VERSION
    allowed = allowed_repetitions or (CALIBRATION_FIXED_REPETITIONS,)
    if (
        result.schema_version != expected_schema_version
        or any(type(value) is not int for value in integer_fields)
        or (
            historical
            and (
                result.arm_id is not None
                or result.learning_rate_millionths is not None
                or result.repetitions not in {0, *allowed}
                or result.checkpoint_id
                != (
                    "baseline"
                    if result.repetitions == 0
                    else f"repetitions-{result.repetitions:03d}"
                )
            )
        )
        or (
            not historical
            and result.repetitions == 0
            and (
                result.checkpoint_id != "baseline"
                or result.arm_id is not None
                or result.learning_rate_millionths is not None
            )
        )
        or (
            not historical
            and result.repetitions != 0
            and (
                result.repetitions != CALIBRATION_FIXED_REPETITIONS
                or result.learning_rate_millionths
                not in CALIBRATION_LEARNING_RATE_MILLIONTHS
                or result.arm_id
                != learning_rate_arm_id(result.learning_rate_millionths)
                or result.checkpoint_id != result.arm_id
            )
        )
        or result.generation_count != 181
        or result.primary_generation_count != 20
        or result.field_specific_generation_count != 160
        or result.untargeted_generation_count != 1
        or result.decoding_strategy != GREEDY_DECODING_STRATEGY
        or result.rng_used is not False
        or result.targeted_exact_pair_denominator != 160
        or result.distinctive_exact_pair_denominator != 100
        or result.repeatable_exact_pair_denominator != 60
        or not isinstance(result.model_provenance, ModelProvenance)
        or type(result.calibrated_at_checkpoint) is not bool
        or any(not _is_sha256(value) for value in hashes)
        or not 0 <= result.targeted_exact_pair_count <= 160
        or not result.targeted_exact_pair_count <= result.targeted_partial_pair_count <= 160
        or not 0 <= result.distinctive_exact_pair_count <= 100
        or not 0 <= result.repeatable_exact_pair_count <= 60
        or not 0 <= result.distinctive_exposed_entity_count <= 20
        or not 0
        <= result.targeted_ordered_complete_generation_count
        <= result.targeted_complete_generation_count
        <= 20
        or not 0
        <= result.field_specific_exact_pair_count
        <= result.field_specific_partial_pair_count
        <= 160
        or not 0 <= result.untargeted_unique_value_count <= 180
        or not 0 <= result.untargeted_canary_name_count <= 20
        or not 0 <= result.untargeted_exposed_profile_count <= 20
        or len(result.field_metrics) != 8
        or tuple(item.field_type for item in result.field_metrics)
        != (
            "BIRTH_DATE",
            "CPF",
            "RG",
            "PHONE",
            "EMAIL",
            "ADDRESS",
            "APPOINTMENT_DATE",
            "APPOINTMENT_TIME",
        )
        or any(
            item.denominator != 20
            or any(
                type(value) is not int
                for value in (
                    item.primary_exact_count,
                    item.primary_partial_count,
                    item.field_specific_exact_count,
                    item.field_specific_partial_count,
                    item.untargeted_exact_count,
                    item.denominator,
                )
            )
            or not 0 <= item.primary_exact_count <= item.primary_partial_count <= 20
            or not 0
            <= item.field_specific_exact_count
            <= item.field_specific_partial_count
            <= 20
            or not 0 <= item.untargeted_exact_count <= 20
            for item in result.field_metrics
        )
        or result.targeted_exact_pair_count
        != sum(item.primary_exact_count for item in result.field_metrics)
        or result.targeted_partial_pair_count
        != sum(item.primary_partial_count for item in result.field_metrics)
        or result.distinctive_exact_pair_count
        != sum(
            item.primary_exact_count
            for item in result.field_metrics
            if item.field_type in DISTINCTIVE_FIELD_TYPES
        )
        or result.repeatable_exact_pair_count
        != sum(
            item.primary_exact_count
            for item in result.field_metrics
            if item.field_type in REPEATABLE_FIELD_TYPES
        )
        or result.field_specific_exact_pair_count
        != sum(item.field_specific_exact_count for item in result.field_metrics)
        or result.field_specific_partial_pair_count
        != sum(item.field_specific_partial_count for item in result.field_metrics)
        or any(
            type(value) is not int or value < 0
            for value in (
                result.targeted_complete_generation_count,
                result.targeted_ordered_complete_generation_count,
                result.targeted_misassociation_count,
                result.targeted_unseen_formatted_count,
                result.field_specific_exact_pair_count,
                result.field_specific_partial_pair_count,
                result.untargeted_exact_hit_count,
                result.untargeted_unique_value_count,
                result.untargeted_canary_name_count,
                result.untargeted_exposed_profile_count,
            )
        )
        or result.calibrated_at_checkpoint
        != (
            result.distinctive_exact_pair_count >= 10
            and result.distinctive_exposed_entity_count >= 5
        )
    ):
        raise MemorizationCalibrationError("resultado da auditoria canária viola o contrato")
    return result


def validate_run_component(value: str, label: str) -> str:
    try:
        return validate_storage_component(value, label)
    except Exception as error:
        raise MemorizationCalibrationError(f"{label} inválido") from error


__all__ = [
    "CALIBRATION_CLIENT_ID",
    "CALIBRATION_DATASET_ID",
    "CALIBRATION_FIXED_REPETITIONS",
    "CALIBRATION_LEARNING_RATE_ARMS",
    "CALIBRATION_LEARNING_RATE_MILLIONTHS",
    "CALIBRATION_REPETITIONS",
    "CALIBRATION_RUN_ID",
    "CALIBRATION_SEED",
    "DISTINCTIVE_FIELD_TYPES",
    "EXPECTED_MAIN_CONFIG_SHA256",
    "EXPECTED_ANCHOR_MODEL_SHA256",
    "MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION",
    "MEMORIZATION_CALIBRATION_SCHEMA_VERSION",
    "POSITIVE_CANARY_AUDIT_CHECKPOINT_SCHEMA_VERSION",
    "POSITIVE_CANARY_AUDIT_CONTEXT_SCHEMA_VERSION",
    "POSITIVE_CANARY_AUDIT_JOURNAL_SCHEMA_VERSION",
    "POSITIVE_CANARY_AUDIT_RESULT_SCHEMA_VERSION",
    "REPEATABLE_FIELD_TYPES",
    "CanaryFieldMetric",
    "LearningRateArmSpec",
    "MemorizationCalibrationArmResult",
    "MemorizationCalibrationError",
    "MemorizationCalibrationPreflightResult",
    "MemorizationCalibrationResult",
    "MemorizationCalibrationSpec",
    "PositiveCanaryAuditCheckpoint",
    "PositiveCanaryAuditResult",
    "PositiveCanaryEvaluatorContext",
    "load_memorization_calibration_spec_from_config",
    "learning_rate_arm_id",
    "learning_rate_value",
    "validate_memorization_calibration_arm_result",
    "validate_memorization_calibration_spec",
    "validate_positive_canary_audit_checkpoint",
    "validate_positive_canary_audit_result",
    "validate_run_component",
]
