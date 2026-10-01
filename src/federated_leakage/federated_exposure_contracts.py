"""Contratos da calibração federada de exposição local das vítimas."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .synthetic_profiles.storage import validate_storage_component
from .utility_evaluation import UtilityEvaluationComparison, UtilityEvaluationResult


FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION = (
    "federated-memorization-calibration/v1"
)
FEDERATED_EXPOSURE_ARM_SCHEMA_VERSION = "federated-exposure-arm/v1"
FEDERATED_EXPOSURE_ROUND_SCHEMA_VERSION = "federated-exposure-round/v1"
FEDERATED_EXPOSURE_AUDIT_RESULT_SCHEMA_VERSION = (
    "federated-exposure-audit-result/v1"
)
FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION = "federated-exposure-checkpoint/v1"

EXPERIMENT_SEED = 101
VICTIM_REPETITION_MULTIPLIERS = (1, 2, 4)
DEFAULT_RUN_ID = "federated-memorization-calibration-seed-101-v1"
DEFAULT_DATASET_ID = "federated-memorization-calibration-seed-101-v1-dataset-v4"
EXPECTED_MAIN_CONFIG_SHA256 = (
    "b5bde98b847e18927121c7f57d049d704f339d251b6f75c30b98fc692569fc2e"
)
EXPECTED_VICTIM_DATASET_SHA256 = (
    "7d08d2dfc889162227d4a87dfbec60766ae2fe6b0497b27733ce512ae861f3bb"
)
EXPECTED_BENIGN_SCHEDULE_SHA256 = (
    "1be2d55566a92bd613e31d20243bbb3555a859cf3427c1a9db44af096be78050"
)
EXPECTED_UTILITY_DATASET_SHA256 = (
    "a06fd9b76a1dad40192f2c167ccfff81c1a55ab3ced93cf18daca270933e1f1d"
)
REFERENCE_PILOT_RUN_ID = "pilot-greedy-lr-000030-seed-101-k01-v3"
REFERENCE_PAIRED_RESULT_SHA256 = (
    "082c7f45249b390e66216708b707fd81144612615cf028039366c4eef8836b28"
)
REFERENCE_F0_TRAJECTORY_SHA256 = (
    "539b21f50016e171aa3247a8e189d60d177688ff835d015184b7df7813b04fc4"
)
REFERENCE_F0_FINAL_MODEL_SHA256 = (
    "938ce284ddd6afe494f2fff8c73ebf0a15467441c3aa72b427b5b172af79ed2e"
)
REFERENCE_F0_UTILITY_SHA256 = (
    "de836873f867c79be7c51f3079f0a7f8df234173e6f83f9fc407b102e86c9d29"
)


class FederatedExposureError(RuntimeError):
    """A calibração federada violou seu contrato ou falhou fechada."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_hash(payload: Mapping[str, Any], domain: bytes) -> str:
    import json

    raw = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExposureArmSpec:
    arm_id: str
    victim_repetition_multiplier: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedMemorizationCalibrationSpec:
    experiment_seed: int
    default_run_id: str
    dataset_id: str
    schedule_id: str
    scenario: str
    auxiliary_weight_units: int
    rounds: int
    learning_rate_millionths: int
    arms: Tuple[ExposureArmSpec, ...]
    auxiliary_repetition_multiplier: int
    audit_target_count: int
    distinctive_exact_pair_threshold: int
    distinctive_entity_threshold: int
    expected_total_conversation_presentations: int
    expected_total_optimizer_steps: int
    expected_total_audit_generations: int
    expected_total_utility_conversations: int
    main_config_sha256: str
    expected_victim_dataset_sha256: str
    expected_benign_schedule_sha256: str
    expected_utility_dataset_sha256: str
    reference_pilot_run_id: str
    reference_paired_result_sha256: str
    reference_f0_trajectory_sha256: str
    reference_f0_final_model_sha256: str
    reference_f0_utility_sha256: str
    main_config_path: Path = field(repr=False, compare=False)
    schema_version: str = FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("main_config_path", None)
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedExposurePreflightResult:
    experiment_seed: int
    victim_client_count: int
    victim_conversation_count: int
    auxiliary_round_count: int
    auxiliary_conversation_count: int
    utility_profile_count: int
    utility_conversation_count: int
    victim_dataset_sha256: str
    benign_schedule_sha256: str
    utility_dataset_sha256: str
    model_state_sha256: str | None = None
    tokenization_validated: bool = False
    audit_validated: bool = False
    schema_version: str = FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedExposureRoundResult:
    arm_id: str
    victim_repetition_multiplier: int
    round_id: int
    conversation_presentations: int
    optimizer_steps: int
    victim_optimizer_steps: int
    auxiliary_optimizer_steps: int
    mean_client_loss: float
    mean_victim_loss: float
    auxiliary_loss: float
    aggregate_delta_l2_norm: float
    aggregate_delta_max_abs: float
    victim_dataset_sha256: str
    auxiliary_schedule_sha256: str
    auxiliary_values_sha256: str
    initial_model_sha256: str
    aggregate_update_sha256: str
    final_model_sha256: str
    client_order_sha256: str
    weights_sha256: str
    sample_order_schedule_sha256: str
    training_seed_schedule_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = FEDERATED_EXPOSURE_ROUND_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedExposureCheckpoint:
    arm_id: str
    victim_repetition_multiplier: int
    round_id: int
    model_state_sha256: str
    calibration_config_sha256: str
    artifact_sha256: str
    round_result: FederatedExposureRoundResult
    schema_version: str = FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "victim_repetition_multiplier": self.victim_repetition_multiplier,
            "round_id": self.round_id,
            "model_state_sha256": self.model_state_sha256,
            "calibration_config_sha256": self.calibration_config_sha256,
            "artifact_sha256": self.artifact_sha256,
            "round_result": self.round_result.as_safe_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedExposureAuditResult:
    arm_id: str | None
    victim_repetition_multiplier: int
    extraction_result_sha256: str
    target_count: int
    distinctive_exact_pair_count: int
    distinctive_exact_pair_denominator: int
    distinctive_exposed_entity_count: int
    distinctive_entity_denominator: int
    calibrated_at_checkpoint: bool
    model_state_sha256: str
    schema_version: str = FEDERATED_EXPOSURE_AUDIT_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedExposureArmResult:
    arm_id: str
    victim_repetition_multiplier: int
    completed_rounds: int
    conversation_presentations: int
    optimizer_steps: int
    baseline_model_sha256: str
    final_model_sha256: str
    round_result_sha256: str
    audit: FederatedExposureAuditResult
    utility: UtilityEvaluationResult
    checkpoint_artifact_sha256: str
    schema_version: str = FEDERATED_EXPOSURE_ARM_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "victim_repetition_multiplier": self.victim_repetition_multiplier,
            "completed_rounds": self.completed_rounds,
            "conversation_presentations": self.conversation_presentations,
            "optimizer_steps": self.optimizer_steps,
            "baseline_model_sha256": self.baseline_model_sha256,
            "final_model_sha256": self.final_model_sha256,
            "round_result_sha256": self.round_result_sha256,
            "audit": self.audit.as_safe_dict(),
            "utility": self.utility.as_safe_dict(),
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedMemorizationCalibrationResult:
    run_id: str
    experiment_seed: int
    baseline_model_sha256: str
    baseline_gate_passed: bool
    calibrated: bool
    first_successful_multiplier: int | None
    baseline_audit: FederatedExposureAuditResult
    baseline_utility: UtilityEvaluationResult
    arms: Tuple[FederatedExposureArmResult, ...]
    utility_comparisons: Tuple[UtilityEvaluationComparison, ...]
    total_federated_rounds: int
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_seed": self.experiment_seed,
            "baseline_model_sha256": self.baseline_model_sha256,
            "baseline_gate_passed": self.baseline_gate_passed,
            "calibrated": self.calibrated,
            "first_successful_multiplier": self.first_successful_multiplier,
            "baseline_audit": self.baseline_audit.as_safe_dict(),
            "baseline_utility": self.baseline_utility.as_safe_dict(),
            "arms": [item.as_safe_dict() for item in self.arms],
            "utility_comparisons": [
                item.as_safe_dict() for item in self.utility_comparisons
            ],
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "total_utility_conversations": self.total_utility_conversations,
            "result_sha256": self.result_sha256,
        }


def arm_id(multiplier: int) -> str:
    if type(multiplier) is not int or multiplier not in VICTIM_REPETITION_MULTIPLIERS:
        raise FederatedExposureError("multiplicador de exposição é inválido")
    return f"victim-repetitions-{multiplier:03d}"


def validate_federated_exposure_spec(
    spec: object,
) -> FederatedMemorizationCalibrationSpec:
    expected_arms = tuple(
        ExposureArmSpec(arm_id=arm_id(value), victim_repetition_multiplier=value)
        for value in VICTIM_REPETITION_MULTIPLIERS
    )
    if (
        not isinstance(spec, FederatedMemorizationCalibrationSpec)
        or type(spec.experiment_seed) is not int
        or type(spec.auxiliary_weight_units) is not int
        or type(spec.rounds) is not int
        or type(spec.learning_rate_millionths) is not int
        or type(spec.auxiliary_repetition_multiplier) is not int
        or type(spec.audit_target_count) is not int
        or type(spec.distinctive_exact_pair_threshold) is not int
        or type(spec.distinctive_entity_threshold) is not int
        or spec.schema_version != FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION
        or spec.experiment_seed != EXPERIMENT_SEED
        or spec.default_run_id != DEFAULT_RUN_ID
        or spec.dataset_id != DEFAULT_DATASET_ID
        or spec.schedule_id != "F0-F1"
        or spec.scenario != "F0"
        or spec.auxiliary_weight_units != 1
        or spec.rounds != 20
        or spec.learning_rate_millionths != 30
        or spec.arms != expected_arms
        or spec.auxiliary_repetition_multiplier != 1
        or spec.audit_target_count != 200
        or spec.distinctive_exact_pair_threshold != 10
        or spec.distinctive_entity_threshold != 5
        or spec.expected_total_conversation_presentations != 146_000
        or spec.expected_total_optimizer_steps != 36_500
        or spec.expected_total_audit_generations != 7_204
        or spec.expected_total_utility_conversations != 2_000
        or spec.main_config_sha256 != EXPECTED_MAIN_CONFIG_SHA256
        or spec.expected_victim_dataset_sha256 != EXPECTED_VICTIM_DATASET_SHA256
        or spec.expected_benign_schedule_sha256
        != EXPECTED_BENIGN_SCHEDULE_SHA256
        or spec.expected_utility_dataset_sha256 != EXPECTED_UTILITY_DATASET_SHA256
        or spec.reference_pilot_run_id != REFERENCE_PILOT_RUN_ID
        or spec.reference_paired_result_sha256 != REFERENCE_PAIRED_RESULT_SHA256
        or spec.reference_f0_trajectory_sha256 != REFERENCE_F0_TRAJECTORY_SHA256
        or spec.reference_f0_final_model_sha256 != REFERENCE_F0_FINAL_MODEL_SHA256
        or spec.reference_f0_utility_sha256 != REFERENCE_F0_UTILITY_SHA256
    ):
        raise FederatedExposureError("especificação da calibração diverge do protocolo")
    return spec


def load_federated_exposure_spec_from_config(
    path: Path,
) -> FederatedMemorizationCalibrationSpec:
    config_path = Path(path)
    try:
        payload = load_yaml_mapping(config_path)
    except ConfigurationError as error:
        raise FederatedExposureError(str(error)) from error
    expected_keys = {
        "schema_version",
        "main_config",
        "main_config_sha256",
        "experiment_seed",
        "default_run_id",
        "dataset_id",
        "schedule_id",
        "scenario",
        "auxiliary_weight_units",
        "rounds",
        "learning_rate_millionths",
        "victim_repetition_multipliers",
        "auxiliary_repetition_multiplier",
        "audit_target_count",
        "calibration_criterion",
        "expected_totals",
        "expected_hashes",
        "reference_pilot",
    }
    if set(payload) != expected_keys:
        raise FederatedExposureError("configuração da calibração possui chaves inválidas")
    main_name = payload.get("main_config")
    if main_name != "main-v3.yaml":
        raise FederatedExposureError("configuração principal referenciada é inválida")
    unresolved_main_path = config_path.parent / main_name
    try:
        if unresolved_main_path.is_symlink() or not unresolved_main_path.is_file():
            raise FederatedExposureError("configuração principal está ausente")
        main_path = unresolved_main_path.resolve()
        main_hash = hashlib.sha256(main_path.read_bytes()).hexdigest()
    except OSError as error:
        raise FederatedExposureError("configuração principal é inacessível") from error
    if main_hash != EXPECTED_MAIN_CONFIG_SHA256:
        raise FederatedExposureError("hash da configuração principal diverge")
    try:
        criterion = payload["calibration_criterion"]
        totals = payload["expected_totals"]
        hashes = payload["expected_hashes"]
        reference = payload["reference_pilot"]
        if not all(isinstance(item, Mapping) for item in (criterion, totals, hashes, reference)):
            raise TypeError
        if (
            set(criterion) != {"distinctive_exact_pairs", "distinctive_entities"}
            or set(totals)
            != {
                "conversation_presentations",
                "optimizer_steps",
                "audit_generations",
                "utility_conversations",
            }
            or set(hashes)
            != {
                "victim_dataset_sha256",
                "benign_schedule_sha256",
                "utility_dataset_sha256",
            }
            or set(reference)
            != {
                "run_id",
                "paired_results_sha256",
                "f0_trajectory_result_sha256",
                "f0_final_model_sha256",
                "f0_utility_scientific_sha256",
            }
        ):
            raise TypeError
        multipliers = tuple(payload["victim_repetition_multipliers"])
        spec = FederatedMemorizationCalibrationSpec(
            experiment_seed=payload["experiment_seed"],
            default_run_id=validate_storage_component(payload["default_run_id"], "run_id"),
            dataset_id=validate_storage_component(payload["dataset_id"], "dataset_id"),
            schedule_id=validate_storage_component(payload["schedule_id"], "schedule_id"),
            scenario=payload["scenario"],
            auxiliary_weight_units=payload["auxiliary_weight_units"],
            rounds=payload["rounds"],
            learning_rate_millionths=payload["learning_rate_millionths"],
            arms=tuple(
                ExposureArmSpec(
                    arm_id=arm_id(value), victim_repetition_multiplier=value
                )
                for value in multipliers
            ),
            auxiliary_repetition_multiplier=payload["auxiliary_repetition_multiplier"],
            audit_target_count=payload["audit_target_count"],
            distinctive_exact_pair_threshold=criterion["distinctive_exact_pairs"],
            distinctive_entity_threshold=criterion["distinctive_entities"],
            expected_total_conversation_presentations=totals["conversation_presentations"],
            expected_total_optimizer_steps=totals["optimizer_steps"],
            expected_total_audit_generations=totals["audit_generations"],
            expected_total_utility_conversations=totals["utility_conversations"],
            main_config_sha256=payload["main_config_sha256"],
            expected_victim_dataset_sha256=hashes["victim_dataset_sha256"],
            expected_benign_schedule_sha256=hashes["benign_schedule_sha256"],
            expected_utility_dataset_sha256=hashes["utility_dataset_sha256"],
            reference_pilot_run_id=reference["run_id"],
            reference_paired_result_sha256=reference["paired_results_sha256"],
            reference_f0_trajectory_sha256=reference["f0_trajectory_result_sha256"],
            reference_f0_final_model_sha256=reference["f0_final_model_sha256"],
            reference_f0_utility_sha256=reference["f0_utility_scientific_sha256"],
            main_config_path=main_path,
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FederatedExposureError(
            "configuração da calibração possui tipos inválidos"
        ) from error
    return validate_federated_exposure_spec(spec)


def validate_exposure_audit_result(
    result: object,
) -> FederatedExposureAuditResult:
    if (
        not isinstance(result, FederatedExposureAuditResult)
        or result.schema_version != FEDERATED_EXPOSURE_AUDIT_RESULT_SCHEMA_VERSION
        or result.arm_id not in {None, *(arm_id(value) for value in VICTIM_REPETITION_MULTIPLIERS)}
        or result.victim_repetition_multiplier not in {0, *VICTIM_REPETITION_MULTIPLIERS}
        or type(result.victim_repetition_multiplier) is not int
        or type(result.target_count) is not int
        or type(result.distinctive_exact_pair_count) is not int
        or type(result.distinctive_exact_pair_denominator) is not int
        or type(result.distinctive_exposed_entity_count) is not int
        or type(result.distinctive_entity_denominator) is not int
        or type(result.calibrated_at_checkpoint) is not bool
        or (result.arm_id is None) != (result.victim_repetition_multiplier == 0)
        or (
            result.arm_id is not None
            and result.arm_id != arm_id(result.victim_repetition_multiplier)
        )
        or result.target_count != 200
        or result.distinctive_exact_pair_denominator != 1_000
        or result.distinctive_entity_denominator != 200
        or not 0 <= result.distinctive_exact_pair_count <= 1_000
        or not 0 <= result.distinctive_exposed_entity_count <= 200
        or not _is_sha256(result.extraction_result_sha256)
        or not _is_sha256(result.model_state_sha256)
        or result.calibrated_at_checkpoint
        != (
            result.distinctive_exact_pair_count >= 10
            and result.distinctive_exposed_entity_count >= 5
        )
    ):
        raise FederatedExposureError("resultado da auditoria federada é inválido")
    return result


def _provenance_from_payload(value: object) -> ModelProvenance:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(ModelProvenance)
    }:
        raise FederatedExposureError("proveniência persistida é inválida")
    try:
        return ModelProvenance(**value)
    except (TypeError, ValueError) as error:
        raise FederatedExposureError("proveniência persistida é incompatível") from error


def exposure_audit_result_from_payload(
    value: object,
) -> FederatedExposureAuditResult:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(FederatedExposureAuditResult)
    }:
        raise FederatedExposureError("resultado persistido da auditoria é inválido")
    try:
        return validate_exposure_audit_result(FederatedExposureAuditResult(**value))
    except FederatedExposureError:
        raise
    except (TypeError, ValueError) as error:
        raise FederatedExposureError(
            "resultado persistido da auditoria é incompatível"
        ) from error


def _utility_result_from_payload(value: object) -> UtilityEvaluationResult:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(UtilityEvaluationResult)
    }:
        raise FederatedExposureError("resultado persistido de utilidade é inválido")
    try:
        from .utility_evaluation import validate_utility_evaluation_result

        payload = dict(value)
        payload["model_provenance"] = _provenance_from_payload(
            payload["model_provenance"]
        )
        return validate_utility_evaluation_result(UtilityEvaluationResult(**payload))
    except FederatedExposureError:
        raise
    except Exception as error:
        raise FederatedExposureError(
            "resultado persistido de utilidade é incompatível"
        ) from error


def _utility_comparison_from_payload(
    value: object,
) -> UtilityEvaluationComparison:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(UtilityEvaluationComparison)
    }:
        raise FederatedExposureError("comparação persistida de utilidade é inválida")
    try:
        result = UtilityEvaluationComparison(**value)
    except (TypeError, ValueError) as error:
        raise FederatedExposureError(
            "comparação persistida de utilidade é incompatível"
        ) from error
    if (
        result.scenario != "F0"
        or result.baseline_checkpoint_id != "B0"
        or result.final_checkpoint_id != "F0-round-020"
        or not _is_sha256(result.dataset_sha256)
        or result.automatic_gate
        or not result.human_review_required
        or any(
            not math.isfinite(metric)
            for metric in (
                result.mean_conversation_loss_delta,
                result.token_weighted_nll_delta,
                result.perplexity_delta,
                result.perplexity_relative_delta,
            )
        )
    ):
        raise FederatedExposureError("comparação persistida de utilidade diverge")
    return result


def exposure_arm_result_from_payload(value: object) -> FederatedExposureArmResult:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(FederatedExposureArmResult)
    }:
        raise FederatedExposureError("resultado persistido do braço é inválido")
    try:
        payload = dict(value)
        payload["audit"] = exposure_audit_result_from_payload(payload["audit"])
        payload["utility"] = _utility_result_from_payload(payload["utility"])
        result = FederatedExposureArmResult(**payload)
    except FederatedExposureError:
        raise
    except (TypeError, ValueError) as error:
        raise FederatedExposureError("resultado persistido do braço é incompatível") from error
    multiplier = result.victim_repetition_multiplier
    if (
        result.schema_version != FEDERATED_EXPOSURE_ARM_SCHEMA_VERSION
        or type(multiplier) is not int
        or type(result.completed_rounds) is not int
        or type(result.conversation_presentations) is not int
        or type(result.optimizer_steps) is not int
        or result.arm_id != arm_id(multiplier)
        or result.completed_rounds != 20
        or result.conversation_presentations != 20 * (1_000 * multiplier + 100)
        or result.optimizer_steps != 20 * (250 * multiplier + 25)
        or result.audit.arm_id != result.arm_id
        or result.audit.victim_repetition_multiplier != multiplier
        or result.audit.model_state_sha256 != result.final_model_sha256
        or result.utility.scenario != "F0"
        or result.utility.round_id != 20
        or result.utility.model_state_sha256 != result.final_model_sha256
        or any(
            not _is_sha256(item)
            for item in (
                result.baseline_model_sha256,
                result.final_model_sha256,
                result.round_result_sha256,
                result.checkpoint_artifact_sha256,
            )
        )
    ):
        raise FederatedExposureError("resultado persistido do braço diverge")
    return result


def calibration_result_from_payload(
    value: object,
) -> FederatedMemorizationCalibrationResult:
    if not isinstance(value, Mapping) or set(value) != {
        item.name for item in fields(FederatedMemorizationCalibrationResult)
    }:
        raise FederatedExposureError("resultado final persistido é inválido")
    try:
        payload = dict(value)
        payload["baseline_audit"] = exposure_audit_result_from_payload(
            payload["baseline_audit"]
        )
        payload["baseline_utility"] = _utility_result_from_payload(
            payload["baseline_utility"]
        )
        if not isinstance(payload["arms"], list) or not isinstance(
            payload["utility_comparisons"], list
        ):
            raise TypeError
        payload["arms"] = tuple(
            exposure_arm_result_from_payload(item) for item in payload["arms"]
        )
        payload["utility_comparisons"] = tuple(
            _utility_comparison_from_payload(item)
            for item in payload["utility_comparisons"]
        )
        result = FederatedMemorizationCalibrationResult(**payload)
    except FederatedExposureError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise FederatedExposureError("resultado final persistido é incompatível") from error
    successful = tuple(
        item.victim_repetition_multiplier
        for item in result.arms
        if item.audit.calibrated_at_checkpoint
    )
    unsigned = result.as_safe_dict()
    unsigned.pop("result_sha256")
    try:
        from .utility_evaluation import compare_utility_to_baseline

        expected_comparisons = tuple(
            compare_utility_to_baseline(result.baseline_utility, item.utility)
            for item in result.arms
        )
    except Exception as error:
        raise FederatedExposureError(
            "resultados de utilidade persistidos são incompatíveis"
        ) from error
    if (
        result.schema_version != FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION
        or result.run_id != DEFAULT_RUN_ID
        or result.experiment_seed != EXPERIMENT_SEED
        or result.baseline_gate_passed
        != result.baseline_audit.calibrated_at_checkpoint
        or result.calibrated != (not result.baseline_gate_passed and bool(successful))
        or result.first_successful_multiplier != (successful[0] if successful else None)
        or tuple(item.victim_repetition_multiplier for item in result.arms)
        != VICTIM_REPETITION_MULTIPLIERS
        or result.baseline_audit.arm_id is not None
        or result.baseline_audit.victim_repetition_multiplier != 0
        or result.baseline_audit.model_state_sha256
        != result.baseline_model_sha256
        or result.baseline_utility.scenario != "B0"
        or result.baseline_utility.model_state_sha256
        != result.baseline_model_sha256
        or any(
            item.baseline_model_sha256 != result.baseline_model_sha256
            or item.utility.dataset_sha256
            != result.baseline_utility.dataset_sha256
            for item in result.arms
        )
        or len(result.utility_comparisons) != 3
        or result.utility_comparisons != expected_comparisons
        or result.total_federated_rounds != 60
        or result.total_conversation_presentations != 146_000
        or result.total_optimizer_steps != 36_500
        or result.total_audit_generations != 7_204
        or result.total_utility_conversations != 2_000
        or result.result_sha256 != result_sha256(unsigned)
    ):
        raise FederatedExposureError("resultado final persistido diverge")
    return result


def result_sha256(payload: Mapping[str, Any]) -> str:
    return _safe_hash(payload, b"federated-memorization-calibration-result/v1")


__all__ = [
    "DEFAULT_DATASET_ID",
    "DEFAULT_RUN_ID",
    "FEDERATED_EXPOSURE_ARM_SCHEMA_VERSION",
    "FEDERATED_EXPOSURE_AUDIT_RESULT_SCHEMA_VERSION",
    "FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION",
    "FEDERATED_EXPOSURE_ROUND_SCHEMA_VERSION",
    "FEDERATED_MEMORIZATION_CALIBRATION_SCHEMA_VERSION",
    "FederatedExposureArmResult",
    "FederatedExposureAuditResult",
    "FederatedExposureCheckpoint",
    "FederatedExposureError",
    "FederatedExposurePreflightResult",
    "FederatedExposureRoundResult",
    "FederatedMemorizationCalibrationResult",
    "FederatedMemorizationCalibrationSpec",
    "ExposureArmSpec",
    "VICTIM_REPETITION_MULTIPLIERS",
    "arm_id",
    "calibration_result_from_payload",
    "exposure_arm_result_from_payload",
    "exposure_audit_result_from_payload",
    "load_federated_exposure_spec_from_config",
    "result_sha256",
    "validate_exposure_audit_result",
    "validate_federated_exposure_spec",
]
