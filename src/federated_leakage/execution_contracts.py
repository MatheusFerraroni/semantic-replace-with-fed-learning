"""Contratos estritos do piloto pareado e de suas trajetórias retomáveis."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple

from .aggregation_contracts import FedAvgRoundResult
from .audit_contracts import ExtractionAuditResult
from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .synthetic_profiles.model import GENERATOR_VERSION
from .synthetic_profiles.storage import validate_storage_component
from .utility_evaluation import (
    UtilityEvaluationComparison,
    UtilityEvaluationResult,
    UtilityEvaluationSpec,
    parse_utility_evaluation_spec,
    validate_utility_evaluation_spec,
)


PILOT_EXECUTION_SCHEMA_VERSION = "pilot-execution/v3"
FEDERATED_TRAJECTORY_SCHEMA_VERSION = "federated-trajectory/v3"
FEDERATED_CHECKPOINT_SCHEMA_VERSION = "federated-checkpoint/v3"
PILOT_DEVELOPMENT_SEED = 101
PILOT_AUXILIARY_WEIGHT_UNITS = 1
PILOT_SCHEDULE_ID = "F0-F1"
PILOT_ROUNDS = 20
PILOT_TARGET_COUNTS = (1, 5, 20, 200)
PILOT_REFERENCE_TARGET_COUNT = 20
PILOT_SENSITIVITY_TARGET_COUNTS = (1, 5, 200)
PILOT_EXPECTED_GENERATION_COUNT = 12_992
PILOT_CALIBRATION_RUN_ID = "memorization-calibration-greedy-lr-seed-101-v4"
PILOT_DEFAULT_RUN_ID = "pilot-greedy-lr-000030-seed-101-k01-v3"
PILOT_LEARNING_RATE_MILLIONTHS = 30
PILOT_CALIBRATION_RESULT_SHA256 = (
    "08d2dbfdda6ea0cd01b44a42d778c691c3587578c84dd07c5eff189b989dcddf"
)
PILOT_CALIBRATION_MAIN_CONFIG_SHA256 = (
    "18e066855ad147c7cc31bdd6221b62275eb8a6c44e0158e83cb610d3b4298d87"
)
PILOT_BASELINE_MODEL_SHA256 = (
    "19266d7e80078389de8073a5ad0d167d9889e72a4cb32475b8197718d81d3be0"
)

TrajectoryScenario = Literal["F0", "F1"]


class PilotExecutionError(RuntimeError):
    """O preflight, execução, persistência ou retomada do piloto falhou fechada."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotExecutionSpec:
    experiment_seed: int
    auxiliary_weight_units: int
    schedule_id: str
    rounds: int
    scenario_order: Tuple[str, ...]
    target_counts: Tuple[int, ...]
    reference_target_count: int
    sensitivity_target_counts: Tuple[int, ...]
    sensitivity_checkpoints: Tuple[str, ...]
    retained_rounds: Tuple[int, ...]
    rolling_resume_checkpoint: bool
    incomplete_round_policy: str
    expected_generation_count: int
    config_sha256: str
    freeze_requires_human_review: bool
    calibration_run_id: str
    calibration_schema_version: str
    calibration_decoding_strategy: str
    calibration_main_config_sha256: str
    calibration_result_sha256: str
    calibration_canary_dataset_sha256: str
    calibration_collision_preflight_sha256: str
    calibration_baseline_model_sha256: str
    calibration_selected_arm_id: str
    calibration_selected_learning_rate_millionths: int
    utility_evaluation: UtilityEvaluationSpec
    schema_version: str = PILOT_EXECUTION_SCHEMA_VERSION
    trajectory_schema_version: str = FEDERATED_TRAJECTORY_SCHEMA_VERSION
    checkpoint_schema_version: str = FEDERATED_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotRunIdentity:
    run_id: str
    dataset_id: str
    experiment_seed: int
    auxiliary_weight_units: int
    schedule_id: str
    config_sha256: str
    calibration_result_sha256: str
    calibration_manifest_sha256: str
    learning_rate_millionths: int
    schema_version: str = PILOT_EXECUTION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotPreflightResult:
    experiment_seed: int
    auxiliary_weight_units: int
    victim_client_count: int
    victim_conversation_count: int
    auxiliary_round_count: int
    auxiliary_conversation_count: int
    victim_dataset_sha256: str
    benign_schedule_sha256: str
    adversarial_schedule_sha256: str
    paired_schedule_sha256: str
    model_state_sha256: str | None = None
    tokenization_validated: bool = False
    audit_target_counts: Tuple[int, ...] = PILOT_TARGET_COUNTS
    calibration_result_sha256: str | None = None
    calibration_manifest_sha256: str | None = None
    utility_profile_count: int = 0
    utility_conversation_count: int = 0
    utility_dataset_sha256: str | None = None
    utility_tokenization_validated: bool = False
    schema_version: str = PILOT_EXECUTION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointAuditMarker:
    target_count: int
    audit_id: str
    result_sha256: str
    generation_schedule_sha256: str
    model_state_sha256: str
    schema_version: str = FEDERATED_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedCheckpointMetadata:
    scenario: TrajectoryScenario
    experiment_seed: int
    auxiliary_weight_units: int
    round_id: int
    config_sha256: str
    victim_dataset_sha256: str
    baseline_model_sha256: str
    baseline_audit_sha256: str
    model_state_sha256: str
    auxiliary_schedule_sha256: str
    auxiliary_values_sha256: str
    canonical_template_sha256: str
    round_result_sha256: str
    audit_markers: Tuple[CheckpointAuditMarker, ...]
    utility_result_sha256: str | None
    model_provenance: ModelProvenance
    seed_derivation: str = "sha256_domain_separated_from_single_experiment_seed"
    schema_version: str = FEDERATED_CHECKPOINT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        result["audit_markers"] = [asdict(marker) for marker in self.audit_markers]
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedFederatedCheckpoint:
    metadata: FederatedCheckpointMetadata
    round_result_payload: Mapping[str, Any] = field(repr=False)
    artifact_sha256: str
    schema_version: str = FEDERATED_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedTrajectoryState:
    scenario: TrajectoryScenario
    completed_round: int
    baseline_model_sha256: str
    current_model_sha256: str
    checkpoint_artifact_sha256: str | None
    schema_version: str = FEDERATED_TRAJECTORY_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedTrajectoryResult:
    scenario: TrajectoryScenario
    experiment_seed: int
    auxiliary_weight_units: int
    completed_rounds: int
    conversation_count: int
    optimizer_steps: int
    baseline_model_sha256: str
    baseline_audit_sha256: str
    final_model_sha256: str
    round_results: Tuple[FedAvgRoundResult, ...]
    audit_results: Tuple[ExtractionAuditResult, ...]
    utility_result: UtilityEvaluationResult
    result_sha256: str
    schema_version: str = FEDERATED_TRAJECTORY_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario": self.scenario,
            "experiment_seed": self.experiment_seed,
            "auxiliary_weight_units": self.auxiliary_weight_units,
            "completed_rounds": self.completed_rounds,
            "conversation_count": self.conversation_count,
            "optimizer_steps": self.optimizer_steps,
            "baseline_model_sha256": self.baseline_model_sha256,
            "baseline_audit_sha256": self.baseline_audit_sha256,
            "final_model_sha256": self.final_model_sha256,
            "round_result_count": len(self.round_results),
            "audit_result_count": len(self.audit_results),
            "utility_result": self.utility_result.as_safe_dict(),
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotExecutionResult:
    identity: PilotRunIdentity
    baseline_model_sha256: str
    baseline_audit_sha256: str
    baseline_audits: Tuple[ExtractionAuditResult, ...]
    baseline_utility: UtilityEvaluationResult
    trajectories: Tuple[FederatedTrajectoryResult, FederatedTrajectoryResult]
    utility_comparisons: Tuple[
        UtilityEvaluationComparison,
        UtilityEvaluationComparison,
    ]
    total_federated_rounds: int
    total_conversation_count: int
    total_optimizer_steps: int
    total_audit_generations: int
    paired_results_sha256: str
    completed: bool
    schema_version: str = PILOT_EXECUTION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.as_safe_dict(),
            "baseline_model_sha256": self.baseline_model_sha256,
            "baseline_audit_sha256": self.baseline_audit_sha256,
            "baseline_audit_count": len(self.baseline_audits),
            "baseline_utility": self.baseline_utility.as_safe_dict(),
            "trajectories": [item.as_safe_dict() for item in self.trajectories],
            "utility_comparisons": [
                item.as_safe_dict() for item in self.utility_comparisons
            ],
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_count": self.total_conversation_count,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "paired_results_sha256": self.paired_results_sha256,
            "completed": self.completed,
        }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise PilotExecutionError(f"configuração deve conter a seção {key}")
    return value


def validate_pilot_execution_spec(spec: object) -> PilotExecutionSpec:
    if (
        not isinstance(spec, PilotExecutionSpec)
        or spec.schema_version != PILOT_EXECUTION_SCHEMA_VERSION
        or spec.trajectory_schema_version != FEDERATED_TRAJECTORY_SCHEMA_VERSION
        or spec.checkpoint_schema_version != FEDERATED_CHECKPOINT_SCHEMA_VERSION
        or spec.experiment_seed != PILOT_DEVELOPMENT_SEED
        or spec.auxiliary_weight_units != PILOT_AUXILIARY_WEIGHT_UNITS
        or spec.schedule_id != PILOT_SCHEDULE_ID
        or spec.rounds != PILOT_ROUNDS
        or spec.scenario_order != ("B0", "F0", "F1")
        or spec.target_counts != PILOT_TARGET_COUNTS
        or spec.reference_target_count != PILOT_REFERENCE_TARGET_COUNT
        or spec.sensitivity_target_counts != PILOT_SENSITIVITY_TARGET_COUNTS
        or spec.sensitivity_checkpoints
        != ("B0", "F0-round-020", "F1-round-020")
        or spec.retained_rounds != (1, 10, 20)
        or spec.rolling_resume_checkpoint is not True
        or spec.incomplete_round_policy != "discard_and_replay"
        or spec.expected_generation_count != PILOT_EXPECTED_GENERATION_COUNT
        or not _is_sha256(spec.config_sha256)
        or spec.freeze_requires_human_review is not True
        or spec.calibration_run_id != PILOT_CALIBRATION_RUN_ID
        or spec.calibration_schema_version != "memorization-calibration/v4"
        or spec.calibration_decoding_strategy != "tokenwise_greedy_argmax/v1"
        or spec.calibration_main_config_sha256
        != PILOT_CALIBRATION_MAIN_CONFIG_SHA256
        or spec.calibration_result_sha256 != PILOT_CALIBRATION_RESULT_SHA256
        or spec.calibration_canary_dataset_sha256
        != "7f7feaaf39603847a81ee7c4e39519ea41ea162f669813e4664811ecd09da4ba"
        or spec.calibration_collision_preflight_sha256
        != "d3ea270b495cc7669006fa2f78a56184c759a97360322caaec66270c8a145295"
        or spec.calibration_baseline_model_sha256 != PILOT_BASELINE_MODEL_SHA256
        or spec.calibration_selected_arm_id != "lr-000030"
        or spec.calibration_selected_learning_rate_millionths
        != PILOT_LEARNING_RATE_MILLIONTHS
    ):
        raise PilotExecutionError("especificação do piloto diverge do protocolo")
    try:
        validate_utility_evaluation_spec(spec.utility_evaluation)
    except Exception as error:
        raise PilotExecutionError("especificação da utilidade diverge") from error
    return spec


def parse_pilot_execution_spec(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
) -> PilotExecutionSpec:
    if not isinstance(config, Mapping):
        raise PilotExecutionError("configuração do piloto deve ser mapeada")
    if config.get("schema_version") != "federated-leakage/main-config/v3":
        raise PilotExecutionError("schema da configuração principal é incompatível")
    pilot = _mapping(config, "pilot")
    checkpoints = _mapping(config, "checkpoints")
    required = {
        "required_before_main_campaign": True,
        "development_seeds": [101],
        "execution_schema_version": PILOT_EXECUTION_SCHEMA_VERSION,
        "trajectory_schema_version": FEDERATED_TRAJECTORY_SCHEMA_VERSION,
        "checkpoint_schema_version": FEDERATED_CHECKPOINT_SCHEMA_VERSION,
        "schedule_id": PILOT_SCHEDULE_ID,
        "auxiliary_weight_units": PILOT_AUXILIARY_WEIGHT_UNITS,
        "federated_rounds": PILOT_ROUNDS,
        "paired_execution": True,
        "scenario_order": ["B0", "F0", "F1"],
        "target_profile_counts": list(PILOT_TARGET_COUNTS),
        "reference_target_profile_count": PILOT_REFERENCE_TARGET_COUNT,
        "reference_audit_rounds": "all",
        "sensitivity_target_profile_counts": list(
            PILOT_SENSITIVITY_TARGET_COUNTS
        ),
        "sensitivity_audit_checkpoints": [
            "B0",
            "F0-round-020",
            "F1-round-020",
        ],
        "expected_generation_count": PILOT_EXPECTED_GENERATION_COUNT,
        "scenarios": ["B0", "F0", "F1"],
        "freeze_reference_recipe_after_pilot": True,
        "freeze_requires_human_review": True,
        "calibration_gate": {
            "required": True,
            "run_id": PILOT_CALIBRATION_RUN_ID,
            "schema_version": "memorization-calibration/v4",
            "decoding_strategy": "tokenwise_greedy_argmax/v1",
            "calibration_main_config_sha256": PILOT_CALIBRATION_MAIN_CONFIG_SHA256,
            "result_sha256": PILOT_CALIBRATION_RESULT_SHA256,
            "canary_dataset_sha256": "7f7feaaf39603847a81ee7c4e39519ea41ea162f669813e4664811ecd09da4ba",
            "collision_preflight_sha256": "d3ea270b495cc7669006fa2f78a56184c759a97360322caaec66270c8a145295",
            "baseline_model_sha256": PILOT_BASELINE_MODEL_SHA256,
            "selected_arm_id": "lr-000030",
            "selected_learning_rate_millionths": PILOT_LEARNING_RATE_MILLIONTHS,
            "require_calibrated": True,
            "require_baseline_gate_passed": False,
        },
    }
    if frozenset(pilot) != frozenset(required):
        raise PilotExecutionError("seção pilot possui chaves desconhecidas")
    for key, expected in required.items():
        if pilot.get(key) != expected:
            raise PilotExecutionError(f"pilot.{key} diverge do protocolo")
    checkpoint_required = {
        "retained_rounds": [1, 10, 20],
        "rolling_resume_checkpoint": True,
        "rolling_resume_retention": 1,
        "resume_granularity": "completed_round",
        "incomplete_round_policy": "discard_and_replay",
        "rolling_resume_contents": [
            "global_model_bfloat16_safetensors",
            "completed_round",
            "safe_federated_round_result",
            "model_fingerprint_and_provenance",
            "shared_baseline_and_B0_audit_hashes",
            "cpu_and_device_rng_states",
            "resolved_config_hash",
            "victim_dataset_hash",
            "auxiliary_schedule_and_values_hashes",
            "canonical_profile_template_hash",
            "completed_audit_markers_and_hashes",
            "completed_utility_result_hash_at_round_20",
        ],
        "forbidden_checkpoint_contents": [
            "optimizer_states",
            "parameter_deltas",
            "tokens",
            "conversation_texts",
            "protected_records",
        ],
        "audit_rounds": "all",
    }
    if frozenset(checkpoints) != frozenset(checkpoint_required):
        raise PilotExecutionError("seção checkpoints possui chaves desconhecidas")
    for key, expected in checkpoint_required.items():
        if checkpoints.get(key) != expected:
            raise PilotExecutionError(f"checkpoints.{key} diverge do protocolo")
    return validate_pilot_execution_spec(
        PilotExecutionSpec(
            experiment_seed=PILOT_DEVELOPMENT_SEED,
            auxiliary_weight_units=PILOT_AUXILIARY_WEIGHT_UNITS,
            schedule_id=PILOT_SCHEDULE_ID,
            rounds=PILOT_ROUNDS,
            scenario_order=("B0", "F0", "F1"),
            target_counts=PILOT_TARGET_COUNTS,
            reference_target_count=PILOT_REFERENCE_TARGET_COUNT,
            sensitivity_target_counts=PILOT_SENSITIVITY_TARGET_COUNTS,
            sensitivity_checkpoints=("B0", "F0-round-020", "F1-round-020"),
            retained_rounds=(1, 10, 20),
            rolling_resume_checkpoint=True,
            incomplete_round_policy="discard_and_replay",
            expected_generation_count=PILOT_EXPECTED_GENERATION_COUNT,
            config_sha256=config_sha256,
            freeze_requires_human_review=True,
            calibration_run_id=PILOT_CALIBRATION_RUN_ID,
            calibration_schema_version="memorization-calibration/v4",
            calibration_decoding_strategy="tokenwise_greedy_argmax/v1",
            calibration_main_config_sha256=PILOT_CALIBRATION_MAIN_CONFIG_SHA256,
            calibration_result_sha256=PILOT_CALIBRATION_RESULT_SHA256,
            calibration_canary_dataset_sha256=(
                "7f7feaaf39603847a81ee7c4e39519ea41ea162f669813e4664811ecd09da4ba"
            ),
            calibration_collision_preflight_sha256=(
                "d3ea270b495cc7669006fa2f78a56184c759a97360322caaec66270c8a145295"
            ),
            calibration_baseline_model_sha256=PILOT_BASELINE_MODEL_SHA256,
            calibration_selected_arm_id="lr-000030",
            calibration_selected_learning_rate_millionths=(
                PILOT_LEARNING_RATE_MILLIONTHS
            ),
            utility_evaluation=parse_utility_evaluation_spec(config),
        )
    )


def load_pilot_execution_spec_from_config(path: Path) -> PilotExecutionSpec:
    resolved_path = Path(path)
    try:
        raw = resolved_path.read_bytes()
        config = load_yaml_mapping(resolved_path)
    except (OSError, ConfigurationError) as error:
        raise PilotExecutionError("configuração do piloto é inválida") from error
    return parse_pilot_execution_spec(
        config,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_pilot_run_identity(
    spec: PilotExecutionSpec,
    *,
    run_id: str | None = None,
    calibration_result_sha256: str,
    calibration_manifest_sha256: str,
) -> PilotRunIdentity:
    resolved = validate_pilot_execution_spec(spec)
    try:
        safe_run_id = validate_storage_component(
            run_id or PILOT_DEFAULT_RUN_ID,
            "run_id",
        )
        version = GENERATOR_VERSION.rsplit("/", 1)[-1]
        dataset_id = validate_storage_component(
            f"{safe_run_id}-dataset-{version}",
            "dataset_id",
        )
    except Exception as error:
        raise PilotExecutionError("identidade do piloto é inválida") from error
    if not _is_sha256(calibration_result_sha256):
        raise PilotExecutionError("gate da calibração possui hash inválido")
    if not _is_sha256(calibration_manifest_sha256):
        raise PilotExecutionError("manifesto da calibração possui hash inválido")
    return PilotRunIdentity(
        run_id=safe_run_id,
        dataset_id=dataset_id,
        experiment_seed=resolved.experiment_seed,
        auxiliary_weight_units=resolved.auxiliary_weight_units,
        schedule_id=resolved.schedule_id,
        config_sha256=resolved.config_sha256,
        calibration_result_sha256=calibration_result_sha256,
        calibration_manifest_sha256=calibration_manifest_sha256,
        learning_rate_millionths=PILOT_LEARNING_RATE_MILLIONTHS,
    )


__all__ = [
    "CheckpointAuditMarker",
    "FEDERATED_CHECKPOINT_SCHEMA_VERSION",
    "FEDERATED_TRAJECTORY_SCHEMA_VERSION",
    "FederatedCheckpointMetadata",
    "FederatedTrajectoryResult",
    "FederatedTrajectoryState",
    "LoadedFederatedCheckpoint",
    "PILOT_AUXILIARY_WEIGHT_UNITS",
    "PILOT_DEVELOPMENT_SEED",
    "PILOT_DEFAULT_RUN_ID",
    "PILOT_EXECUTION_SCHEMA_VERSION",
    "PILOT_EXPECTED_GENERATION_COUNT",
    "PILOT_LEARNING_RATE_MILLIONTHS",
    "PILOT_REFERENCE_TARGET_COUNT",
    "PILOT_ROUNDS",
    "PILOT_SCHEDULE_ID",
    "PILOT_SENSITIVITY_TARGET_COUNTS",
    "PILOT_TARGET_COUNTS",
    "PilotExecutionError",
    "PilotExecutionResult",
    "PilotExecutionSpec",
    "PilotPreflightResult",
    "PilotRunIdentity",
    "TrajectoryScenario",
    "build_pilot_run_identity",
    "load_pilot_execution_spec_from_config",
    "parse_pilot_execution_spec",
    "validate_pilot_execution_spec",
]
