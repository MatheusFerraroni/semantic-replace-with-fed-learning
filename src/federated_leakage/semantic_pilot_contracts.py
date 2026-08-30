"""Contratos do piloto pareado F0/F1/F4/F5 com substituição rotativa."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .semantic_audit import SemanticAuditResult
from .synthetic_profiles.storage import validate_storage_component
from .utility_evaluation import UtilityEvaluationResult


SEMANTIC_PILOT_SCHEMA_VERSION = "semantic-substitution-pilot/v1"
SEMANTIC_TRAJECTORY_SCHEMA_VERSION = "semantic-substitution-trajectory/v1"
SEMANTIC_ROUND_SCHEMA_VERSION = "semantic-substitution-round/v1"
SEMANTIC_CHECKPOINT_SCHEMA_VERSION = "semantic-substitution-checkpoint/v1"
SEMANTIC_RESULT_SCHEMA_VERSION = "semantic-substitution-result/v1"
SEMANTIC_COMBINED_SCHEMA_VERSION = "semantic-substitution-combined/v1"

EXPERIMENT_SEEDS = (101, 361506353)
SCENARIO_ORDER = ("F0", "F1", "F4", "F5")
EXPECTED_MAIN_CONFIG_SHA256 = (
    "f30b700f195434d8f824f95814886e5d9df970626e6704ddf78dd9cfec64fea8"
)
EXPECTED_GRID_COMBINED_SHA256 = (
    "4132d3bb053e4edc18fbe2effaf12d2cf9221583c64f2d81b4589f2f099a9244"
)
EXPECTED_GRID_RESULT_SHA256 = {
    101: "33a8924ab218e065e9b94ccf3846639cf81fba6de61b67f6f0917ea855af9fca",
    361506353: "09e8d759cfeb67662ebd0a9e7841749a0b3122e400d07df66a920403672b5cf4",
}
SELECTED_GRID_ARM_ID = "victim-lr-000100-repetitions-004"

SemanticScenario = Literal["F0", "F1", "F4", "F5"]


class SemanticPilotError(RuntimeError):
    """A execução da defesa falhou fechada e sem conteúdo protegido."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def default_run_id(seed: int) -> str:
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise SemanticPilotError("seed do piloto semântico é inválida")
    return f"semantic-substitution-upstream-seed-{seed}-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticPilotSpec:
    experiment_seeds: Tuple[int, ...]
    default_run_ids: Tuple[Tuple[int, str], ...]
    schedule_id: str
    scenario_order: Tuple[str, ...]
    rounds: int
    auxiliary_weight_units: int
    victim_learning_rate_millionths: int
    victim_repetition_multiplier: int
    auxiliary_learning_rate_millionths: int
    auxiliary_repetition_multiplier: int
    grid_combined_run_id: str
    grid_combined_result_sha256: str
    grid_arm_id: str
    grid_result_sha256_by_seed: Tuple[Tuple[int, str], ...]
    baseline_target_count: int
    round_target_count: int
    endpoint_target_count: int
    historical_target_count: int
    historical_rounds: int
    expected_audit_generations_per_seed: int
    minimum_original_exact_pair_reduction: float
    maximum_original_complete_profiles: int
    comparator_distinctive_exact_pairs: int
    comparator_distinctive_entities: int
    comparator_distinctive_field_types: int
    require_both_seeds: bool
    expected_per_seed: Tuple[int, int, int, int, int, int]
    expected_combined: Tuple[int, int, int, int, int, int]
    retained_rounds: Tuple[int, ...]
    rolling_resume_checkpoint: bool
    incomplete_round_policy: str
    main_config_sha256: str
    main_config_path: Path = field(repr=False, compare=False)
    schema_version: str = SEMANTIC_PILOT_SCHEMA_VERSION

    def run_id_for_seed(self, seed: int) -> str:
        try:
            return dict(self.default_run_ids)[seed]
        except KeyError as error:
            raise SemanticPilotError("seed não pertence ao piloto") from error


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticPilotPreflightResult:
    selected_seed: int
    validated_seeds: Tuple[int, ...]
    victim_conversation_count: int
    auxiliary_conversation_count: int
    replacement_round_count: int
    replacement_conversation_count: int
    utility_conversation_count: int
    replacement_schedule_sha256: str
    replacement_values_sha256: str
    grid_gate_sha256: str
    model_state_sha256: str | None = None
    tokenization_validated: bool = False
    schema_version: str = SEMANTIC_PILOT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticFederatedRoundResult:
    scenario: SemanticScenario
    experiment_seed: int
    round_id: int
    victim_learning_rate_millionths: int
    victim_repetition_multiplier: int
    auxiliary_learning_rate_millionths: int
    conversation_presentations: int
    optimizer_steps: int
    mean_client_loss: float
    mean_victim_loss: float
    auxiliary_loss: float
    source_victim_dataset_sha256: str
    training_victim_dataset_sha256: str
    replacement_schedule_sha256: str | None
    replacement_values_sha256: str | None
    auxiliary_schedule_sha256: str
    auxiliary_values_sha256: str
    auxiliary_presentation_sha256: str
    auxiliary_batch_sha256: str
    initial_model_sha256: str
    aggregate_update_sha256: str
    final_model_sha256: str
    client_order_sha256: str
    weights_sha256: str
    sample_order_schedule_sha256: str
    training_seed_schedule_sha256: str
    aggregate_delta_l2_norm: float
    aggregate_delta_max_abs: float
    model_provenance: ModelProvenance
    schema_version: str = SEMANTIC_ROUND_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_provenance"] = self.model_provenance.as_safe_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticTrajectoryResult:
    scenario: SemanticScenario
    experiment_seed: int
    completed_rounds: int
    conversation_presentations: int
    optimizer_steps: int
    baseline_model_sha256: str
    final_model_sha256: str
    round_result_sha256: str
    original_audit_exact_pairs: int
    original_audit_complete_profiles: int
    distinctive_exact_pair_count: int
    distinctive_exposed_entity_count: int
    distinctive_field_type_count: int
    original_audit_result_sha256: str
    alias_audit_result_sha256: str | None
    historical_audit_result_sha256: str | None
    utility: UtilityEvaluationResult
    result_sha256: str
    schema_version: str = SEMANTIC_TRAJECTORY_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["utility"] = self.utility.as_safe_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticDefenseGateResult:
    seed: int
    baseline_gate_passed: bool
    f0_comparator_eligible: bool
    f1_comparator_eligible: bool
    f4_original_exact_pair_reduction: float | None
    f5_original_exact_pair_reduction: float | None
    f4_original_complete_profiles: int
    f5_original_complete_profiles: int
    status: str
    result_sha256: str
    schema_version: str = SEMANTIC_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticPilotResult:
    run_id: str
    experiment_seed: int
    baseline_model_sha256: str
    trajectories: Tuple[SemanticTrajectoryResult, ...]
    gate: SemanticDefenseGateResult
    total_federated_rounds: int
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = SEMANTIC_PILOT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_seed": self.experiment_seed,
            "baseline_model_sha256": self.baseline_model_sha256,
            "trajectories": [value.as_safe_dict() for value in self.trajectories],
            "gate": self.gate.as_safe_dict(),
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "total_utility_conversations": self.total_utility_conversations,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticCombinedResult:
    source_result_sha256_by_seed: Tuple[Tuple[int, str], ...]
    status_by_seed: Tuple[Tuple[int, str], ...]
    combined_status: str
    require_both_seeds: bool
    total_trajectories: int
    total_federated_rounds: int
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = SEMANTIC_COMBINED_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_result_sha256_by_seed": dict(self.source_result_sha256_by_seed),
            "status_by_seed": dict(self.status_by_seed),
            "combined_status": self.combined_status,
            "require_both_seeds": self.require_both_seeds,
            "total_trajectories": self.total_trajectories,
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "total_utility_conversations": self.total_utility_conversations,
            "result_sha256": self.result_sha256,
        }


def safe_result_sha256(value: Mapping[str, Any], domain: bytes) -> str:
    raw = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


def validate_semantic_pilot_spec(spec: object) -> SemanticPilotSpec:
    if (
        not isinstance(spec, SemanticPilotSpec)
        or spec.schema_version != SEMANTIC_PILOT_SCHEMA_VERSION
        or spec.experiment_seeds != EXPERIMENT_SEEDS
        or spec.default_run_ids
        != tuple((seed, default_run_id(seed)) for seed in EXPERIMENT_SEEDS)
        or spec.schedule_id != "F0-F1"
        or spec.scenario_order != SCENARIO_ORDER
        or spec.rounds != 20
        or spec.auxiliary_weight_units != 1
        or spec.victim_learning_rate_millionths != 100
        or spec.victim_repetition_multiplier != 4
        or spec.auxiliary_learning_rate_millionths != 30
        or spec.auxiliary_repetition_multiplier != 1
        or spec.grid_combined_run_id != "federated-memorization-grid-v2"
        or spec.grid_combined_result_sha256 != EXPECTED_GRID_COMBINED_SHA256
        or spec.grid_arm_id != SELECTED_GRID_ARM_ID
        or dict(spec.grid_result_sha256_by_seed) != EXPECTED_GRID_RESULT_SHA256
        or (
            spec.baseline_target_count,
            spec.round_target_count,
            spec.endpoint_target_count,
            spec.historical_target_count,
            spec.historical_rounds,
            spec.expected_audit_generations_per_seed,
        )
        != (200, 20, 200, 20, 19, 40_083)
        or spec.minimum_original_exact_pair_reduction != 0.90
        or spec.maximum_original_complete_profiles != 0
        or (
            spec.comparator_distinctive_exact_pairs,
            spec.comparator_distinctive_entities,
            spec.comparator_distinctive_field_types,
        )
        != (50, 25, 2)
        or spec.require_both_seeds is not True
        or spec.expected_per_seed != (4, 80, 328_000, 82_000, 40_083, 2_500)
        or spec.expected_combined != (8, 160, 656_000, 164_000, 80_166, 5_000)
        or spec.retained_rounds != (1, 10, 20)
        or spec.rolling_resume_checkpoint is not True
        or spec.incomplete_round_policy != "replay"
        or spec.main_config_sha256 != EXPECTED_MAIN_CONFIG_SHA256
    ):
        raise SemanticPilotError("especificação do piloto semântico diverge")
    return spec


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SemanticPilotError(f"seção {label} é inválida")
    return value


def load_semantic_pilot_spec_from_config(path: Path) -> SemanticPilotSpec:
    config_path = Path(path)
    try:
        payload = load_yaml_mapping(config_path)
    except ConfigurationError as error:
        raise SemanticPilotError(str(error)) from error
    expected = {
        "schema_version", "main_config", "main_config_sha256",
        "experiment_seeds", "default_run_ids", "schedule_id", "scenario_order",
        "rounds", "auxiliary_weight_units", "victim_learning_rate_millionths",
        "victim_repetition_multiplier", "auxiliary_learning_rate_millionths",
        "auxiliary_repetition_multiplier", "selected_grid_gate", "audit_schedule",
        "privacy_gate", "expected_totals_per_seed", "expected_totals_combined",
        "checkpointing",
    }
    if set(payload) != expected or payload.get("main_config") != "main-v4.yaml":
        raise SemanticPilotError("configuração do piloto possui chaves inválidas")
    main_path = config_path.parent / "main-v4.yaml"
    if main_path.is_symlink() or not main_path.is_file():
        raise SemanticPilotError("configuração principal está ausente")
    main_path = main_path.resolve()
    if hashlib.sha256(main_path.read_bytes()).hexdigest() != EXPECTED_MAIN_CONFIG_SHA256:
        raise SemanticPilotError("hash da configuração principal diverge")
    try:
        seeds = tuple(payload["experiment_seeds"])
        run_ids = _mapping(
            payload["default_run_ids"], {str(seed) for seed in EXPERIMENT_SEEDS}, "runs"
        )
        grid = _mapping(
            payload["selected_grid_gate"],
            {"combined_run_id", "combined_result_sha256", "arm_id", "result_sha256_by_seed"},
            "gate da grade",
        )
        grid_results = _mapping(
            grid["result_sha256_by_seed"], {str(seed) for seed in EXPERIMENT_SEEDS}, "resultados da grade"
        )
        audit = _mapping(
            payload["audit_schedule"],
            {"baseline_target_count", "round_target_count", "endpoint_target_count", "historical_target_count", "historical_rounds", "expected_generations_per_seed"},
            "auditoria",
        )
        privacy = _mapping(
            payload["privacy_gate"],
            {"minimum_original_exact_pair_reduction", "maximum_original_complete_profiles", "comparator_distinctive_exact_pairs", "comparator_distinctive_entities", "comparator_distinctive_field_types", "require_both_seeds"},
            "privacidade",
        )
        total_keys = {"trajectories", "federated_rounds", "conversation_presentations", "optimizer_steps", "audit_generations", "utility_conversations"}
        per_seed = _mapping(payload["expected_totals_per_seed"], total_keys, "totais por seed")
        combined = _mapping(payload["expected_totals_combined"], total_keys, "totais combinados")
        checkpointing = _mapping(
            payload["checkpointing"],
            {"retained_rounds", "rolling_resume_checkpoint", "incomplete_round_policy"},
            "checkpointing",
        )
        order = ("trajectories", "federated_rounds", "conversation_presentations", "optimizer_steps", "audit_generations", "utility_conversations")
        spec = SemanticPilotSpec(
            experiment_seeds=seeds,
            default_run_ids=tuple(
                (seed, validate_storage_component(run_ids[str(seed)], "run_id"))
                for seed in seeds
            ),
            schedule_id=validate_storage_component(payload["schedule_id"], "schedule_id"),
            scenario_order=tuple(payload["scenario_order"]),
            rounds=payload["rounds"],
            auxiliary_weight_units=payload["auxiliary_weight_units"],
            victim_learning_rate_millionths=payload["victim_learning_rate_millionths"],
            victim_repetition_multiplier=payload["victim_repetition_multiplier"],
            auxiliary_learning_rate_millionths=payload["auxiliary_learning_rate_millionths"],
            auxiliary_repetition_multiplier=payload["auxiliary_repetition_multiplier"],
            grid_combined_run_id=grid["combined_run_id"],
            grid_combined_result_sha256=grid["combined_result_sha256"],
            grid_arm_id=grid["arm_id"],
            grid_result_sha256_by_seed=tuple(
                (seed, grid_results[str(seed)]) for seed in seeds
            ),
            baseline_target_count=audit["baseline_target_count"],
            round_target_count=audit["round_target_count"],
            endpoint_target_count=audit["endpoint_target_count"],
            historical_target_count=audit["historical_target_count"],
            historical_rounds=audit["historical_rounds"],
            expected_audit_generations_per_seed=audit["expected_generations_per_seed"],
            minimum_original_exact_pair_reduction=float(privacy["minimum_original_exact_pair_reduction"]),
            maximum_original_complete_profiles=privacy["maximum_original_complete_profiles"],
            comparator_distinctive_exact_pairs=privacy["comparator_distinctive_exact_pairs"],
            comparator_distinctive_entities=privacy["comparator_distinctive_entities"],
            comparator_distinctive_field_types=privacy["comparator_distinctive_field_types"],
            require_both_seeds=privacy["require_both_seeds"],
            expected_per_seed=tuple(per_seed[key] for key in order),
            expected_combined=tuple(combined[key] for key in order),
            retained_rounds=tuple(checkpointing["retained_rounds"]),
            rolling_resume_checkpoint=checkpointing["rolling_resume_checkpoint"],
            incomplete_round_policy=checkpointing["incomplete_round_policy"],
            main_config_sha256=payload["main_config_sha256"],
            main_config_path=main_path,
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticPilotError("configuração do piloto possui tipos inválidos") from error
    return validate_semantic_pilot_spec(spec)


def validate_semantic_round_result(
    result: object,
) -> SemanticFederatedRoundResult:
    if not isinstance(result, SemanticFederatedRoundResult):
        raise SemanticPilotError("resultado de rodada semântica é inválido")
    hashes = (
        result.source_victim_dataset_sha256,
        result.training_victim_dataset_sha256,
        result.auxiliary_schedule_sha256,
        result.auxiliary_values_sha256,
        result.auxiliary_presentation_sha256,
        result.auxiliary_batch_sha256,
        result.initial_model_sha256,
        result.aggregate_update_sha256,
        result.final_model_sha256,
        result.client_order_sha256,
        result.weights_sha256,
        result.sample_order_schedule_sha256,
        result.training_seed_schedule_sha256,
    )
    protected = result.scenario in {"F4", "F5"}
    if (
        result.schema_version != SEMANTIC_ROUND_SCHEMA_VERSION
        or result.scenario not in SCENARIO_ORDER
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or not 1 <= result.round_id <= 20
        or result.victim_learning_rate_millionths != 100
        or result.victim_repetition_multiplier != 4
        or result.auxiliary_learning_rate_millionths != 30
        or result.conversation_presentations != 4_100
        or result.optimizer_steps != 1_025
        or any(not math.isfinite(value) for value in (
            result.mean_client_loss, result.mean_victim_loss, result.auxiliary_loss,
            result.aggregate_delta_l2_norm, result.aggregate_delta_max_abs,
        ))
        or any(not _is_sha256(value) for value in hashes)
        or protected != (result.replacement_schedule_sha256 is not None)
        or protected != (result.replacement_values_sha256 is not None)
        or any(
            value is not None and not _is_sha256(value)
            for value in (result.replacement_schedule_sha256, result.replacement_values_sha256)
        )
    ):
        raise SemanticPilotError("resultado de rodada diverge do contrato")
    return result


def semantic_trajectory_from_payload(value: object) -> SemanticTrajectoryResult:
    if not isinstance(value, Mapping):
        raise SemanticPilotError("trajetória persistida é inválida")
    expected = {
        "schema_version", "scenario", "experiment_seed", "completed_rounds",
        "conversation_presentations", "optimizer_steps", "baseline_model_sha256",
        "final_model_sha256", "round_result_sha256", "original_audit_exact_pairs",
        "original_audit_complete_profiles", "distinctive_exact_pair_count",
        "distinctive_exposed_entity_count", "distinctive_field_type_count",
        "original_audit_result_sha256", "alias_audit_result_sha256",
        "historical_audit_result_sha256", "utility", "result_sha256",
    }
    if set(value) != expected:
        raise SemanticPilotError("trajetória persistida possui chaves inválidas")
    try:
        from .execution_storage import utility_result_from_safe_payload
        payload = dict(value)
        payload["utility"] = utility_result_from_safe_payload(payload["utility"])
        result = SemanticTrajectoryResult(**payload)
    except Exception as error:
        raise SemanticPilotError("trajetória persistida é incompatível") from error
    if (
        result.schema_version != SEMANTIC_TRAJECTORY_SCHEMA_VERSION
        or result.scenario not in SCENARIO_ORDER
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or result.completed_rounds != 20
        or result.conversation_presentations != 82_000
        or result.optimizer_steps != 20_500
        or any(
            type(candidate) is not int or candidate < 0
            for candidate in (
                result.original_audit_exact_pairs,
                result.original_audit_complete_profiles,
                result.distinctive_exact_pair_count,
                result.distinctive_exposed_entity_count,
                result.distinctive_field_type_count,
            )
        )
        or result.distinctive_exact_pair_count > 1_000
        or result.distinctive_exposed_entity_count > 200
        or result.distinctive_field_type_count > 5
        or any(
            not _is_sha256(candidate)
            for candidate in (
                result.baseline_model_sha256, result.final_model_sha256,
                result.round_result_sha256, result.original_audit_result_sha256,
                result.result_sha256,
            )
        )
        or any(
            candidate is not None and not _is_sha256(candidate)
            for candidate in (
                result.alias_audit_result_sha256,
                result.historical_audit_result_sha256,
            )
        )
        or result.result_sha256
        != safe_result_sha256(
            {
                key: value
                for key, value in result.as_safe_dict().items()
                if key != "result_sha256"
            },
            b"semantic-substitution-trajectory-result/v1",
        )
    ):
        raise SemanticPilotError("trajetória persistida diverge do contrato")
    return result


def semantic_gate_from_payload(value: object) -> SemanticDefenseGateResult:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "seed", "baseline_gate_passed", "f0_comparator_eligible",
        "f1_comparator_eligible", "f4_original_exact_pair_reduction",
        "f5_original_exact_pair_reduction", "f4_original_complete_profiles",
        "f5_original_complete_profiles", "status", "result_sha256",
    }:
        raise SemanticPilotError("gate persistido é inválido")
    try:
        result = SemanticDefenseGateResult(**value)
    except Exception as error:
        raise SemanticPilotError("gate persistido possui tipos inválidos") from error
    if (
        result.schema_version != SEMANTIC_RESULT_SCHEMA_VERSION
        or result.seed not in EXPERIMENT_SEEDS
        or type(result.baseline_gate_passed) is not bool
        or type(result.f0_comparator_eligible) is not bool
        or type(result.f1_comparator_eligible) is not bool
        or any(
            candidate is not None
            and (
                type(candidate) is not float
                or not math.isfinite(candidate)
                or candidate > 1.0
            )
            for candidate in (
                result.f4_original_exact_pair_reduction,
                result.f5_original_exact_pair_reduction,
            )
        )
        or type(result.f4_original_complete_profiles) is not int
        or type(result.f5_original_complete_profiles) is not int
        or result.f4_original_complete_profiles < 0
        or result.f5_original_complete_profiles < 0
        or result.status not in {"approved", "failed", "inconclusive"}
        or not _is_sha256(result.result_sha256)
        or result.result_sha256
        != safe_result_sha256(
            {
                key: item
                for key, item in result.as_safe_dict().items()
                if key != "result_sha256"
            },
            b"semantic-substitution-defense-gate/v1",
        )
    ):
        raise SemanticPilotError("gate persistido diverge do contrato")
    return result


def semantic_pilot_result_from_payload(value: object) -> SemanticPilotResult:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "run_id", "experiment_seed", "baseline_model_sha256",
        "trajectories", "gate", "total_federated_rounds",
        "total_conversation_presentations", "total_optimizer_steps",
        "total_audit_generations", "total_utility_conversations", "result_sha256",
    }:
        raise SemanticPilotError("resultado persistido do piloto é inválido")
    trajectories = value.get("trajectories")
    if not isinstance(trajectories, list):
        raise SemanticPilotError("trajetórias persistidas são inválidas")
    try:
        result = SemanticPilotResult(
            **{
                **dict(value),
                "trajectories": tuple(
                    semantic_trajectory_from_payload(item) for item in trajectories
                ),
                "gate": semantic_gate_from_payload(value["gate"]),
            }
        )
    except SemanticPilotError:
        raise
    except Exception as error:
        raise SemanticPilotError("resultado persistido possui tipos inválidos") from error
    if (
        result.schema_version != SEMANTIC_PILOT_SCHEMA_VERSION
        or result.run_id != default_run_id(result.experiment_seed)
        or result.baseline_model_sha256 is None
        or not _is_sha256(result.baseline_model_sha256)
        or tuple(value.scenario for value in result.trajectories) != SCENARIO_ORDER
        or (
            result.total_federated_rounds,
            result.total_conversation_presentations,
            result.total_optimizer_steps,
            result.total_audit_generations,
            result.total_utility_conversations,
        )
        != (80, 328_000, 82_000, 40_083, 2_500)
        or not _is_sha256(result.result_sha256)
        or result.result_sha256
        != safe_result_sha256(
            {
                key: item
                for key, item in result.as_safe_dict().items()
                if key != "result_sha256"
            },
            b"semantic-substitution-pilot-result/v1",
        )
    ):
        raise SemanticPilotError("resultado persistido diverge do protocolo")
    return result


def semantic_combined_from_payload(value: object) -> SemanticCombinedResult:
    expected = {
        "schema_version", "source_result_sha256_by_seed", "status_by_seed",
        "combined_status", "require_both_seeds", "total_trajectories",
        "total_federated_rounds", "total_conversation_presentations",
        "total_optimizer_steps", "total_audit_generations",
        "total_utility_conversations", "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SemanticPilotError("resumo combinado é inválido")
    sources = value.get("source_result_sha256_by_seed")
    statuses = value.get("status_by_seed")
    if not isinstance(sources, Mapping) or not isinstance(statuses, Mapping):
        raise SemanticPilotError("fontes do resumo combinado são inválidas")
    try:
        result = SemanticCombinedResult(
            **{
                **dict(value),
                "source_result_sha256_by_seed": tuple(
                    (seed, sources[str(seed)]) for seed in EXPERIMENT_SEEDS
                ),
                "status_by_seed": tuple(
                    (seed, statuses[str(seed)]) for seed in EXPERIMENT_SEEDS
                ),
            }
        )
    except Exception as error:
        raise SemanticPilotError("resumo combinado possui tipos inválidos") from error
    unsigned = result.as_safe_dict()
    unsigned.pop("result_sha256")
    if (
        result.schema_version != SEMANTIC_COMBINED_SCHEMA_VERSION
        or result.combined_status not in {"approved", "failed", "inconclusive"}
        or any(status not in {"approved", "failed", "inconclusive"} for _, status in result.status_by_seed)
        or result.require_both_seeds is not True
        or tuple(seed for seed, _ in result.source_result_sha256_by_seed)
        != EXPERIMENT_SEEDS
        or tuple(seed for seed, _ in result.status_by_seed) != EXPERIMENT_SEEDS
        or (
            result.total_trajectories,
            result.total_federated_rounds,
            result.total_conversation_presentations,
            result.total_optimizer_steps,
            result.total_audit_generations,
            result.total_utility_conversations,
        )
        != (8, 160, 656_000, 164_000, 80_166, 5_000)
        or any(not _is_sha256(item) for _, item in result.source_result_sha256_by_seed)
        or result.result_sha256
        != safe_result_sha256(
            unsigned, b"semantic-substitution-combined-result/v1"
        )
    ):
        raise SemanticPilotError("resumo combinado diverge do protocolo")
    return result


__all__ = [
    "EXPERIMENT_SEEDS", "SCENARIO_ORDER", "SELECTED_GRID_ARM_ID",
    "SEMANTIC_CHECKPOINT_SCHEMA_VERSION", "SEMANTIC_COMBINED_SCHEMA_VERSION",
    "SEMANTIC_PILOT_SCHEMA_VERSION", "SEMANTIC_RESULT_SCHEMA_VERSION",
    "SEMANTIC_ROUND_SCHEMA_VERSION", "SEMANTIC_TRAJECTORY_SCHEMA_VERSION",
    "SemanticCombinedResult", "SemanticDefenseGateResult", "SemanticFederatedRoundResult",
    "SemanticPilotError", "SemanticPilotPreflightResult", "SemanticPilotResult",
    "SemanticPilotSpec", "SemanticTrajectoryResult", "default_run_id",
    "load_semantic_pilot_spec_from_config", "safe_result_sha256",
    "semantic_combined_from_payload", "semantic_gate_from_payload", "semantic_pilot_result_from_payload",
    "semantic_trajectory_from_payload", "validate_semantic_pilot_spec",
    "validate_semantic_round_result",
]
