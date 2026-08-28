"""Contratos estritos da grade federada de intensidade com duas seeds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .synthetic_profiles.storage import validate_storage_component
from .utility_evaluation import UtilityEvaluationComparison, UtilityEvaluationResult


GRID_SCHEMA_VERSION = "federated-memorization-grid/v2"
GRID_ARM_SCHEMA_VERSION = "federated-exposure-grid-arm/v2"
GRID_ROUND_SCHEMA_VERSION = "federated-exposure-grid-round/v2"
GRID_AUDIT_SCHEMA_VERSION = "federated-exposure-grid-audit-result/v2"
GRID_CHECKPOINT_SCHEMA_VERSION = "federated-exposure-grid-checkpoint/v2"
GRID_COMBINED_SCHEMA_VERSION = "federated-memorization-grid-combined/v2"

EXPERIMENT_SEEDS = (101, 361506353)
VICTIM_LEARNING_RATE_MILLIONTHS = (30, 100)
VICTIM_REPETITION_MULTIPLIERS = (4, 8, 16)
DISTINCTIVE_FIELD_TYPES = ("CPF", "RG", "PHONE", "EMAIL", "ADDRESS")
EXPECTED_MAIN_CONFIG_SHA256 = (
    "b5bde98b847e18927121c7f57d049d704f339d251b6f75c30b98fc692569fc2e"
)
EXPECTED_HASHES = {
    101: (
        "7d08d2dfc889162227d4a87dfbec60766ae2fe6b0497b27733ce512ae861f3bb",
        "1be2d55566a92bd613e31d20243bbb3555a859cf3427c1a9db44af096be78050",
        "a06fd9b76a1dad40192f2c167ccfff81c1a55ab3ced93cf18daca270933e1f1d",
    ),
    361506353: (
        "7ca28596cabcbce705d48d3c841a1cf8148c25ec7cbbf4b45a3c6e2df3e97dc0",
        "97e9a0d6cf636477dae4577c5b39fd5291a7b097826f1c9ff061dab1ca3dcfb8",
        "f6332605986244161cb25c87baf6a374f1711408cfb99c52f4b317021c9fab3c",
    ),
}
REFERENCE_V1_RESULT_SHA256 = (
    "6ecb06f1fa5c5090015e9e6a45f680c6d3f428d1d23b12b7e8211e1d80a6c5c3"
)
REFERENCE_V1_CONFIG_SHA256 = (
    "89237c8358e574c0185ec7eeb537676f2773a599390f760f8d823e493805950c"
)
REFERENCE_V1_FINAL_MODEL_SHA256 = (
    "4ed1d34bb3e0bf15e07e5a8f6bcfd9dcc52fb5ff5939b25d76d0183fbed90486"
)


class FederatedGridError(RuntimeError):
    """A grade violou o protocolo ou falhou sem publicar resultado parcial."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def grid_arm_id(learning_rate_millionths: int, multiplier: int) -> str:
    if (
        type(learning_rate_millionths) is not int
        or learning_rate_millionths not in VICTIM_LEARNING_RATE_MILLIONTHS
        or type(multiplier) is not int
        or multiplier not in VICTIM_REPETITION_MULTIPLIERS
    ):
        raise FederatedGridError("identidade do braço da grade é inválida")
    return (
        f"victim-lr-{learning_rate_millionths:06d}"
        f"-repetitions-{multiplier:03d}"
    )


def default_run_id(seed: int) -> str:
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise FederatedGridError("seed da grade é inválida")
    return f"federated-memorization-grid-seed-{seed}-v2"


def default_dataset_id(seed: int) -> str:
    return f"{default_run_id(seed)}-dataset-v4"


@dataclass(frozen=True, slots=True, kw_only=True)
class GridArmSpec:
    arm_id: str
    victim_learning_rate_millionths: int
    victim_repetition_multiplier: int

    @property
    def victim_learning_rate(self) -> float:
        return self.victim_learning_rate_millionths / 1_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class GridSeedHashes:
    seed: int
    victim_dataset_sha256: str
    benign_schedule_sha256: str
    utility_dataset_sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridSpec:
    experiment_seeds: Tuple[int, ...]
    default_run_ids: Tuple[Tuple[int, str], ...]
    dataset_ids: Tuple[Tuple[int, str], ...]
    schedule_id: str
    scenario: str
    auxiliary_weight_units: int
    rounds: int
    auxiliary_learning_rate_millionths: int
    auxiliary_repetition_multiplier: int
    arms: Tuple[GridArmSpec, ...]
    audit_target_count: int
    distinctive_exact_pair_threshold: int
    distinctive_entity_threshold: int
    distinctive_field_type_threshold: int
    expected_per_seed: Tuple[int, int, int, int, int, int]
    expected_combined: Tuple[int, int, int, int, int, int]
    seed_hashes: Tuple[GridSeedHashes, ...]
    main_config_sha256: str
    reference_v1_run_id: str
    reference_v1_config_sha256: str
    reference_v1_result_sha256: str
    reference_v1_final_model_sha256: str
    reference_v1_distinctive_exact_pairs: int
    reference_v1_distinctive_entities: int
    main_config_path: Path = field(repr=False, compare=False)
    schema_version: str = GRID_SCHEMA_VERSION

    def run_id_for_seed(self, seed: int) -> str:
        try:
            return dict(self.default_run_ids)[seed]
        except KeyError as error:
            raise FederatedGridError("seed da execução não pertence à grade") from error

    def dataset_id_for_seed(self, seed: int) -> str:
        try:
            return dict(self.dataset_ids)[seed]
        except KeyError as error:
            raise FederatedGridError("seed da execução não pertence à grade") from error

    def hashes_for_seed(self, seed: int) -> GridSeedHashes:
        for value in self.seed_hashes:
            if value.seed == seed:
                return value
        raise FederatedGridError("hashes da seed não pertencem à grade")

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_seeds": list(self.experiment_seeds),
            "default_run_ids": {str(key): value for key, value in self.default_run_ids},
            "dataset_ids": {str(key): value for key, value in self.dataset_ids},
            "schedule_id": self.schedule_id,
            "scenario": self.scenario,
            "auxiliary_weight_units": self.auxiliary_weight_units,
            "rounds": self.rounds,
            "auxiliary_learning_rate_millionths": self.auxiliary_learning_rate_millionths,
            "auxiliary_repetition_multiplier": self.auxiliary_repetition_multiplier,
            "arms": [asdict(value) for value in self.arms],
            "audit_target_count": self.audit_target_count,
            "distinctive_exact_pair_threshold": self.distinctive_exact_pair_threshold,
            "distinctive_entity_threshold": self.distinctive_entity_threshold,
            "distinctive_field_type_threshold": self.distinctive_field_type_threshold,
            "expected_per_seed": list(self.expected_per_seed),
            "expected_combined": list(self.expected_combined),
            "seed_hashes": [asdict(value) for value in self.seed_hashes],
            "main_config_sha256": self.main_config_sha256,
            "reference_v1_run_id": self.reference_v1_run_id,
            "reference_v1_config_sha256": self.reference_v1_config_sha256,
            "reference_v1_result_sha256": self.reference_v1_result_sha256,
            "reference_v1_final_model_sha256": self.reference_v1_final_model_sha256,
            "reference_v1_distinctive_exact_pairs": self.reference_v1_distinctive_exact_pairs,
            "reference_v1_distinctive_entities": self.reference_v1_distinctive_entities,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridPreflightResult:
    selected_seed: int
    validated_seeds: Tuple[int, ...]
    victim_conversation_count: int
    auxiliary_conversation_count: int
    utility_conversation_count: int
    cross_seed_collision_preflight_sha256: str
    selected_victim_dataset_sha256: str
    selected_benign_schedule_sha256: str
    selected_utility_dataset_sha256: str
    model_state_sha256: str | None = None
    schema_version: str = GRID_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridRoundResult:
    experiment_seed: int
    arm_id: str
    victim_learning_rate_millionths: int
    auxiliary_learning_rate_millionths: int
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
    schema_version: str = GRID_ROUND_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_provenance"] = self.model_provenance.as_safe_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridCheckpoint:
    experiment_seed: int
    arm_id: str
    victim_learning_rate_millionths: int
    victim_repetition_multiplier: int
    round_id: int
    model_state_sha256: str
    grid_config_sha256: str
    artifact_sha256: str
    round_result: FederatedGridRoundResult
    schema_version: str = GRID_CHECKPOINT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_seed": self.experiment_seed,
            "arm_id": self.arm_id,
            "victim_learning_rate_millionths": self.victim_learning_rate_millionths,
            "victim_repetition_multiplier": self.victim_repetition_multiplier,
            "round_id": self.round_id,
            "model_state_sha256": self.model_state_sha256,
            "grid_config_sha256": self.grid_config_sha256,
            "artifact_sha256": self.artifact_sha256,
            "round_result": self.round_result.as_safe_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridAuditResult:
    experiment_seed: int
    arm_id: str | None
    victim_learning_rate_millionths: int
    victim_repetition_multiplier: int
    extraction_result_sha256: str
    target_count: int
    distinctive_exact_pair_count: int
    distinctive_exposed_entity_count: int
    distinctive_exact_pairs_by_type: Tuple[Tuple[str, int], ...]
    distinctive_field_type_count: int
    gate_passed: bool
    model_state_sha256: str
    schema_version: str = GRID_AUDIT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["distinctive_exact_pairs_by_type"] = dict(
            self.distinctive_exact_pairs_by_type
        )
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridArmResult:
    experiment_seed: int
    arm_id: str
    victim_learning_rate_millionths: int
    auxiliary_learning_rate_millionths: int
    victim_repetition_multiplier: int
    completed_rounds: int
    conversation_presentations: int
    optimizer_steps: int
    baseline_model_sha256: str
    final_model_sha256: str
    round_result_sha256: str
    audit: FederatedGridAuditResult
    utility: UtilityEvaluationResult
    checkpoint_artifact_sha256: str
    schema_version: str = GRID_ARM_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key not in {"audit", "utility"}},
            "audit": self.audit.as_safe_dict(),
            "utility": self.utility.as_safe_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridSeedResult:
    run_id: str
    experiment_seed: int
    baseline_model_sha256: str
    baseline_gate_passed: bool
    any_arm_passed: bool
    first_successful_arm: str | None
    baseline_audit: FederatedGridAuditResult
    baseline_utility: UtilityEvaluationResult
    arms: Tuple[FederatedGridArmResult, ...]
    utility_comparisons: Tuple[UtilityEvaluationComparison, ...]
    total_federated_rounds: int
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = GRID_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_seed": self.experiment_seed,
            "baseline_model_sha256": self.baseline_model_sha256,
            "baseline_gate_passed": self.baseline_gate_passed,
            "any_arm_passed": self.any_arm_passed,
            "first_successful_arm": self.first_successful_arm,
            "baseline_audit": self.baseline_audit.as_safe_dict(),
            "baseline_utility": self.baseline_utility.as_safe_dict(),
            "arms": [value.as_safe_dict() for value in self.arms],
            "utility_comparisons": [value.as_safe_dict() for value in self.utility_comparisons],
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "total_utility_conversations": self.total_utility_conversations,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GridArmClassification:
    arm_id: str
    classification: str
    passed_seeds: Tuple[int, ...]
    distinctive_exact_pair_min: int
    distinctive_exact_pair_max: int
    distinctive_exact_pair_difference: int
    distinctive_entity_min: int
    distinctive_entity_max: int
    utility_perplexity_min: float
    utility_perplexity_max: float
    utility_perplexity_difference: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedGridCombinedResult:
    source_result_sha256_by_seed: Tuple[Tuple[int, str], ...]
    classifications: Tuple[GridArmClassification, ...]
    first_robust_arm: str | None
    human_review_required: bool
    total_arms: int
    total_federated_rounds: int
    total_conversation_presentations: int
    total_optimizer_steps: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = GRID_COMBINED_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_result_sha256_by_seed": dict(self.source_result_sha256_by_seed),
            "classifications": [asdict(value) for value in self.classifications],
            "first_robust_arm": self.first_robust_arm,
            "human_review_required": self.human_review_required,
            "total_arms": self.total_arms,
            "total_federated_rounds": self.total_federated_rounds,
            "total_conversation_presentations": self.total_conversation_presentations,
            "total_optimizer_steps": self.total_optimizer_steps,
            "total_audit_generations": self.total_audit_generations,
            "total_utility_conversations": self.total_utility_conversations,
            "result_sha256": self.result_sha256,
        }


def _expected_arms() -> Tuple[GridArmSpec, ...]:
    return tuple(
        GridArmSpec(
            arm_id=grid_arm_id(lr, multiplier),
            victim_learning_rate_millionths=lr,
            victim_repetition_multiplier=multiplier,
        )
        for lr in VICTIM_LEARNING_RATE_MILLIONTHS
        for multiplier in VICTIM_REPETITION_MULTIPLIERS
    )


def validate_federated_grid_spec(spec: object) -> FederatedGridSpec:
    expected_run_ids = tuple((seed, default_run_id(seed)) for seed in EXPERIMENT_SEEDS)
    expected_dataset_ids = tuple((seed, default_dataset_id(seed)) for seed in EXPERIMENT_SEEDS)
    if (
        not isinstance(spec, FederatedGridSpec)
        or spec.schema_version != GRID_SCHEMA_VERSION
        or spec.experiment_seeds != EXPERIMENT_SEEDS
        or spec.default_run_ids != expected_run_ids
        or spec.dataset_ids != expected_dataset_ids
        or spec.schedule_id != "F0-F1"
        or spec.scenario != "F0"
        or spec.auxiliary_weight_units != 1
        or spec.rounds != 20
        or spec.auxiliary_learning_rate_millionths != 30
        or spec.auxiliary_repetition_multiplier != 1
        or spec.arms != _expected_arms()
        or spec.audit_target_count != 200
        or spec.distinctive_exact_pair_threshold != 50
        or spec.distinctive_entity_threshold != 25
        or spec.distinctive_field_type_threshold != 2
        or spec.expected_per_seed != (6, 120, 1_132_000, 283_000, 12_607, 3_500)
        or spec.expected_combined != (12, 240, 2_264_000, 566_000, 25_214, 7_000)
        or spec.main_config_sha256 != EXPECTED_MAIN_CONFIG_SHA256
        or tuple(
            (item.seed, item.victim_dataset_sha256, item.benign_schedule_sha256, item.utility_dataset_sha256)
            for item in spec.seed_hashes
        )
        != tuple((seed, *EXPECTED_HASHES[seed]) for seed in EXPERIMENT_SEEDS)
        or spec.reference_v1_run_id != "federated-memorization-calibration-seed-101-v1"
        or spec.reference_v1_config_sha256 != REFERENCE_V1_CONFIG_SHA256
        or spec.reference_v1_result_sha256 != REFERENCE_V1_RESULT_SHA256
        or spec.reference_v1_final_model_sha256 != REFERENCE_V1_FINAL_MODEL_SHA256
        or spec.reference_v1_distinctive_exact_pairs != 15
        or spec.reference_v1_distinctive_entities != 15
    ):
        raise FederatedGridError("especificação da grade diverge do protocolo")
    return spec


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise FederatedGridError(f"seção {label} da grade é inválida")
    return value


def load_federated_grid_spec_from_config(path: Path) -> FederatedGridSpec:
    config_path = Path(path)
    try:
        payload = load_yaml_mapping(config_path)
    except ConfigurationError as error:
        raise FederatedGridError(str(error)) from error
    expected = {
        "schema_version", "main_config", "main_config_sha256", "experiment_seeds",
        "default_run_ids", "dataset_ids", "schedule_id", "scenario",
        "auxiliary_weight_units", "rounds", "auxiliary_learning_rate_millionths",
        "auxiliary_repetition_multiplier", "victim_grid", "audit_target_count",
        "intense_gate", "expected_totals_per_seed", "expected_totals_combined",
        "expected_hashes", "reference_v1",
    }
    if set(payload) != expected or payload.get("main_config") != "main-v3.yaml":
        raise FederatedGridError("configuração da grade possui chaves inválidas")
    main_path = config_path.parent / "main-v3.yaml"
    if main_path.is_symlink() or not main_path.is_file():
        raise FederatedGridError("configuração principal da grade está ausente")
    main_path = main_path.resolve()
    if hashlib.sha256(main_path.read_bytes()).hexdigest() != EXPECTED_MAIN_CONFIG_SHA256:
        raise FederatedGridError("hash da configuração principal diverge")
    try:
        seeds = tuple(payload["experiment_seeds"])
        runs = _mapping(payload["default_run_ids"], {str(value) for value in EXPERIMENT_SEEDS}, "runs")
        datasets = _mapping(payload["dataset_ids"], {str(value) for value in EXPERIMENT_SEEDS}, "datasets")
        grid = _mapping(payload["victim_grid"], {"learning_rate_millionths", "repetition_multipliers"}, "victim_grid")
        gate = _mapping(payload["intense_gate"], {"distinctive_exact_pairs", "distinctive_entities", "distinctive_field_types"}, "gate")
        total_keys = {"arms", "federated_rounds", "conversation_presentations", "optimizer_steps", "audit_generations", "utility_conversations"}
        per_seed = _mapping(payload["expected_totals_per_seed"], total_keys, "totais por seed")
        combined = _mapping(payload["expected_totals_combined"], total_keys, "totais combinados")
        hashes = _mapping(payload["expected_hashes"], {str(value) for value in EXPERIMENT_SEEDS}, "hashes")
        reference = _mapping(payload["reference_v1"], {"run_id", "config_sha256", "result_sha256", "final_model_sha256", "distinctive_exact_pairs", "distinctive_entities"}, "referência v1")
        arms = tuple(
            GridArmSpec(
                arm_id=grid_arm_id(lr, multiplier),
                victim_learning_rate_millionths=lr,
                victim_repetition_multiplier=multiplier,
            )
            for lr in tuple(grid["learning_rate_millionths"])
            for multiplier in tuple(grid["repetition_multipliers"])
        )
        seed_hashes = tuple(
            GridSeedHashes(
                seed=seed,
                **_mapping(hashes[str(seed)], {"victim_dataset_sha256", "benign_schedule_sha256", "utility_dataset_sha256"}, "hash da seed"),
            )
            for seed in seeds
        )
        spec = FederatedGridSpec(
            experiment_seeds=seeds,
            default_run_ids=tuple((seed, validate_storage_component(runs[str(seed)], "run_id")) for seed in seeds),
            dataset_ids=tuple((seed, validate_storage_component(datasets[str(seed)], "dataset_id")) for seed in seeds),
            schedule_id=validate_storage_component(payload["schedule_id"], "schedule_id"),
            scenario=payload["scenario"],
            auxiliary_weight_units=payload["auxiliary_weight_units"],
            rounds=payload["rounds"],
            auxiliary_learning_rate_millionths=payload["auxiliary_learning_rate_millionths"],
            auxiliary_repetition_multiplier=payload["auxiliary_repetition_multiplier"],
            arms=arms,
            audit_target_count=payload["audit_target_count"],
            distinctive_exact_pair_threshold=gate["distinctive_exact_pairs"],
            distinctive_entity_threshold=gate["distinctive_entities"],
            distinctive_field_type_threshold=gate["distinctive_field_types"],
            expected_per_seed=tuple(per_seed[key] for key in ("arms", "federated_rounds", "conversation_presentations", "optimizer_steps", "audit_generations", "utility_conversations")),
            expected_combined=tuple(combined[key] for key in ("arms", "federated_rounds", "conversation_presentations", "optimizer_steps", "audit_generations", "utility_conversations")),
            seed_hashes=seed_hashes,
            main_config_sha256=payload["main_config_sha256"],
            reference_v1_run_id=reference["run_id"],
            reference_v1_config_sha256=reference["config_sha256"],
            reference_v1_result_sha256=reference["result_sha256"],
            reference_v1_final_model_sha256=reference["final_model_sha256"],
            reference_v1_distinctive_exact_pairs=reference["distinctive_exact_pairs"],
            reference_v1_distinctive_entities=reference["distinctive_entities"],
            main_config_path=main_path,
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FederatedGridError("configuração da grade possui tipos inválidos") from error
    return validate_federated_grid_spec(spec)


def safe_result_sha256(payload: Mapping[str, Any], domain: bytes) -> str:
    raw = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


def validate_grid_round_result(result: object) -> FederatedGridRoundResult:
    if not isinstance(result, FederatedGridRoundResult):
        raise FederatedGridError("resultado da rodada da grade é inválido")
    multiplier = result.victim_repetition_multiplier
    if (
        result.schema_version != GRID_ROUND_SCHEMA_VERSION
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or result.arm_id != grid_arm_id(result.victim_learning_rate_millionths, multiplier)
        or result.auxiliary_learning_rate_millionths != 30
        or not 1 <= result.round_id <= 20
        or result.conversation_presentations != 1_000 * multiplier + 100
        or result.optimizer_steps != 250 * multiplier + 25
        or result.victim_optimizer_steps != 250 * multiplier
        or result.auxiliary_optimizer_steps != 25
        or any(not math.isfinite(value) for value in (result.mean_client_loss, result.mean_victim_loss, result.auxiliary_loss, result.aggregate_delta_l2_norm, result.aggregate_delta_max_abs))
        or any(not _is_sha256(value) for value in (result.victim_dataset_sha256, result.auxiliary_schedule_sha256, result.auxiliary_values_sha256, result.initial_model_sha256, result.aggregate_update_sha256, result.final_model_sha256, result.client_order_sha256, result.weights_sha256, result.sample_order_schedule_sha256, result.training_seed_schedule_sha256))
    ):
        raise FederatedGridError("resultado da rodada diverge do contrato da grade")
    return result


def validate_grid_audit_result(result: object, spec: FederatedGridSpec) -> FederatedGridAuditResult:
    if not isinstance(result, FederatedGridAuditResult):
        raise FederatedGridError("resultado de auditoria da grade é inválido")
    by_type = dict(result.distinctive_exact_pairs_by_type)
    passed = (
        result.distinctive_exact_pair_count >= spec.distinctive_exact_pair_threshold
        and result.distinctive_exposed_entity_count >= spec.distinctive_entity_threshold
        and result.distinctive_field_type_count >= spec.distinctive_field_type_threshold
    )
    if (
        result.schema_version != GRID_AUDIT_SCHEMA_VERSION
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or result.target_count != 200
        or set(by_type) != set(DISTINCTIVE_FIELD_TYPES)
        or any(type(value) is not int or not 0 <= value <= 200 for value in by_type.values())
        or result.distinctive_exact_pair_count != sum(by_type.values())
        or result.distinctive_field_type_count != sum(value > 0 for value in by_type.values())
        or not 0 <= result.distinctive_exposed_entity_count <= 200
        or result.gate_passed != passed
        or not _is_sha256(result.extraction_result_sha256)
        or not _is_sha256(result.model_state_sha256)
    ):
        raise FederatedGridError("resultado de auditoria diverge do contrato da grade")
    if result.arm_id is None:
        if result.victim_learning_rate_millionths != 0 or result.victim_repetition_multiplier != 0:
            raise FederatedGridError("identidade da auditoria baseline é inválida")
    elif result.arm_id != grid_arm_id(result.victim_learning_rate_millionths, result.victim_repetition_multiplier):
        raise FederatedGridError("identidade da auditoria do braço é inválida")
    return result


def grid_audit_result_from_payload(value: object, spec: FederatedGridSpec) -> FederatedGridAuditResult:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(FederatedGridAuditResult)}:
        raise FederatedGridError("resultado persistido da auditoria da grade é inválido")
    try:
        payload = dict(value)
        breakdown = payload["distinctive_exact_pairs_by_type"]
        if not isinstance(breakdown, Mapping):
            raise TypeError
        payload["distinctive_exact_pairs_by_type"] = tuple(
            (field_type, breakdown[field_type]) for field_type in DISTINCTIVE_FIELD_TYPES
        )
        return validate_grid_audit_result(FederatedGridAuditResult(**payload), spec)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("resultado persistido da auditoria é incompatível") from error


def grid_arm_result_from_payload(value: object, spec: FederatedGridSpec) -> FederatedGridArmResult:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(FederatedGridArmResult)}:
        raise FederatedGridError("resultado persistido do braço da grade é inválido")
    try:
        from .execution_storage import utility_result_from_safe_payload
        payload = dict(value)
        payload["audit"] = grid_audit_result_from_payload(payload["audit"], spec)
        payload["utility"] = utility_result_from_safe_payload(payload["utility"])
        result = FederatedGridArmResult(**payload)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("resultado persistido do braço é incompatível") from error
    arm = next((item for item in spec.arms if item.arm_id == result.arm_id), None)
    if (
        arm is None
        or result.schema_version != GRID_ARM_SCHEMA_VERSION
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or result.victim_learning_rate_millionths != arm.victim_learning_rate_millionths
        or result.auxiliary_learning_rate_millionths != 30
        or result.victim_repetition_multiplier != arm.victim_repetition_multiplier
        or result.completed_rounds != 20
        or result.conversation_presentations != 20 * (1_000 * arm.victim_repetition_multiplier + 100)
        or result.optimizer_steps != 20 * (250 * arm.victim_repetition_multiplier + 25)
        or result.audit.arm_id != result.arm_id
        or result.audit.experiment_seed != result.experiment_seed
        or result.audit.model_state_sha256 != result.final_model_sha256
        or result.utility.scenario != "F0"
        or result.utility.round_id != 20
        or result.utility.experiment_seed != result.experiment_seed
        or result.utility.model_state_sha256 != result.final_model_sha256
        or any(not _is_sha256(item) for item in (result.baseline_model_sha256, result.final_model_sha256, result.round_result_sha256, result.checkpoint_artifact_sha256))
    ):
        raise FederatedGridError("resultado persistido do braço diverge")
    return result


def grid_seed_result_from_payload(value: object, spec: FederatedGridSpec) -> FederatedGridSeedResult:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(FederatedGridSeedResult)}:
        raise FederatedGridError("resultado final persistido da grade é inválido")
    try:
        from .execution_storage import utility_result_from_safe_payload
        from .federated_exposure_contracts import _utility_comparison_from_payload
        payload = dict(value)
        seed = payload["experiment_seed"]
        payload["baseline_audit"] = grid_audit_result_from_payload(payload["baseline_audit"], spec)
        payload["baseline_utility"] = utility_result_from_safe_payload(payload["baseline_utility"])
        payload["arms"] = tuple(grid_arm_result_from_payload(item, spec) for item in payload["arms"])
        payload["utility_comparisons"] = tuple(_utility_comparison_from_payload(item) for item in payload["utility_comparisons"])
        result = FederatedGridSeedResult(**payload)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("resultado final persistido da grade é incompatível") from error
    successful = tuple(item.arm_id for item in result.arms if item.audit.gate_passed)
    unsigned = result.as_safe_dict()
    unsigned.pop("result_sha256")
    if (
        result.schema_version != GRID_SCHEMA_VERSION
        or result.experiment_seed not in EXPERIMENT_SEEDS
        or result.run_id != spec.run_id_for_seed(result.experiment_seed)
        or result.baseline_audit.experiment_seed != result.experiment_seed
        or result.baseline_audit.arm_id is not None
        or result.baseline_audit.model_state_sha256 != result.baseline_model_sha256
        or result.baseline_utility.experiment_seed != result.experiment_seed
        or result.baseline_utility.scenario != "B0"
        or result.baseline_utility.round_id != 0
        or result.baseline_utility.model_state_sha256 != result.baseline_model_sha256
        or tuple(item.arm_id for item in result.arms) != tuple(item.arm_id for item in spec.arms)
        or any(
            item.experiment_seed != result.experiment_seed
            or item.baseline_model_sha256 != result.baseline_model_sha256
            for item in result.arms
        )
        or result.baseline_gate_passed != result.baseline_audit.gate_passed
        or result.any_arm_passed != (not result.baseline_gate_passed and bool(successful))
        or result.first_successful_arm != (successful[0] if successful and not result.baseline_gate_passed else None)
        or (result.total_federated_rounds, result.total_conversation_presentations, result.total_optimizer_steps, result.total_audit_generations, result.total_utility_conversations) != spec.expected_per_seed[1:]
        or result.result_sha256 != safe_result_sha256(unsigned, b"federated-memorization-grid-result/v2")
    ):
        raise FederatedGridError("resultado final persistido diverge da grade")
    return result


__all__ = [
    "DISTINCTIVE_FIELD_TYPES", "EXPERIMENT_SEEDS", "GRID_ARM_SCHEMA_VERSION",
    "GRID_AUDIT_SCHEMA_VERSION", "GRID_CHECKPOINT_SCHEMA_VERSION", "GRID_COMBINED_SCHEMA_VERSION",
    "GRID_ROUND_SCHEMA_VERSION", "GRID_SCHEMA_VERSION", "FederatedGridArmResult",
    "FederatedGridAuditResult", "FederatedGridCombinedResult", "FederatedGridError",
    "FederatedGridCheckpoint", "FederatedGridPreflightResult", "FederatedGridRoundResult", "FederatedGridSeedResult",
    "FederatedGridSpec", "GridArmClassification", "GridArmSpec", "GridSeedHashes",
    "default_dataset_id", "default_run_id", "grid_arm_id", "load_federated_grid_spec_from_config",
    "grid_arm_result_from_payload", "grid_audit_result_from_payload", "grid_seed_result_from_payload",
    "safe_result_sha256", "validate_federated_grid_spec", "validate_grid_audit_result",
    "validate_grid_round_result",
]
