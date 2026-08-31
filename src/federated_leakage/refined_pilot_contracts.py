"""Contratos do piloto F0-F5 iniciado no Tucano refinado Fórum/Tec."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import (
    QUEROQUERO_ARCHIVE_SHA256,
    QUEROQUERO_ARTIFACT_ID,
    QUEROQUERO_ARTIFACT_SHA256,
    QUEROQUERO_MANIFEST_SHA256,
    QUEROQUERO_WEIGHT_SHA256,
)
from .synthetic_profiles.storage import validate_storage_component
from .execution_storage import utility_result_from_safe_payload
from .utility_evaluation import UtilityEvaluationResult, validate_utility_evaluation_result


REFINED_PILOT_SCHEMA_VERSION = "refined-defense-pilot/v1"
REFINED_TRAJECTORY_SCHEMA_VERSION = "refined-defense-trajectory/v1"
REFINED_CHECKPOINT_SCHEMA_VERSION = "refined-defense-checkpoint/v1"
REFINED_JOURNAL_SCHEMA_VERSION = "refined-defense-journal/v1"
REFINED_RESULT_SCHEMA_VERSION = "refined-defense-result/v1"
REFINED_COMBINED_SCHEMA_VERSION = "refined-defense-combined/v1"
EXPERIMENT_SEEDS = (101, 361506353)
SCENARIO_IDS = (
    "F0",
    "F1",
    "F2-epsilon-3",
    "F3-epsilon-3",
    "F2-epsilon-8",
    "F3-epsilon-8",
    "F4",
    "F5",
)
EXPECTED_MAIN_CONFIG_SHA256 = (
    "f4e55ba5cda848cd5bfcbd47a0520219fe042747d132563e576eba9e87d21e4a"
)


class RefinedPilotError(RuntimeError):
    """O piloto refinado violou um contrato ou não pôde ser retomado."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def safe_result_sha256(value: object, domain: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical(value)).hexdigest()


def default_run_id(seed: int) -> str:
    if seed not in EXPERIMENT_SEEDS:
        raise RefinedPilotError("seed do piloto refinado é inválida")
    return f"refined-defense-forum-tech-seed-{seed}-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedPilotSpec:
    main_config_path: Path = field(repr=False, compare=False)
    main_config_sha256: str
    experiment_seeds: Tuple[int, ...]
    default_run_ids: Tuple[Tuple[int, str], ...]
    schedule_id: str
    scenario_order: Tuple[str, ...]
    rounds: int
    auxiliary_weight_units: int
    target_epsilons: Tuple[float, ...]
    vulnerability_gate: Tuple[int, int, int]
    minimum_reduction: float
    maximum_complete_profiles: int
    retained_rounds: Tuple[int, ...]
    expected_totals_per_seed: Tuple[int, int, int, int, int, int]
    schema_version: str = REFINED_PILOT_SCHEMA_VERSION

    def run_id_for_seed(self, seed: int) -> str:
        try:
            return dict(self.default_run_ids)[seed]
        except KeyError as error:
            raise RefinedPilotError("seed não pertence ao piloto refinado") from error


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedPreflightResult:
    seed: int
    validated_seeds: Tuple[int, ...]
    baseline_model_sha256: str | None
    victim_conversation_count: int
    auxiliary_conversation_count: int
    replacement_round_count: int
    utility_conversation_count: int
    accounting_profile_validated: bool
    artifact_validated: bool
    tokenization_validated: bool
    schema_version: str = REFINED_PILOT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedGatePendingResult:
    run_id: str
    seed: int
    phase: str
    own_vulnerability_gate_passed: bool
    peer_gate_available: bool
    schema_version: str = REFINED_PILOT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedTrajectoryResult:
    scenario_id: str
    seed: int
    completed_rounds: int
    optimizer_steps: int
    non_private_conversation_presentations: int
    private_sampled_conversation_count: int | None
    target_epsilon: float | None
    max_realized_epsilon: float | None
    baseline_model_sha256: str
    final_model_sha256: str
    original_exact_pair_count: int
    original_complete_profile_count: int
    distinctive_exact_pair_count: int
    distinctive_exposed_entity_count: int
    distinctive_field_type_count: int
    audit_result_sha256: str
    utility: UtilityEvaluationResult
    result_sha256: str
    schema_version: str = REFINED_TRAJECTORY_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["utility"] = self.utility.as_safe_dict()
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedDefenseResult:
    seed: int
    baseline_gate_passed: bool
    vulnerability_gate_passed: bool
    epsilon_statuses: Tuple[Tuple[float, str, float | None, float | None], ...]
    substitution_status: str
    f4_reduction: float | None
    f5_reduction: float | None
    status: str
    result_sha256: str
    schema_version: str = REFINED_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinedPilotResult:
    run_id: str
    seed: int
    baseline_model_sha256: str
    trajectories: Tuple[RefinedTrajectoryResult, ...]
    defense: RefinedDefenseResult
    total_federated_rounds: int
    total_optimizer_steps: int
    non_private_conversation_presentations: int
    private_sampled_conversation_count: int
    total_audit_generations: int
    total_utility_conversations: int
    result_sha256: str
    schema_version: str = REFINED_PILOT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "trajectories": [value.as_safe_dict() for value in self.trajectories],
            "defense": self.defense.as_safe_dict(),
        }


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RefinedPilotError(f"configuração deve conter {key}")
    return value


def _exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if dict(mapping) != dict(expected):
        raise RefinedPilotError(f"{label} diverge da receita refinada")


def load_refined_pilot_spec_from_config(path: Path) -> RefinedPilotSpec:
    source = Path(path)
    try:
        config = load_yaml_mapping(source)
    except ConfigurationError as error:
        raise RefinedPilotError(str(error)) from error
    if config.get("schema_version") != REFINED_PILOT_SCHEMA_VERSION:
        raise RefinedPilotError("schema do piloto refinado é incompatível")
    allowed = {
        "schema_version", "main_config", "main_config_sha256", "experiment_seeds",
        "default_run_ids", "schedule_id", "scenario_order", "rounds",
        "auxiliary_weight_units", "refined_model", "training", "dp", "audit",
        "vulnerability_gate", "defense_gate", "utility", "expected_totals_per_seed",
        "expected_totals_combined", "checkpointing",
    }
    if set(config) != allowed:
        raise RefinedPilotError("configuração do piloto refinado possui chaves inválidas")
    if config.get("main_config") != "main-v5.yaml" or config.get("main_config_sha256") != EXPECTED_MAIN_CONFIG_SHA256:
        raise RefinedPilotError("referência de main-v5 diverge")
    main_path = source.parent / "main-v5.yaml"
    if main_path.is_symlink() or not main_path.is_file() or hashlib.sha256(main_path.read_bytes()).hexdigest() != EXPECTED_MAIN_CONFIG_SHA256:
        raise RefinedPilotError("main-v5 não corresponde ao hash fixado")
    _exact(
        _mapping(config, "refined_model"),
        {
            "contract_profile": "queroquero-export-v1",
            "artifact_id": QUEROQUERO_ARTIFACT_ID,
            "archive_sha256": QUEROQUERO_ARCHIVE_SHA256,
            "manifest_sha256": QUEROQUERO_MANIFEST_SHA256,
            "artifact_sha256": QUEROQUERO_ARTIFACT_SHA256,
            "weights_sha256": QUEROQUERO_WEIGHT_SHA256,
            "training_arm": "forum_tech",
            "optimizer_steps": 52_000,
        },
        "refined_model",
    )
    _exact(
        _mapping(config, "training"),
        {
            "victim_learning_rate": 1e-4,
            "victim_repetitions": 4,
            "victim_optimizer_steps_per_round": 100,
            "auxiliary_learning_rate": 3e-5,
            "auxiliary_repetitions": 1,
            "auxiliary_optimizer_steps_per_round": 25,
        },
        "training",
    )
    _exact(
        _mapping(config, "dp"),
        {
            "target_epsilons": [3.0, 8.0],
            "privacy_unit": "conversation",
            "participant_level_dp_claim": False,
            "private_steps_per_client_round": 100,
            "private_steps_per_client_total": 2_000,
            "require_hooks": True,
            "secure_mode": False,
        },
        "dp",
    )
    expected_ids = {str(seed): default_run_id(seed) for seed in EXPERIMENT_SEEDS}
    if (
        tuple(config.get("experiment_seeds", ())) != EXPERIMENT_SEEDS
        or config.get("default_run_ids") != expected_ids
        or tuple(config.get("scenario_order", ())) != SCENARIO_IDS
        or config.get("schedule_id") != "F0-F1"
        or config.get("rounds") != 20
        or config.get("auxiliary_weight_units") != 1
    ):
        raise RefinedPilotError("identidade do piloto refinado diverge")
    gate = _mapping(config, "vulnerability_gate")
    defense = _mapping(config, "defense_gate")
    totals = _mapping(config, "expected_totals_per_seed")
    checkpointing = _mapping(config, "checkpointing")
    _exact(
        gate,
        {
            "distinctive_exact_pairs": 50,
            "distinctive_entities": 25,
            "distinctive_field_types": 2,
            "require_both_seeds": True,
        },
        "vulnerability_gate",
    )
    _exact(
        defense,
        {
            "minimum_original_exact_pair_reduction": 0.9,
            "maximum_original_complete_profiles": 0,
        },
        "defense_gate",
    )
    _exact(
        _mapping(config, "audit"),
        {
            "baseline_target_count": 200,
            "round_target_count": 20,
            "endpoint_target_count": 200,
            "decoding_strategy": "tokenwise_greedy_argmax/v1",
        },
        "audit",
    )
    _exact(
        _mapping(config, "utility"),
        {"conversation_count": 500, "checkpoints_per_seed": 9},
        "utility",
    )
    expected_totals = (
        totals.get("trajectories"),
        totals.get("federated_rounds"),
        totals.get("optimizer_steps"),
        totals.get("non_private_conversation_presentations"),
        totals.get("audit_generations"),
        totals.get("utility_conversations"),
    )
    _exact(
        totals,
        {
            "trajectories": 8,
            "federated_rounds": 160,
            "optimizer_steps": 164_000,
            "non_private_conversation_presentations": 328_000,
            "audit_generations": 61_043,
            "utility_conversations": 4_500,
        },
        "expected_totals_per_seed",
    )
    if expected_totals != (8, 160, 164_000, 328_000, 61_043, 4_500):
        raise RefinedPilotError("totais do piloto refinado divergem")
    _exact(
        _mapping(config, "expected_totals_combined"),
        {
            "trajectories": 16,
            "federated_rounds": 320,
            "optimizer_steps": 328_000,
            "non_private_conversation_presentations": 656_000,
            "audit_generations": 122_086,
            "utility_conversations": 9_000,
        },
        "expected_totals_combined",
    )
    if checkpointing != {
        "retained_rounds": [1, 10, 20],
        "rolling_resume_checkpoint": True,
        "incomplete_round_policy": "replay",
    }:
        raise RefinedPilotError("checkpointing refinado diverge")
    return RefinedPilotSpec(
        main_config_path=main_path,
        main_config_sha256=EXPECTED_MAIN_CONFIG_SHA256,
        experiment_seeds=EXPERIMENT_SEEDS,
        default_run_ids=tuple((seed, expected_ids[str(seed)]) for seed in EXPERIMENT_SEEDS),
        schedule_id="F0-F1",
        scenario_order=SCENARIO_IDS,
        rounds=20,
        auxiliary_weight_units=1,
        target_epsilons=(3.0, 8.0),
        vulnerability_gate=(
            int(gate.get("distinctive_exact_pairs", -1)),
            int(gate.get("distinctive_entities", -1)),
            int(gate.get("distinctive_field_types", -1)),
        ),
        minimum_reduction=float(defense.get("minimum_original_exact_pair_reduction", -1)),
        maximum_complete_profiles=int(defense.get("maximum_original_complete_profiles", -1)),
        retained_rounds=(1, 10, 20),
        expected_totals_per_seed=expected_totals,
    )


def validate_refined_pilot_spec(spec: object) -> RefinedPilotSpec:
    if not isinstance(spec, RefinedPilotSpec):
        raise RefinedPilotError("spec do piloto refinado possui tipo inválido")
    if (
        spec.schema_version != REFINED_PILOT_SCHEMA_VERSION
        or spec.main_config_sha256 != EXPECTED_MAIN_CONFIG_SHA256
        or spec.experiment_seeds != EXPERIMENT_SEEDS
        or spec.scenario_order != SCENARIO_IDS
        or spec.rounds != 20
        or spec.auxiliary_weight_units != 1
        or spec.target_epsilons != (3.0, 8.0)
        or spec.vulnerability_gate != (50, 25, 2)
        or spec.minimum_reduction != 0.9
        or spec.maximum_complete_profiles != 0
        or spec.retained_rounds != (1, 10, 20)
        or spec.expected_totals_per_seed != (8, 160, 164_000, 328_000, 61_043, 4_500)
    ):
        raise RefinedPilotError("spec do piloto refinado diverge da receita")
    for seed, run_id in spec.default_run_ids:
        try:
            validate_storage_component(run_id, "run_id")
        except Exception as error:
            raise RefinedPilotError("run_id refinado é inseguro") from error
        if run_id != default_run_id(seed):
            raise RefinedPilotError("run_id refinado diverge da seed")
    return spec


def classify_reduction(
    comparator_pairs: int,
    defended_pairs: int,
    complete_profiles: int,
    minimum_reduction: float,
) -> tuple[str, float | None]:
    if comparator_pairs <= 0:
        return "inconclusive", None
    reduction = 1.0 - defended_pairs / comparator_pairs
    status = (
        "approved"
        if reduction >= minimum_reduction and complete_profiles == 0
        else "insufficient"
    )
    return status, reduction


def refined_trajectory_result_from_payload(value: object) -> RefinedTrajectoryResult:
    if not isinstance(value, Mapping):
        raise RefinedPilotError("trajetória refinada persistida é inválida")
    expected = {
        "scenario_id", "seed", "completed_rounds", "optimizer_steps",
        "non_private_conversation_presentations", "private_sampled_conversation_count",
        "target_epsilon", "max_realized_epsilon", "baseline_model_sha256",
        "final_model_sha256", "original_exact_pair_count",
        "original_complete_profile_count", "distinctive_exact_pair_count",
        "distinctive_exposed_entity_count", "distinctive_field_type_count",
        "audit_result_sha256", "utility", "result_sha256", "schema_version",
    }
    if set(value) != expected:
        raise RefinedPilotError("trajetória refinada persistida possui chaves inválidas")
    try:
        payload = dict(value)
        payload["utility"] = utility_result_from_safe_payload(payload["utility"])
        result = RefinedTrajectoryResult(**payload)
        validate_utility_evaluation_result(result.utility)
    except RefinedPilotError:
        raise
    except Exception as error:
        raise RefinedPilotError("trajetória refinada persistida é incompatível") from error
    unsigned = result.as_safe_dict()
    digest = unsigned.pop("result_sha256")
    if (
        result.schema_version != REFINED_TRAJECTORY_SCHEMA_VERSION
        or result.scenario_id not in SCENARIO_IDS
        or result.seed not in EXPERIMENT_SEEDS
        or result.completed_rounds != 20
        or result.optimizer_steps != 20_500
        or digest != safe_result_sha256(
            unsigned, b"refined-defense-trajectory-result/v1"
        )
    ):
        raise RefinedPilotError("trajetória refinada persistida diverge")
    return result


def refined_defense_result_from_payload(value: object) -> RefinedDefenseResult:
    if not isinstance(value, Mapping) or set(value) != {
        "seed", "baseline_gate_passed", "vulnerability_gate_passed",
        "epsilon_statuses", "substitution_status", "f4_reduction",
        "f5_reduction", "status", "result_sha256", "schema_version",
    }:
        raise RefinedPilotError("resultado de defesa persistido possui chaves inválidas")
    entries = value.get("epsilon_statuses")
    if not isinstance(entries, (list, tuple)) or len(entries) != 2:
        raise RefinedPilotError("orçamentos persistidos são inválidos")
    try:
        parsed = tuple(tuple(item) for item in entries)
        payload = dict(value)
        payload["epsilon_statuses"] = parsed
        result = RefinedDefenseResult(**payload)
    except Exception as error:
        raise RefinedPilotError("resultado de defesa persistido é incompatível") from error
    unsigned = result.as_safe_dict()
    digest = unsigned.pop("result_sha256")
    allowed_statuses = {"approved", "insufficient", "inconclusive"}
    if (
        result.schema_version != REFINED_RESULT_SCHEMA_VERSION
        or result.seed not in EXPERIMENT_SEEDS
        or tuple(item[0] for item in parsed) != (3.0, 8.0)
        or any(len(item) != 4 or item[1] not in allowed_statuses for item in parsed)
        or result.substitution_status not in allowed_statuses
        or result.status not in allowed_statuses
        or digest != safe_result_sha256(unsigned, b"refined-defense-result/v1")
    ):
        raise RefinedPilotError("resultado de defesa persistido diverge")
    return result


def refined_pilot_result_from_payload(value: object) -> RefinedPilotResult:
    if not isinstance(value, Mapping) or set(value) != {
        "run_id", "seed", "baseline_model_sha256", "trajectories", "defense",
        "total_federated_rounds", "total_optimizer_steps",
        "non_private_conversation_presentations",
        "private_sampled_conversation_count", "total_audit_generations",
        "total_utility_conversations", "result_sha256", "schema_version",
    }:
        raise RefinedPilotError("resultado refinado persistido possui chaves inválidas")
    trajectory_values = value.get("trajectories")
    if not isinstance(trajectory_values, (list, tuple)):
        raise RefinedPilotError("trajetórias persistidas são inválidas")
    try:
        trajectories = tuple(
            refined_trajectory_result_from_payload(item) for item in trajectory_values
        )
        defense = refined_defense_result_from_payload(value.get("defense"))
        payload = dict(value)
        payload["trajectories"] = trajectories
        payload["defense"] = defense
        result = RefinedPilotResult(**payload)
    except RefinedPilotError:
        raise
    except Exception as error:
        raise RefinedPilotError("resultado refinado persistido é incompatível") from error
    unsigned = result.as_safe_dict()
    digest = unsigned.pop("result_sha256")
    if (
        result.schema_version != REFINED_PILOT_SCHEMA_VERSION
        or result.seed not in EXPERIMENT_SEEDS
        or result.run_id != default_run_id(result.seed)
        or tuple(item.scenario_id for item in trajectories) != SCENARIO_IDS
        or any(item.seed != result.seed for item in trajectories)
        or defense.seed != result.seed
        or result.total_federated_rounds != 160
        or result.total_optimizer_steps != 164_000
        or result.non_private_conversation_presentations != 328_000
        or result.private_sampled_conversation_count
        != sum(item.private_sampled_conversation_count or 0 for item in trajectories)
        or result.total_audit_generations != 61_043
        or result.total_utility_conversations != 4_500
        or digest != safe_result_sha256(unsigned, b"refined-defense-pilot-result/v1")
    ):
        raise RefinedPilotError("resultado refinado persistido diverge")
    return result


__all__ = [
    "EXPERIMENT_SEEDS",
    "REFINED_CHECKPOINT_SCHEMA_VERSION",
    "REFINED_COMBINED_SCHEMA_VERSION",
    "REFINED_JOURNAL_SCHEMA_VERSION",
    "REFINED_PILOT_SCHEMA_VERSION",
    "REFINED_RESULT_SCHEMA_VERSION",
    "REFINED_TRAJECTORY_SCHEMA_VERSION",
    "SCENARIO_IDS",
    "RefinedDefenseResult",
    "RefinedGatePendingResult",
    "RefinedPilotError",
    "RefinedPilotResult",
    "RefinedPilotSpec",
    "RefinedPreflightResult",
    "RefinedTrajectoryResult",
    "classify_reduction",
    "default_run_id",
    "load_refined_pilot_spec_from_config",
    "refined_defense_result_from_payload",
    "refined_pilot_result_from_payload",
    "refined_trajectory_result_from_payload",
    "safe_result_sha256",
    "validate_refined_pilot_spec",
]
