"""Contratos fail-closed do DP-AdamW por conversa."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance


DP_ACCOUNTING_SCHEMA_VERSION = "dp-accounting-profile/v1"
PRIVATE_LOCAL_TRAINING_SCHEMA_VERSION = "private-local-training/v1"
PRIVATE_MODEL_UPDATE_SCHEMA_VERSION = "private-model-update/v1"
PRIVATE_FEDERATED_ROUND_SCHEMA_VERSION = "private-federated-round/v1"
EXPECTED_MAIN_SCHEMA_VERSION = "federated-leakage/main-config/v5"
EXPECTED_TARGET_EPSILONS = (3.0, 8.0)
EXPECTED_SIGMA_BY_EPSILON = ((3.0, 2.81), (8.0, 1.36))
EXPECTED_REALIZED_BY_EPSILON = (
    (3.0, 2.98777705562, 7.4),
    (8.0, 7.96431428079, 3.7),
)


class PrivateTrainingError(RuntimeError):
    """A receita, a execução ou a contabilização privada falhou fechada."""


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DPAccountingSpec:
    """Receita normativa de privacidade por conversa."""

    library: str
    library_version: str
    privacy_unit: str
    participant_level_dp_claim: bool
    records_per_client: int
    poisson_sampling: bool
    sample_rate: float
    private_steps_per_round: int
    total_private_steps: int
    max_physical_conversations: int
    expected_batch_size: int
    max_grad_norm: float
    clipping: str
    accountant: str
    delta: float
    target_epsilons: Tuple[float, ...]
    sigma_by_epsilon: Tuple[Tuple[float, float], ...]
    realized_by_epsilon: Tuple[Tuple[float, float, float], ...]
    optimizer: str
    victim_learning_rate: float
    betas: Tuple[float, float]
    optimizer_epsilon: float
    weight_decay: float
    optimizer_state: str
    grad_sample_mode: str
    secure_mode: bool
    empty_batch_policy: str
    schema_version: str = DP_ACCOUNTING_SCHEMA_VERSION

    def sigma_for(self, epsilon: float) -> float:
        try:
            return dict(self.sigma_by_epsilon)[float(epsilon)]
        except (KeyError, TypeError, ValueError) as error:
            raise PrivateTrainingError("epsilon privado não pertence à receita") from error

    def realized_for(self, epsilon: float) -> tuple[float, float]:
        for target, realized, order in self.realized_by_epsilon:
            if target == float(epsilon):
                return realized, order
        raise PrivateTrainingError("epsilon privado não pertence à receita")


@dataclass(frozen=True, slots=True, kw_only=True)
class DPAccountantState:
    """Estado RDP mínimo e seguro persistido entre rodadas de um cliente."""

    client_id: str
    target_epsilon: float
    history: Tuple[Tuple[float, float, int], ...]
    completed_steps: int
    realized_epsilon: float
    optimal_order: float
    state_sha256: str
    schema_version: str = DP_ACCOUNTING_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateLocalTrainingResult:
    """Recibo técnico sem perdas, normas ou métricas individuais da vítima."""

    client_id: str
    role: str
    round_id: int
    conversation_count: int
    optimizer_steps: int
    sampled_conversation_count: int
    target_epsilon: float
    noise_multiplier: float
    sample_rate: float
    max_grad_norm: float
    delta: float
    accountant_steps_total: int
    realized_epsilon: float
    optimal_order: float
    sample_schedule_sha256: str
    noise_schedule_sha256: str
    training_seed_sha256: str
    accountant_state_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = PRIVATE_LOCAL_TRAINING_SCHEMA_VERSION
    update_schema_version: str = PRIVATE_MODEL_UPDATE_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_provenance"] = self.model_provenance.as_safe_dict()
        return value


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise PrivateTrainingError(f"configuração deve conter a seção {key}")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: object, label: str) -> None:
    if mapping.get(key) != expected:
        raise PrivateTrainingError(f"{label}.{key} diverge da receita privada")


def _number(mapping: Mapping[str, Any], key: str, expected: float, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateTrainingError(f"{label}.{key} deve ser numérico")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved != expected:
        raise PrivateTrainingError(f"{label}.{key} diverge da receita privada")
    return resolved


def parse_dp_accounting_spec(config: Mapping[str, Any]) -> DPAccountingSpec:
    """Valida a seção v5 sem recalcular o accountant ou carregar o modelo."""

    if not isinstance(config, Mapping) or config.get("schema_version") != EXPECTED_MAIN_SCHEMA_VERSION:
        raise PrivateTrainingError("schema principal não pertence ao piloto DP refinado")
    dp = _mapping(config, "dp_sgd")
    expected_keys = {
        "schema_version", "library", "library_version", "privacy_unit",
        "participant_level_dp_claim", "participant_conversations",
        "protected_record_conversations_per_participant",
        "participant_level_accounting_requirement", "privacy_records_per_client",
        "protected_value_bearing_privacy_records_per_client", "target_epsilons",
        "delta", "max_grad_norm", "clipping", "accountant", "poisson_sampling",
        "logical_batch_size", "max_physical_conversations", "sampling_rate",
        "accounting_steps_per_round", "accounting_total_steps", "optimizer",
        "victim_learning_rate", "betas", "optimizer_epsilon", "weight_decay",
        "noise_multiplier_by_target_epsilon", "realized_epsilon_by_target",
        "optimal_rdp_order_by_target", "pre_campaign_accountant_reproduction_required",
        "invalidate_sigma_when_sampling_contract_changes", "noise_multiplier_rounding",
        "epsilon_tolerance", "composition_rounds", "accountant_state",
        "optimizer_state", "empty_poisson_batch_policy", "grad_sample_mode",
        "secure_mode",
    }
    if set(dp) != expected_keys:
        raise PrivateTrainingError("dp_sgd possui chaves inválidas")
    fixed = {
        "schema_version": DP_ACCOUNTING_SCHEMA_VERSION,
        "library": "opacus",
        "library_version": "1.6.0",
        "privacy_unit": "conversation",
        "participant_level_dp_claim": False,
        "privacy_records_per_client": 100,
        "poisson_sampling": True,
        "accounting_steps_per_round": 100,
        "accounting_total_steps": 2000,
        "max_physical_conversations": 1,
        "logical_batch_size": 4,
        "clipping": "flat",
        "accountant": "rdp",
        "optimizer": "adamw",
        "optimizer_state": "reset_each_client_round",
        "grad_sample_mode": "hooks",
        "secure_mode": False,
        "empty_poisson_batch_policy": "preserve_private_step",
        "participant_conversations": 5,
        "protected_record_conversations_per_participant": 4,
        "participant_level_accounting_requirement": "aggregate_all_five_conversations_or_version_explicit_group_composition",
        "protected_value_bearing_privacy_records_per_client": 80,
        "pre_campaign_accountant_reproduction_required": True,
        "invalidate_sigma_when_sampling_contract_changes": True,
        "noise_multiplier_rounding": "conservative_two_decimal_ceiling",
        "composition_rounds": 20,
        "accountant_state": "persist_across_rounds",
    }
    for key, expected in fixed.items():
        _require(dp, key, expected, "dp_sgd")
    if tuple(dp.get("target_epsilons", ())) != EXPECTED_TARGET_EPSILONS:
        raise PrivateTrainingError("dp_sgd.target_epsilons diverge da receita privada")
    sigma_mapping = dp.get("noise_multiplier_by_target_epsilon")
    realized_mapping = dp.get("realized_epsilon_by_target")
    order_mapping = dp.get("optimal_rdp_order_by_target")
    if not all(isinstance(value, Mapping) for value in (sigma_mapping, realized_mapping, order_mapping)):
        raise PrivateTrainingError("parâmetros do accountant privado são inválidos")
    if any(set(value) != {"3.0", "8.0"} for value in (
        sigma_mapping, realized_mapping, order_mapping
    )):
        raise PrivateTrainingError("orçamentos do accountant privado possuem chaves inválidas")
    for target, sigma in EXPECTED_SIGMA_BY_EPSILON:
        key = f"{target:.1f}"
        _number(sigma_mapping, key, sigma, "dp_sgd.noise_multiplier_by_target_epsilon")
    for target, realized, order in EXPECTED_REALIZED_BY_EPSILON:
        key = f"{target:.1f}"
        _number(realized_mapping, key, realized, "dp_sgd.realized_epsilon_by_target")
        _number(order_mapping, key, order, "dp_sgd.optimal_rdp_order_by_target")
    betas = dp.get("betas")
    if not isinstance(betas, (list, tuple)) or tuple(betas) != (0.9, 0.95):
        raise PrivateTrainingError("dp_sgd.betas diverge da receita privada")
    _number(dp, "epsilon_tolerance", 0.01, "dp_sgd")
    return DPAccountingSpec(
        library="opacus",
        library_version="1.6.0",
        privacy_unit="conversation",
        participant_level_dp_claim=False,
        records_per_client=100,
        poisson_sampling=True,
        sample_rate=_number(dp, "sampling_rate", 0.04, "dp_sgd"),
        private_steps_per_round=100,
        total_private_steps=2000,
        max_physical_conversations=1,
        expected_batch_size=4,
        max_grad_norm=_number(dp, "max_grad_norm", 1.0, "dp_sgd"),
        clipping="flat",
        accountant="rdp",
        delta=_number(dp, "delta", 1e-5, "dp_sgd"),
        target_epsilons=EXPECTED_TARGET_EPSILONS,
        sigma_by_epsilon=EXPECTED_SIGMA_BY_EPSILON,
        realized_by_epsilon=EXPECTED_REALIZED_BY_EPSILON,
        optimizer="adamw",
        victim_learning_rate=_number(dp, "victim_learning_rate", 1e-4, "dp_sgd"),
        betas=(0.9, 0.95),
        optimizer_epsilon=_number(dp, "optimizer_epsilon", 1e-8, "dp_sgd"),
        weight_decay=_number(dp, "weight_decay", 0.01, "dp_sgd"),
        optimizer_state="reset_each_client_round",
        grad_sample_mode="hooks",
        secure_mode=False,
        empty_batch_policy="preserve_private_step",
    )


def load_dp_accounting_spec_from_config(path: Path) -> DPAccountingSpec:
    try:
        return parse_dp_accounting_spec(load_yaml_mapping(Path(path)))
    except ConfigurationError as error:
        raise PrivateTrainingError(str(error)) from error


def validate_dp_accounting_spec(spec: object) -> DPAccountingSpec:
    if not isinstance(spec, DPAccountingSpec):
        raise PrivateTrainingError("spec DP possui tipo inválido")
    if (
        spec.schema_version != DP_ACCOUNTING_SCHEMA_VERSION
        or spec.library != "opacus"
        or spec.library_version != "1.6.0"
        or spec.privacy_unit != "conversation"
        or spec.participant_level_dp_claim
        or spec.records_per_client != 100
        or not spec.poisson_sampling
        or spec.sample_rate != 0.04
        or spec.private_steps_per_round != 100
        or spec.total_private_steps != 2000
        or spec.max_physical_conversations != 1
        or spec.expected_batch_size != 4
        or spec.max_grad_norm != 1.0
        or spec.clipping != "flat"
        or spec.accountant != "rdp"
        or spec.delta != 1e-5
        or spec.target_epsilons != EXPECTED_TARGET_EPSILONS
        or spec.sigma_by_epsilon != EXPECTED_SIGMA_BY_EPSILON
        or spec.realized_by_epsilon != EXPECTED_REALIZED_BY_EPSILON
        or spec.optimizer != "adamw"
        or spec.victim_learning_rate != 1e-4
        or spec.betas != (0.9, 0.95)
        or spec.optimizer_epsilon != 1e-8
        or spec.weight_decay != 0.01
        or spec.optimizer_state != "reset_each_client_round"
        or spec.grad_sample_mode != "hooks"
        or spec.secure_mode
        or spec.empty_batch_policy != "preserve_private_step"
    ):
        raise PrivateTrainingError("spec DP diverge da receita privada")
    return spec


def validate_private_result(result: object, spec: DPAccountingSpec) -> PrivateLocalTrainingResult:
    validated_spec = validate_dp_accounting_spec(spec)
    if not isinstance(result, PrivateLocalTrainingResult):
        raise PrivateTrainingError("resultado privado possui tipo inválido")
    expected_realized, expected_order = validated_spec.realized_for(result.target_epsilon)
    if (
        result.schema_version != PRIVATE_LOCAL_TRAINING_SCHEMA_VERSION
        or result.update_schema_version != PRIVATE_MODEL_UPDATE_SCHEMA_VERSION
        or result.role != "victim"
        or not result.client_id.startswith("victim-")
        or type(result.round_id) is not int
        or not 1 <= result.round_id <= 20
        or result.conversation_count != 100
        or result.optimizer_steps != 100
        or type(result.sampled_conversation_count) is not int
        or result.sampled_conversation_count < 0
        or result.noise_multiplier != validated_spec.sigma_for(result.target_epsilon)
        or result.sample_rate != 0.04
        or result.max_grad_norm != 1.0
        or result.delta != 1e-5
        or result.accountant_steps_total != result.round_id * 100
        or not math.isfinite(result.realized_epsilon)
        or result.realized_epsilon <= 0
        or result.realized_epsilon > expected_realized + 1e-10
        or result.optimal_order <= 0
        or (result.round_id == 20 and abs(result.optimal_order - expected_order) > 1e-10)
        or any(
            not _sha256(value)
            for value in (
                result.sample_schedule_sha256,
                result.noise_schedule_sha256,
                result.training_seed_sha256,
                result.accountant_state_sha256,
            )
        )
        or not isinstance(result.model_provenance, ModelProvenance)
    ):
        raise PrivateTrainingError("resultado privado diverge da receita")
    return result


def accountant_state_sha256(
    client_id: str,
    target_epsilon: float,
    history: Tuple[Tuple[float, float, int], ...],
) -> str:
    digest = hashlib.sha256(b"dp-accountant-state/v1\0")
    digest.update(client_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{target_epsilon:.1f}".encode("ascii"))
    for sigma, rate, steps in history:
        digest.update(f"\0{sigma:.17g}\0{rate:.17g}\0{steps}".encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "DP_ACCOUNTING_SCHEMA_VERSION",
    "PRIVATE_FEDERATED_ROUND_SCHEMA_VERSION",
    "PRIVATE_LOCAL_TRAINING_SCHEMA_VERSION",
    "PRIVATE_MODEL_UPDATE_SCHEMA_VERSION",
    "DPAccountantState",
    "DPAccountingSpec",
    "PrivateLocalTrainingResult",
    "PrivateTrainingError",
    "accountant_state_sha256",
    "load_dp_accounting_spec_from_config",
    "parse_dp_accounting_spec",
    "validate_dp_accounting_spec",
    "validate_private_result",
]
