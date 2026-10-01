"""Contratos estritos da agregação FedAvg e de uma rodada federada."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .training_contracts import TrainingRole


FEDAVG_AGGREGATION_SCHEMA_VERSION = "fedavg-aggregation/v1"
FEDERATED_ROUND_SCHEMA_VERSION = "federated-round/v1"

FedAvgScenario = Literal["F0", "F1"]


class FedAvgError(RuntimeError):
    """A configuração ou execução FedAvg violou o contrato experimental."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FedAvgSpec:
    """Receita normativa da agregação federada não privada."""

    victim_clients: int
    auxiliary_slots: int
    total_clients: int
    rounds: int
    participation_rate: float
    aggregation: str
    aggregation_form: str
    aggregation_weighting: str
    aggregation_dtype: str
    trainable_parameters: str
    client_execution: str
    client_order: str
    accumulator_device: str
    global_update_application: str
    physical_auxiliary_slots: int
    total_victim_weight_units: int
    auxiliary_weight_units: Tuple[int, ...]
    maximum_auxiliary_share: float
    minimum_total_victim_share: float
    submitted_delta_scale: float
    update_transformation: str
    schema_version: str = FEDAVG_AGGREGATION_SCHEMA_VERSION
    round_schema_version: str = FEDERATED_ROUND_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class FedAvgClientWeight:
    """Peso racional de um cliente, mantido em unidades inteiras."""

    client_id: str
    role: TrainingRole
    numerator_units: int
    denominator_units: int
    schema_version: str = FEDAVG_AGGREGATION_SCHEMA_VERSION

    @property
    def value(self) -> float:
        return self.numerator_units / self.denominator_units


@dataclass(frozen=True, slots=True, kw_only=True)
class FedAvgRoundResult:
    """Resultado agregado seguro de uma única rodada F0 ou F1."""

    scenario: FedAvgScenario
    experiment_seed: int
    round_id: int
    auxiliary_weight_units: int
    victim_client_count: int
    auxiliary_client_count: int
    conversation_count: int
    optimizer_steps: int
    supervised_token_count: int
    mean_client_loss: float
    mean_victim_loss: float
    auxiliary_loss: float
    mean_client_gradient_norm: float
    max_client_gradient_norm: float
    aggregate_delta_l2_norm: float
    aggregate_delta_max_abs: float
    client_order_sha256: str
    weights_sha256: str
    sample_order_schedule_sha256: str
    training_seed_schedule_sha256: str
    victim_dataset_sha256: str
    auxiliary_schedule_sha256: str
    auxiliary_values_sha256: str
    auxiliary_presentation_sha256: str
    auxiliary_batch_sha256: str
    initial_model_sha256: str
    aggregate_update_sha256: str
    final_model_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = FEDERATED_ROUND_SCHEMA_VERSION
    aggregation_schema_version: str = FEDAVG_AGGREGATION_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        return result


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise FedAvgError(f"configuração deve conter a seção {key}")
    return value


def _require(
    mapping: Mapping[str, Any], key: str, expected: object, label: str
) -> None:
    if mapping.get(key) != expected:
        raise FedAvgError(f"{label}.{key} diverge da receita FedAvg")


def parse_fedavg_spec(config: Mapping[str, Any]) -> FedAvgSpec:
    """Valida a configuração normativa e resolve o contrato FedAvg."""

    if not isinstance(config, Mapping):
        raise FedAvgError("configuração FedAvg deve ser mapeada")
    if config.get("schema_version") not in {
        "federated-leakage/main-config/v1",
        "federated-leakage/main-config/v2",
        "federated-leakage/main-config/v3",
        "federated-leakage/main-config/v4",
        "federated-leakage/main-config/v5",
    }:
        raise FedAvgError("schema da configuração principal é incompatível")

    federated = _mapping(config, "federated")
    attack = _mapping(config, "attack")
    sweep = _mapping(attack, "aggregation_share_sweep")
    training = _mapping(config, "training")

    fixed_federated = {
        "aggregation_schema_version": FEDAVG_AGGREGATION_SCHEMA_VERSION,
        "round_schema_version": FEDERATED_ROUND_SCHEMA_VERSION,
        "victim_clients": 10,
        "auxiliary_slots": 1,
        "total_clients": 11,
        "rounds": 20,
        "participation_rate": 1.0,
        "aggregation": "fedavg",
        "aggregation_form": "normalized_weighted_model_deltas",
        "aggregation_weighting": "configured_aggregation_share",
        "aggregation_dtype": "float32",
        "trainable_parameters": "all",
        "client_execution": "sequential",
        "client_order": "victim_01_to_10_then_auxiliary",
        "accumulator_device": "cpu",
        "global_update_application": "atomic_float32_then_bfloat16",
    }
    fixed_sweep = {
        "physical_auxiliary_slots": 1,
        "interpretation": "one_auxiliary_slot_weighted_as_k_virtual_units",
        "victim_weight_units": 10,
        "auxiliary_weight_units": list(range(1, 11)),
        "auxiliary_share_formula": "k_over_10_plus_k",
        "victim_share_each_formula": "one_over_10_plus_k",
        "maximum_auxiliary_share": 0.5,
        "minimum_total_victim_share": 0.5,
        "submitted_delta_scale": 1.0,
        "update_transformation": "none",
    }
    fixed_training = {
        "update_dtype": "float32",
        "update_persistence": "forbidden",
    }
    for key, expected in fixed_federated.items():
        _require(federated, key, expected, "federated")
    for key, expected in fixed_sweep.items():
        _require(sweep, key, expected, "attack.aggregation_share_sweep")
    for key, expected in fixed_training.items():
        _require(training, key, expected, "training")

    return FedAvgSpec(
        victim_clients=10,
        auxiliary_slots=1,
        total_clients=11,
        rounds=20,
        participation_rate=1.0,
        aggregation="fedavg",
        aggregation_form="normalized_weighted_model_deltas",
        aggregation_weighting="configured_aggregation_share",
        aggregation_dtype="float32",
        trainable_parameters="all",
        client_execution="sequential",
        client_order="victim_01_to_10_then_auxiliary",
        accumulator_device="cpu",
        global_update_application="atomic_float32_then_bfloat16",
        physical_auxiliary_slots=1,
        total_victim_weight_units=10,
        auxiliary_weight_units=tuple(range(1, 11)),
        maximum_auxiliary_share=0.5,
        minimum_total_victim_share=0.5,
        submitted_delta_scale=1.0,
        update_transformation="none",
    )


def load_fedavg_spec_from_config(path: Path) -> FedAvgSpec:
    """Carrega o YAML compartilhado sem aceitar chaves duplicadas."""

    try:
        config = load_yaml_mapping(Path(path))
    except ConfigurationError as error:
        raise FedAvgError(str(error)) from error
    return parse_fedavg_spec(config)


def validate_fedavg_spec(spec: object) -> FedAvgSpec:
    """Impede construção direta com valores fora da receita normativa."""

    if not isinstance(spec, FedAvgSpec):
        raise FedAvgError("especificação FedAvg é inválida")
    expected = FedAvgSpec(
        victim_clients=10,
        auxiliary_slots=1,
        total_clients=11,
        rounds=20,
        participation_rate=1.0,
        aggregation="fedavg",
        aggregation_form="normalized_weighted_model_deltas",
        aggregation_weighting="configured_aggregation_share",
        aggregation_dtype="float32",
        trainable_parameters="all",
        client_execution="sequential",
        client_order="victim_01_to_10_then_auxiliary",
        accumulator_device="cpu",
        global_update_application="atomic_float32_then_bfloat16",
        physical_auxiliary_slots=1,
        total_victim_weight_units=10,
        auxiliary_weight_units=tuple(range(1, 11)),
        maximum_auxiliary_share=0.5,
        minimum_total_victim_share=0.5,
        submitted_delta_scale=1.0,
        update_transformation="none",
    )
    if spec != expected or not all(
        math.isfinite(value)
        for value in (
            spec.participation_rate,
            spec.maximum_auxiliary_share,
            spec.minimum_total_victim_share,
            spec.submitted_delta_scale,
        )
    ):
        raise FedAvgError("especificação FedAvg diverge da receita fixada")
    return spec


__all__ = [
    "FEDAVG_AGGREGATION_SCHEMA_VERSION",
    "FEDERATED_ROUND_SCHEMA_VERSION",
    "FedAvgClientWeight",
    "FedAvgError",
    "FedAvgRoundResult",
    "FedAvgScenario",
    "FedAvgSpec",
    "load_fedavg_spec_from_config",
    "parse_fedavg_spec",
    "validate_fedavg_spec",
]
