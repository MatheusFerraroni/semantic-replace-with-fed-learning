"""Contratos imutáveis do treinamento local causal."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import ModelProvenance
from .reproducibility import EXPECTED_CUBLAS_WORKSPACE_CONFIG


LOCAL_TRAINING_SCHEMA_VERSION = "local-training/v1"
LOCAL_MODEL_UPDATE_SCHEMA_VERSION = "local-model-update/v1"
EXPECTED_CONVERSATION_COUNT = 100
EXPECTED_OPTIMIZER_STEPS = 25

TrainingRole = Literal[
    "victim",
    "auxiliary_benign",
    "auxiliary_adversarial",
]


class LocalTrainingError(RuntimeError):
    """A receita ou a execução local violou o contrato experimental."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTrainingSpec:
    """Receita fixa do treinamento local não privado."""

    deterministic_algorithms: bool
    dataloader_workers: int
    rounds: int
    honest_local_epochs: int
    client_execution: str
    trainable_parameters: str
    objective: str
    label_ignore_index: int
    loss_reduction: str
    optimizer: str
    learning_rate: float
    betas: Tuple[float, float]
    optimizer_epsilon: float
    optimizer_state: str
    weight_decay: float
    scheduler: str
    warmup_steps: int
    logical_batch_size: int
    max_physical_conversations: int
    precision: str
    loss_computation_dtype: str
    tf32: bool
    torch_compile: bool
    use_cache: bool
    packing: bool
    sample_order: str
    incomplete_logical_batch_policy: str
    expected_conversation_count: int
    optimizer_steps: int
    submitted_delta_scale: float
    update_dtype: str
    update_persistence: str
    schema_version: str = LOCAL_TRAINING_SCHEMA_VERSION
    update_schema_version: str = LOCAL_MODEL_UPDATE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTrainingResult:
    """Métricas agregadas que não carregam amostras nem tensores protegidos."""

    client_id: str
    role: TrainingRole
    round_id: int
    conversation_count: int
    optimizer_steps: int
    supervised_token_count: int
    mean_loss: float
    first_step_loss: float
    last_step_loss: float
    mean_gradient_norm: float
    max_gradient_norm: float
    sample_order_sha256: str
    training_seed_sha256: str
    model_provenance: ModelProvenance
    schema_version: str = LOCAL_TRAINING_SCHEMA_VERSION
    update_schema_version: str = LOCAL_MODEL_UPDATE_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        """Serializa somente metadados agregados e proveniência já sanitizada."""

        return {
            "schema_version": self.schema_version,
            "update_schema_version": self.update_schema_version,
            "client_id": self.client_id,
            "role": self.role,
            "round_id": self.round_id,
            "conversation_count": self.conversation_count,
            "optimizer_steps": self.optimizer_steps,
            "supervised_token_count": self.supervised_token_count,
            "mean_loss": self.mean_loss,
            "first_step_loss": self.first_step_loss,
            "last_step_loss": self.last_step_loss,
            "mean_gradient_norm": self.mean_gradient_norm,
            "max_gradient_norm": self.max_gradient_norm,
            "sample_order_sha256": self.sample_order_sha256,
            "training_seed_sha256": self.training_seed_sha256,
            "model_provenance": self.model_provenance.as_safe_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelParameterSnapshot:
    """Cópia inicial efêmera dos parâmetros, nunca serializável pelo projeto."""

    model_identity: int
    parameter_names: Tuple[str, ...]
    parameters: Tuple[Any, ...] = field(repr=False)
    parameter_count: int
    source_device: str
    dtype: str
    schema_version: str = LOCAL_MODEL_UPDATE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterDelta:
    """Um delta transitório em CPU/float32 para consumo imediato pelo FedAvg."""

    name: str
    tensor: Any = field(repr=False)
    numel: int
    schema_version: str = LOCAL_MODEL_UPDATE_SCHEMA_VERSION


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise LocalTrainingError(f"configuração deve conter a seção {key}")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: object, label: str) -> None:
    if mapping.get(key) != expected:
        raise LocalTrainingError(f"{label}.{key} diverge da receita fixada")


def _finite_float(mapping: Mapping[str, Any], key: str, expected: float, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalTrainingError(f"{label}.{key} deve ser numérico")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved != expected:
        raise LocalTrainingError(f"{label}.{key} diverge da receita fixada")
    return resolved


def parse_local_training_spec(config: Mapping[str, Any]) -> LocalTrainingSpec:
    """Valida as seções normativas e devolve a receita local única."""

    if not isinstance(config, Mapping):
        raise LocalTrainingError("configuração de treinamento deve ser mapeada")
    main_schema_version = config.get("schema_version")
    if main_schema_version not in {
        "federated-leakage/main-config/v1",
        "federated-leakage/main-config/v2",
        "federated-leakage/main-config/v3",
        "federated-leakage/main-config/v4",
    }:
        raise LocalTrainingError("schema da configuração principal é incompatível")
    expected_learning_rate = (
        3e-5
        if main_schema_version in {
            "federated-leakage/main-config/v3",
            "federated-leakage/main-config/v4",
        }
        else 1e-5
    )

    reproducibility = _mapping(config, "reproducibility")
    federated = _mapping(config, "federated")
    training = _mapping(config, "training")
    attack = _mapping(config, "attack")
    local_data = _mapping(attack, "local_data_generation")
    synthetic_data = _mapping(config, "synthetic_data")

    fixed_reproducibility = {
        "deterministic_algorithms": True,
        "cuda_cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "dataloader_workers": 0,
    }
    fixed_federated = {
        "rounds": 20,
        "honest_local_epochs": 1,
        "training_unit": "conversation",
        "aggregation_dtype": "float32",
        "trainable_parameters": "all",
        "client_execution": "sequential",
    }
    fixed_training = {
        "schema_version": LOCAL_TRAINING_SCHEMA_VERSION,
        "update_schema_version": LOCAL_MODEL_UPDATE_SCHEMA_VERSION,
        "objective": "causal_language_modeling",
        "tokenized_conversation_schema_version": "tokenized-conversation/v1",
        "label_ignore_index": -100,
        "loss_reduction": "mean_per_conversation_then_mean_batch",
        "optimizer": "adamw",
        "optimizer_state": "reset_each_client_round",
        "scheduler": "constant",
        "warmup_steps": 0,
        "logical_batch_size": 4,
        "max_physical_conversations": 1,
        "precision": "bf16",
        "loss_computation_dtype": "float32",
        "tf32": False,
        "torch_compile": False,
        "attention_implementation": "eager",
        "use_cache": False,
        "dynamic_padding": True,
        "packing": False,
        "overlength_policy": "reject",
        "sample_order": "preserve_input",
        "incomplete_logical_batch_policy": "reject",
        "update_dtype": "float32",
        "update_persistence": "forbidden",
    }
    for key, expected in fixed_reproducibility.items():
        _require(reproducibility, key, expected, "reproducibility")
    for key, expected in fixed_federated.items():
        _require(federated, key, expected, "federated")
    for key, expected in fixed_training.items():
        _require(training, key, expected, "training")

    _require(attack, "optimizer", "adamw", "attack")
    _require(attack, "logical_batch_size", 4, "attack")
    _require(attack, "local_steps", EXPECTED_OPTIMIZER_STEPS, "attack")
    _require(attack, "optimizer_state", "reset_each_round", "attack")
    _require(attack, "reference_update_transformation", "none", "attack")
    _require(
        local_data,
        "records_per_round",
        EXPECTED_CONVERSATION_COUNT,
        "attack.local_data_generation",
    )
    _finite_float(
        attack,
        "learning_rate",
        expected_learning_rate,
        "attack",
    )
    submitted_scale = _finite_float(
        attack,
        "reference_submitted_delta_scale",
        1.0,
        "attack",
    )

    profiles = synthetic_data.get("profiles_per_victim_client")
    conversations_per_profile = synthetic_data.get("conversations_per_profile")
    if (
        type(profiles) is not int
        or type(conversations_per_profile) is not int
        or profiles * conversations_per_profile != EXPECTED_CONVERSATION_COUNT
    ):
        raise LocalTrainingError("contagem local das vítimas diverge da receita")

    betas = training.get("betas")
    if not isinstance(betas, (list, tuple)) or tuple(betas) != (0.9, 0.95):
        raise LocalTrainingError("training.betas diverge da receita fixada")

    learning_rate = _finite_float(
        training,
        "learning_rate",
        expected_learning_rate,
        "training",
    )
    optimizer_epsilon = _finite_float(
        training, "optimizer_epsilon", 1e-8, "training"
    )
    weight_decay = _finite_float(training, "weight_decay", 0.01, "training")
    if learning_rate != float(attack["learning_rate"]):
        raise LocalTrainingError("receitas honesta e auxiliar usam learning rates distintos")

    return LocalTrainingSpec(
        deterministic_algorithms=True,
        dataloader_workers=0,
        rounds=20,
        honest_local_epochs=1,
        client_execution="sequential",
        trainable_parameters="all",
        objective="causal_language_modeling",
        label_ignore_index=-100,
        loss_reduction="mean_per_conversation_then_mean_batch",
        optimizer="adamw",
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        optimizer_epsilon=optimizer_epsilon,
        optimizer_state="reset_each_client_round",
        weight_decay=weight_decay,
        scheduler="constant",
        warmup_steps=0,
        logical_batch_size=4,
        max_physical_conversations=1,
        precision="bf16",
        loss_computation_dtype="float32",
        tf32=False,
        torch_compile=False,
        use_cache=False,
        packing=False,
        sample_order="preserve_input",
        incomplete_logical_batch_policy="reject",
        expected_conversation_count=EXPECTED_CONVERSATION_COUNT,
        optimizer_steps=EXPECTED_OPTIMIZER_STEPS,
        submitted_delta_scale=submitted_scale,
        update_dtype="float32",
        update_persistence="forbidden",
    )


def load_local_training_spec_from_config(path: Path) -> LocalTrainingSpec:
    """Carrega o YAML principal sem aceitar valores duplicados ou implícitos."""

    try:
        config = load_yaml_mapping(Path(path))
    except ConfigurationError as error:
        raise LocalTrainingError(str(error)) from error
    return parse_local_training_spec(config)


def validate_local_training_spec(spec: object) -> LocalTrainingSpec:
    """Impede que construção direta contorne os valores normativos do parser."""

    if not isinstance(spec, LocalTrainingSpec):
        raise LocalTrainingError("especificação do treinamento local é inválida")
    expected = {
        "schema_version": LOCAL_TRAINING_SCHEMA_VERSION,
        "update_schema_version": LOCAL_MODEL_UPDATE_SCHEMA_VERSION,
        "deterministic_algorithms": True,
        "dataloader_workers": 0,
        "rounds": 20,
        "honest_local_epochs": 1,
        "client_execution": "sequential",
        "trainable_parameters": "all",
        "objective": "causal_language_modeling",
        "label_ignore_index": -100,
        "loss_reduction": "mean_per_conversation_then_mean_batch",
        "optimizer": "adamw",
        "betas": (0.9, 0.95),
        "optimizer_epsilon": 1e-8,
        "optimizer_state": "reset_each_client_round",
        "weight_decay": 0.01,
        "scheduler": "constant",
        "warmup_steps": 0,
        "logical_batch_size": 4,
        "max_physical_conversations": 1,
        "precision": "bf16",
        "loss_computation_dtype": "float32",
        "tf32": False,
        "torch_compile": False,
        "use_cache": False,
        "packing": False,
        "sample_order": "preserve_input",
        "incomplete_logical_batch_policy": "reject",
        "expected_conversation_count": EXPECTED_CONVERSATION_COUNT,
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "submitted_delta_scale": 1.0,
        "update_dtype": "float32",
        "update_persistence": "forbidden",
    }
    if spec.learning_rate not in {1e-5, 3e-5} or any(
        getattr(spec, key, None) != value for key, value in expected.items()
    ):
        raise LocalTrainingError("especificação local diverge da receita fixada")
    return spec


__all__ = [
    "EXPECTED_CONVERSATION_COUNT",
    "EXPECTED_OPTIMIZER_STEPS",
    "LOCAL_MODEL_UPDATE_SCHEMA_VERSION",
    "LOCAL_TRAINING_SCHEMA_VERSION",
    "LocalTrainingError",
    "LocalTrainingResult",
    "LocalTrainingSpec",
    "ModelParameterSnapshot",
    "ParameterDelta",
    "TrainingRole",
    "load_local_training_spec_from_config",
    "parse_local_training_spec",
    "validate_local_training_spec",
]
