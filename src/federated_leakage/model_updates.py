"""Snapshots efêmeros e deltas em streaming do treinamento local."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Tuple

from .model_contracts import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_WEIGHT_DTYPE,
    LoadedModelBundle,
)
from .dp_contracts import (
    PRIVATE_MODEL_UPDATE_SCHEMA_VERSION,
    PrivateLocalTrainingResult,
)
from .training_contracts import (
    LOCAL_MODEL_UPDATE_SCHEMA_VERSION,
    LocalTrainingError,
    LocalTrainingResult,
    ModelParameterSnapshot,
    ParameterDelta,
)


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise LocalTrainingError(
            "PyTorch ausente; instale o projeto com .[model]"
        ) from error
    return torch


def _named_parameters(bundle: LoadedModelBundle) -> Tuple[Tuple[str, Any], ...]:
    if not isinstance(bundle, LoadedModelBundle):
        raise LocalTrainingError("bundle de modelo local é inválido")
    model = bundle.model
    if (
        bundle.provenance.architecture != EXPECTED_ARCHITECTURE
        or bundle.provenance.parameter_count != EXPECTED_PARAMETER_COUNT
    ):
        raise LocalTrainingError("proveniência do modelo local é incompatível")
    if model.__class__.__name__ != EXPECTED_ARCHITECTURE:
        raise LocalTrainingError("arquitetura do modelo local é incompatível")
    try:
        named = tuple(model.named_parameters())
    except Exception as error:
        raise LocalTrainingError("parâmetros do modelo local são inacessíveis") from error
    if not named or any(not isinstance(name, str) or not name for name, _ in named):
        raise LocalTrainingError("nomes dos parâmetros locais são inválidos")
    names = tuple(name for name, _ in named)
    if len(set(names)) != len(names):
        raise LocalTrainingError("modelo local possui parâmetros duplicados")
    if sum(parameter.numel() for _, parameter in named) != EXPECTED_PARAMETER_COUNT:
        raise LocalTrainingError("contagem de parâmetros locais é incompatível")
    return named


def _validate_model_parameters(
    bundle: LoadedModelBundle,
    *,
    require_finite: bool,
) -> Tuple[Tuple[str, Any], ...]:
    torch = _load_torch()
    named = _named_parameters(bundle)
    expected_device = bundle.provenance.device.split(":", 1)[0]
    if bundle.provenance.weight_dtype != EXPECTED_WEIGHT_DTYPE:
        raise LocalTrainingError("dtype declarado do modelo local é incompatível")
    for _, parameter in named:
        if parameter.dtype != torch.bfloat16:
            raise LocalTrainingError("parâmetro local não está em bfloat16")
        if parameter.device.type != expected_device:
            raise LocalTrainingError("parâmetro local está no dispositivo incorreto")
        if not parameter.requires_grad:
            raise LocalTrainingError("modelo local possui parâmetro congelado")
        if require_finite and not bool(torch.isfinite(parameter.detach()).all().item()):
            raise LocalTrainingError("modelo local possui parâmetro não finito")
    return named


def capture_model_parameter_snapshot(
    model_bundle: LoadedModelBundle,
) -> ModelParameterSnapshot:
    """Copia para CPU/BF16 o estado inicial de um modelo exclusivo do cliente."""

    torch = _load_torch()
    named = _validate_model_parameters(model_bundle, require_finite=True)
    try:
        parameters = tuple(
            parameter.detach().to(device="cpu", dtype=torch.bfloat16).clone()
            for _, parameter in named
        )
    except Exception as error:
        raise LocalTrainingError("falha ao capturar snapshot do modelo local") from error
    return ModelParameterSnapshot(
        model_identity=id(model_bundle.model),
        parameter_names=tuple(name for name, _ in named),
        parameters=parameters,
        parameter_count=sum(parameter.numel() for parameter in parameters),
        source_device=model_bundle.provenance.device,
        dtype="bfloat16",
    )


def _validate_snapshot(
    model_bundle: LoadedModelBundle,
    snapshot: ModelParameterSnapshot,
) -> Tuple[Tuple[str, Any], ...]:
    torch = _load_torch()
    if not isinstance(snapshot, ModelParameterSnapshot):
        raise LocalTrainingError("snapshot inicial é inválido")
    if snapshot.schema_version != LOCAL_MODEL_UPDATE_SCHEMA_VERSION:
        raise LocalTrainingError("schema do snapshot inicial é incompatível")
    if snapshot.model_identity != id(model_bundle.model):
        raise LocalTrainingError("snapshot pertence a outro modelo local")
    if (
        snapshot.dtype != "bfloat16"
        or snapshot.parameter_count != EXPECTED_PARAMETER_COUNT
        or snapshot.source_device != model_bundle.provenance.device
    ):
        raise LocalTrainingError("metadados do snapshot inicial são incompatíveis")

    named = _validate_model_parameters(model_bundle, require_finite=False)
    if snapshot.parameter_names != tuple(name for name, _ in named):
        raise LocalTrainingError("estrutura do snapshot inicial diverge do modelo")
    if len(snapshot.parameters) != len(named):
        raise LocalTrainingError("quantidade de tensores do snapshot inicial diverge")
    for base, (_, current) in zip(snapshot.parameters, named):
        if not isinstance(base, torch.Tensor):
            raise LocalTrainingError("snapshot inicial possui tensor inválido")
        if base.device.type != "cpu" or base.dtype != torch.bfloat16:
            raise LocalTrainingError("snapshot inicial não está em CPU/bfloat16")
        if base.shape != current.shape:
            raise LocalTrainingError("forma do snapshot inicial diverge do modelo")
        if not bool(torch.isfinite(base).all().item()):
            raise LocalTrainingError("snapshot inicial possui parâmetro não finito")
    return named


def restore_model_parameter_snapshot(
    model_bundle: LoadedModelBundle,
    snapshot: ModelParameterSnapshot,
) -> None:
    """Restaura exatamente o modelo associado ao snapshot efêmero."""

    torch = _load_torch()
    named = _validate_snapshot(model_bundle, snapshot)
    try:
        with torch.no_grad():
            for base, (_, current) in zip(snapshot.parameters, named):
                current.copy_(base.to(device=current.device, dtype=current.dtype))
    except Exception as error:
        raise LocalTrainingError("falha ao restaurar snapshot do modelo local") from error


def iter_local_parameter_deltas(
    model_bundle: LoadedModelBundle,
    snapshot: ModelParameterSnapshot,
    result: LocalTrainingResult | PrivateLocalTrainingResult,
) -> Iterator[ParameterDelta]:
    """Emite o delta não escalado em CPU/float32, um parâmetro por vez."""

    torch = _load_torch()
    if not isinstance(result, (LocalTrainingResult, PrivateLocalTrainingResult)):
        raise LocalTrainingError("resultado do treinamento local é inválido")
    expected_update_schema = (
        PRIVATE_MODEL_UPDATE_SCHEMA_VERSION
        if isinstance(result, PrivateLocalTrainingResult)
        else LOCAL_MODEL_UPDATE_SCHEMA_VERSION
    )
    if (
        result.update_schema_version != expected_update_schema
        or result.model_provenance != model_bundle.provenance
    ):
        raise LocalTrainingError("metadados da atualização local são incompatíveis")
    named = _validate_snapshot(model_bundle, snapshot)
    _validate_model_parameters(model_bundle, require_finite=True)
    for base, (name, current) in zip(snapshot.parameters, named):
        try:
            delta = current.detach().to(device="cpu", dtype=torch.float32)
            delta.sub_(base)
        except Exception as error:
            raise LocalTrainingError("falha ao calcular delta do modelo local") from error
        if not bool(torch.isfinite(delta).all().item()):
            raise LocalTrainingError("atualização local possui delta não finito")
        yield ParameterDelta(
            name=name,
            tensor=delta,
            numel=delta.numel(),
            schema_version=expected_update_schema,
        )


__all__ = [
    "capture_model_parameter_snapshot",
    "iter_local_parameter_deltas",
]
