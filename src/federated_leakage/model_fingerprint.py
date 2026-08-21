"""Fingerprint canônico compartilhado de estados e atualizações do modelo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .model_contracts import LoadedModelBundle
from .model_updates import _validate_model_parameters


MODEL_STATE_FINGERPRINT_DOMAIN = b"federated-model-state/v1\0"


class ModelFingerprintError(RuntimeError):
    """O estado do modelo não pôde ser identificado com segurança."""


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise ModelFingerprintError(
            "PyTorch ausente; instale o projeto com .[model]"
        ) from error
    return torch


def _tensor_header(name: str, tensor: Any) -> bytes:
    return json.dumps(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "name": name,
            "shape": list(tensor.shape),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _update_tensor_digest(digest: Any, name: str, tensor: Any) -> None:
    torch = _load_torch()
    try:
        resolved = tensor.detach().to(device="cpu").contiguous()
        header = _tensor_header(name, resolved)
        raw = resolved.reshape(-1).view(torch.uint8).numpy()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(resolved.numel().to_bytes(8, "big"))
        digest.update(memoryview(raw))
    except Exception as error:
        raise ModelFingerprintError(
            "falha ao calcular fingerprint de parâmetros"
        ) from error


def fingerprint_named_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    domain: bytes = MODEL_STATE_FINGERPRINT_DOMAIN,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    count = 0
    for name, tensor in named_tensors:
        if not isinstance(name, str) or not name:
            raise ModelFingerprintError("fingerprint possui nome de parâmetro inválido")
        _update_tensor_digest(digest, name, tensor)
        count += 1
    if count == 0:
        raise ModelFingerprintError("fingerprint não contém parâmetros")
    return digest.hexdigest()


def fingerprint_model_parameters(model_bundle: LoadedModelBundle) -> str:
    """Identifica o estado atual usando o mesmo domínio empregado pelo FedAvg."""

    try:
        named = _validate_model_parameters(model_bundle, require_finite=True)
        return fingerprint_named_tensors(
            (name, parameter.detach()) for name, parameter in named
        )
    except ModelFingerprintError:
        raise
    except Exception as error:
        raise ModelFingerprintError("modelo não pôde ser identificado") from error


__all__ = [
    "MODEL_STATE_FINGERPRINT_DOMAIN",
    "ModelFingerprintError",
    "fingerprint_model_parameters",
    "fingerprint_named_tensors",
]
