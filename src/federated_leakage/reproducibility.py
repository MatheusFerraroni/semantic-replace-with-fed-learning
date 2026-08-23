"""Validações operacionais compartilhadas para execução reprodutível."""

from __future__ import annotations

import os
from typing import Any


EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class ReproducibilityEnvironmentError(RuntimeError):
    """O ambiente de execução não satisfaz a receita reprodutível."""


def validate_cuda_reproducibility_environment(device: Any) -> None:
    """Exige a configuração cuBLAS fixada somente para dispositivos CUDA."""

    device_type = device if isinstance(device, str) else getattr(device, "type", None)
    if isinstance(device_type, str):
        device_type = device_type.split(":", 1)[0]
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ReproducibilityEnvironmentError(
            "dispositivo da execução reprodutível é inválido"
        )
    if device_type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        EXPECTED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ReproducibilityEnvironmentError(
            "CUDA determinístico exige CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "antes do processo"
        )


__all__ = [
    "EXPECTED_CUBLAS_WORKSPACE_CONFIG",
    "ReproducibilityEnvironmentError",
    "validate_cuda_reproducibility_environment",
]
