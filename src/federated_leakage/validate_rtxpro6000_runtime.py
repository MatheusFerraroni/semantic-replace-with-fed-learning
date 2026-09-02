"""Validação isolada do runtime Blackwell, sem carregar modelo ou dados."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .runtime_profile import (
    ExecutionRuntimeError,
    capture_execution_runtime,
    load_execution_runtime_spec,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida o runtime RTX PRO 6000 pinado.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refined-runtime-rtxpro6000-cu128-v1.yaml"),
    )
    arguments = parser.parse_args(argv)
    try:
        value = capture_execution_runtime(load_execution_runtime_spec(arguments.config))
    except ExecutionRuntimeError as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada ao validar runtime RTX", file=sys.stderr)
        return 1
    print("status: runtime RTX PRO 6000 validado")
    print(f"gpu: {value.gpu_name}")
    print(f"compute_capability: {value.compute_capability[0]}.{value.compute_capability[1]}")
    print(f"vram_gib: {value.total_memory_bytes / 1024**3:.2f}")
    print(f"torch: {value.torch_version}")
    print(f"cuda_build: {value.torch_cuda_version}")
    print(f"driver: {value.driver_version}")
    print(f"runtime_sha256: {value.runtime_sha256}")
    print("escrita: nao")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
