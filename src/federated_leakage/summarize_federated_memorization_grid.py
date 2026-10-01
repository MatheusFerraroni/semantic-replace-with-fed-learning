"""CLI do resumo conjunto seguro da grade v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .federated_grid_contracts import FederatedGridError, load_federated_grid_spec_from_config
from .federated_grid_summary import build_federated_grid_combined_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combina estritamente os dois runs concluídos da grade v2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    arguments = parser.parse_args(argv)
    try:
        spec = load_federated_grid_spec_from_config(arguments.config)
        result = build_federated_grid_combined_result(spec, output_root=arguments.output_root)
    except (FileNotFoundError, FederatedGridError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada ao combinar a grade", file=sys.stderr)
        return 1
    print("status: resumo conjunto da grade concluído")
    print(f"primeiro_braco_robusto: {result.first_robust_arm}")
    for value in result.classifications:
        print(f"{value.arm_id}: {value.classification}")
    print(f"resultado_sha256: {result.result_sha256}")
    print(f"saida: {arguments.output_root / 'runs' / 'federated-memorization-grid-v2' / 'combined.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
