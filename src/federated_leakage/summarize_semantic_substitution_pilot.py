"""CLI de resumo agregado das duas seeds do piloto semântico."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .semantic_pilot_contracts import (
    SemanticPilotError,
    load_semantic_pilot_spec_from_config,
)
from .semantic_pilot_summary import build_semantic_substitution_combined_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume as duas seeds da defesa.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = build_semantic_substitution_combined_result(
            load_semantic_pilot_spec_from_config(arguments.config),
            output_root=arguments.output_root,
        )
    except (FileNotFoundError, SemanticPilotError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("erro: falha inesperada ao resumir o piloto semântico", file=sys.stderr)
        return 1
    print("status: resumo combinado concluído")
    print(f"classificacao: {result.combined_status}")
    print(f"resultado_sha256: {result.result_sha256}")
    print(
        "saida: "
        + str(
            arguments.output_root
            / "runs"
            / "semantic-substitution-upstream-combined-v1"
            / "combined.json"
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
