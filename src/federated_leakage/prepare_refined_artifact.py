"""CLI de preparação do artefato refinado Fórum/Tec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .model_contracts import LocalArtifactModelSpec, ModelArtifactError, ModelConfigurationError
from .model_loading import load_model_spec_from_config
from .queroquero_artifact import prepare_queroquero_artifact_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida e extrai atomicamente o artefato Fórum/Tec fixado."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/models")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = load_model_spec_from_config(arguments.config)
        if not isinstance(spec, LocalArtifactModelSpec) or spec.contract_profile != "queroquero-export-v1":
            raise ModelConfigurationError("configuração não seleciona o artefato Fórum/Tec")
        validated = prepare_queroquero_artifact_archive(
            spec, arguments.archive, arguments.output_root
        )
    except (ModelArtifactError, ModelConfigurationError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    print("status: artefato refinado preparado")
    print(f"identificador: {validated.manifest['artifact_id']}")
    print(f"artifact_sha256: {validated.manifest['artifact_sha256']}")
    print(f"destino: {validated.directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
