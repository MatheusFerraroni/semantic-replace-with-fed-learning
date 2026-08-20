"""CLI para preparar e validar o Tucano 2 sem iniciar treinamento."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .configuration import ConfigurationError, load_yaml_mapping

from .model_loading import (
    DEFAULT_MODEL_CACHE,
    HuggingFaceModelSpec,
    ModelArtifactError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
    ModelSpec,
    load_model_bundle,
    parse_model_spec,
    prepare_huggingface_model,
)


def load_model_spec_from_config(path: Path) -> ModelSpec:
    """Carrega somente a seção pública `model` de um YAML."""

    try:
        config = load_yaml_mapping(Path(path))
    except ConfigurationError as error:
        raise ModelConfigurationError(str(error)) from error
    if not isinstance(config.get("model"), dict):
        raise ModelConfigurationError("configuração deve conter o objeto model")
    return parse_model_spec(config["model"])


def _print_safe_summary(bundle) -> None:
    provenance = bundle.provenance
    print("status: modelo validado")
    print(f"origem: {provenance.source_kind}")
    print(f"identificador: {provenance.source_identifier}")
    if provenance.revision is not None:
        print(f"revisao: {provenance.revision}")
    if provenance.artifact_sha256 is not None:
        print(f"artifact_sha256: {provenance.artifact_sha256}")
    print(f"variante: {provenance.result_variant}")
    print(f"arquitetura: {provenance.architecture}")
    print(f"parametros: {provenance.parameter_count}")
    print(f"vocabulario: {provenance.vocab_size}")
    print(f"contexto_nativo: {provenance.native_context_length}")
    print(f"comprimento_treinamento: {provenance.training_sequence_length}")
    print(f"dtype: {provenance.weight_dtype}")
    print(f"dispositivo: {provenance.device}")
    print(
        "tokenizer_fingerprint_sha256: "
        f"{provenance.tokenizer_fingerprint_sha256}"
    )
    print(f"torch: {provenance.torch_version}")
    print(f"transformers: {provenance.transformers_version}")
    print(f"tokenizers: {provenance.tokenizers_version}")
    print(f"safetensors: {provenance.safetensors_version}")
    print(f"huggingface_hub: {provenance.huggingface_hub_version}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara ou valida offline o Tucano 2 fixado pelo protocolo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuração YAML que contém a seção model.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="Cache local do snapshot Hugging Face.",
    )
    parser.add_argument(
        "--model-artifact-dir",
        type=Path,
        help="Diretório absoluto do artefato local refinado.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Dispositivo exato para a validação da carga.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Proíbe download e exige o snapshot já preparado.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        spec = load_model_spec_from_config(arguments.config)
        if isinstance(spec, HuggingFaceModelSpec):
            if arguments.model_artifact_dir is not None:
                raise ModelConfigurationError(
                    "--model-artifact-dir não é permitido no modo huggingface"
                )
            if arguments.offline:
                bundle = load_model_bundle(
                    spec,
                    cache_dir=arguments.cache_dir,
                    device=arguments.device,
                )
            else:
                bundle = prepare_huggingface_model(
                    spec,
                    cache_dir=arguments.cache_dir,
                    device=arguments.device,
                )
        else:
            if arguments.model_artifact_dir is None:
                raise ModelConfigurationError(
                    "--model-artifact-dir é obrigatório no modo local_artifact"
                )
            bundle = load_model_bundle(
                spec,
                model_artifact_dir=arguments.model_artifact_dir,
                device=arguments.device,
            )
    except (
        ModelArtifactError,
        ModelConfigurationError,
        ModelDependencyError,
        ModelLoadError,
    ) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    _print_safe_summary(bundle)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
