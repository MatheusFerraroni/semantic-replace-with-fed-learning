"""Contratos imutáveis do carregamento do Tucano 2 0.6B."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union


BASE_MODEL_ID = "Polygl0t/Tucano2-0.6B-Base"
BASE_MODEL_REVISION = "dad97dc864a8f9a1d240fb9351d098f3af9511d7"
BASE_RESULT_VARIANT = "upstream_baseline"
MODEL_ARTIFACT_SCHEMA_VERSION = "tucano2-model-artifact/v1"
MODEL_LOADING_SCHEMA_VERSION = "tucano2-model-loading/v1"
DEFAULT_MODEL_CACHE = Path("artifacts/huggingface")
TRAINING_SEQUENCE_LENGTH = 1_024

EXPECTED_ARCHITECTURE = "LlamaForCausalLM"
EXPECTED_MODEL_TYPE = "llama"
EXPECTED_PARAMETER_COUNT = 670_127_616
EXPECTED_NATIVE_CONTEXT_LENGTH = 4_096
EXPECTED_HIDDEN_SIZE = 1_536
EXPECTED_INTERMEDIATE_SIZE = 3_072
EXPECTED_HIDDEN_LAYERS = 28
EXPECTED_ATTENTION_HEADS = 16
EXPECTED_KEY_VALUE_HEADS = 8
EXPECTED_VOCAB_SIZE = 49_152
EXPECTED_WEIGHT_DTYPE = "bfloat16"
EXPECTED_TOKENIZER_FINGERPRINT = (
    "069e8fecbf6a1e7adc2941a53408306827516f11418998a295e2c4d0e24d3ae7"
)
EXPECTED_TOKENIZER_FILES = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED_TOKEN_IDS = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 49_109,
    "unk_token_id": 0,
}
EXPECTED_TOKEN_TEXT = {
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
SNAPSHOT_ALLOW_PATTERNS = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ModelConfigurationError(ValueError):
    """A configuração de origem do modelo viola o contrato."""


class ModelArtifactError(ValueError):
    """O artefato local está incompleto, adulterado ou incompatível."""


class ModelLoadError(RuntimeError):
    """O modelo não pôde ser preparado ou carregado com segurança."""


class ModelDependencyError(RuntimeError):
    """As dependências opcionais do carregador não estão instaladas."""


@dataclass(frozen=True, slots=True)
class HuggingFaceModelSpec:
    """Origem pública imutável do baseline."""

    model_id: str
    revision: str
    result_variant: str
    max_sequence_length: int
    kind: str = "huggingface"


@dataclass(frozen=True, slots=True)
class LocalArtifactModelSpec:
    """Origem local validada pelo contrato de artefato v1."""

    expected_schema: str
    expected_artifact_sha256: str
    max_sequence_length: int
    kind: str = "local_artifact"


ModelSpec = Union[HuggingFaceModelSpec, LocalArtifactModelSpec]


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Metadados seguros e suficientes para identificar o modelo carregado."""

    schema_version: str
    source_kind: str
    source_identifier: str
    revision: str | None
    artifact_sha256: str | None
    result_variant: str
    architecture: str
    parameter_count: int
    native_context_length: int
    training_sequence_length: int
    vocab_size: int
    tokenizer_fingerprint_sha256: str
    weight_dtype: str
    device: str
    torch_version: str
    transformers_version: str
    tokenizers_version: str
    safetensors_version: str
    huggingface_hub_version: str

    def as_safe_dict(self) -> Dict[str, Any]:
        """Serializa somente campos que nunca incluem caminhos locais."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedModelBundle:
    """Modelo, tokenizador e proveniência validados como uma unidade."""

    model: Any
    tokenizer: Any
    max_sequence_length: int
    provenance: ModelProvenance


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if frozenset(value) != expected:
        raise ModelConfigurationError(f"{label} possui chaves inválidas")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelConfigurationError(f"{label} deve ser uma string não vazia")
    return value


def _require_sequence_length(value: object) -> int:
    if type(value) is not int or value != TRAINING_SEQUENCE_LENGTH:
        raise ModelConfigurationError(
            "model.max_sequence_length deve ser exatamente 1024"
        )
    return value


def parse_model_spec(model_config: Mapping[str, Any]) -> ModelSpec:
    """Converte a seção `model` em uma origem estrita e imutável."""

    if not isinstance(model_config, Mapping):
        raise ModelConfigurationError("model deve ser um objeto")
    kind = _require_string(model_config.get("kind"), "model.kind")

    if kind == "huggingface":
        _require_exact_keys(
            model_config,
            frozenset(
                {
                    "kind",
                    "model_id",
                    "revision",
                    "result_variant",
                    "max_sequence_length",
                }
            ),
            "model",
        )
        model_id = _require_string(model_config["model_id"], "model.model_id")
        revision = _require_string(model_config["revision"], "model.revision")
        result_variant = _require_string(
            model_config["result_variant"], "model.result_variant"
        )
        if model_id != BASE_MODEL_ID:
            raise ModelConfigurationError("model.model_id não é o baseline fixado")
        if not _GIT_SHA_PATTERN.fullmatch(revision):
            raise ModelConfigurationError(
                "model.revision deve ser um SHA Git completo em minúsculas"
            )
        if revision != BASE_MODEL_REVISION:
            raise ModelConfigurationError("model.revision não é a revisão fixada")
        if result_variant != BASE_RESULT_VARIANT:
            raise ModelConfigurationError(
                "model.result_variant deve ser upstream_baseline"
            )
        return HuggingFaceModelSpec(
            model_id=model_id,
            revision=revision,
            result_variant=result_variant,
            max_sequence_length=_require_sequence_length(
                model_config["max_sequence_length"]
            ),
        )

    if kind == "local_artifact":
        _require_exact_keys(
            model_config,
            frozenset(
                {
                    "kind",
                    "expected_schema",
                    "expected_artifact_sha256",
                    "max_sequence_length",
                }
            ),
            "model",
        )
        expected_schema = _require_string(
            model_config["expected_schema"], "model.expected_schema"
        )
        expected_hash = _require_string(
            model_config["expected_artifact_sha256"],
            "model.expected_artifact_sha256",
        )
        if expected_schema != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ModelConfigurationError("model.expected_schema é desconhecido")
        if not _SHA256_PATTERN.fullmatch(expected_hash):
            raise ModelConfigurationError(
                "model.expected_artifact_sha256 deve ser SHA-256 em minúsculas"
            )
        return LocalArtifactModelSpec(
            expected_schema=expected_schema,
            expected_artifact_sha256=expected_hash,
            max_sequence_length=_require_sequence_length(
                model_config["max_sequence_length"]
            ),
        )

    raise ModelConfigurationError("model.kind deve ser huggingface ou local_artifact")


def validate_model_spec(spec: object) -> ModelSpec:
    """Impede que a construção direta dos dataclasses contorne o parser."""

    if isinstance(spec, HuggingFaceModelSpec):
        parsed = parse_model_spec(
            {
                "kind": spec.kind,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "result_variant": spec.result_variant,
                "max_sequence_length": spec.max_sequence_length,
            }
        )
    elif isinstance(spec, LocalArtifactModelSpec):
        parsed = parse_model_spec(
            {
                "kind": spec.kind,
                "expected_schema": spec.expected_schema,
                "expected_artifact_sha256": spec.expected_artifact_sha256,
                "max_sequence_length": spec.max_sequence_length,
            }
        )
    else:
        raise ModelConfigurationError("spec do modelo possui tipo inválido")
    if parsed != spec:
        raise ModelConfigurationError("spec do modelo não é canônico")
    return parsed
