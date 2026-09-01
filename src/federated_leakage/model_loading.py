"""Preparação explícita e carga sempre offline do Tucano 2 0.6B."""

from __future__ import annotations

import importlib.metadata
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .configuration import ConfigurationError, load_yaml_mapping
from .model_artifact import (
    tokenizer_fingerprint as _tokenizer_fingerprint,
    validate_local_artifact as _validate_local_artifact,
)
from .model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    DEFAULT_MODEL_CACHE,
    EXPECTED_ARCHITECTURE,
    EXPECTED_ATTENTION_HEADS,
    EXPECTED_HIDDEN_LAYERS,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_INTERMEDIATE_SIZE,
    EXPECTED_KEY_VALUE_HEADS,
    EXPECTED_MODEL_TYPE,
    EXPECTED_NATIVE_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_FILES,
    EXPECTED_TOKENIZER_FINGERPRINT,
    EXPECTED_TOKEN_IDS,
    EXPECTED_TOKEN_TEXT,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_WEIGHT_DTYPE,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    MODEL_LOADING_SCHEMA_VERSION,
    QUEROQUERO_ARTIFACT_CONTRACT_PROFILE,
    SNAPSHOT_ALLOW_PATTERNS,
    HuggingFaceModelSpec,
    LoadedModelBundle,
    LocalArtifactModelSpec,
    ModelArtifactError,
    ModelConfigurationError,
    ModelDependencyError,
    ModelLoadError,
    ModelProvenance,
    ModelSpec,
    parse_model_spec,
    validate_model_spec,
)
from .queroquero_artifact import validate_queroquero_artifact_directory
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
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


@dataclass(frozen=True, slots=True)
class _ModelDependencies:
    torch: Any
    transformers: Any
    tokenizers: Any
    snapshot_download: Any
    jsonschema: Any


def _load_model_dependencies() -> _ModelDependencies:
    try:
        import jsonschema
        import tokenizers
        import torch
        import transformers
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelDependencyError(
            "dependências de modelo ausentes; instale o projeto com .[model]"
        ) from error
    return _ModelDependencies(
        torch=torch,
        transformers=transformers,
        tokenizers=tokenizers,
        snapshot_download=snapshot_download,
        jsonschema=jsonschema,
    )


def validate_local_artifact(
    spec: LocalArtifactModelSpec,
    artifact_directory: Path,
    *,
    dependencies: _ModelDependencies | None = None,
):
    """Expõe a validação local usando o conjunto opcional de dependências."""

    resolved_dependencies = dependencies or _load_model_dependencies()
    if spec.contract_profile == QUEROQUERO_ARTIFACT_CONTRACT_PROFILE:
        return validate_queroquero_artifact_directory(spec, artifact_directory)
    return _validate_local_artifact(
        spec,
        artifact_directory,
        jsonschema=resolved_dependencies.jsonschema,
        fingerprint_function=_tokenizer_fingerprint,
    )


def _validate_config(
    config: Any,
    *,
    expected_declared_dtype: str = EXPECTED_WEIGHT_DTYPE,
    allowed_use_cache: tuple[bool, ...] = (False,),
) -> None:
    expected = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "model_type": EXPECTED_MODEL_TYPE,
        "head_dim": 96,
        "hidden_act": "silu",
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
        "mlp_bias": False,
        "num_hidden_layers": EXPECTED_HIDDEN_LAYERS,
        "num_attention_heads": EXPECTED_ATTENTION_HEADS,
        "num_key_value_heads": EXPECTED_KEY_VALUE_HEADS,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 50_000.0,
        "tie_word_embeddings": True,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "max_position_embeddings": EXPECTED_NATIVE_CONTEXT_LENGTH,
        "bos_token_id": EXPECTED_TOKEN_IDS["bos_token_id"],
        "eos_token_id": EXPECTED_TOKEN_IDS["eos_token_id"],
        "pad_token_id": EXPECTED_TOKEN_IDS["pad_token_id"],
    }
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise ModelLoadError("configuração carregada do modelo é incompatível")
    if getattr(config, "use_cache", None) not in allowed_use_cache:
        raise ModelLoadError("política de cache causal declarada é incompatível")
    if tuple(getattr(config, "architectures", ())) != (EXPECTED_ARCHITECTURE,):
        raise ModelLoadError("classe causal declarada é incompatível")
    configured_dtype = str(getattr(config, "torch_dtype", "")).replace("torch.", "")
    if configured_dtype != expected_declared_dtype:
        raise ModelLoadError("dtype declarado do modelo é incompatível")


def _validate_tokenizer(tokenizer: Any) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise ModelLoadError("tokenizador carregado não é fast")
    try:
        tokenizer_size = len(tokenizer)
    except TypeError as error:
        raise ModelLoadError("vocabulário do tokenizador é inválido") from error
    if tokenizer_size != EXPECTED_VOCAB_SIZE:
        raise ModelLoadError("vocabulário carregado é incompatível")
    if any(getattr(tokenizer, key, None) != value for key, value in EXPECTED_TOKEN_IDS.items()):
        raise ModelLoadError("IDs especiais carregados são incompatíveis")
    if any(getattr(tokenizer, key, None) != value for key, value in EXPECTED_TOKEN_TEXT.items()):
        raise ModelLoadError("tokens especiais carregados são incompatíveis")
    try:
        special_wrapped = tokenizer.build_inputs_with_special_tokens([10, 11])
    except Exception as error:
        raise ModelLoadError("comportamento de tokens especiais é inválido") from error
    if list(special_wrapped) != [10, 11]:
        raise ModelLoadError("tokenizador adicionaria BOS ou EOS automaticamente")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ModelLoadError("lado de padding do tokenizador é incompatível")
    if getattr(tokenizer, "model_max_length", 0) < 1_024:
        raise ModelLoadError("contexto do tokenizador é menor que 1024")


def _backend_json(tokenizer: Any) -> Any:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if not callable(serializer):
        raise ModelLoadError("backend do tokenizador fast está ausente")
    try:
        return json.loads(serializer())
    except Exception as error:
        raise ModelLoadError("backend do tokenizador não pode ser normalizado") from error


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ModelLoadError(
                "configuração do tokenizador refinado possui chave duplicada"
            )
        value[key] = item
    return value


def _validate_refined_tokenizer_config(artifact_directory: Path) -> None:
    path = Path(artifact_directory) / "tokenizer_config.json"
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ModelLoadError("configuração do tokenizador refinado é inválida")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(
            "configuração do tokenizador refinado não pode ser validada"
        ) from error
    expected = {
        "backend": "tokenizers",
        "tokenizer_class": "TokenizersBackend",
        "bos_token": EXPECTED_TOKEN_TEXT["bos_token"],
        "eos_token": EXPECTED_TOKEN_TEXT["eos_token"],
        "pad_token": EXPECTED_TOKEN_TEXT["pad_token"],
        "unk_token": EXPECTED_TOKEN_TEXT["unk_token"],
        "bos_token_id": EXPECTED_TOKEN_IDS["bos_token_id"],
        "eos_token_id": EXPECTED_TOKEN_IDS["eos_token_id"],
        "pad_token_id": EXPECTED_TOKEN_IDS["pad_token_id"],
        "unk_token_id": EXPECTED_TOKEN_IDS["unk_token_id"],
        "model_max_length": EXPECTED_NATIVE_CONTEXT_LENGTH,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": False,
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ModelLoadError(
            "configuração semântica do tokenizador refinado diverge"
        )


def _load_raw_tokenizer_backend(
    directory: Path,
    dependencies: _ModelDependencies,
    *,
    source: str,
) -> Any:
    if source not in {"refined", "upstream"}:
        raise ValueError("origem interna do tokenizador é inválida")
    source_label = "refinado" if source == "refined" else "upstream pinado"
    path = Path(directory) / "tokenizer.json"
    try:
        mode = path.lstat().st_mode if source == "refined" else path.stat().st_mode
        if (source == "refined" and stat.S_ISLNK(mode)) or not stat.S_ISREG(mode):
            raise ModelLoadError(
                f"backend bruto do tokenizador {source_label} é inválido"
            )
        backend = dependencies.tokenizers.Tokenizer.from_file(str(path))
        json.loads(backend.to_str())
        return backend
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(
            f"backend bruto do tokenizador {source_label} é incompatível "
            "com o runtime fixado"
        ) from error


def _validate_refined_tokenizer_equivalence(
    artifact_directory: Path,
    reference_directory: Path,
    dependencies: _ModelDependencies,
) -> Any:
    arguments = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
    }
    try:
        reference_fingerprint = _tokenizer_fingerprint(reference_directory)
    except ModelArtifactError as error:
        raise ModelLoadError(
            "tokenizador upstream pinado não pode ser validado offline"
        ) from error
    if reference_fingerprint != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ModelLoadError("tokenizador upstream pinado em cache é incompatível")
    _validate_refined_tokenizer_config(artifact_directory)
    artifact = _load_raw_tokenizer_backend(
        artifact_directory,
        dependencies,
        source="refined",
    )
    reference_raw = _load_raw_tokenizer_backend(
        reference_directory,
        dependencies,
        source="upstream",
    )
    try:
        reference = dependencies.transformers.AutoTokenizer.from_pretrained(
            reference_directory, **arguments
        )
    except Exception as error:
        raise ModelLoadError(
            "tokenizador upstream pinado não pode ser carregado offline"
        ) from error
    _validate_tokenizer(reference)
    probes = (
        "Olá, mundo!",
        "AÇÃO e informação em português.",
        "PERSON_NAME: Pessoa Sintética\nCPF: 000.000.000-00",
        "09:15 2026-12-31 endereço@example.com",
    )
    try:
        artifact_vocab = artifact.get_vocab(with_added_tokens=True)
        reference_raw_vocab = reference_raw.get_vocab(with_added_tokens=True)
        artifact_backend = json.loads(artifact.to_str())
        reference_backend = _backend_json(reference)
        for value in probes:
            artifact_ids = artifact.encode(value, add_special_tokens=False).ids
            reference_raw_ids = reference_raw.encode(
                value,
                add_special_tokens=False,
            ).ids
            reference_ids = reference.encode(value, add_special_tokens=False)
            if artifact_ids != reference_raw_ids or artifact_ids != reference_ids:
                raise ModelLoadError(
                    "codificação do tokenizador refinado diverge do upstream"
                )
            artifact_text = artifact.decode(
                artifact_ids,
                skip_special_tokens=False,
            )
            reference_text = reference.decode(
                reference_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            reference_raw_text = reference_raw.decode(
                reference_raw_ids,
                skip_special_tokens=False,
            )
            if artifact_text != reference_raw_text or artifact_text != reference_text:
                raise ModelLoadError(
                    "decodificação do tokenizador refinado diverge do upstream"
                )
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(
            "backend do tokenizador refinado não pôde ser comparado"
        ) from error
    if artifact_vocab != reference_raw_vocab or artifact_vocab != reference.get_vocab():
        raise ModelLoadError("vocabulário do tokenizador refinado diverge do upstream")
    if artifact_backend != reference_backend:
        raise ModelLoadError("backend do tokenizador refinado diverge do upstream")
    return reference


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("dispositivo cuda solicitado não está disponível")
        return torch.device("cuda")
    if requested == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None or not backend.is_available():
            raise ModelLoadError("dispositivo mps solicitado não está disponível")
        return torch.device("mps")
    raise ModelLoadError("dispositivo deve ser cpu, cuda ou mps")


def _validate_loaded_model(model: Any, torch: Any) -> None:
    if model.__class__.__name__ != EXPECTED_ARCHITECTURE:
        raise ModelLoadError("modelo carregado não é LlamaForCausalLM")
    parameters = tuple(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != EXPECTED_PARAMETER_COUNT:
        raise ModelLoadError("contagem de parâmetros carregada é incompatível")
    if any(parameter.dtype != torch.bfloat16 for parameter in parameters):
        raise ModelLoadError("parâmetros carregados não estão em bfloat16")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ModelLoadError("nem todos os parâmetros podem ser treinados")
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if getattr(input_embeddings, "num_embeddings", None) != EXPECTED_VOCAB_SIZE:
        raise ModelLoadError("embedding de entrada é incompatível")
    output_weight = getattr(output_embeddings, "weight", None)
    if output_weight is None or output_weight.shape[0] != EXPECTED_VOCAB_SIZE:
        raise ModelLoadError("cabeça de saída é incompatível")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_pretrained_directory(
    directory: Path,
    *,
    spec: ModelSpec,
    device: str,
    tokenizer_fingerprint: str,
    artifact_manifest: Mapping[str, Any] | None,
    dependencies: _ModelDependencies,
    prevalidated_tokenizer: Any | None = None,
) -> LoadedModelBundle:
    torch = dependencies.torch
    transformers = dependencies.transformers
    resolved_device = _resolve_device(torch, device)
    common_arguments = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    try:
        config = transformers.AutoConfig.from_pretrained(directory, **common_arguments)
        refined_profile = (
            isinstance(spec, LocalArtifactModelSpec)
            and spec.contract_profile == QUEROQUERO_ARTIFACT_CONTRACT_PROFILE
        )
        _validate_config(
            config,
            expected_declared_dtype=(
                "float32" if refined_profile else EXPECTED_WEIGHT_DTYPE
            ),
            allowed_use_cache=(True,) if refined_profile else (False,),
        )
        if refined_profile:
            config.use_cache = False
        tokenizer = prevalidated_tokenizer
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                directory,
                use_fast=True,
                **common_arguments,
            )
        _validate_tokenizer(tokenizer)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            directory,
            config=config,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            **common_arguments,
        )
        model.to(resolved_device)
    except (ModelLoadError, ModelArtifactError):
        raise
    except Exception as error:
        raise ModelLoadError("falha ao carregar o modelo validado") from error
    _validate_loaded_model(model, torch)

    if isinstance(spec, HuggingFaceModelSpec):
        source_identifier = spec.model_id
        revision = spec.revision
        artifact_sha256 = None
        result_variant = spec.result_variant
    else:
        if artifact_manifest is None:
            raise ModelLoadError("manifesto validado do artefato está ausente")
        source_identifier = artifact_manifest["artifact_id"]
        revision = None
        artifact_sha256 = spec.expected_artifact_sha256
        result_variant = f"local-artifact-sha256-{artifact_sha256}"

    provenance = ModelProvenance(
        schema_version=MODEL_LOADING_SCHEMA_VERSION,
        source_kind=spec.kind,
        source_identifier=source_identifier,
        revision=revision,
        artifact_sha256=artifact_sha256,
        result_variant=result_variant,
        architecture=EXPECTED_ARCHITECTURE,
        parameter_count=EXPECTED_PARAMETER_COUNT,
        native_context_length=EXPECTED_NATIVE_CONTEXT_LENGTH,
        training_sequence_length=spec.max_sequence_length,
        vocab_size=EXPECTED_VOCAB_SIZE,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint,
        weight_dtype=EXPECTED_WEIGHT_DTYPE,
        device=str(resolved_device),
        torch_version=_package_version("torch"),
        transformers_version=_package_version("transformers"),
        tokenizers_version=_package_version("tokenizers"),
        safetensors_version=_package_version("safetensors"),
        huggingface_hub_version=_package_version("huggingface-hub"),
    )
    return LoadedModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_sequence_length=spec.max_sequence_length,
        provenance=provenance,
    )


def _resolve_cached_snapshot(
    spec: HuggingFaceModelSpec,
    cache_dir: Path,
    dependencies: _ModelDependencies,
) -> Path:
    try:
        snapshot = dependencies.snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            cache_dir=str(cache_dir),
            allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
            local_files_only=True,
        )
    except Exception as error:
        raise ModelLoadError(
            "snapshot pinado ausente; execute prepare_model sem --offline"
        ) from error
    snapshot_path = Path(snapshot)
    if snapshot_path.name != spec.revision:
        raise ModelLoadError("cache não resolveu para a revisão fixada")
    return snapshot_path


def prepare_huggingface_model(
    spec: HuggingFaceModelSpec,
    *,
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    device: str = "cpu",
    dependencies: _ModelDependencies | None = None,
) -> LoadedModelBundle:
    """Baixa explicitamente o snapshot pinado e valida a carga offline."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise ModelLoadError(str(error)) from error
    validated_spec = validate_model_spec(spec)
    if not isinstance(validated_spec, HuggingFaceModelSpec):
        raise ModelConfigurationError("prepare_huggingface_model exige origem pública")
    resolved_dependencies = dependencies or _load_model_dependencies()
    try:
        snapshot = resolved_dependencies.snapshot_download(
            repo_id=validated_spec.model_id,
            revision=validated_spec.revision,
            cache_dir=str(Path(cache_dir)),
            allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
            local_files_only=False,
        )
    except Exception as error:
        raise ModelLoadError("falha na preparação explícita do snapshot pinado") from error
    snapshot_path = Path(snapshot)
    if snapshot_path.name != validated_spec.revision:
        raise ModelLoadError("download não resolveu para a revisão fixada")
    fingerprint = _tokenizer_fingerprint(snapshot_path)
    if fingerprint != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ModelLoadError("tokenizador baixado não corresponde à revisão fixada")
    return _load_pretrained_directory(
        snapshot_path,
        spec=validated_spec,
        device=device,
        tokenizer_fingerprint=fingerprint,
        artifact_manifest=None,
        dependencies=resolved_dependencies,
    )


def load_model_bundle(
    spec: ModelSpec,
    *,
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    model_artifact_dir: Path | None = None,
    device: str = "cpu",
    dependencies: _ModelDependencies | None = None,
) -> LoadedModelBundle:
    """Carrega somente de cache ou diretório local, sem permitir rede."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise ModelLoadError(str(error)) from error
    validated_spec = validate_model_spec(spec)
    resolved_dependencies = dependencies or _load_model_dependencies()
    if isinstance(validated_spec, HuggingFaceModelSpec):
        if model_artifact_dir is not None:
            raise ModelConfigurationError(
                "model_artifact_dir não é permitido no modo huggingface"
            )
        snapshot_path = _resolve_cached_snapshot(
            validated_spec,
            Path(cache_dir),
            resolved_dependencies,
        )
        fingerprint = _tokenizer_fingerprint(snapshot_path)
        if fingerprint != EXPECTED_TOKENIZER_FINGERPRINT:
            raise ModelLoadError("tokenizador em cache é incompatível")
        return _load_pretrained_directory(
            snapshot_path,
            spec=validated_spec,
            device=device,
            tokenizer_fingerprint=fingerprint,
            artifact_manifest=None,
            dependencies=resolved_dependencies,
        )

    if model_artifact_dir is None:
        raise ModelConfigurationError(
            "model_artifact_dir é obrigatório no modo local_artifact"
        )
    validated = validate_local_artifact(
        validated_spec,
        model_artifact_dir,
        dependencies=resolved_dependencies,
    )
    refined_tokenizer = None
    if validated_spec.contract_profile == QUEROQUERO_ARTIFACT_CONTRACT_PROFILE:
        reference_spec = HuggingFaceModelSpec(
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            result_variant=BASE_RESULT_VARIANT,
            max_sequence_length=validated_spec.max_sequence_length,
        )
        reference = _resolve_cached_snapshot(
            reference_spec, Path(cache_dir), resolved_dependencies
        )
        refined_tokenizer = _validate_refined_tokenizer_equivalence(
            validated.directory, reference, resolved_dependencies
        )
    return _load_pretrained_directory(
        validated.directory,
        spec=validated_spec,
        device=device,
        tokenizer_fingerprint=EXPECTED_TOKENIZER_FINGERPRINT,
        artifact_manifest=validated.manifest,
        dependencies=resolved_dependencies,
        prevalidated_tokenizer=refined_tokenizer,
    )


__all__ = [
    "BASE_MODEL_ID",
    "BASE_MODEL_REVISION",
    "BASE_RESULT_VARIANT",
    "DEFAULT_MODEL_CACHE",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_TOKENIZER_FINGERPRINT",
    "HuggingFaceModelSpec",
    "LoadedModelBundle",
    "LocalArtifactModelSpec",
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "MODEL_LOADING_SCHEMA_VERSION",
    "ModelArtifactError",
    "ModelConfigurationError",
    "ModelDependencyError",
    "ModelLoadError",
    "ModelProvenance",
    "ModelSpec",
    "load_model_bundle",
    "load_model_spec_from_config",
    "parse_model_spec",
    "prepare_huggingface_model",
    "validate_local_artifact",
]
