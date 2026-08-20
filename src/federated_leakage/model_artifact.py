"""Validação fail-closed do artefato local do Tucano 2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Tuple

from .model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    EXPECTED_MODEL_TYPE,
    EXPECTED_NATIVE_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_FILES,
    EXPECTED_TOKENIZER_FINGERPRINT,
    EXPECTED_TOKEN_IDS,
    EXPECTED_VOCAB_SIZE,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    TRAINING_SEQUENCE_LENGTH,
    LocalArtifactModelSpec,
    ModelArtifactError,
    ModelLoadError,
    validate_model_spec,
)


_SAFE_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ValidatedLocalArtifact:
    """Artefato cuja estrutura e conteúdo já foram validados."""

    directory: Path
    manifest: Mapping[str, Any]


def _strict_json_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelArtifactError("manifesto contém chave JSON duplicada")
        result[key] = value
    return result


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ModelArtifactError("manifesto do modelo ausente ou ilegível") from error
    try:
        value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except ModelArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError("manifesto do modelo não é JSON UTF-8 válido") from error
    if not isinstance(value, dict):
        raise ModelArtifactError("manifesto do modelo deve ser um objeto JSON")
    return value


def _schema_path() -> Path:
    return Path(__file__).with_name("schemas") / "model-artifact-v1.schema.json"


def _validate_manifest_schema(manifest: Mapping[str, Any], jsonschema: Any) -> None:
    try:
        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        errors = tuple(validator.iter_errors(manifest))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelLoadError("schema interno do artefato está indisponível") from error
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path)
        suffix = f" no campo {location}" if location else ""
        raise ModelArtifactError(f"manifesto viola o schema v1{suffix}")


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ModelArtifactError("manifesto contém caminho de arquivo inválido")
    if any(character in value for character in ("\\", "\t", "\r", "\n")):
        raise ModelArtifactError("manifesto contém caminho de arquivo inválido")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
        raise ModelArtifactError("manifesto contém caminho de arquivo inválido")
    if pure_path.as_posix() != value:
        raise ModelArtifactError("manifesto contém caminho de arquivo inválido")
    return value


def _artifact_files(directory: Path) -> Tuple[str, ...]:
    result = []
    for root, directory_names, file_names in os.walk(directory, followlinks=False):
        root_path = Path(root)
        for name in tuple(directory_names):
            candidate = root_path / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as error:
                raise ModelArtifactError("artefato local não pode ser inspecionado") from error
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ModelArtifactError("artefato local contém link ou entrada inválida")
        for name in file_names:
            candidate = root_path / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as error:
                raise ModelArtifactError("artefato local não pode ser inspecionado") from error
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ModelArtifactError("artefato local contém link ou entrada inválida")
            relative = candidate.relative_to(directory).as_posix()
            if relative != "model_artifact_manifest.json":
                result.append(relative)
    return tuple(sorted(result))


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 em blocos sem carregar pesos inteiros em memória."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ModelArtifactError("arquivo declarado não pode ser lido") from error
    return digest.hexdigest()


def tokenizer_fingerprint(
    directory: Path,
    tokenizer_files: Sequence[str] = EXPECTED_TOKENIZER_FILES,
) -> str:
    """Reproduz a serialização do fingerprint definida no contrato v1."""

    digest = hashlib.sha256()
    for relative_path in tokenizer_files:
        path = directory / relative_path
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            raise ModelArtifactError("arquivo do tokenizador ausente") from error
        file_digest = sha256_file(path)
        digest.update(
            f"{file_digest}\t{size_bytes}\t{relative_path}\n".encode("utf-8")
        )
    digest.update(b"TOKEN_IDS\t49152\t1\t2\t49109\t0\n")
    return digest.hexdigest()


def _validate_weight_layout(file_paths: Sequence[str]) -> None:
    names = frozenset(file_paths)
    forbidden_suffixes = (".bin", ".ckpt", ".pt", ".pth")
    if any(path.endswith(forbidden_suffixes) for path in names):
        raise ModelArtifactError("artefato contém formato de peso proibido")
    if any(PurePosixPath(path).name.startswith("adapter_") for path in names):
        raise ModelArtifactError("artefato contém adaptador em vez do modelo completo")

    has_single = "model.safetensors" in names
    has_index = "model.safetensors.index.json" in names
    shards = tuple(
        sorted(
            path
            for path in names
            if PurePosixPath(path).name.startswith("model-")
            and path.endswith(".safetensors")
        )
    )
    if has_single:
        if has_index or shards:
            raise ModelArtifactError("layout de pesos safetensors é ambíguo")
        return
    if not has_index or not shards:
        raise ModelArtifactError("pesos safetensors completos estão ausentes")


def _validate_shard_index(directory: Path, file_paths: Sequence[str]) -> None:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.exists():
        return
    try:
        index = json.loads(index_path.read_bytes(), object_pairs_hook=_strict_json_object)
    except ModelArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError("índice de shards não é JSON válido") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ModelArtifactError("índice de shards não contém weight_map")
    referenced = {
        _validate_relative_path(value) for value in index["weight_map"].values()
    }
    declared_shards = {
        path
        for path in file_paths
        if PurePosixPath(path).name.startswith("model-")
        and path.endswith(".safetensors")
    }
    if referenced != declared_shards:
        raise ModelArtifactError("índice e shards safetensors divergem")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest["schema_version"] != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError("versão do manifesto do modelo é desconhecida")
    if manifest["format"] != "transformers_pretrained":
        raise ModelArtifactError("formato do artefato não é transformers_pretrained")
    if not _SAFE_ARTIFACT_ID_PATTERN.fullmatch(manifest["artifact_id"]):
        raise ModelArtifactError("artifact_id é inválido")

    parent = manifest["parent_model"]
    if (
        parent["model_id"] != BASE_MODEL_ID
        or parent["revision"] != BASE_MODEL_REVISION
        or parent["license"] != "Apache-2.0"
    ):
        raise ModelArtifactError("modelo pai do artefato é incompatível")

    architecture = manifest["architecture"]
    expected_architecture = {
        "model_type": EXPECTED_MODEL_TYPE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "native_context_length": EXPECTED_NATIVE_CONTEXT_LENGTH,
        "training_sequence_length": TRAINING_SEQUENCE_LENGTH,
    }
    if any(architecture[key] != value for key, value in expected_architecture.items()):
        raise ModelArtifactError("arquitetura declarada é incompatível")

    tokenizer = manifest["tokenizer"]
    if tuple(tokenizer["files"]) != EXPECTED_TOKENIZER_FILES:
        raise ModelArtifactError("arquivos declarados do tokenizador são incompatíveis")
    if tokenizer["fingerprint_sha256"] != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ModelArtifactError("fingerprint declarado do tokenizador é incompatível")
    if tokenizer["vocab_size"] != EXPECTED_VOCAB_SIZE:
        raise ModelArtifactError("vocabulário declarado é incompatível")
    if any(tokenizer[key] != value for key, value in EXPECTED_TOKEN_IDS.items()):
        raise ModelArtifactError("IDs especiais declarados são incompatíveis")


def validate_local_artifact(
    spec: LocalArtifactModelSpec,
    artifact_directory: Path,
    *,
    jsonschema: Any,
    fingerprint_function=tokenizer_fingerprint,
) -> ValidatedLocalArtifact:
    """Valida integralmente um artefato local antes de carregar seu conteúdo."""

    validate_model_spec(spec)
    directory = Path(artifact_directory)
    if not directory.is_absolute() or ".." in directory.parts:
        raise ModelArtifactError("diretório do artefato deve ser absoluto e sem travessia")
    try:
        root_mode = directory.lstat().st_mode
    except OSError as error:
        raise ModelArtifactError("diretório do artefato está ausente") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ModelArtifactError("diretório do artefato não pode ser link simbólico")

    manifest = _load_manifest(directory / "model_artifact_manifest.json")
    _validate_manifest_schema(manifest, jsonschema)
    _validate_manifest_contract(manifest)

    entries = manifest["files"]
    declared_paths = tuple(_validate_relative_path(entry["path"]) for entry in entries)
    if declared_paths != tuple(sorted(declared_paths)) or len(set(declared_paths)) != len(
        declared_paths
    ):
        raise ModelArtifactError("lista de arquivos deve ser única e ordenada")
    actual_paths = _artifact_files(directory)
    if declared_paths != actual_paths:
        raise ModelArtifactError("arquivos reais e declarados no manifesto divergem")

    aggregate = hashlib.sha256()
    for entry in entries:
        relative_path = entry["path"]
        path = directory / relative_path
        try:
            size_bytes = path.stat().st_size
        except OSError as error:
            raise ModelArtifactError("arquivo declarado está ausente") from error
        if size_bytes != entry["size_bytes"]:
            raise ModelArtifactError(f"tamanho divergente em {relative_path}")
        file_digest = sha256_file(path)
        if file_digest != entry["sha256"]:
            raise ModelArtifactError(f"hash divergente em {relative_path}")
        aggregate.update(
            f"{file_digest}\t{size_bytes}\t{relative_path}\n".encode("utf-8")
        )
    aggregate_digest = aggregate.hexdigest()
    if aggregate_digest != manifest["artifact_sha256"]:
        raise ModelArtifactError("hash agregado do artefato é divergente")
    if aggregate_digest != spec.expected_artifact_sha256:
        raise ModelArtifactError("artefato não corresponde ao hash esperado")

    required_files = {
        "added_tokens.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required_files.issubset(actual_paths):
        raise ModelArtifactError("artefato não contém os arquivos Hugging Face obrigatórios")
    _validate_weight_layout(actual_paths)
    _validate_shard_index(directory, actual_paths)
    if fingerprint_function(directory) != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ModelArtifactError("arquivos locais do tokenizador são incompatíveis")
    return ValidatedLocalArtifact(directory=directory, manifest=manifest)
