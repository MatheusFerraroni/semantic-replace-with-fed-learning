"""Preparação fail-closed do artefato Fórum/Tec produzido pelo Quero-Quero."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_artifact import ValidatedLocalArtifact, sha256_file
from .model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    EXPECTED_MODEL_TYPE,
    EXPECTED_NATIVE_CONTEXT_LENGTH,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKEN_IDS,
    EXPECTED_VOCAB_SIZE,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    QUEROQUERO_ARCHIVE_SHA256,
    QUEROQUERO_ARTIFACT_ID,
    QUEROQUERO_ARTIFACT_SHA256,
    QUEROQUERO_MANIFEST_SHA256,
    QUEROQUERO_TOKENIZER_FILE_FINGERPRINT,
    QUEROQUERO_TOKENIZER_PREPARED_FINGERPRINT,
    QUEROQUERO_WEIGHT_SHA256,
    LocalArtifactModelSpec,
    ModelArtifactError,
    validate_model_spec,
)


QUEROQUERO_EXPECTED_FILES = {
    "config.json": (
        781,
        "d13cfc6eb1ef50b4ecff73657e890f4403428b55dc304ff22703bd0332a2ae20",
    ),
    "generation_config.json": (
        287,
        "c14dbce126c9953cfa69c93517fa89d041f132e9570fe0950992d1d474cc210f",
    ),
    "model.safetensors": (2_680_539_488, QUEROQUERO_WEIGHT_SHA256),
    "tokenizer.json": (
        6_151_542,
        "c417d50d55ea3acd32bb0a14e833adedb411fe6ac6e2b5f64e12c9aef4a2a686",
    ),
    "tokenizer_config.json": (
        667,
        "637d6290e43b9c3ec832212df00a977edb843e0e2b6e81df218d52ba615adeab",
    ),
}


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelArtifactError("manifesto refinado contém chave duplicada")
        result[key] = value
    return result


def _load_manifest_bytes(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != QUEROQUERO_MANIFEST_SHA256:
        raise ModelArtifactError("hash do manifesto refinado é divergente")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except ModelArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError("manifesto refinado não é JSON UTF-8 válido") from error
    if not isinstance(value, dict):
        raise ModelArtifactError("manifesto refinado deve ser um objeto")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _producer_aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in sorted(records, key=lambda item: item["path"])
    ]
    return hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()


def _validate_manifest(manifest: Mapping[str, Any], spec: LocalArtifactModelSpec) -> None:
    expected_top = {
        "architecture",
        "artifact_id",
        "artifact_sha256",
        "environment",
        "files",
        "format",
        "parent_model",
        "redistribution_status",
        "schema_version",
        "tokenizer",
        "training",
    }
    if set(manifest) != expected_top:
        raise ModelArtifactError("estrutura do manifesto refinado é incompatível")
    if (
        manifest.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION
        or manifest.get("artifact_id") != QUEROQUERO_ARTIFACT_ID
        or manifest.get("artifact_sha256") != QUEROQUERO_ARTIFACT_SHA256
        or manifest.get("format") != "transformers_pretrained"
        or manifest.get("redistribution_status") != "internal_research_only"
        or spec.expected_artifact_sha256 != QUEROQUERO_ARTIFACT_SHA256
        or spec.expected_artifact_id != QUEROQUERO_ARTIFACT_ID
        or spec.expected_manifest_sha256 != QUEROQUERO_MANIFEST_SHA256
        or spec.expected_weight_sha256 != QUEROQUERO_WEIGHT_SHA256
        or spec.expected_training_arm != "forum_tech"
    ):
        raise ModelArtifactError("identidade do artefato refinado é incompatível")
    if manifest.get("parent_model") != {
        "license": "Apache-2.0",
        "model_id": BASE_MODEL_ID,
        "revision": BASE_MODEL_REVISION,
    }:
        raise ModelArtifactError("modelo pai do artefato refinado é incompatível")
    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping) or dict(architecture) != {
        "model_type": EXPECTED_MODEL_TYPE,
        "native_context_length": EXPECTED_NATIVE_CONTEXT_LENGTH,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "training_sequence_length": 1_024,
        "weights_dtype": "float32",
    }:
        raise ModelArtifactError("arquitetura do artefato refinado é incompatível")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or any(
            tokenizer.get(key) != value for key, value in EXPECTED_TOKEN_IDS.items()
        )
        or tokenizer.get("vocab_size") != EXPECTED_VOCAB_SIZE
        or tokenizer.get("model_id") != BASE_MODEL_ID
        or tokenizer.get("revision") != BASE_MODEL_REVISION
        or tokenizer.get("fingerprint_sha256")
        != QUEROQUERO_TOKENIZER_FILE_FINGERPRINT
        or tokenizer.get("prepared_fingerprint_sha256")
        != QUEROQUERO_TOKENIZER_PREPARED_FINGERPRINT
    ):
        raise ModelArtifactError("tokenizador declarado pelo artefato é incompatível")
    training = manifest.get("training")
    if not isinstance(training, Mapping) or (
        training.get("method") != "full_parameter_continual_pretraining"
        or training.get("profile") != "real"
        or training.get("seed") != 42
        or training.get("optimizer_steps") != 52_000
        or not isinstance(training.get("experiment"), Mapping)
        or training["experiment"].get("arm") != "forum_tech"
        or not isinstance(training.get("data_mixture"), Mapping)
        or training["data_mixture"].get("arm") != "forum_tech"
    ):
        raise ModelArtifactError("proveniência de treinamento é incompatível")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ModelArtifactError("lista de arquivos do artefato é inválida")
    expected_records = [
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, (size, digest) in sorted(QUEROQUERO_EXPECTED_FILES.items())
    ]
    if records != expected_records or _producer_aggregate(records) != QUEROQUERO_ARTIFACT_SHA256:
        raise ModelArtifactError("inventário do artefato refinado é incompatível")


def validate_queroquero_artifact_directory(
    spec: LocalArtifactModelSpec, directory: Path
) -> ValidatedLocalArtifact:
    validated_spec = validate_model_spec(spec)
    root = Path(directory)
    if not root.is_absolute() or ".." in root.parts:
        raise ModelArtifactError("diretório refinado deve ser absoluto e sem travessia")
    try:
        root_mode = root.lstat().st_mode
    except OSError as error:
        raise ModelArtifactError("diretório refinado está ausente") from error
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ModelArtifactError("diretório refinado não pode ser link simbólico")
    actual: set[str] = set()
    for item in root.iterdir():
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ModelArtifactError("artefato refinado contém entrada inválida")
        actual.add(item.name)
    expected = set(QUEROQUERO_EXPECTED_FILES) | {"model_artifact_manifest.json"}
    if actual != expected:
        raise ModelArtifactError("arquivos reais do artefato refinado divergem")
    try:
        raw_manifest = (root / "model_artifact_manifest.json").read_bytes()
    except OSError as error:
        raise ModelArtifactError("manifesto refinado não pode ser lido") from error
    manifest = _load_manifest_bytes(raw_manifest)
    _validate_manifest(manifest, validated_spec)
    for path, (expected_size, expected_hash) in QUEROQUERO_EXPECTED_FILES.items():
        candidate = root / path
        if candidate.stat().st_size != expected_size or sha256_file(candidate) != expected_hash:
            raise ModelArtifactError("arquivo declarado do artefato refinado diverge")
    return ValidatedLocalArtifact(directory=root, manifest=manifest)


def prepare_queroquero_artifact_archive(
    spec: LocalArtifactModelSpec,
    archive: Path,
    output_root: Path = Path("artifacts/models"),
) -> ValidatedLocalArtifact:
    """Valida e publica o ZIP fixado sem deixar diretório parcial."""

    validated_spec = validate_model_spec(spec)
    source = Path(archive)
    if not source.is_absolute() or ".." in source.parts:
        raise ModelArtifactError("arquivo ZIP deve ser absoluto e sem travessia")
    try:
        mode = source.lstat().st_mode
    except OSError as error:
        raise ModelArtifactError("arquivo ZIP refinado está ausente") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ModelArtifactError("arquivo ZIP refinado deve ser regular")
    if sha256_file(source) != QUEROQUERO_ARCHIVE_SHA256:
        raise ModelArtifactError("hash do ZIP refinado é divergente")
    requested_root = Path(output_root)
    if requested_root.exists() and requested_root.is_symlink():
        raise ModelArtifactError("raiz de destino do artefato refinado é inválida")
    destination_root = requested_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise ModelArtifactError("raiz de destino do artefato refinado é inválida")
    destination = destination_root / QUEROQUERO_ARTIFACT_ID
    if destination.exists():
        raise ModelArtifactError("destino do artefato refinado já existe")
    staging = Path(tempfile.mkdtemp(prefix=".refined-partial-", dir=destination_root))
    published = staging / QUEROQUERO_ARTIFACT_ID
    published.mkdir(mode=0o700)
    expected_members = {
        f"{QUEROQUERO_ARTIFACT_ID}/",
        f"{QUEROQUERO_ARTIFACT_ID}/model_artifact_manifest.json",
        *(f"{QUEROQUERO_ARTIFACT_ID}/{name}" for name in QUEROQUERO_EXPECTED_FILES),
    }
    try:
        with zipfile.ZipFile(source) as zipped:
            infos = zipped.infolist()
            if {item.filename for item in infos} != expected_members:
                raise ModelArtifactError("inventário do ZIP refinado é incompatível")
            for info in infos:
                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == stat.S_IFLNK:
                    raise ModelArtifactError("ZIP refinado contém link simbólico")
                if info.is_dir():
                    continue
                relative = info.filename.split("/", 1)[1]
                target = published / relative
                digest = hashlib.sha256()
                size = 0
                with zipped.open(info) as source_file, target.open("xb") as target_file:
                    while True:
                        chunk = source_file.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                        target_file.write(chunk)
                os.chmod(target, 0o600)
                if relative == "model_artifact_manifest.json":
                    expected_size = info.file_size
                    expected_hash = QUEROQUERO_MANIFEST_SHA256
                else:
                    expected_size, expected_hash = QUEROQUERO_EXPECTED_FILES[relative]
                if size != expected_size or digest.hexdigest() != expected_hash:
                    raise ModelArtifactError("conteúdo extraído do ZIP refinado diverge")
        validated = validate_queroquero_artifact_directory(validated_spec, published)
        published.replace(destination)
        staging.rmdir()
        return ValidatedLocalArtifact(directory=destination, manifest=validated.manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "QUEROQUERO_EXPECTED_FILES",
    "prepare_queroquero_artifact_archive",
    "validate_queroquero_artifact_directory",
]
