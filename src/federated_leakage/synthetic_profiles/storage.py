"""Persistência local, estrita e atômica de conversas sintéticas validadas."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .manifest import build_victim_dataset_manifest
from .model import (
    AUXILIARY_ROUND_SCHEMA_VERSION,
    CONVERSATION_GENERATOR_VERSION,
    CONVERSATION_SCHEMA_VERSION,
    GENERATOR_VERSION,
    VICTIM_DATASET_SCHEMA_VERSION,
    AuxiliaryPresentation,
    AuxiliaryRound,
    FieldAnnotation,
    TrainingConversation,
    VictimClientDataset,
)
from .validation import (
    validate_auxiliary_round,
    validate_training_conversation,
    validate_victim_dataset,
)


ARTIFACT_METADATA_SCHEMA_VERSION = "conversation-artifact-metadata/v1"
CONVERSATION_JSONL_SCHEMA_VERSION = "training-conversation-jsonl/v1"

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONVERSATION_KEYS = frozenset(
    {
        "schema_version",
        "text",
        "entity_id",
        "client_id",
        "round_id",
        "sample_index",
        "kind",
        "template_id",
        "annotations",
        "prefix_length",
        "loss_scope",
    }
)
_ANNOTATION_KEYS = frozenset(
    {"entity_id", "field_type", "start", "end", "value"}
)
_COMMON_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "jsonl_schema_version",
        "container_schema_version",
        "conversation_schema_version",
        "profile_generator_version",
        "conversation_generator_version",
        "dataset_id",
        "role",
        "client_id",
        "record_count",
        "content_sha256",
    }
)
_VICTIM_METADATA_KEYS = _COMMON_METADATA_KEYS
_AUXILIARY_METADATA_KEYS = _COMMON_METADATA_KEYS | frozenset(
    {"schedule_id", "presentation", "round_id"}
)


class DatasetStorageError(ValueError):
    """Falha fechada de serialização ou leitura sem conteúdo protegido."""


def _validate_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise DatasetStorageError(f"{label} inválido")
    return value


def validate_storage_component(value: str, label: str) -> str:
    """Valida um componente de caminho sem criar diretórios ou arquivos."""

    return _validate_component(value, label)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetStorageError("JSON contém chave duplicada")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes) -> Dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetStorageError("JSON inválido") from exc
    if not isinstance(value, dict):
        raise DatasetStorageError("JSON não contém um objeto")
    if _canonical_json_bytes(value) != raw:
        raise DatasetStorageError("JSON não usa a codificação canônica")
    return value


def _annotation_to_dict(annotation: FieldAnnotation) -> Dict[str, Any]:
    return {
        "entity_id": annotation.entity_id,
        "field_type": annotation.field_type,
        "start": annotation.start,
        "end": annotation.end,
        "value": annotation.value,
    }


def _conversation_to_dict(conversation: TrainingConversation) -> Dict[str, Any]:
    validate_training_conversation(conversation)
    return {
        "schema_version": conversation.schema_version,
        "text": conversation.text,
        "entity_id": conversation.entity_id,
        "client_id": conversation.client_id,
        "round_id": conversation.round_id,
        "sample_index": conversation.sample_index,
        "kind": conversation.kind,
        "template_id": conversation.template_id,
        "annotations": [
            _annotation_to_dict(annotation)
            for annotation in conversation.annotations
        ],
        "prefix_length": conversation.prefix_length,
        "loss_scope": conversation.loss_scope,
    }


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str]) -> None:
    if set(value) != expected:
        raise DatasetStorageError("artefato contém campos desconhecidos ou ausentes")


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise DatasetStorageError("artefato contém tipo inválido")
    return value


def _require_integer(value: object) -> int:
    if type(value) is not int:
        raise DatasetStorageError("artefato contém tipo inválido")
    return value


def _annotation_from_dict(value: object) -> FieldAnnotation:
    if not isinstance(value, dict):
        raise DatasetStorageError("anotação inválida")
    _require_exact_keys(value, _ANNOTATION_KEYS)
    return FieldAnnotation(
        entity_id=_require_string(value["entity_id"]),
        field_type=_require_string(value["field_type"]),
        start=_require_integer(value["start"]),
        end=_require_integer(value["end"]),
        value=_require_string(value["value"]),
    )


def _conversation_from_dict(value: Mapping[str, Any]) -> TrainingConversation:
    _require_exact_keys(value, _CONVERSATION_KEYS)
    annotations = value["annotations"]
    if not isinstance(annotations, list):
        raise DatasetStorageError("anotações inválidas")

    round_id = value["round_id"]
    if round_id is not None and type(round_id) is not int:
        raise DatasetStorageError("round_id inválido")
    prefix_length = value["prefix_length"]
    if prefix_length is not None and type(prefix_length) is not int:
        raise DatasetStorageError("prefix_length inválido")

    conversation = TrainingConversation(
        schema_version=_require_string(value["schema_version"]),
        text=_require_string(value["text"]),
        entity_id=_require_string(value["entity_id"]),
        client_id=_require_string(value["client_id"]),
        round_id=round_id,
        sample_index=_require_integer(value["sample_index"]),
        kind=_require_string(value["kind"]),
        template_id=_require_string(value["template_id"]),
        annotations=tuple(
            _annotation_from_dict(annotation) for annotation in annotations
        ),
        prefix_length=prefix_length,
        loss_scope=_require_string(value["loss_scope"]),
    )
    validate_training_conversation(conversation)
    return conversation


def _conversations_to_jsonl(
    conversations: Iterable[TrainingConversation],
) -> bytes:
    return b"".join(
        _canonical_json_bytes(_conversation_to_dict(conversation))
        for conversation in conversations
    )


def _conversations_from_jsonl(
    raw: bytes,
    *,
    expected_count: int,
) -> Tuple[TrainingConversation, ...]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise DatasetStorageError("JSONL não usa LF canônico")
    lines = raw.splitlines()
    if len(lines) != expected_count or any(not line for line in lines):
        raise DatasetStorageError("quantidade de registros JSONL inválida")

    conversations = []
    for line in lines:
        payload = _load_json_bytes(line + b"\n")
        conversation = _conversation_from_dict(payload)
        if _canonical_json_bytes(_conversation_to_dict(conversation)) != line + b"\n":
            raise DatasetStorageError("registro JSONL não é canônico")
        conversations.append(conversation)
    return tuple(conversations)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_bytes_exclusive(path: Path, raw: bytes) -> None:
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def _write_artifact_files(
    artifact_directory: Path,
    conversations: Sequence[TrainingConversation],
    metadata: Mapping[str, Any],
) -> None:
    artifact_directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(artifact_directory, 0o700)
    content = _conversations_to_jsonl(conversations)
    resolved_metadata = dict(metadata)
    resolved_metadata["record_count"] = len(conversations)
    resolved_metadata["content_sha256"] = _sha256(content)
    _write_bytes_exclusive(artifact_directory / "conversations.jsonl", content)
    _write_bytes_exclusive(
        artifact_directory / "metadata.json",
        _canonical_json_bytes(resolved_metadata),
    )


def _load_artifact(
    artifact_directory: Path,
    *,
    expected_metadata_keys: frozenset[str],
) -> Tuple[Dict[str, Any], Tuple[TrainingConversation, ...]]:
    metadata_path = artifact_directory / "metadata.json"
    content_path = artifact_directory / "conversations.jsonl"
    try:
        metadata_raw = metadata_path.read_bytes()
        content = content_path.read_bytes()
    except OSError as exc:
        raise DatasetStorageError("artefato de conversas ausente") from exc

    metadata = _load_json_bytes(metadata_raw)
    _require_exact_keys(metadata, expected_metadata_keys)
    record_count = _require_integer(metadata["record_count"])
    if _require_string(metadata["content_sha256"]) != _sha256(content):
        raise DatasetStorageError("hash do conteúdo diverge")
    conversations = _conversations_from_jsonl(
        content,
        expected_count=record_count,
    )
    return metadata, conversations


def _common_metadata(
    *,
    dataset_id: str,
    role: str,
    client_id: str,
    container_schema_version: str,
) -> Dict[str, Any]:
    return {
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "jsonl_schema_version": CONVERSATION_JSONL_SCHEMA_VERSION,
        "container_schema_version": container_schema_version,
        "conversation_schema_version": CONVERSATION_SCHEMA_VERSION,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "dataset_id": dataset_id,
        "role": role,
        "client_id": client_id,
    }


def _victim_artifact_directory(
    output_root: Path,
    dataset_id: str,
    client_id: str,
) -> Path:
    return output_root / dataset_id / "clients" / "victim" / client_id


def _auxiliary_artifact_directory(
    output_root: Path,
    dataset_id: str,
    schedule_id: str,
    presentation: AuxiliaryPresentation,
    round_id: int,
) -> Path:
    return (
        output_root
        / dataset_id
        / "clients"
        / "auxiliary"
        / schedule_id
        / presentation
        / f"round-{round_id:03d}"
    )


def write_victim_datasets(
    output_root: Path,
    dataset_id: str,
    datasets: Sequence[VictimClientDataset],
) -> Tuple[Path, ...]:
    """Publica os dez clientes de uma vez e nunca sobrescreve um bundle."""

    resolved_dataset_id = _validate_component(dataset_id, "dataset_id")
    resolved_datasets = tuple(datasets)
    manifest = build_victim_dataset_manifest(resolved_datasets)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target_root = output_root / resolved_dataset_id
    if target_root.exists():
        raise FileExistsError(target_root)

    staging_root = Path(
        tempfile.mkdtemp(prefix=".dataset-staging-", dir=output_root)
    )
    try:
        os.chmod(staging_root, 0o700)
        for dataset in resolved_datasets:
            _validate_component(dataset.client_id, "client_id")
            validate_victim_dataset(dataset)
            artifact_directory = (
                staging_root / "clients" / "victim" / dataset.client_id
            )
            metadata = _common_metadata(
                dataset_id=resolved_dataset_id,
                role="victim",
                client_id=dataset.client_id,
                container_schema_version=dataset.schema_version,
            )
            _write_artifact_files(
                artifact_directory,
                dataset.conversations,
                metadata,
            )
            _read_victim_artifact(
                artifact_directory,
                expected_dataset_id=resolved_dataset_id,
                expected_client_id=dataset.client_id,
            )

        manifest_directory = staging_root / "trusted" / "manifests"
        manifest_directory.mkdir(parents=True, mode=0o700)
        os.chmod(manifest_directory, 0o700)
        _write_bytes_exclusive(
            manifest_directory / "victim_dataset_manifest.json",
            _canonical_json_bytes(manifest),
        )

        if target_root.exists():
            raise FileExistsError(target_root)
        staging_root.rename(target_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return tuple(
        _victim_artifact_directory(
            output_root,
            resolved_dataset_id,
            dataset.client_id,
        )
        / "conversations.jsonl"
        for dataset in resolved_datasets
    )


def _read_victim_artifact(
    artifact_directory: Path,
    *,
    expected_dataset_id: str,
    expected_client_id: str,
) -> VictimClientDataset:
    metadata, conversations = _load_artifact(
        artifact_directory,
        expected_metadata_keys=_VICTIM_METADATA_KEYS,
    )
    expected = {
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "jsonl_schema_version": CONVERSATION_JSONL_SCHEMA_VERSION,
        "container_schema_version": VICTIM_DATASET_SCHEMA_VERSION,
        "conversation_schema_version": CONVERSATION_SCHEMA_VERSION,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "dataset_id": expected_dataset_id,
        "role": "victim",
        "client_id": expected_client_id,
    }
    if any(metadata[key] != value for key, value in expected.items()):
        raise DatasetStorageError("metadados do cliente-vítima divergem")
    dataset = VictimClientDataset(
        client_id=expected_client_id,
        conversations=conversations,
    )
    validate_victim_dataset(dataset)
    return dataset


def read_victim_client_dataset(
    output_root: Path,
    dataset_id: str,
    client_id: str,
) -> VictimClientDataset:
    """Lê somente o cliente explicitamente solicitado e o valida por completo."""

    resolved_dataset_id = _validate_component(dataset_id, "dataset_id")
    resolved_client_id = _validate_component(client_id, "client_id")
    return _read_victim_artifact(
        _victim_artifact_directory(
            Path(output_root),
            resolved_dataset_id,
            resolved_client_id,
        ),
        expected_dataset_id=resolved_dataset_id,
        expected_client_id=resolved_client_id,
    )


def write_auxiliary_round(
    output_root: Path,
    dataset_id: str,
    schedule_id: str,
    round_data: AuxiliaryRound,
) -> Path:
    """Publica atomicamente apenas uma apresentação de uma rodada auxiliar."""

    resolved_dataset_id = _validate_component(dataset_id, "dataset_id")
    resolved_schedule_id = _validate_component(schedule_id, "schedule_id")
    validate_auxiliary_round(round_data)
    dataset_root = Path(output_root) / resolved_dataset_id
    if not dataset_root.is_dir():
        raise DatasetStorageError("bundle de vítimas deve ser publicado primeiro")

    target_directory = _auxiliary_artifact_directory(
        Path(output_root),
        resolved_dataset_id,
        resolved_schedule_id,
        round_data.presentation,
        round_data.round_id,
    )
    if target_directory.exists():
        raise FileExistsError(target_directory)
    target_directory.parent.mkdir(parents=True, exist_ok=True)

    staging_directory = Path(
        tempfile.mkdtemp(prefix=".round-staging-", dir=target_directory.parent)
    )
    try:
        os.chmod(staging_directory, 0o700)
        metadata = _common_metadata(
            dataset_id=resolved_dataset_id,
            role="auxiliary",
            client_id="auxiliary",
            container_schema_version=round_data.schema_version,
        )
        metadata.update(
            {
                "schedule_id": resolved_schedule_id,
                "presentation": round_data.presentation,
                "round_id": round_data.round_id,
            }
        )
        artifact_directory = staging_directory / "artifact"
        _write_artifact_files(
            artifact_directory,
            round_data.conversations,
            metadata,
        )
        _read_auxiliary_artifact(
            artifact_directory,
            expected_dataset_id=resolved_dataset_id,
            expected_schedule_id=resolved_schedule_id,
            expected_presentation=round_data.presentation,
            expected_round_id=round_data.round_id,
        )
        if target_directory.exists():
            raise FileExistsError(target_directory)
        artifact_directory.rename(target_directory)
        staging_directory.rmdir()
    except Exception:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise

    return target_directory / "conversations.jsonl"


def _read_auxiliary_artifact(
    artifact_directory: Path,
    *,
    expected_dataset_id: str,
    expected_schedule_id: str,
    expected_presentation: AuxiliaryPresentation,
    expected_round_id: int,
) -> AuxiliaryRound:
    metadata, conversations = _load_artifact(
        artifact_directory,
        expected_metadata_keys=_AUXILIARY_METADATA_KEYS,
    )
    expected = {
        "schema_version": ARTIFACT_METADATA_SCHEMA_VERSION,
        "jsonl_schema_version": CONVERSATION_JSONL_SCHEMA_VERSION,
        "container_schema_version": AUXILIARY_ROUND_SCHEMA_VERSION,
        "conversation_schema_version": CONVERSATION_SCHEMA_VERSION,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "dataset_id": expected_dataset_id,
        "role": "auxiliary",
        "client_id": "auxiliary",
        "schedule_id": expected_schedule_id,
        "presentation": expected_presentation,
        "round_id": expected_round_id,
    }
    if any(metadata[key] != value for key, value in expected.items()):
        raise DatasetStorageError("metadados da rodada auxiliar divergem")
    round_data = AuxiliaryRound(
        round_id=expected_round_id,
        presentation=expected_presentation,
        conversations=conversations,
    )
    validate_auxiliary_round(round_data)
    return round_data


def read_auxiliary_round(
    output_root: Path,
    dataset_id: str,
    schedule_id: str,
    presentation: AuxiliaryPresentation,
    round_id: int,
) -> AuxiliaryRound:
    """Lê uma rodada identificada sem expor outras árvores de clientes."""

    resolved_dataset_id = _validate_component(dataset_id, "dataset_id")
    resolved_schedule_id = _validate_component(schedule_id, "schedule_id")
    if presentation not in {"benign", "adversarial"}:
        raise DatasetStorageError("presentation inválida")
    if type(round_id) is not int or not 1 <= round_id <= 20:
        raise DatasetStorageError("round_id inválido")
    return _read_auxiliary_artifact(
        _auxiliary_artifact_directory(
            Path(output_root),
            resolved_dataset_id,
            resolved_schedule_id,
            presentation,
            round_id,
        ),
        expected_dataset_id=resolved_dataset_id,
        expected_schedule_id=resolved_schedule_id,
        expected_presentation=presentation,
        expected_round_id=round_id,
    )
