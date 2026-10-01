"""Manifestos sem nomes, identificadores, textos ou valores renderizados."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from .conversations import conversation_catalog_sha256
from .generator import (
    AUXILIARY_ROUNDS,
    EXPECTED_FAKER_VERSION,
    GENERAL_RECORDS_PER_ROUND,
    PROFILES_PER_ROUND,
)
from .model import (
    CONVERSATION_GENERATOR_VERSION,
    CONVERSATION_SCHEMA_VERSION,
    GENERATOR_VERSION,
    PROFILE_SCHEMA_VERSION,
    AuxiliaryRound,
    TrainingConversation,
    VictimClientDataset,
)
from .rendering import CANONICAL_PROFILE_TEMPLATE
from .validation import validate_auxiliary_round, validate_victim_dataset
from .victims import (
    CONVERSATIONS_PER_VICTIM_CLIENT,
    PROFILES_PER_VICTIM_CLIENT,
    VICTIM_CLIENTS,
)


MANIFEST_SCHEMA_VERSION = "auxiliary-round-manifest/v2"
VICTIM_MANIFEST_SCHEMA_VERSION = "victim-dataset-manifest/v1"
GENERATION_MANIFEST_SCHEMA_VERSION = "dataset-generation-manifest/v1"

AUXILIARY_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "profile_schema_version",
        "conversation_schema_version",
        "profile_generator_version",
        "conversation_generator_version",
        "faker_version",
        "round",
        "presentation",
        "conversation_records",
        "protected_records",
        "general_records",
        "schedule_sha256",
        "values_sha256",
        "presentation_sha256",
        "batch_sha256",
        "canonical_template_sha256",
        "catalog_sha256",
    }
)
VICTIM_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "profile_schema_version",
        "conversation_schema_version",
        "profile_generator_version",
        "conversation_generator_version",
        "faker_version",
        "client_count",
        "profiles_per_client",
        "conversations_per_client",
        "protected_records_per_client",
        "general_records_per_client",
        "client_schedule_sha256",
        "client_batch_sha256",
        "dataset_sha256",
        "canonical_template_sha256",
        "catalog_sha256",
    }
)
GENERATION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "experiment_seed",
        "dataset_id",
        "schedule_id",
        "profile_generator_version",
        "conversation_generator_version",
        "round_count",
        "presentations",
        "victim_conversation_records",
        "auxiliary_conversation_records",
        "total_conversation_records",
        "victim_dataset_sha256",
        "auxiliary_schedule_sha256",
        "auxiliary_values_sha256",
        "auxiliary_presentation_sha256",
        "auxiliary_batch_sha256",
    }
)
AUXILIARY_HASH_KEYS = (
    "schedule_sha256",
    "values_sha256",
    "presentation_sha256",
    "batch_sha256",
    "canonical_template_sha256",
    "catalog_sha256",
)
VICTIM_HASH_KEYS = (
    "dataset_sha256",
    "canonical_template_sha256",
    "catalog_sha256",
)
GENERATION_HASH_KEYS = (
    "victim_dataset_sha256",
    "auxiliary_schedule_sha256",
    "auxiliary_values_sha256",
    "auxiliary_presentation_sha256",
    "auxiliary_batch_sha256",
)


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _faker_version() -> str:
    version = importlib.metadata.version("Faker")
    if version != EXPECTED_FAKER_VERSION:
        raise RuntimeError("a versão do Faker diverge do contrato do manifesto")
    return version


def _schedule_lines(conversations: Sequence[TrainingConversation]) -> Iterable[str]:
    for conversation in conversations:
        catalog_entry = (
            conversation.template_id if conversation.kind == "general" else "protected"
        )
        yield (
            f"{conversation.kind}\t{conversation.sample_index}\t"
            f"{conversation.entity_id}\t{catalog_entry}"
        )


def _value_lines(conversations: Sequence[TrainingConversation]) -> Iterable[str]:
    for conversation in conversations:
        if conversation.kind != "protected":
            continue
        for annotation in conversation.annotations:
            yield (
                f"{conversation.entity_id}\t{annotation.field_type}\t"
                f"{annotation.value}"
            )


def _presentation_lines(
    conversations: Sequence[TrainingConversation],
) -> Iterable[str]:
    for conversation in conversations:
        yield (
            f"{conversation.sample_index}\t{conversation.template_id}\t"
            f"{conversation.loss_scope}\t{conversation.prefix_length}"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_round_manifest(round_data: AuxiliaryRound) -> Dict[str, Any]:
    """Calcula hashes em memória para uma apresentação da rodada auxiliar."""

    validate_auxiliary_round(round_data)
    protected_records = sum(
        conversation.kind == "protected" for conversation in round_data.conversations
    )
    general_records = len(round_data.conversations) - protected_records
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "conversation_schema_version": CONVERSATION_SCHEMA_VERSION,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "faker_version": _faker_version(),
        "round": round_data.round_id,
        "presentation": round_data.presentation,
        "conversation_records": len(round_data.conversations),
        "protected_records": protected_records,
        "general_records": general_records,
        "schedule_sha256": _sha256_lines(
            _schedule_lines(round_data.conversations)
        ),
        "values_sha256": _sha256_lines(_value_lines(round_data.conversations)),
        "presentation_sha256": _sha256_lines(
            _presentation_lines(round_data.conversations)
        ),
        "batch_sha256": _sha256_lines(
            conversation.text for conversation in round_data.conversations
        ),
        "canonical_template_sha256": hashlib.sha256(
            CANONICAL_PROFILE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "catalog_sha256": conversation_catalog_sha256(),
    }
    _validate_round_manifest(manifest)
    return manifest


def build_victim_dataset_manifest(
    datasets: Sequence[VictimClientDataset],
) -> Dict[str, Any]:
    """Resume os dez datasets sem serializar seu conteúdo ou identificadores."""

    if len(datasets) != VICTIM_CLIENTS:
        raise ValueError("quantidade de clientes-vítima inválida")
    if tuple(dataset.client_id for dataset in datasets) != tuple(
        f"victim-{index:02d}" for index in range(1, VICTIM_CLIENTS + 1)
    ):
        raise ValueError("ordem dos clientes-vítima inválida")
    for dataset in datasets:
        validate_victim_dataset(dataset)

    client_schedule_hashes = [
        _sha256_lines(_schedule_lines(dataset.conversations)) for dataset in datasets
    ]
    client_batch_hashes = [
        _sha256_lines(conversation.text for conversation in dataset.conversations)
        for dataset in datasets
    ]
    manifest = {
        "schema_version": VICTIM_MANIFEST_SCHEMA_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "conversation_schema_version": CONVERSATION_SCHEMA_VERSION,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "faker_version": _faker_version(),
        "client_count": VICTIM_CLIENTS,
        "profiles_per_client": PROFILES_PER_VICTIM_CLIENT,
        "conversations_per_client": CONVERSATIONS_PER_VICTIM_CLIENT,
        "protected_records_per_client": 80,
        "general_records_per_client": 20,
        "client_schedule_sha256": client_schedule_hashes,
        "client_batch_sha256": client_batch_hashes,
        "dataset_sha256": _sha256_lines(
            [*client_schedule_hashes, *client_batch_hashes]
        ),
        "canonical_template_sha256": hashlib.sha256(
            CANONICAL_PROFILE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "catalog_sha256": conversation_catalog_sha256(),
    }
    _validate_victim_manifest(manifest)
    return manifest


def build_generation_manifest(
    *,
    experiment_seed: int,
    dataset_id: str,
    schedule_id: str,
    victim_manifest: Dict[str, Any],
    round_manifests: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resume um bundle completo sem incluir seu conteúdo protegido."""

    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ValueError("a seed experimental deve ser um inteiro não negativo")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id do manifesto de geração inválido")
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValueError("schedule_id do manifesto de geração inválido")

    _validate_victim_manifest(victim_manifest)
    resolved_round_manifests = tuple(round_manifests)
    expected_order = tuple(
        (round_id, presentation)
        for round_id in range(1, AUXILIARY_ROUNDS + 1)
        for presentation in ("benign", "adversarial")
    )
    if len(resolved_round_manifests) != len(expected_order):
        raise ValueError("quantidade de manifestos auxiliares inválida")
    for round_manifest in resolved_round_manifests:
        _validate_round_manifest(round_manifest)
    if tuple(
        (manifest["round"], manifest["presentation"])
        for manifest in resolved_round_manifests
    ) != expected_order:
        raise ValueError("ordem dos manifestos auxiliares inválida")
    for index in range(0, len(resolved_round_manifests), 2):
        benign = resolved_round_manifests[index]
        adversarial = resolved_round_manifests[index + 1]
        if (
            benign["schedule_sha256"] != adversarial["schedule_sha256"]
            or benign["values_sha256"] != adversarial["values_sha256"]
        ):
            raise ValueError("manifestos auxiliares pareados divergem")

    victim_records = (
        victim_manifest["client_count"]
        * victim_manifest["conversations_per_client"]
    )
    auxiliary_records = sum(
        manifest["conversation_records"]
        for manifest in resolved_round_manifests
    )
    manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "experiment_seed": experiment_seed,
        "dataset_id": dataset_id,
        "schedule_id": schedule_id,
        "profile_generator_version": GENERATOR_VERSION,
        "conversation_generator_version": CONVERSATION_GENERATOR_VERSION,
        "round_count": AUXILIARY_ROUNDS,
        "presentations": ["benign", "adversarial"],
        "victim_conversation_records": victim_records,
        "auxiliary_conversation_records": auxiliary_records,
        "total_conversation_records": victim_records + auxiliary_records,
        "victim_dataset_sha256": victim_manifest["dataset_sha256"],
        "auxiliary_schedule_sha256": _sha256_lines(
            manifest["schedule_sha256"]
            for manifest in resolved_round_manifests
        ),
        "auxiliary_values_sha256": _sha256_lines(
            manifest["values_sha256"]
            for manifest in resolved_round_manifests
        ),
        "auxiliary_presentation_sha256": _sha256_lines(
            manifest["presentation_sha256"]
            for manifest in resolved_round_manifests
        ),
        "auxiliary_batch_sha256": _sha256_lines(
            manifest["batch_sha256"]
            for manifest in resolved_round_manifests
        ),
    }
    _validate_generation_manifest(manifest)
    return manifest


def _validate_round_manifest(manifest: Dict[str, Any]) -> None:
    if set(manifest) != AUXILIARY_MANIFEST_KEYS:
        raise ValueError("o manifesto auxiliar contém campos não autorizados")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema_version do manifesto auxiliar inválida")
    if manifest["profile_schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("profile_schema_version do manifesto inválida")
    if manifest["conversation_schema_version"] != CONVERSATION_SCHEMA_VERSION:
        raise ValueError("conversation_schema_version do manifesto inválida")
    if manifest["profile_generator_version"] != GENERATOR_VERSION:
        raise ValueError("profile_generator_version do manifesto inválida")
    if manifest["conversation_generator_version"] != CONVERSATION_GENERATOR_VERSION:
        raise ValueError("conversation_generator_version do manifesto inválida")
    if manifest["faker_version"] != EXPECTED_FAKER_VERSION:
        raise ValueError("faker_version do manifesto inválida")
    if not isinstance(manifest["round"], int) or not 1 <= manifest["round"] <= AUXILIARY_ROUNDS:
        raise ValueError("rodada do manifesto inválida")
    if manifest["presentation"] not in {"benign", "adversarial"}:
        raise ValueError("apresentação do manifesto inválida")
    if (
        manifest["conversation_records"] != 100
        or manifest["protected_records"] != PROFILES_PER_ROUND
        or manifest["general_records"] != GENERAL_RECORDS_PER_ROUND
    ):
        raise ValueError("contagens do manifesto auxiliar inválidas")
    if any(not _is_sha256(manifest[key]) for key in AUXILIARY_HASH_KEYS):
        raise ValueError("hash do manifesto auxiliar inválido")


def _validate_victim_manifest(manifest: Dict[str, Any]) -> None:
    if set(manifest) != VICTIM_MANIFEST_KEYS:
        raise ValueError("o manifesto vítima contém campos não autorizados")
    if manifest["schema_version"] != VICTIM_MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema_version do manifesto vítima inválida")
    if manifest["profile_schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("profile_schema_version do manifesto vítima inválida")
    if manifest["conversation_schema_version"] != CONVERSATION_SCHEMA_VERSION:
        raise ValueError("conversation_schema_version do manifesto vítima inválida")
    if manifest["profile_generator_version"] != GENERATOR_VERSION:
        raise ValueError("profile_generator_version do manifesto vítima inválida")
    if manifest["conversation_generator_version"] != CONVERSATION_GENERATOR_VERSION:
        raise ValueError("conversation_generator_version do manifesto vítima inválida")
    if manifest["faker_version"] != EXPECTED_FAKER_VERSION:
        raise ValueError("faker_version do manifesto vítima inválida")
    if (
        manifest["client_count"] != VICTIM_CLIENTS
        or manifest["profiles_per_client"] != PROFILES_PER_VICTIM_CLIENT
        or manifest["conversations_per_client"] != CONVERSATIONS_PER_VICTIM_CLIENT
        or manifest["protected_records_per_client"] != 80
        or manifest["general_records_per_client"] != 20
    ):
        raise ValueError("contagens do manifesto vítima inválidas")
    for key in ("client_schedule_sha256", "client_batch_sha256"):
        hashes = manifest[key]
        if not isinstance(hashes, list) or len(hashes) != VICTIM_CLIENTS:
            raise ValueError("lista de hashes do manifesto vítima inválida")
        if any(not _is_sha256(value) for value in hashes):
            raise ValueError("hash por cliente do manifesto vítima inválido")
    if any(not _is_sha256(manifest[key]) for key in VICTIM_HASH_KEYS):
        raise ValueError("hash do manifesto vítima inválido")


def _validate_generation_manifest(manifest: Dict[str, Any]) -> None:
    if set(manifest) != GENERATION_MANIFEST_KEYS:
        raise ValueError("o manifesto de geração contém campos não autorizados")
    if manifest["schema_version"] != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema_version do manifesto de geração inválida")
    if (
        type(manifest["experiment_seed"]) is not int
        or manifest["experiment_seed"] < 0
    ):
        raise ValueError("seed do manifesto de geração inválida")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        raise ValueError("dataset_id do manifesto de geração inválido")
    if not isinstance(manifest["schedule_id"], str) or not manifest["schedule_id"]:
        raise ValueError("schedule_id do manifesto de geração inválido")
    if manifest["profile_generator_version"] != GENERATOR_VERSION:
        raise ValueError(
            "profile_generator_version do manifesto de geração inválida"
        )
    if manifest["conversation_generator_version"] != CONVERSATION_GENERATOR_VERSION:
        raise ValueError(
            "conversation_generator_version do manifesto de geração inválida"
        )
    if (
        manifest["round_count"] != AUXILIARY_ROUNDS
        or manifest["presentations"] != ["benign", "adversarial"]
        or manifest["victim_conversation_records"] != 1_000
        or manifest["auxiliary_conversation_records"] != 4_000
        or manifest["total_conversation_records"] != 5_000
    ):
        raise ValueError("contagens do manifesto de geração inválidas")
    if any(not _is_sha256(manifest[key]) for key in GENERATION_HASH_KEYS):
        raise ValueError("hash do manifesto de geração inválido")


def append_round_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Acrescenta uma linha auxiliar validada ao JSONL externo da execução."""

    _validate_round_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
        output.write("\n")


def write_victim_dataset_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Cria sem sobrescrever o manifesto agregado das vítimas."""

    _validate_victim_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, sort_keys=True, ensure_ascii=False)
        output.write("\n")


def write_generation_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Cria sem sobrescrever o manifesto seguro do bundle completo."""

    _validate_generation_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(
            manifest,
            output,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output.write("\n")
