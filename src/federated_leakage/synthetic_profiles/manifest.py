"""Manifesto de rodada sem nomes, documentos ou textos renderizados."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .generator import (
    AUXILIARY_ROUNDS,
    EXPECTED_FAKER_VERSION,
    GENERAL_RECORDS_PER_ROUND,
    PROFILES_PER_ROUND,
)
from .model import AuxiliaryRound, GENERATOR_VERSION, PROFILE_SCHEMA_VERSION
from .rendering import CANONICAL_PROFILE_TEMPLATE


MANIFEST_SCHEMA_VERSION = "auxiliary-round-manifest/v1"
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "profile_schema_version",
        "generator_version",
        "faker_version",
        "round",
        "profile_records",
        "general_records",
        "schedule_sha256",
        "batch_sha256",
        "template_sha256",
    }
)
_HASH_KEYS = ("schedule_sha256", "batch_sha256", "template_sha256")


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_round_manifest(round_data: AuxiliaryRound) -> Dict[str, Any]:
    """Calcula hashes em memória e devolve somente metadados permitidos."""

    faker_version = importlib.metadata.version("Faker")
    if faker_version != EXPECTED_FAKER_VERSION:
        raise RuntimeError("a versão do Faker diverge do contrato do manifesto")

    schedule_hash = _sha256_lines(
        sample.profile.entity_id for sample in round_data.profile_samples
    )
    batch_hash = _sha256_lines(
        [sample.rendered.text for sample in round_data.profile_samples]
        + list(round_data.general_records)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "faker_version": faker_version,
        "round": round_data.round_id,
        "profile_records": len(round_data.profile_samples),
        "general_records": len(round_data.general_records),
        "schedule_sha256": schedule_hash,
        "batch_sha256": batch_hash,
        "template_sha256": hashlib.sha256(
            CANONICAL_PROFILE_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("o manifesto contém campos não autorizados")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema_version do manifesto inválida")
    if manifest["profile_schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("profile_schema_version do manifesto inválida")
    if manifest["generator_version"] != GENERATOR_VERSION:
        raise ValueError("generator_version do manifesto inválida")
    if manifest["faker_version"] != EXPECTED_FAKER_VERSION:
        raise ValueError("faker_version do manifesto inválida")
    if (
        not isinstance(manifest["round"], int)
        or manifest["round"] < 1
        or manifest["round"] > AUXILIARY_ROUNDS
    ):
        raise ValueError("rodada do manifesto inválida")
    if (
        manifest["profile_records"] != PROFILES_PER_ROUND
        or manifest["general_records"] != GENERAL_RECORDS_PER_ROUND
    ):
        raise ValueError("contagens do manifesto inválidas")
    for key in _HASH_KEYS:
        value = manifest[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{key} não contém um SHA-256 hexadecimal")


def append_round_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Acrescenta uma linha JSON sem aceitar chaves ou valores extras."""

    _validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
        output.write("\n")
