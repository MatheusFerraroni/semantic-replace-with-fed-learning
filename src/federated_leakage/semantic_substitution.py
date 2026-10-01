"""Substituição semântica rotativa aplicada somente aos clientes-vítima."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping, Sequence, Tuple

from .federated_round import PreparedVictimTrainingInputs, prepare_victim_training_inputs
from .model_contracts import LoadedModelBundle
from .synthetic_profiles.conversations import render_protected_conversation
from .synthetic_profiles.generator import _SyntheticProfileFactory, _assert_faker_version
from .synthetic_profiles.model import (
    PROFILE_FIELD_ORDER,
    UNIQUE_FIELD_TYPES,
    SyntheticProfile,
    VictimClientDataset,
    profile_field_values,
)
from .synthetic_profiles.seeding import derive_seed_material
from .synthetic_profiles.validation import validate_profile, validate_victim_dataset


ROTATING_REPLACEMENT_SCHEMA_VERSION = "rotating-semantic-replacement/v1"
SUBSTITUTED_VICTIM_ROUND_SCHEMA_VERSION = "substituted-victim-client-round/v1"
PREPARED_SUBSTITUTED_INPUTS_SCHEMA_VERSION = (
    "prepared-substituted-victim-training-inputs/v1"
)
REPLACEMENT_SCHEDULE_VERSION = "rotating-profile/v3"
REPLACEMENT_ROUNDS = 20
_PROFILES_PER_ROUND = 200
_MAX_CANDIDATE_ATTEMPTS = 256
_PROFILE_POSITION_DOMAIN = 26**3


class SemanticSubstitutionError(RuntimeError):
    """A substituição falhou sem expor valores ou entidades."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticReplacementEntry:
    """Associação privada entre uma entidade técnica e seu perfil substituto."""

    client_id: str
    round_id: int
    source_entity_id: str = field(repr=False)
    replacement_profile: SyntheticProfile = field(repr=False)
    schema_version: str = ROTATING_REPLACEMENT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class SubstitutedVictimClientRound:
    """As 100 conversas substituídas de um cliente em uma rodada."""

    client_id: str
    round_id: int
    dataset: VictimClientDataset = field(repr=False)
    schema_version: str = SUBSTITUTED_VICTIM_ROUND_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticReplacementRound:
    """Agenda completa de substituição de uma rodada, mantida em memória."""

    round_id: int
    clients: Tuple[SubstitutedVictimClientRound, ...] = field(repr=False)
    entries: Tuple[SemanticReplacementEntry, ...] = field(repr=False)
    schedule_sha256: str
    values_sha256: str
    collision_counts: Tuple[Tuple[str, int], ...]
    ambiguous_name_count: int
    schema_version: str = ROTATING_REPLACEMENT_SCHEMA_VERSION

    @property
    def datasets(self) -> Tuple[VictimClientDataset, ...]:
        return tuple(value.dataset for value in self.clients)

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "round_id": self.round_id,
            "client_count": len(self.clients),
            "entity_count": len(self.entries),
            "conversation_count": sum(
                len(value.dataset.conversations) for value in self.clients
            ),
            "schedule_sha256": self.schedule_sha256,
            "values_sha256": self.values_sha256,
            "collision_counts": dict(self.collision_counts),
            "ambiguous_name_count": self.ambiguous_name_count,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedSubstitutedVictimTrainingInputs:
    """Entradas tokenizadas uma vez para uma rodada protegida."""

    round_id: int
    prepared: PreparedVictimTrainingInputs = field(repr=False)
    replacement_schedule_sha256: str
    replacement_values_sha256: str
    schema_version: str = PREPARED_SUBSTITUTED_INPUTS_SCHEMA_VERSION


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash(value: Any, domain: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json_bytes(value)).hexdigest()


def _profile_from_conversations(dataset: VictimClientDataset) -> Tuple[SyntheticProfile, ...]:
    by_entity: dict[str, list[Any]] = {}
    for conversation in dataset.conversations:
        by_entity.setdefault(conversation.entity_id, []).append(conversation)
    ordered: list[tuple[int, SyntheticProfile]] = []
    for entity_id, conversations in by_entity.items():
        protected = tuple(value for value in conversations if value.kind == "protected")
        general = tuple(value for value in conversations if value.kind == "general")
        if len(protected) != 4 or len(general) != 1:
            raise SemanticSubstitutionError("agrupamento de entidade é inválido")
        reference = tuple(
            (annotation.field_type, annotation.value)
            for annotation in protected[0].annotations
        )
        if (
            tuple(field_type for field_type, _ in reference) != PROFILE_FIELD_ORDER
            or any(
                tuple(
                    (annotation.field_type, annotation.value)
                    for annotation in conversation.annotations
                )
                != reference
                for conversation in protected[1:]
            )
        ):
            raise SemanticSubstitutionError("registro protegido de entidade diverge")
        values = dict(reference)
        try:
            profile = SyntheticProfile(
                entity_id=entity_id,
                person_name=values["PERSON_NAME"],
                birth_date=datetime.strptime(values["BIRTH_DATE"], "%d/%m/%Y").date(),
                cpf=values["CPF"],
                rg=values["RG"],
                phone=values["PHONE"],
                email=values["EMAIL"],
                address=values["ADDRESS"],
                appointment_date=datetime.strptime(
                    values["APPOINTMENT_DATE"], "%d/%m/%Y"
                ).date(),
                appointment_time=datetime.strptime(
                    values["APPOINTMENT_TIME"], "%H:%M"
                ).time(),
            )
            validate_profile(profile)
        except Exception as error:
            raise SemanticSubstitutionError("registro protegido é inválido") from error
        ordered.append((min(value.sample_index for value in conversations), profile))
    ordered.sort(key=lambda item: item[0])
    if len(ordered) != 20:
        raise SemanticSubstitutionError("cliente não contém vinte entidades")
    return tuple(profile for _, profile in ordered)


def _source_profiles(
    datasets: Sequence[VictimClientDataset],
) -> Tuple[Tuple[SyntheticProfile, ...], ...]:
    resolved = tuple(datasets)
    if len(resolved) != 10:
        raise SemanticSubstitutionError("substituição exige dez clientes-vítima")
    result = []
    for index, dataset in enumerate(resolved, start=1):
        try:
            validate_victim_dataset(dataset)
        except Exception as error:
            raise SemanticSubstitutionError("dataset original é inválido") from error
        if dataset.client_id != f"victim-{index:02d}":
            raise SemanticSubstitutionError("ordem dos clientes-vítima é inválida")
        result.append(_profile_from_conversations(dataset))
    return tuple(result)


class RotatingVictimSubstitutionGenerator:
    """Reconstrói perfis falsos por rodada sem compartilhar mapas entre papéis."""

    def __init__(self, seed: int, *, locale: str = "pt_BR") -> None:
        if type(seed) is not int or seed < 0:
            raise SemanticSubstitutionError("seed deve ser inteira não negativa")
        _assert_faker_version()
        self._seed = seed
        self._seed_material = derive_seed_material(
            seed,
            namespace="victim-substitution",
            schedule_id=REPLACEMENT_SCHEDULE_VERSION,
        )
        self._factory = _SyntheticProfileFactory(self._seed_material, locale=locale)

    @staticmethod
    def _candidate_position(base_position: int, attempt: int) -> int:
        return (
            base_position
            + attempt * (REPLACEMENT_ROUNDS * _PROFILES_PER_ROUND)
        ) % _PROFILE_POSITION_DOMAIN

    def _replacement_for_round(
        self,
        *,
        original: SyntheticProfile,
        original_unique_values: Mapping[str, frozenset[str]],
        client_index: int,
        entity_index: int,
        target_round: int,
    ) -> SyntheticProfile:
        used = {
            field_type: {value}
            for field_type, value in profile_field_values(original).items()
        }
        selected: SyntheticProfile | None = None
        for round_id in range(1, target_round + 1):
            base_position = (
                (round_id - 1) * _PROFILES_PER_ROUND
                + client_index * 20
                + entity_index
            )
            selected = None
            for attempt in range(_MAX_CANDIDATE_ATTEMPTS):
                candidate = self._factory.generate(
                    self._candidate_position(base_position, attempt),
                    "victim-substitution",
                    client_index,
                    entity_index,
                    round_id,
                    attempt,
                )
                candidate = replace(candidate, entity_id=original.entity_id)
                values = profile_field_values(candidate)
                if any(values[field_type] in used[field_type] for field_type in PROFILE_FIELD_ORDER):
                    continue
                if any(
                    values[field_type] in original_unique_values[field_type]
                    for field_type in UNIQUE_FIELD_TYPES
                ):
                    continue
                selected = candidate
                break
            if selected is None:
                raise SemanticSubstitutionError(
                    "não foi possível gerar uma substituição válida"
                )
            for field_type, value in profile_field_values(selected).items():
                used[field_type].add(value)
        assert selected is not None
        return selected

    def generate_round(
        self,
        datasets: Sequence[VictimClientDataset],
        round_id: int,
    ) -> SemanticReplacementRound:
        """Gera e valida os dez datasets substituídos da rodada solicitada."""

        if type(round_id) is not int or not 1 <= round_id <= REPLACEMENT_ROUNDS:
            raise SemanticSubstitutionError("round_id deve estar entre 1 e 20")
        sources = _source_profiles(datasets)
        original_unique_values = {
            field_type: frozenset(
                profile_field_values(profile)[field_type]
                for client in sources
                for profile in client
            )
            for field_type in UNIQUE_FIELD_TYPES
        }
        replacements: dict[str, SyntheticProfile] = {}
        entries: list[SemanticReplacementEntry] = []
        for client_index, profiles in enumerate(sources):
            client_id = f"victim-{client_index + 1:02d}"
            for entity_index, original in enumerate(profiles):
                replacement = self._replacement_for_round(
                    original=original,
                    original_unique_values=original_unique_values,
                    client_index=client_index,
                    entity_index=entity_index,
                    target_round=round_id,
                )
                replacements[original.entity_id] = replacement
                entries.append(
                    SemanticReplacementEntry(
                        client_id=client_id,
                        round_id=round_id,
                        source_entity_id=original.entity_id,
                        replacement_profile=replacement,
                    )
                )

        clients: list[SubstitutedVictimClientRound] = []
        for source_dataset in datasets:
            conversations = []
            for conversation in source_dataset.conversations:
                if conversation.kind == "general":
                    conversations.append(conversation)
                    continue
                replacement = replacements[conversation.entity_id]
                rendered = render_protected_conversation(
                    replacement,
                    client_id=conversation.client_id,
                    round_id=None,
                    sample_index=conversation.sample_index,
                    presentation="benign",
                    template_id=conversation.template_id,
                )
                conversations.append(rendered)
            dataset = VictimClientDataset(
                client_id=source_dataset.client_id,
                conversations=tuple(conversations),
            )
            try:
                validate_victim_dataset(dataset)
            except Exception as error:
                raise SemanticSubstitutionError(
                    "dataset substituído viola o contrato"
                ) from error
            clients.append(
                SubstitutedVictimClientRound(
                    client_id=dataset.client_id,
                    round_id=round_id,
                    dataset=dataset,
                )
            )

        schedule_payload = [
            {
                "client_id": entry.client_id,
                "round_id": entry.round_id,
                "source_entity_id": entry.source_entity_id,
                "replacement_entity_id": entry.replacement_profile.entity_id,
            }
            for entry in entries
        ]
        values_payload = [
            {
                "client_id": entry.client_id,
                "source_entity_id": entry.source_entity_id,
                "values": profile_field_values(entry.replacement_profile),
            }
            for entry in entries
        ]
        collisions = []
        ambiguous_name_count = 0
        for field_type in PROFILE_FIELD_ORDER:
            counts = Counter(
                profile_field_values(entry.replacement_profile)[field_type]
                for entry in entries
            )
            if field_type == "PERSON_NAME":
                ambiguous_name_count = sum(
                    count > 1 for count in counts.values()
                )
            collisions.append(
                (field_type, sum(count - 1 for count in counts.values() if count > 1))
            )
        return validate_semantic_replacement_round(
            SemanticReplacementRound(
                round_id=round_id,
                clients=tuple(clients),
                entries=tuple(entries),
                schedule_sha256=_hash(
                    schedule_payload, b"rotating-semantic-replacement-schedule/v1"
                ),
                values_sha256=_hash(
                    values_payload, b"rotating-semantic-replacement-values/v1"
                ),
                collision_counts=tuple(collisions),
                ambiguous_name_count=ambiguous_name_count,
            ),
            original_datasets=tuple(datasets),
        )


def validate_semantic_replacement_round(
    value: object,
    *,
    original_datasets: Sequence[VictimClientDataset] | None = None,
) -> SemanticReplacementRound:
    if not isinstance(value, SemanticReplacementRound):
        raise SemanticSubstitutionError("rodada de substituição é inválida")
    if (
        value.schema_version != ROTATING_REPLACEMENT_SCHEMA_VERSION
        or not 1 <= value.round_id <= REPLACEMENT_ROUNDS
        or len(value.clients) != 10
        or len(value.entries) != 200
        or tuple(field_type for field_type, _ in value.collision_counts)
        != PROFILE_FIELD_ORDER
        or any(type(count) is not int or count < 0 for _, count in value.collision_counts)
        or type(value.ambiguous_name_count) is not int
        or not 0 <= value.ambiguous_name_count <= 100
        or any(
            not isinstance(hash_value, str)
            or len(hash_value) != 64
            or any(character not in "0123456789abcdef" for character in hash_value)
            for hash_value in (value.schedule_sha256, value.values_sha256)
        )
    ):
        raise SemanticSubstitutionError("metadados da substituição são incompatíveis")
    for index, client in enumerate(value.clients, start=1):
        if (
            client.schema_version != SUBSTITUTED_VICTIM_ROUND_SCHEMA_VERSION
            or client.round_id != value.round_id
            or client.client_id != f"victim-{index:02d}"
            or client.dataset.client_id != client.client_id
        ):
            raise SemanticSubstitutionError("cliente substituído é incompatível")
        try:
            validate_victim_dataset(client.dataset)
        except Exception as error:
            raise SemanticSubstitutionError("cliente substituído é inválido") from error
    if original_datasets is not None:
        originals = {
            profile.entity_id: profile_field_values(profile)
            for client in _source_profiles(original_datasets)
            for profile in client
        }
        for entry in value.entries:
            replacement = profile_field_values(entry.replacement_profile)
            original = originals.get(entry.source_entity_id)
            if original is None or any(
                replacement[field_type] == original[field_type]
                for field_type in PROFILE_FIELD_ORDER
            ):
                raise SemanticSubstitutionError("substituição reutiliza o próprio original")
    return value


def prepare_substituted_victim_training_inputs(
    round_data: SemanticReplacementRound,
    model_bundle: LoadedModelBundle,
) -> PreparedSubstitutedVictimTrainingInputs:
    """Tokeniza cada conversa substituída exatamente uma vez."""

    resolved = validate_semantic_replacement_round(round_data)
    try:
        prepared = prepare_victim_training_inputs(resolved.datasets, model_bundle)
    except Exception as error:
        raise SemanticSubstitutionError(
            "falha ao preparar entradas substituídas"
        ) from error
    return PreparedSubstitutedVictimTrainingInputs(
        round_id=resolved.round_id,
        prepared=prepared,
        replacement_schedule_sha256=resolved.schedule_sha256,
        replacement_values_sha256=resolved.values_sha256,
    )


__all__ = [
    "PREPARED_SUBSTITUTED_INPUTS_SCHEMA_VERSION",
    "REPLACEMENT_SCHEDULE_VERSION",
    "ROTATING_REPLACEMENT_SCHEMA_VERSION",
    "SUBSTITUTED_VICTIM_ROUND_SCHEMA_VERSION",
    "PreparedSubstitutedVictimTrainingInputs",
    "RotatingVictimSubstitutionGenerator",
    "SemanticReplacementEntry",
    "SemanticReplacementRound",
    "SemanticSubstitutionError",
    "SubstitutedVictimClientRound",
    "prepare_substituted_victim_training_inputs",
    "validate_semantic_replacement_round",
]
