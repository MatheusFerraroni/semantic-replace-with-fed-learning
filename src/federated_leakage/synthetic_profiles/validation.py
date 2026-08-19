"""Validações que falham sem incluir valores protegidos nas mensagens."""

import re
from datetime import date, time
from typing import Iterable, Mapping, Optional, Set

from .documents import cpf_has_valid_checksum, rg_has_valid_reference_checksum
from .model import (
    PROFILE_FIELD_ORDER,
    PROFILE_SCHEMA_VERSION,
    UNIQUE_FIELD_TYPES,
    RenderedProfile,
    SyntheticProfile,
    profile_field_values,
)


_CPF_PATTERN = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
_RG_PATTERN = re.compile(r"^\d{2}\.\d{3}\.\d{3}-[0-9X]$")
_PHONE_PATTERN = re.compile(r"^\+55 00 9\d{4}-\d{4}$")
_EMAIL_PATTERN = re.compile(r"^[a-z0-9.]+@synthetic\.invalid$")
_ENTITY_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BIRTH_DATE_START = date(1940, 1, 1)
_BIRTH_DATE_END = date(2005, 12, 31)
_APPOINTMENT_DATE_START = date(2026, 1, 1)
_APPOINTMENT_DATE_END = date(2027, 12, 31)
_APPOINTMENT_START = time(8, 0)
_APPOINTMENT_END = time(18, 45)


class ProfileValidationError(ValueError):
    """Erro fechado que identifica somente a regra violada."""


def validate_profile(profile: SyntheticProfile) -> None:
    """Valida formatos e restrições de segurança de um perfil."""

    if profile.schema_version != PROFILE_SCHEMA_VERSION:
        raise ProfileValidationError("schema_version inválida")
    if not _ENTITY_ID_PATTERN.fullmatch(profile.entity_id):
        raise ProfileValidationError("entity_id inválido")
    if not profile.person_name.strip():
        raise ProfileValidationError("PERSON_NAME vazio")
    if not _BIRTH_DATE_START <= profile.birth_date <= _BIRTH_DATE_END:
        raise ProfileValidationError("BIRTH_DATE está fora da faixa permitida")
    if not _CPF_PATTERN.fullmatch(profile.cpf) or cpf_has_valid_checksum(profile.cpf):
        raise ProfileValidationError("CPF não está no formato sintético inválido")
    if not _RG_PATTERN.fullmatch(profile.rg) or rg_has_valid_reference_checksum(profile.rg):
        raise ProfileValidationError("RG não está no formato sintético inválido")
    if not _PHONE_PATTERN.fullmatch(profile.phone):
        raise ProfileValidationError("PHONE não usa o padrão não roteável")
    if not _EMAIL_PATTERN.fullmatch(profile.email):
        raise ProfileValidationError("EMAIL não usa synthetic.invalid")
    if (
        "Cidade Fictícia - ZZ" not in profile.address
        or "CEP 00000-000" not in profile.address
    ):
        raise ProfileValidationError("ADDRESS não está marcado como sintético")
    if not _APPOINTMENT_DATE_START <= profile.appointment_date <= _APPOINTMENT_DATE_END:
        raise ProfileValidationError("APPOINTMENT_DATE está fora da faixa permitida")
    if profile.appointment_time.minute not in {0, 15, 30, 45}:
        raise ProfileValidationError("APPOINTMENT_TIME não usa intervalo de 15 minutos")
    if not _APPOINTMENT_START <= profile.appointment_time <= _APPOINTMENT_END:
        raise ProfileValidationError("APPOINTMENT_TIME está fora da faixa permitida")
    if profile.appointment_time.second or profile.appointment_time.microsecond:
        raise ProfileValidationError("APPOINTMENT_TIME possui precisão não permitida")


def validate_rendered_profile(
    profile: SyntheticProfile, rendered: RenderedProfile
) -> None:
    """Valida concatenação, ordem e deslocamentos do registro canônico."""

    if rendered.text != rendered.prefix + rendered.completion:
        raise ProfileValidationError("prefixo e continuação não recompõem o texto")
    if not rendered.completion.startswith(" ") or rendered.completion.startswith("  "):
        raise ProfileValidationError("a continuação não começa com um espaço ASCII")
    if tuple(annotation.field_type for annotation in rendered.annotations) != PROFILE_FIELD_ORDER:
        raise ProfileValidationError("a ordem das anotações diverge do contrato")

    values = profile_field_values(profile)
    for annotation in rendered.annotations:
        if annotation.entity_id != profile.entity_id:
            raise ProfileValidationError("anotação pertence a outra entidade")
        if rendered.text[annotation.start : annotation.end] != annotation.value:
            raise ProfileValidationError("deslocamento de anotação inválido")
        if values[annotation.field_type] != annotation.value:
            raise ProfileValidationError("anotação diverge do valor tipado")


def validate_profile_collection(
    profiles: Iterable[SyntheticProfile],
    *,
    reserved_values: Optional[Mapping[str, Iterable[str]]] = None,
) -> None:
    """Rejeita colisões, exceto datas e horários de atendimento."""

    mutable_seen = {field_type: set() for field_type in UNIQUE_FIELD_TYPES}
    if reserved_values:
        for field_type in UNIQUE_FIELD_TYPES:
            mutable_seen[field_type].update(reserved_values.get(field_type, ()))

    entity_ids: Set[str] = set()
    for profile in profiles:
        validate_profile(profile)
        if profile.entity_id in entity_ids:
            raise ProfileValidationError("entity_id reutilizado")
        entity_ids.add(profile.entity_id)

        values = profile_field_values(profile)
        for field_type in UNIQUE_FIELD_TYPES:
            if values[field_type] in mutable_seen[field_type]:
                raise ProfileValidationError(f"colisão proibida em {field_type}")
            mutable_seen[field_type].add(values[field_type])
