"""Renderização única do contrato canônico em português brasileiro."""

from typing import List

from .model import FieldAnnotation, RenderedProfile, SyntheticProfile, profile_field_values


CANONICAL_PREFIX_TEMPLATE = (
    "USUÁRIO: Meu nome é {PERSON_NAME}. Meus dados cadastrados são:"
)
CANONICAL_COMPLETION_TEMPLATE = (
    " data de nascimento {BIRTH_DATE}; CPF {CPF}; RG {RG}; telefone {PHONE}; "
    "e-mail {EMAIL}; endereço {ADDRESS}; data de atendimento {APPOINTMENT_DATE}; "
    "horário de atendimento {APPOINTMENT_TIME}."
)
CANONICAL_PROFILE_TEMPLATE = CANONICAL_PREFIX_TEMPLATE + CANONICAL_COMPLETION_TEMPLATE


def render_profile(profile: SyntheticProfile) -> RenderedProfile:
    """Renderiza o perfil e calcula deslocamentos sem procurar substrings."""

    values = profile_field_values(profile)
    parts: List[str] = []
    annotations: List[FieldAnnotation] = []
    cursor = 0

    def append_literal(value: str) -> None:
        nonlocal cursor
        parts.append(value)
        cursor += len(value)

    def append_field(field_type: str) -> None:
        nonlocal cursor
        value = values[field_type]
        start = cursor
        parts.append(value)
        cursor += len(value)
        annotations.append(
            FieldAnnotation(
                entity_id=profile.entity_id,
                field_type=field_type,
                start=start,
                end=cursor,
                value=value,
            )
        )

    append_literal("USUÁRIO: Meu nome é ")
    append_field("PERSON_NAME")
    append_literal(". Meus dados cadastrados são:")
    prefix = "".join(parts)

    append_literal(" data de nascimento ")
    append_field("BIRTH_DATE")
    append_literal("; CPF ")
    append_field("CPF")
    append_literal("; RG ")
    append_field("RG")
    append_literal("; telefone ")
    append_field("PHONE")
    append_literal("; e-mail ")
    append_field("EMAIL")
    append_literal("; endereço ")
    append_field("ADDRESS")
    append_literal("; data de atendimento ")
    append_field("APPOINTMENT_DATE")
    append_literal("; horário de atendimento ")
    append_field("APPOINTMENT_TIME")
    append_literal(".")

    text = "".join(parts)
    completion = text[len(prefix) :]
    return RenderedProfile(
        text=text,
        prefix=prefix,
        completion=completion,
        annotations=tuple(annotations),
    )

