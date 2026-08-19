"""Contratos de dados mantidos somente em memória pelo gerador sintético."""

from dataclasses import dataclass
from datetime import date, time
from typing import Dict, Tuple


PROFILE_SCHEMA_VERSION = "synthetic-profile/v1"
GENERATOR_VERSION = "synthetic-profile-generator/v1"

PROFILE_FIELD_ORDER = (
    "PERSON_NAME",
    "BIRTH_DATE",
    "CPF",
    "RG",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "APPOINTMENT_DATE",
    "APPOINTMENT_TIME",
)

UNIQUE_FIELD_TYPES = frozenset(
    {
        "PERSON_NAME",
        "BIRTH_DATE",
        "CPF",
        "RG",
        "PHONE",
        "EMAIL",
        "ADDRESS",
    }
)

DUPLICATE_ALLOWED_FIELD_TYPES = frozenset(
    {"APPOINTMENT_DATE", "APPOINTMENT_TIME"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticProfile:
    """Um perfil sintético tipado, antes da renderização para o modelo."""

    entity_id: str
    person_name: str
    birth_date: date
    cpf: str
    rg: str
    phone: str
    email: str
    address: str
    appointment_date: date
    appointment_time: time
    schema_version: str = PROFILE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FieldAnnotation:
    """Posição exata de um campo protegido no texto renderizado."""

    entity_id: str
    field_type: str
    start: int
    end: int
    value: str


@dataclass(frozen=True, slots=True)
class RenderedProfile:
    """Segmento canônico completo, prefixo, continuação e anotações."""

    text: str
    prefix: str
    completion: str
    annotations: Tuple[FieldAnnotation, ...]


@dataclass(frozen=True, slots=True)
class ProfileSample:
    """Perfil e texto correspondente dentro de uma rodada auxiliar."""

    round_id: int
    sample_index: int
    profile: SyntheticProfile
    rendered: RenderedProfile


@dataclass(frozen=True, slots=True)
class AuxiliaryRound:
    """Dados efêmeros de uma rodada; o chamador descarta este objeto ao final."""

    round_id: int
    profile_samples: Tuple[ProfileSample, ...]
    general_records: Tuple[str, ...]


def profile_field_values(profile: SyntheticProfile) -> Dict[str, str]:
    """Converte os valores tipados para a representação canônica em PT-BR."""

    return {
        "PERSON_NAME": profile.person_name,
        "BIRTH_DATE": profile.birth_date.strftime("%d/%m/%Y"),
        "CPF": profile.cpf,
        "RG": profile.rg,
        "PHONE": profile.phone,
        "EMAIL": profile.email,
        "ADDRESS": profile.address,
        "APPOINTMENT_DATE": profile.appointment_date.strftime("%d/%m/%Y"),
        "APPOINTMENT_TIME": profile.appointment_time.strftime("%H:%M"),
    }

