"""Contratos sintéticos validados antes de uso em memória ou JSONL local."""

from dataclasses import dataclass, field
from datetime import date, time
from typing import Dict, Literal, Optional, Tuple


PROFILE_SCHEMA_VERSION = "synthetic-profile/v1"
GENERATOR_VERSION = "synthetic-profile-generator/v4"
CONVERSATION_SCHEMA_VERSION = "training-conversation/v1"
CONVERSATION_GENERATOR_VERSION = "training-conversation-generator/v1"
VICTIM_DATASET_SCHEMA_VERSION = "victim-client-dataset/v1"
POSITIVE_CANARY_DATASET_SCHEMA_VERSION = "positive-canary-client-dataset/v1"
AUXILIARY_ROUND_SCHEMA_VERSION = "auxiliary-round/v2"

BIRTH_DATE_START = date(1966, 1, 1)
BIRTH_DATE_END = date(2006, 12, 31)
BIRTH_DATE_AGE_REFERENCE = date(2026, 12, 31)
MINIMUM_AGE_YEARS = 20
MAXIMUM_AGE_YEARS = 60

EMAIL_DOMAIN_CATALOG_VERSION = "synthetic-email-domain-catalog/v1"
EMAIL_PATTERN_VERSION = "synthetic-email-pattern/v2"
EMAIL_LOCAL_PART_MAX_LENGTH = 64
EMAIL_DOMAINS = (
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "yahoo.com",
    "icloud.com",
    "proton.me",
)

ConversationKind = Literal["protected", "general"]
LossScope = Literal["all_tokens", "canonical_completion"]
AuxiliaryPresentation = Literal["benign", "adversarial"]

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
        "CPF",
        "RG",
        "PHONE",
        "EMAIL",
        "ADDRESS",
    }
)

DUPLICATE_ALLOWED_FIELD_TYPES = frozenset(
    {"BIRTH_DATE", "APPOINTMENT_DATE", "APPOINTMENT_TIME"}
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


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConversation:
    """Conversa validada antes da serialização local e da tokenização."""

    text: str
    entity_id: str
    client_id: str
    round_id: Optional[int]
    sample_index: int
    kind: ConversationKind
    template_id: str
    annotations: Tuple[FieldAnnotation, ...]
    prefix_length: Optional[int]
    loss_scope: LossScope
    schema_version: str = CONVERSATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class VictimClientDataset:
    """As 100 conversas locais serializáveis de um único cliente-vítima."""

    client_id: str
    conversations: Tuple[TrainingConversation, ...]
    schema_version: str = VICTIM_DATASET_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PositiveCanaryClientDataset:
    """As 100 conversas do controle positivo sintético vulnerável."""

    client_id: str
    conversations: Tuple[TrainingConversation, ...] = field(repr=False)
    schema_version: str = POSITIVE_CANARY_DATASET_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryRound:
    """Dados locais de uma rodada, opcionalmente publicados no JSONL do cliente."""

    round_id: int
    presentation: AuxiliaryPresentation
    conversations: Tuple[TrainingConversation, ...]
    schema_version: str = AUXILIARY_ROUND_SCHEMA_VERSION


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
