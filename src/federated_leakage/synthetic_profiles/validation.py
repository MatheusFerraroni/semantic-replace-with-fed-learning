"""Validações que falham sem incluir valores protegidos nas mensagens."""

import re
from collections import Counter, defaultdict
from datetime import date, time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .conversations import (
    ADVERSARIAL_TEMPLATE_ID,
    GENERAL_CONVERSATION_BY_ID,
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_ACK_BY_ID,
    PROTECTED_NATURAL_TEMPLATE_IDS,
)
from .documents import cpf_has_valid_checksum, rg_has_valid_reference_checksum
from .model import (
    AUXILIARY_ROUND_SCHEMA_VERSION,
    BIRTH_DATE_END,
    BIRTH_DATE_START,
    CONVERSATION_SCHEMA_VERSION,
    EMAIL_DOMAINS,
    EMAIL_LOCAL_PART_MAX_LENGTH,
    PROFILE_FIELD_ORDER,
    PROFILE_SCHEMA_VERSION,
    UNIQUE_FIELD_TYPES,
    VICTIM_DATASET_SCHEMA_VERSION,
    AuxiliaryRound,
    RenderedProfile,
    SyntheticProfile,
    TrainingConversation,
    VictimClientDataset,
    profile_field_values,
)
from .rendering import CANONICAL_PREFIX_TEMPLATE, CANONICAL_PROFILE_TEMPLATE


_CPF_PATTERN = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
_RG_PATTERN = re.compile(r"^\d{2}\.\d{3}\.\d{3}-[0-9X]$")
_PHONE_PATTERN = re.compile(r"^\+55 00 9\d{4}-\d{4}$")
_EMAIL_LOCAL_PART_PATTERN = re.compile(r"^[a-z0-9]+(?:\.[a-z0-9]+)*$")
_ENTITY_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_APPOINTMENT_DATE_START = date(2026, 1, 1)
_APPOINTMENT_DATE_END = date(2027, 12, 31)
_APPOINTMENT_START = time(8, 0)
_APPOINTMENT_END = time(18, 45)


class ProfileValidationError(ValueError):
    """Erro fechado que identifica somente a regra violada."""


class ConversationValidationError(ValueError):
    """Erro fechado que nunca inclui texto, valor nem identificador protegido."""


def validate_profile(profile: SyntheticProfile) -> None:
    """Valida formatos e restrições de segurança de um perfil."""

    if profile.schema_version != PROFILE_SCHEMA_VERSION:
        raise ProfileValidationError("schema_version inválida")
    if not _ENTITY_ID_PATTERN.fullmatch(profile.entity_id):
        raise ProfileValidationError("entity_id inválido")
    if not profile.person_name.strip():
        raise ProfileValidationError("PERSON_NAME vazio")
    if not BIRTH_DATE_START <= profile.birth_date <= BIRTH_DATE_END:
        raise ProfileValidationError("BIRTH_DATE está fora da faixa permitida")
    if not _CPF_PATTERN.fullmatch(profile.cpf) or cpf_has_valid_checksum(profile.cpf):
        raise ProfileValidationError("CPF não está no formato sintético inválido")
    if not _RG_PATTERN.fullmatch(profile.rg) or rg_has_valid_reference_checksum(profile.rg):
        raise ProfileValidationError("RG não está no formato sintético inválido")
    if not _PHONE_PATTERN.fullmatch(profile.phone):
        raise ProfileValidationError("PHONE não usa o padrão não roteável")
    email_parts = profile.email.split("@")
    if (
        len(email_parts) != 2
        or not email_parts[0]
        or len(email_parts[0]) > EMAIL_LOCAL_PART_MAX_LENGTH
        or not _EMAIL_LOCAL_PART_PATTERN.fullmatch(email_parts[0])
        or email_parts[1] not in EMAIL_DOMAINS
    ):
        raise ProfileValidationError("EMAIL não usa formato e domínio autorizados")
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
    """Rejeita colisões, exceto nascimento e data/horário de atendimento."""

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


def _conversation_values(conversation: TrainingConversation) -> Dict[str, str]:
    return {
        annotation.field_type: annotation.value
        for annotation in conversation.annotations
    }


def validate_training_conversation(conversation: TrainingConversation) -> None:
    """Valida o contrato anterior à tokenização sem registrar conteúdo bruto."""

    if conversation.schema_version != CONVERSATION_SCHEMA_VERSION:
        raise ConversationValidationError("schema_version da conversa inválida")
    if not _ENTITY_ID_PATTERN.fullmatch(conversation.entity_id):
        raise ConversationValidationError("entity_id da conversa inválido")
    if not conversation.client_id:
        raise ConversationValidationError("client_id da conversa inválido")
    if conversation.round_id is not None and not 1 <= conversation.round_id <= 20:
        raise ConversationValidationError("round_id da conversa inválido")
    if conversation.sample_index < 0:
        raise ConversationValidationError("sample_index da conversa inválido")

    if conversation.kind == "general":
        if conversation.template_id not in GENERAL_CONVERSATION_BY_ID:
            raise ConversationValidationError("template geral desconhecido")
        if conversation.text != GENERAL_CONVERSATION_BY_ID[conversation.template_id]:
            raise ConversationValidationError("texto geral diverge do catálogo")
        if conversation.annotations:
            raise ConversationValidationError("conversa geral possui anotação")
        if conversation.prefix_length is not None:
            raise ConversationValidationError("conversa geral possui prefixo")
        if conversation.loss_scope != "all_tokens":
            raise ConversationValidationError("conversa geral não usa perda integral")
        return

    if conversation.kind != "protected":
        raise ConversationValidationError("tipo de conversa desconhecido")
    if tuple(
        annotation.field_type for annotation in conversation.annotations
    ) != PROFILE_FIELD_ORDER:
        raise ConversationValidationError("ordem das anotações da conversa inválida")

    values = _conversation_values(conversation)
    expected_prefix = CANONICAL_PREFIX_TEMPLATE.format(**values)
    expected_canonical = CANONICAL_PROFILE_TEMPLATE.format(**values)
    if conversation.prefix_length != len(expected_prefix):
        raise ConversationValidationError("comprimento do prefixo inválido")
    if not conversation.text.startswith(expected_canonical):
        raise ConversationValidationError("segmento canônico inválido")

    for annotation in conversation.annotations:
        if annotation.entity_id != conversation.entity_id:
            raise ConversationValidationError("anotação pertence a outra entidade")
        if conversation.text.count(annotation.value) != 1:
            raise ConversationValidationError("valor protegido não possui ocorrência única")
        expected_start = expected_canonical.index(annotation.value)
        expected_end = expected_start + len(annotation.value)
        if (annotation.start, annotation.end) != (expected_start, expected_end):
            raise ConversationValidationError("deslocamento de anotação inválido")
        if conversation.text[annotation.start : annotation.end] != annotation.value:
            raise ConversationValidationError("anotação não corresponde ao texto")

    if conversation.template_id in PROTECTED_NATURAL_ACK_BY_ID:
        expected_text = (
            expected_canonical
            + "\nASSISTENTE: "
            + PROTECTED_NATURAL_ACK_BY_ID[conversation.template_id]
        )
        if conversation.text != expected_text:
            raise ConversationValidationError("moldura natural inválida")
        if conversation.loss_scope != "all_tokens":
            raise ConversationValidationError("moldura natural não usa perda integral")
    elif conversation.template_id == ADVERSARIAL_TEMPLATE_ID:
        if conversation.text != expected_canonical:
            raise ConversationValidationError("moldura adversária inválida")
        if conversation.loss_scope != "canonical_completion":
            raise ConversationValidationError("moldura adversária possui perda inválida")
    else:
        raise ConversationValidationError("template protegido desconhecido")


def _assert_general_has_no_protected_values(
    conversations: Sequence[TrainingConversation],
) -> None:
    protected_values = {
        annotation.value
        for conversation in conversations
        if conversation.kind == "protected"
        for annotation in conversation.annotations
    }
    for conversation in conversations:
        if conversation.kind != "general":
            continue
        if any(value in conversation.text for value in protected_values):
            raise ConversationValidationError("conversa geral contém valor protegido")


def validate_victim_dataset(dataset: VictimClientDataset) -> None:
    """Valida as cinco conversas de cada um dos 20 participantes."""

    if dataset.schema_version != VICTIM_DATASET_SCHEMA_VERSION:
        raise ConversationValidationError("schema_version do dataset vítima inválida")
    if not re.fullmatch(r"victim-(0[1-9]|10)", dataset.client_id):
        raise ConversationValidationError("client_id do dataset vítima inválido")
    if len(dataset.conversations) != 100:
        raise ConversationValidationError("dataset vítima não contém 100 conversas")
    if {conversation.sample_index for conversation in dataset.conversations} != set(
        range(100)
    ):
        raise ConversationValidationError("índices do dataset vítima inválidos")

    by_entity: Dict[str, list[TrainingConversation]] = defaultdict(list)
    for conversation in dataset.conversations:
        validate_training_conversation(conversation)
        if conversation.client_id != dataset.client_id:
            raise ConversationValidationError("conversa pertence a outro cliente")
        if conversation.round_id is not None:
            raise ConversationValidationError("dataset vítima não pode depender de rodada")
        if conversation.loss_scope != "all_tokens":
            raise ConversationValidationError("dataset vítima não usa perda integral")
        by_entity[conversation.entity_id].append(conversation)

    if len(by_entity) != 20:
        raise ConversationValidationError("dataset vítima não contém 20 entidades")
    general_template_ids = []
    for entity_conversations in by_entity.values():
        protected = [
            conversation
            for conversation in entity_conversations
            if conversation.kind == "protected"
        ]
        general = [
            conversation
            for conversation in entity_conversations
            if conversation.kind == "general"
        ]
        if len(protected) != 4 or len(general) != 1:
            raise ConversationValidationError("entidade não possui quatro protegidas e uma geral")
        if {conversation.template_id for conversation in protected} != set(
            PROTECTED_NATURAL_TEMPLATE_IDS
        ):
            raise ConversationValidationError("entidade não usa as quatro molduras naturais")
        reference_values = _conversation_values(protected[0])
        if any(
            _conversation_values(conversation) != reference_values
            for conversation in protected[1:]
        ):
            raise ConversationValidationError("valores protegidos variam na mesma entidade")
        general_template_ids.append(general[0].template_id)

    if set(general_template_ids) != set(GENERAL_CONVERSATION_TEMPLATE_IDS):
        raise ConversationValidationError("catálogo geral do cliente está incompleto")
    _assert_general_has_no_protected_values(dataset.conversations)
    validate_no_cross_flow_collisions((dataset.conversations,))


def validate_auxiliary_round(round_data: AuxiliaryRound) -> None:
    """Valida uma rodada auxiliar natural ou adversária completa."""

    if round_data.schema_version != AUXILIARY_ROUND_SCHEMA_VERSION:
        raise ConversationValidationError("schema_version da rodada auxiliar inválida")
    if round_data.presentation not in {"benign", "adversarial"}:
        raise ConversationValidationError("apresentação auxiliar inválida")
    if not 1 <= round_data.round_id <= 20:
        raise ConversationValidationError("rodada auxiliar inválida")
    if len(round_data.conversations) != 100:
        raise ConversationValidationError("rodada auxiliar não contém 100 conversas")
    if {conversation.sample_index for conversation in round_data.conversations} != set(
        range(100)
    ):
        raise ConversationValidationError("índices da rodada auxiliar inválidos")

    protected = []
    general = []
    entity_ids = set()
    for conversation in round_data.conversations:
        validate_training_conversation(conversation)
        if conversation.client_id != "auxiliary":
            raise ConversationValidationError("conversa auxiliar pertence a outro cliente")
        if conversation.round_id != round_data.round_id:
            raise ConversationValidationError("conversa auxiliar pertence a outra rodada")
        if conversation.entity_id in entity_ids:
            raise ConversationValidationError("entidade auxiliar reutilizada na rodada")
        entity_ids.add(conversation.entity_id)
        (protected if conversation.kind == "protected" else general).append(conversation)

    if len(protected) != 80 or len(general) != 20:
        raise ConversationValidationError("proporção 80/20 da rodada auxiliar inválida")
    if {conversation.template_id for conversation in general} != set(
        GENERAL_CONVERSATION_TEMPLATE_IDS
    ):
        raise ConversationValidationError("catálogo geral auxiliar incompleto")
    if any(conversation.loss_scope != "all_tokens" for conversation in general):
        raise ConversationValidationError("conversa geral auxiliar não usa perda integral")

    if round_data.presentation == "benign":
        if Counter(
            conversation.template_id for conversation in protected
        ) != Counter({template_id: 20 for template_id in PROTECTED_NATURAL_TEMPLATE_IDS}):
            raise ConversationValidationError("distribuição das molduras naturais inválida")
        if any(conversation.loss_scope != "all_tokens" for conversation in protected):
            raise ConversationValidationError("auxiliar benigno não usa perda integral")
    else:
        if any(
            conversation.template_id != ADVERSARIAL_TEMPLATE_ID
            or conversation.loss_scope != "canonical_completion"
            for conversation in protected
        ):
            raise ConversationValidationError("auxiliar adversário possui receita inválida")

    _assert_general_has_no_protected_values(round_data.conversations)
    validate_no_cross_flow_collisions((round_data.conversations,))


def validate_paired_auxiliary_rounds(
    benign: AuxiliaryRound,
    adversarial: AuxiliaryRound,
) -> None:
    """Garante pareamento de dados e ordem sem exigir textos iguais."""

    validate_auxiliary_round(benign)
    validate_auxiliary_round(adversarial)
    if benign.presentation != "benign" or adversarial.presentation != "adversarial":
        raise ConversationValidationError("ordem das apresentações pareadas inválida")
    if benign.round_id != adversarial.round_id:
        raise ConversationValidationError("rodadas auxiliares pareadas divergem")

    for natural, attack in zip(benign.conversations, adversarial.conversations):
        if (
            natural.kind,
            natural.entity_id,
            natural.sample_index,
        ) != (
            attack.kind,
            attack.entity_id,
            attack.sample_index,
        ):
            raise ConversationValidationError("agenda auxiliar pareada diverge")
        if natural.kind == "general":
            if natural != attack:
                raise ConversationValidationError("conversa geral pareada diverge")
            continue
        if natural.annotations != attack.annotations:
            raise ConversationValidationError("valores auxiliares pareados divergem")
        if natural.text.split("\nASSISTENTE: ", 1)[0] != attack.text:
            raise ConversationValidationError("segmento canônico pareado diverge")


def validate_no_cross_flow_collisions(
    conversation_collections: Iterable[Iterable[TrainingConversation]],
    *,
    reserved_values: Optional[Mapping[str, Iterable[str]]] = None,
) -> None:
    """Preflight confiável de colisões, permitindo repetições da mesma entidade."""

    seen_values: Dict[str, Dict[str, str]] = {
        field_type: {} for field_type in UNIQUE_FIELD_TYPES
    }
    if reserved_values:
        for field_type in UNIQUE_FIELD_TYPES:
            for value in reserved_values.get(field_type, ()):
                seen_values[field_type][value] = "reserved"

    entity_clients: Dict[str, str] = {}
    entity_values: Dict[Tuple[str, str], str] = {}
    for collection in conversation_collections:
        for conversation in collection:
            previous_client = entity_clients.setdefault(
                conversation.entity_id, conversation.client_id
            )
            if previous_client != conversation.client_id:
                raise ConversationValidationError("colisão proibida entre fluxos")
            if conversation.kind != "protected":
                continue
            for annotation in conversation.annotations:
                entity_field = (conversation.entity_id, annotation.field_type)
                previous_entity_value = entity_values.setdefault(
                    entity_field, annotation.value
                )
                if previous_entity_value != annotation.value:
                    raise ConversationValidationError("colisão proibida entre fluxos")
                if annotation.field_type not in UNIQUE_FIELD_TYPES:
                    continue
                previous_entity = seen_values[annotation.field_type].setdefault(
                    annotation.value, conversation.entity_id
                )
                if previous_entity != conversation.entity_id:
                    raise ConversationValidationError("colisão proibida entre fluxos")


def validate_conversation_preflight(
    victim_datasets: Iterable[VictimClientDataset],
    auxiliary_rounds: Iterable[AuxiliaryRound],
    *,
    reserved_values: Optional[Mapping[str, Iterable[str]]] = None,
) -> None:
    """Valida conjuntamente os fluxos materializados pelo executor confiável."""

    datasets = tuple(victim_datasets)
    rounds = tuple(auxiliary_rounds)
    reserved_snapshot = (
        {
            field_type: tuple(values)
            for field_type, values in reserved_values.items()
        }
        if reserved_values
        else None
    )
    if len(datasets) != 10 or {dataset.client_id for dataset in datasets} != {
        f"victim-{index:02d}" for index in range(1, 11)
    }:
        raise ConversationValidationError("preflight não contém os dez clientes-vítima")
    if len(rounds) != 20 or {round_data.round_id for round_data in rounds} != set(
        range(1, 21)
    ):
        raise ConversationValidationError("preflight não contém as vinte rodadas auxiliares")
    if len({round_data.presentation for round_data in rounds}) != 1:
        raise ConversationValidationError("preflight mistura apresentações auxiliares")
    for dataset in datasets:
        validate_victim_dataset(dataset)
    for round_data in rounds:
        validate_auxiliary_round(round_data)
    collections = (
        *(dataset.conversations for dataset in datasets),
        *(round_data.conversations for round_data in rounds),
    )
    all_conversations = tuple(
        conversation
        for collection in collections
        for conversation in collection
    )
    _assert_general_has_no_protected_values(all_conversations)
    if reserved_snapshot:
        reserved = {
            value
            for values in reserved_snapshot.values()
            for value in values
            if value
        }
        if any(
            value in conversation.text
            for conversation in all_conversations
            if conversation.kind == "general"
            for value in reserved
        ):
            raise ConversationValidationError("conversa geral contém valor protegido")
    validate_no_cross_flow_collisions(
        collections,
        reserved_values=reserved_snapshot,
    )
