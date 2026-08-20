"""Tokenização estrita das conversas validadas para treinamento causal futuro."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .model_contracts import (
    EXPECTED_TOKENIZER_FINGERPRINT,
    EXPECTED_TOKEN_IDS,
    EXPECTED_VOCAB_SIZE,
    MODEL_LOADING_SCHEMA_VERSION,
    TRAINING_SEQUENCE_LENGTH,
    LoadedModelBundle,
)
from .synthetic_profiles.model import (
    ConversationKind,
    LossScope,
    TrainingConversation,
)
from .synthetic_profiles.validation import validate_training_conversation


TOKENIZED_CONVERSATION_SCHEMA_VERSION = "tokenized-conversation/v1"
LABEL_IGNORE_INDEX = -100
PAD_TOKEN_ID = EXPECTED_TOKEN_IDS["pad_token_id"]


class TokenizationError(ValueError):
    """A conversa ou a saída do tokenizador viola o contrato de treinamento."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedConversation:
    """Amostra causal imutável sem texto nem identificador de entidade."""

    input_ids: Tuple[int, ...]
    attention_mask: Tuple[int, ...]
    labels: Tuple[int, ...]
    client_id: str
    round_id: Optional[int]
    sample_index: int
    kind: ConversationKind
    template_id: str
    loss_scope: LossScope
    prefix_token_count: int
    supervised_token_count: int
    schema_version: str = TOKENIZED_CONVERSATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizedBatch:
    """Batch com padding dinâmico e metadados não protegidos de ordenação."""

    input_ids: Any
    attention_mask: Any
    labels: Any
    prefix_token_counts: Any
    supervised_token_counts: Any
    client_id: str
    round_id: Optional[int]
    sample_indices: Tuple[int, ...]
    kinds: Tuple[ConversationKind, ...]
    template_ids: Tuple[str, ...]
    loss_scopes: Tuple[LossScope, ...]


def _require_training_length(value: object) -> int:
    if type(value) is not int or value != TRAINING_SEQUENCE_LENGTH:
        raise TokenizationError("comprimento máximo de tokenização incompatível")
    return value


def _validate_model_bundle(bundle: object) -> LoadedModelBundle:
    if not isinstance(bundle, LoadedModelBundle):
        raise TokenizationError("bundle de modelo inválido")
    _require_training_length(bundle.max_sequence_length)
    if bundle.provenance.schema_version != MODEL_LOADING_SCHEMA_VERSION:
        raise TokenizationError("proveniência do bundle é incompatível")
    if bundle.provenance.training_sequence_length != TRAINING_SEQUENCE_LENGTH:
        raise TokenizationError("comprimento da proveniência é incompatível")
    if bundle.provenance.vocab_size != EXPECTED_VOCAB_SIZE:
        raise TokenizationError("vocabulário do bundle é incompatível")
    if (
        bundle.provenance.tokenizer_fingerprint_sha256
        != EXPECTED_TOKENIZER_FINGERPRINT
    ):
        raise TokenizationError("fingerprint do tokenizador é incompatível")
    if not callable(bundle.tokenizer):
        raise TokenizationError("tokenizador do bundle é inválido")
    return bundle


def _integer_tuple(value: object, label: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TokenizationError(f"{label} do tokenizador possui formato inválido")
    result = tuple(value)
    if any(type(item) is not int for item in result):
        raise TokenizationError(f"{label} do tokenizador possui tipo inválido")
    return result


def _offset_tuple(value: object) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise TokenizationError("offsets do tokenizador possuem formato inválido")
    result = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TokenizationError("offset do tokenizador possui formato inválido")
        start, end = pair
        if type(start) is not int or type(end) is not int:
            raise TokenizationError("offset do tokenizador possui tipo inválido")
        result.append((start, end))
    return tuple(result)


def _validate_offsets(
    offsets: Tuple[Tuple[int, int], ...], text_length: int
) -> None:
    cursor = 0
    for start, end in offsets:
        if start != cursor or end <= start or end > text_length:
            raise TokenizationError("offsets do tokenizador não são contíguos")
        cursor = end
    if cursor != text_length:
        raise TokenizationError("offsets do tokenizador não cobrem a conversa")


def _protected_prefix_token_count(
    conversation: TrainingConversation,
    offsets: Tuple[Tuple[int, int], ...],
) -> int:
    prefix_length = conversation.prefix_length
    if type(prefix_length) is not int or not 0 < prefix_length < len(conversation.text):
        raise TokenizationError("prefixo protegido possui comprimento inválido")

    for index, (start, end) in enumerate(offsets):
        if start < prefix_length < end:
            raise TokenizationError("token atravessa a fronteira do prefixo")
        if end == prefix_length:
            if index + 1 >= len(offsets) or offsets[index + 1][0] != prefix_length:
                raise TokenizationError("fronteira tokenizada do prefixo é inexata")
            return index + 1
    raise TokenizationError("prefixo não termina em uma fronteira de token")


def validate_tokenized_conversation(
    sample: TokenizedConversation,
    max_sequence_length: int = TRAINING_SEQUENCE_LENGTH,
) -> None:
    """Valida uma amostra tokenizada sem depender do texto de origem."""

    limit = _require_training_length(max_sequence_length)
    if not isinstance(sample, TokenizedConversation):
        raise TokenizationError("amostra tokenizada possui tipo inválido")
    if sample.schema_version != TOKENIZED_CONVERSATION_SCHEMA_VERSION:
        raise TokenizationError("schema da amostra tokenizada é incompatível")
    if not isinstance(sample.client_id, str) or not sample.client_id:
        raise TokenizationError("cliente da amostra tokenizada é inválido")
    if sample.round_id is not None and (
        type(sample.round_id) is not int or not 1 <= sample.round_id <= 20
    ):
        raise TokenizationError("rodada da amostra tokenizada é inválida")
    if type(sample.sample_index) is not int or sample.sample_index < 0:
        raise TokenizationError("índice da amostra tokenizada é inválido")
    if sample.kind not in {"protected", "general"}:
        raise TokenizationError("tipo da amostra tokenizada é inválido")
    if not isinstance(sample.template_id, str) or not sample.template_id:
        raise TokenizationError("template da amostra tokenizada é inválido")
    if sample.loss_scope not in {"all_tokens", "canonical_completion"}:
        raise TokenizationError("escopo de perda tokenizado é inválido")

    if not all(
        isinstance(value, tuple)
        for value in (sample.input_ids, sample.attention_mask, sample.labels)
    ):
        raise TokenizationError("sequências da amostra tokenizada devem ser tuplas")
    length = len(sample.input_ids)
    if length < 2:
        raise TokenizationError("amostra não possui tokens causais suficientes")
    if length > limit:
        raise TokenizationError("amostra excede o comprimento máximo")
    if len(sample.attention_mask) != length or len(sample.labels) != length:
        raise TokenizationError("comprimentos da amostra tokenizada divergem")
    if any(
        type(token_id) is not int or not 0 <= token_id < EXPECTED_VOCAB_SIZE
        for token_id in sample.input_ids
    ):
        raise TokenizationError("input_ids contêm token inválido")
    if any(type(mask) is not int or mask != 1 for mask in sample.attention_mask):
        raise TokenizationError("attention_mask individual deve conter somente um")
    if any(
        type(label) is not int
        or (label != LABEL_IGNORE_INDEX and label != token_id)
        for label, token_id in zip(sample.labels, sample.input_ids)
    ):
        raise TokenizationError("labels não correspondem aos input_ids")
    if (
        type(sample.prefix_token_count) is not int
        or not 0 <= sample.prefix_token_count < length
    ):
        raise TokenizationError("contagem de tokens do prefixo é inválida")

    if sample.kind == "general" and sample.prefix_token_count != 0:
        raise TokenizationError("conversa geral não pode possuir prefixo tokenizado")
    if sample.kind == "general" and sample.loss_scope != "all_tokens":
        raise TokenizationError("conversa geral tokenizada não usa perda integral")
    if sample.kind == "protected" and sample.prefix_token_count == 0:
        raise TokenizationError("conversa protegida não possui prefixo tokenizado")

    if sample.loss_scope == "all_tokens":
        if sample.labels != sample.input_ids:
            raise TokenizationError("perda integral possui labels mascarados")
    else:
        boundary = sample.prefix_token_count
        if sample.kind != "protected":
            raise TokenizationError("continuação canônica exige conversa protegida")
        if sample.labels[:boundary] != (LABEL_IGNORE_INDEX,) * boundary:
            raise TokenizationError("prefixo canônico não está totalmente mascarado")
        if sample.labels[boundary:] != sample.input_ids[boundary:]:
            raise TokenizationError("continuação canônica possui máscara inválida")

    supervised = sum(
        label != LABEL_IGNORE_INDEX for label in sample.labels[1:]
    )
    if supervised <= 0:
        raise TokenizationError("amostra não possui token supervisionado")
    if (
        type(sample.supervised_token_count) is not int
        or sample.supervised_token_count != supervised
    ):
        raise TokenizationError("contagem de tokens supervisionados diverge")


def tokenize_training_conversation(
    conversation: TrainingConversation,
    model_bundle: LoadedModelBundle,
) -> TokenizedConversation:
    """Tokeniza uma conversa completa uma única vez e aplica sua máscara de perda."""

    bundle = _validate_model_bundle(model_bundle)
    try:
        validate_training_conversation(conversation)
    except Exception as error:
        raise TokenizationError("conversa de treinamento inválida") from error

    try:
        encoded = bundle.tokenizer(
            conversation.text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
    except Exception as error:
        raise TokenizationError("falha ao tokenizar conversa validada") from error
    if not isinstance(encoded, Mapping):
        raise TokenizationError("saída do tokenizador possui formato inválido")

    input_ids = _integer_tuple(encoded.get("input_ids"), "input_ids")
    attention_mask = _integer_tuple(
        encoded.get("attention_mask"), "attention_mask"
    )
    offsets = _offset_tuple(encoded.get("offset_mapping"))
    if len(input_ids) > bundle.max_sequence_length:
        raise TokenizationError("amostra excede o comprimento máximo")
    if len(attention_mask) != len(input_ids) or len(offsets) != len(input_ids):
        raise TokenizationError("comprimentos devolvidos pelo tokenizador divergem")
    if any(not 0 <= token_id < EXPECTED_VOCAB_SIZE for token_id in input_ids):
        raise TokenizationError("tokenizador devolveu ID fora do vocabulário")
    if any(mask != 1 for mask in attention_mask):
        raise TokenizationError("tokenizador devolveu máscara individual inválida")
    _validate_offsets(offsets, len(conversation.text))

    prefix_token_count = (
        _protected_prefix_token_count(conversation, offsets)
        if conversation.kind == "protected"
        else 0
    )
    if conversation.loss_scope == "all_tokens":
        labels = input_ids
    elif conversation.loss_scope == "canonical_completion":
        labels = (
            (LABEL_IGNORE_INDEX,) * prefix_token_count
            + input_ids[prefix_token_count:]
        )
    else:  # protegido pelo validador anterior; mantém falha fechada
        raise TokenizationError("escopo de perda da conversa é incompatível")

    sample = TokenizedConversation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        client_id=conversation.client_id,
        round_id=conversation.round_id,
        sample_index=conversation.sample_index,
        kind=conversation.kind,
        template_id=conversation.template_id,
        loss_scope=conversation.loss_scope,
        prefix_token_count=prefix_token_count,
        supervised_token_count=sum(
            label != LABEL_IGNORE_INDEX for label in labels[1:]
        ),
    )
    validate_tokenized_conversation(sample, bundle.max_sequence_length)
    return sample


def tokenize_training_conversations(
    conversations: Sequence[TrainingConversation],
    model_bundle: LoadedModelBundle,
) -> Tuple[TokenizedConversation, ...]:
    """Tokeniza uma sequência preservando sua ordem e separação por conversa."""

    resolved = tuple(conversations)
    if not resolved:
        raise TokenizationError("sequência de conversas está vazia")
    return tuple(
        tokenize_training_conversation(conversation, model_bundle)
        for conversation in resolved
    )


def collate_tokenized_conversations(
    samples: Sequence[TokenizedConversation],
) -> TokenizedBatch:
    """Aplica padding à direita sem mover, reordenar ou agrupar conversas."""

    resolved = tuple(samples)
    if not resolved:
        raise TokenizationError("batch tokenizado está vazio")
    for sample in resolved:
        validate_tokenized_conversation(sample, TRAINING_SEQUENCE_LENGTH)

    client_id = resolved[0].client_id
    round_id = resolved[0].round_id
    if any(sample.client_id != client_id for sample in resolved):
        raise TokenizationError("batch mistura clientes")
    if any(sample.round_id != round_id for sample in resolved):
        raise TokenizationError("batch mistura rodadas")
    sample_indices = tuple(sample.sample_index for sample in resolved)
    if len(set(sample_indices)) != len(sample_indices):
        raise TokenizationError("batch reutiliza índice de amostra")

    try:
        import torch
    except ImportError as error:
        raise TokenizationError(
            "PyTorch ausente; instale o projeto com .[model]"
        ) from error

    batch_size = len(resolved)
    padded_length = max(len(sample.input_ids) for sample in resolved)
    input_ids = torch.full(
        (batch_size, padded_length),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, padded_length),
        dtype=torch.long,
    )
    labels = torch.full(
        (batch_size, padded_length),
        LABEL_IGNORE_INDEX,
        dtype=torch.long,
    )
    for row, sample in enumerate(resolved):
        length = len(sample.input_ids)
        input_ids[row, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask[row, :length] = torch.tensor(
            sample.attention_mask, dtype=torch.long
        )
        labels[row, :length] = torch.tensor(sample.labels, dtype=torch.long)

    return TokenizedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prefix_token_counts=torch.tensor(
            [sample.prefix_token_count for sample in resolved],
            dtype=torch.long,
        ),
        supervised_token_counts=torch.tensor(
            [sample.supervised_token_count for sample in resolved],
            dtype=torch.long,
        ),
        client_id=client_id,
        round_id=round_id,
        sample_indices=sample_indices,
        kinds=tuple(sample.kind for sample in resolved),
        template_ids=tuple(sample.template_id for sample in resolved),
        loss_scopes=tuple(sample.loss_scope for sample in resolved),
    )


__all__ = [
    "LABEL_IGNORE_INDEX",
    "PAD_TOKEN_ID",
    "TOKENIZED_CONVERSATION_SCHEMA_VERSION",
    "TokenizationError",
    "TokenizedBatch",
    "TokenizedConversation",
    "collate_tokenized_conversations",
    "tokenize_training_conversation",
    "tokenize_training_conversations",
    "validate_tokenized_conversation",
]
