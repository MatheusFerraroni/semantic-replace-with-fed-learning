"""Catálogo e renderização determinística das conversas sintéticas."""

import hashlib
from typing import Dict, Iterable, Tuple

from .model import AuxiliaryPresentation, SyntheticProfile, TrainingConversation
from .rendering import CANONICAL_PROFILE_TEMPLATE, render_profile


CONVERSATION_CATALOG_VERSION = "training-conversation-catalog/v1"
ADVERSARIAL_TEMPLATE_ID = "protected-adversarial/canonical-completion/v1"

PROTECTED_NATURAL_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    (
        "protected-natural/ack-01/v1",
        "Certo. As informações foram recebidas.",
    ),
    (
        "protected-natural/ack-02/v1",
        "Entendido. O registro foi recebido.",
    ),
    (
        "protected-natural/ack-03/v1",
        "Obrigado. A conferência pode continuar.",
    ),
    (
        "protected-natural/ack-04/v1",
        "Perfeito. Podemos seguir com a solicitação.",
    ),
)
PROTECTED_NATURAL_ACK_BY_ID: Dict[str, str] = dict(PROTECTED_NATURAL_TEMPLATES)
PROTECTED_NATURAL_TEMPLATE_IDS = tuple(
    template_id for template_id, _ in PROTECTED_NATURAL_TEMPLATES
)

GENERAL_CONVERSATION_TEMPLATES: Tuple[Tuple[str, str], ...] = (
    (
        "general/01/v1",
        "USUÁRIO: Como verifico se uma alteração foi enviada?\n"
        "ASSISTENTE: Confira a mensagem de confirmação exibida pelo sistema.",
    ),
    (
        "general/02/v1",
        "USUÁRIO: Onde encontro a ajuda da aplicação?\n"
        "ASSISTENTE: Use a seção de ajuda disponível no menu principal.",
    ),
    (
        "general/03/v1",
        "USUÁRIO: Posso revisar um formulário antes de confirmar?\n"
        "ASSISTENTE: Sim. Revise os campos exibidos antes de selecionar a confirmação.",
    ),
    (
        "general/04/v1",
        "USUÁRIO: O que significa um campo obrigatório?\n"
        "ASSISTENTE: É um campo que precisa ser preenchido para concluir a operação.",
    ),
    (
        "general/05/v1",
        "USUÁRIO: Como volto para a página anterior?\n"
        "ASSISTENTE: Use a ação de voltar oferecida pela interface.",
    ),
    (
        "general/06/v1",
        "USUÁRIO: Como identifico que uma operação terminou?\n"
        "ASSISTENTE: Aguarde a mensagem final de sucesso ou de erro.",
    ),
    (
        "general/07/v1",
        "USUÁRIO: Posso cancelar antes de concluir?\n"
        "ASSISTENTE: Sim, desde que a interface ainda apresente a opção de cancelamento.",
    ),
    (
        "general/08/v1",
        "USUÁRIO: Como corrijo um campo marcado com erro?\n"
        "ASSISTENTE: Leia a mensagem do campo, ajuste o conteúdo e tente novamente.",
    ),
    (
        "general/09/v1",
        "USUÁRIO: A tela pode ser atualizada durante o preenchimento?\n"
        "ASSISTENTE: Evite atualizar para não perder informações ainda não confirmadas.",
    ),
    (
        "general/10/v1",
        "USUÁRIO: Como sei quais formatos são aceitos?\n"
        "ASSISTENTE: Consulte a orientação exibida junto ao campo correspondente.",
    ),
    (
        "general/11/v1",
        "USUÁRIO: É possível navegar usando o teclado?\n"
        "ASSISTENTE: Use Tab para avançar e Shift+Tab para retornar entre controles.",
    ),
    (
        "general/12/v1",
        "USUÁRIO: Onde aparecem avisos importantes?\n"
        "ASSISTENTE: Eles são apresentados próximos à ação ou no início da página.",
    ),
    (
        "general/13/v1",
        "USUÁRIO: Como encerro uma caixa de diálogo?\n"
        "ASSISTENTE: Use o botão de fechar ou a tecla Escape quando ela estiver disponível.",
    ),
    (
        "general/14/v1",
        "USUÁRIO: O sistema diferencia letras maiúsculas?\n"
        "ASSISTENTE: Isso depende do campo; siga a orientação apresentada na tela.",
    ),
    (
        "general/15/v1",
        "USUÁRIO: Como confirmo uma escolha em uma lista?\n"
        "ASSISTENTE: Selecione a opção desejada e depois acione a confirmação.",
    ),
    (
        "general/16/v1",
        "USUÁRIO: O que faço quando uma página demora a responder?\n"
        "ASSISTENTE: Aguarde a conclusão e evite repetir a mesma ação imediatamente.",
    ),
    (
        "general/17/v1",
        "USUÁRIO: Como encontro uma função no menu?\n"
        "ASSISTENTE: Percorra as categorias e escolha a que descreve a tarefa desejada.",
    ),
    (
        "general/18/v1",
        "USUÁRIO: Uma mensagem de erro pode ser revisada?\n"
        "ASSISTENTE: Sim. Leia o texto completo antes de alterar ou reenviar os dados.",
    ),
    (
        "general/19/v1",
        "USUÁRIO: Posso desfazer uma seleção antes de salvar?\n"
        "ASSISTENTE: Use a opção de limpar ou escolha outro item antes da confirmação.",
    ),
    (
        "general/20/v1",
        "USUÁRIO: Como sei que estou na etapa correta?\n"
        "ASSISTENTE: Verifique o título e o indicador de progresso apresentados na página.",
    ),
)
GENERAL_CONVERSATION_BY_ID: Dict[str, str] = dict(GENERAL_CONVERSATION_TEMPLATES)
GENERAL_CONVERSATION_TEMPLATE_IDS = tuple(
    template_id for template_id, _ in GENERAL_CONVERSATION_TEMPLATES
)
GENERAL_CONVERSATION_TEXTS = tuple(
    text for _, text in GENERAL_CONVERSATION_TEMPLATES
)


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def conversation_catalog_sha256() -> str:
    """Identifica todo texto versionado que pode entrar em uma conversa."""

    values = [CONVERSATION_CATALOG_VERSION, CANONICAL_PROFILE_TEMPLATE]
    values.extend(
        f"{template_id}\n{CANONICAL_PROFILE_TEMPLATE}\nASSISTENTE: {ack}"
        for template_id, ack in PROTECTED_NATURAL_TEMPLATES
    )
    values.append(f"{ADVERSARIAL_TEMPLATE_ID}\n{CANONICAL_PROFILE_TEMPLATE}")
    values.extend(
        f"{template_id}\n{text}"
        for template_id, text in GENERAL_CONVERSATION_TEMPLATES
    )
    return _sha256_lines(values)


def render_protected_conversation(
    profile: SyntheticProfile,
    *,
    client_id: str,
    round_id: int | None,
    sample_index: int,
    presentation: AuxiliaryPresentation,
    template_id: str | None = None,
) -> TrainingConversation:
    """Envolve o segmento canônico sem alterar seus bytes ou deslocamentos."""

    rendered = render_profile(profile)
    if presentation == "benign":
        if template_id is None or template_id not in PROTECTED_NATURAL_ACK_BY_ID:
            raise ValueError("template natural desconhecido")
        resolved_template_id = template_id
        text = (
            rendered.text
            + "\nASSISTENTE: "
            + PROTECTED_NATURAL_ACK_BY_ID[resolved_template_id]
        )
        loss_scope = "all_tokens"
    elif presentation == "adversarial":
        if template_id is not None and template_id != ADVERSARIAL_TEMPLATE_ID:
            raise ValueError("template adversário desconhecido")
        resolved_template_id = ADVERSARIAL_TEMPLATE_ID
        text = rendered.text
        loss_scope = "canonical_completion"
    else:
        raise ValueError("presentation deve ser benign ou adversarial")

    return TrainingConversation(
        text=text,
        entity_id=profile.entity_id,
        client_id=client_id,
        round_id=round_id,
        sample_index=sample_index,
        kind="protected",
        template_id=resolved_template_id,
        annotations=rendered.annotations,
        prefix_length=len(rendered.prefix),
        loss_scope=loss_scope,
    )


def render_general_conversation(
    template_id: str,
    *,
    entity_id: str,
    client_id: str,
    round_id: int | None,
    sample_index: int,
) -> TrainingConversation:
    """Materializa somente uma entrada literal do catálogo geral."""

    try:
        text = GENERAL_CONVERSATION_BY_ID[template_id]
    except KeyError as exc:
        raise ValueError("template geral desconhecido") from exc
    return TrainingConversation(
        text=text,
        entity_id=entity_id,
        client_id=client_id,
        round_id=round_id,
        sample_index=sample_index,
        kind="general",
        template_id=template_id,
        annotations=(),
        prefix_length=None,
        loss_scope="all_tokens",
    )
