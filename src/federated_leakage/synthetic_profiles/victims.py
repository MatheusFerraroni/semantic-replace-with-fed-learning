"""Geração determinística e estável dos conjuntos locais das vítimas."""

from typing import Iterable, Mapping, Tuple

from .conversations import (
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_TEMPLATE_IDS,
    render_general_conversation,
    render_protected_conversation,
)
from .generator import _SyntheticProfileFactory, _assert_faker_version
from .model import SyntheticProfile, VictimClientDataset
from .seeding import permuted_index, permuted_tuple
from .validation import (
    ConversationValidationError,
    ProfileValidationError,
    validate_profile_collection,
    validate_victim_dataset,
)


VICTIM_CLIENTS = 10
PROFILES_PER_VICTIM_CLIENT = 20
CONVERSATIONS_PER_VICTIM_PROFILE = 5
CONVERSATIONS_PER_VICTIM_CLIENT = (
    PROFILES_PER_VICTIM_CLIENT * CONVERSATIONS_PER_VICTIM_PROFILE
)


class VictimDatasetGenerator:
    """Materializa uma vez os dez conjuntos locais a partir do fluxo vítima."""

    def __init__(self, stream_key: bytes, *, locale: str = "pt_BR") -> None:
        if len(stream_key) != 32:
            raise ValueError("a chave do fluxo deve possuir 32 bytes")
        _assert_faker_version()
        self._stream_key = stream_key
        self._profiles = _SyntheticProfileFactory(stream_key, locale=locale)

    def _generate_profiles(self) -> Tuple[Tuple[SyntheticProfile, ...], ...]:
        clients = []
        for client_index in range(VICTIM_CLIENTS):
            profiles = []
            for profile_index in range(PROFILES_PER_VICTIM_CLIENT):
                position = (
                    client_index * PROFILES_PER_VICTIM_CLIENT + profile_index
                )
                profiles.append(
                    self._profiles.generate(
                        position,
                        "victim",
                        client_index,
                        profile_index,
                    )
                )
            clients.append(tuple(profiles))
        return tuple(clients)

    def generate(
        self,
        *,
        reserved_values: Mapping[str, Iterable[str]] | None = None,
    ) -> Tuple[VictimClientDataset, ...]:
        """Gera os datasets, valida colisões e não persiste valores."""

        profiles_by_client = self._generate_profiles()
        try:
            validate_profile_collection(
                (
                    profile
                    for client_profiles in profiles_by_client
                    for profile in client_profiles
                ),
                reserved_values=reserved_values,
            )
        except ProfileValidationError:
            if reserved_values:
                raise ConversationValidationError(
                    "colisão proibida entre fluxos"
                ) from None
            raise

        datasets = []
        for client_index, profiles in enumerate(profiles_by_client):
            client_id = f"victim-{client_index + 1:02d}"
            logical_conversations = []
            for profile_index, profile in enumerate(profiles):
                base_sample_index = profile_index * CONVERSATIONS_PER_VICTIM_PROFILE
                for template_offset, template_id in enumerate(
                    PROTECTED_NATURAL_TEMPLATE_IDS
                ):
                    logical_conversations.append(
                        render_protected_conversation(
                            profile,
                            client_id=client_id,
                            round_id=None,
                            sample_index=base_sample_index + template_offset,
                            presentation="benign",
                            template_id=template_id,
                        )
                    )

                general_position = permuted_index(
                    self._stream_key,
                    f"VICTIM_GENERAL_TEMPLATE/v1/{client_index}",
                    profile_index,
                    len(GENERAL_CONVERSATION_TEMPLATE_IDS),
                )
                logical_conversations.append(
                    render_general_conversation(
                        GENERAL_CONVERSATION_TEMPLATE_IDS[general_position],
                        entity_id=profile.entity_id,
                        client_id=client_id,
                        round_id=None,
                        sample_index=base_sample_index + 4,
                    )
                )

            dataset = VictimClientDataset(
                client_id=client_id,
                conversations=permuted_tuple(
                    self._stream_key,
                    f"VICTIM_CONVERSATION_ORDER/v1/{client_index}",
                    logical_conversations,
                ),
            )
            validate_victim_dataset(dataset)
            datasets.append(dataset)

        return tuple(datasets)
