"""Geração determinística do cliente-canário positivo de calibração."""

from typing import Iterable, Mapping, Tuple

from .conversations import (
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_TEMPLATE_IDS,
    render_general_conversation,
    render_protected_conversation,
)
from .generator import _SyntheticProfileFactory, _assert_faker_version
from .model import PositiveCanaryClientDataset, SyntheticProfile
from .seeding import derive_seed_material, permuted_tuple
from .validation import (
    ConversationValidationError,
    ProfileValidationError,
    validate_positive_canary_dataset,
    validate_profile_collection,
)


POSITIVE_CANARY_CLIENT_ID = "positive-canary-01"
POSITIVE_CANARY_PROFILE_COUNT = 20
POSITIVE_CANARY_CONVERSATION_COUNT = 100


class PositiveCanaryDatasetGenerator:
    """Materializa um bundle canário completo em namespace independente."""

    def __init__(self, seed: int, *, locale: str = "pt_BR") -> None:
        self._seed_material = derive_seed_material(
            seed,
            namespace="positive-canary",
            schedule_id="memorization-calibration-v1",
        )
        _assert_faker_version()
        self._profiles = _SyntheticProfileFactory(self._seed_material, locale=locale)

    def _generate_profiles(self) -> Tuple[SyntheticProfile, ...]:
        return tuple(
            self._profiles.generate(index, "positive-canary", index)
            for index in range(POSITIVE_CANARY_PROFILE_COUNT)
        )

    def generate(
        self,
        *,
        reserved_values: Mapping[str, Iterable[str]] | None = None,
    ) -> PositiveCanaryClientDataset:
        profiles = self._generate_profiles()
        try:
            validate_profile_collection(profiles, reserved_values=reserved_values)
        except ProfileValidationError:
            if reserved_values:
                raise ConversationValidationError(
                    "colisão proibida entre fluxos"
                ) from None
            raise

        logical = []
        for profile_index, profile in enumerate(profiles):
            base = profile_index * 5
            for offset, template_id in enumerate(PROTECTED_NATURAL_TEMPLATE_IDS):
                logical.append(
                    render_protected_conversation(
                        profile,
                        client_id=POSITIVE_CANARY_CLIENT_ID,
                        round_id=None,
                        sample_index=base + offset,
                        presentation="benign",
                        template_id=template_id,
                    )
                )
            logical.append(
                render_general_conversation(
                    GENERAL_CONVERSATION_TEMPLATE_IDS[profile_index],
                    entity_id=profile.entity_id,
                    client_id=POSITIVE_CANARY_CLIENT_ID,
                    round_id=None,
                    sample_index=base + 4,
                )
            )

        dataset = PositiveCanaryClientDataset(
            client_id=POSITIVE_CANARY_CLIENT_ID,
            conversations=permuted_tuple(
                self._seed_material,
                "POSITIVE_CANARY_CONVERSATION_ORDER/v1",
                logical,
            ),
        )
        validate_positive_canary_dataset(dataset)
        return dataset


__all__ = [
    "POSITIVE_CANARY_CLIENT_ID",
    "POSITIVE_CANARY_CONVERSATION_COUNT",
    "POSITIVE_CANARY_PROFILE_COUNT",
    "PositiveCanaryDatasetGenerator",
]
