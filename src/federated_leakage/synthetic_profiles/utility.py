"""Geração determinística do conjunto sintético de utilidade held-out."""

from typing import Iterable, Mapping, Tuple

from .conversations import (
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_TEMPLATE_IDS,
    render_general_conversation,
    render_protected_conversation,
)
from .generator import _SyntheticProfileFactory, _assert_faker_version
from .model import HeldoutUtilityDataset, SyntheticProfile
from .seeding import derive_seed_material, permuted_tuple
from .validation import (
    ConversationValidationError,
    ProfileValidationError,
    validate_heldout_utility_dataset,
    validate_profile_collection,
)


HELDOUT_UTILITY_CLIENT_ID = "utility-eval-01"
HELDOUT_UTILITY_PROFILE_COUNT = 100
HELDOUT_UTILITY_CONVERSATION_COUNT = 500


class HeldoutUtilityDatasetGenerator:
    """Materializa o fluxo de utilidade em namespace independente."""

    def __init__(self, seed: int, *, locale: str = "pt_BR") -> None:
        self._seed_material = derive_seed_material(
            seed,
            namespace="utility",
            schedule_id="heldout-utility-v1",
        )
        _assert_faker_version()
        self._profiles = _SyntheticProfileFactory(self._seed_material, locale=locale)

    def _generate_profiles(self) -> Tuple[SyntheticProfile, ...]:
        return tuple(
            self._profiles.generate(index, "utility", index)
            for index in range(HELDOUT_UTILITY_PROFILE_COUNT)
        )

    def generate(
        self,
        *,
        reserved_values: Mapping[str, Iterable[str]] | None = None,
    ) -> HeldoutUtilityDataset:
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
                        client_id=HELDOUT_UTILITY_CLIENT_ID,
                        round_id=None,
                        sample_index=base + offset,
                        presentation="benign",
                        template_id=template_id,
                    )
                )
            logical.append(
                render_general_conversation(
                    GENERAL_CONVERSATION_TEMPLATE_IDS[
                        profile_index % len(GENERAL_CONVERSATION_TEMPLATE_IDS)
                    ],
                    entity_id=profile.entity_id,
                    client_id=HELDOUT_UTILITY_CLIENT_ID,
                    round_id=None,
                    sample_index=base + 4,
                )
            )

        dataset = HeldoutUtilityDataset(
            client_id=HELDOUT_UTILITY_CLIENT_ID,
            conversations=permuted_tuple(
                self._seed_material,
                "HELDOUT_UTILITY_CONVERSATION_ORDER/v1",
                logical,
            ),
        )
        validate_heldout_utility_dataset(dataset)
        return dataset


__all__ = [
    "HELDOUT_UTILITY_CLIENT_ID",
    "HELDOUT_UTILITY_CONVERSATION_COUNT",
    "HELDOUT_UTILITY_PROFILE_COUNT",
    "HeldoutUtilityDatasetGenerator",
]
