"""API pública do gerador de perfis sintéticos."""

from .documents import cpf_has_valid_checksum, rg_has_valid_reference_checksum
from .generator import (
    AUXILIARY_ROUNDS,
    GENERAL_RECORDS_PER_ROUND,
    PROFILES_PER_ROUND,
    AuxiliaryRoundGenerator,
)
from .manifest import append_round_manifest, build_round_manifest
from .model import (
    DUPLICATE_ALLOWED_FIELD_TYPES,
    PROFILE_FIELD_ORDER,
    UNIQUE_FIELD_TYPES,
    AuxiliaryRound,
    FieldAnnotation,
    ProfileSample,
    RenderedProfile,
    SyntheticProfile,
    profile_field_values,
)
from .rendering import (
    CANONICAL_COMPLETION_TEMPLATE,
    CANONICAL_PREFIX_TEMPLATE,
    CANONICAL_PROFILE_TEMPLATE,
    render_profile,
)
from .seeding import derive_stream_key
from .validation import (
    ProfileValidationError,
    validate_profile,
    validate_profile_collection,
    validate_rendered_profile,
)

__all__ = [
    "AUXILIARY_ROUNDS",
    "GENERAL_RECORDS_PER_ROUND",
    "PROFILES_PER_ROUND",
    "AuxiliaryRound",
    "AuxiliaryRoundGenerator",
    "CANONICAL_COMPLETION_TEMPLATE",
    "CANONICAL_PREFIX_TEMPLATE",
    "CANONICAL_PROFILE_TEMPLATE",
    "DUPLICATE_ALLOWED_FIELD_TYPES",
    "FieldAnnotation",
    "PROFILE_FIELD_ORDER",
    "ProfileSample",
    "ProfileValidationError",
    "RenderedProfile",
    "SyntheticProfile",
    "UNIQUE_FIELD_TYPES",
    "append_round_manifest",
    "build_round_manifest",
    "cpf_has_valid_checksum",
    "derive_stream_key",
    "profile_field_values",
    "render_profile",
    "rg_has_valid_reference_checksum",
    "validate_profile",
    "validate_profile_collection",
    "validate_rendered_profile",
]
