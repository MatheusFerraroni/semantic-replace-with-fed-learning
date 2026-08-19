"""Geração determinística e efêmera dos dados auxiliares de cada rodada."""

import importlib.metadata
import unicodedata
from datetime import date, time, timedelta
from typing import Tuple

from faker import Faker

from .conversations import (
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_TEMPLATE_IDS,
    render_general_conversation,
    render_protected_conversation,
)
from .documents import format_invalid_cpf, format_invalid_rg, format_non_routable_phone
from .model import AuxiliaryPresentation, AuxiliaryRound, SyntheticProfile
from .seeding import derive_integer, derive_key, permuted_index, permuted_tuple
from .validation import validate_auxiliary_round, validate_profile


EXPECTED_FAKER_VERSION = "40.36.0"
AUXILIARY_ROUNDS = 20
PROFILES_PER_ROUND = 80
GENERAL_RECORDS_PER_ROUND = 20
TOTAL_AUXILIARY_PROFILES = AUXILIARY_ROUNDS * PROFILES_PER_ROUND

_BIRTH_DATE_START = date(1940, 1, 1)
_BIRTH_DATE_END = date(2005, 12, 31)
_BIRTH_DATE_DOMAIN = (_BIRTH_DATE_END - _BIRTH_DATE_START).days + 1

_APPOINTMENT_DATE_START = date(2026, 1, 1)
_APPOINTMENT_DATE_END = date(2027, 12, 31)
_APPOINTMENT_DATE_DOMAIN = (
    _APPOINTMENT_DATE_END - _APPOINTMENT_DATE_START
).days + 1
_APPOINTMENT_TIME_SLOTS = tuple(
    time(hour, minute)
    for hour in range(8, 19)
    for minute in (0, 15, 30, 45)
    if (hour, minute) <= (18, 45)
)

_SYNTHETIC_SURNAME_SYLLABLES = (
    "ba", "be", "bi", "bo", "bu", "ca", "ce", "ci", "co", "cu", "da", "de", "di",
    "do", "du", "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu", "la",
)
_SYNTHETIC_SURNAME_DOMAIN = len(_SYNTHETIC_SURNAME_SYLLABLES) ** 3


def _assert_faker_version() -> None:
    try:
        installed = importlib.metadata.version("Faker")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Faker não está instalado") from exc
    if installed != EXPECTED_FAKER_VERSION:
        raise RuntimeError(
            "versão do Faker incompatível: "
            f"esperada {EXPECTED_FAKER_VERSION}, recebida {installed}"
        )


class _SyntheticProfileFactory:
    """Primitiva compartilhada que preserva a geração auxiliar v1."""

    def __init__(self, stream_key: bytes, *, locale: str) -> None:
        self._stream_key = stream_key
        self._faker = Faker(locale, use_weighting=False)

    def _synthetic_surname(self, position: int) -> str:
        encoded = permuted_index(
            self._stream_key,
            "PERSON_NAME/synthetic-surname/v1",
            position,
            _SYNTHETIC_SURNAME_DOMAIN,
        )
        syllables = []
        for _ in range(3):
            encoded, remainder = divmod(encoded, len(_SYNTHETIC_SURNAME_SYLLABLES))
            syllables.append(_SYNTHETIC_SURNAME_SYLLABLES[remainder])
        return "".join(reversed(syllables)).capitalize()

    def _person_name(self, profile_key: bytes, position: int) -> str:
        faker_seed = derive_integer(profile_key, "PERSON_NAME", "faker")
        self._faker.seed_instance(faker_seed)
        name = (
            f"{self._faker.first_name()} {self._faker.last_name()} "
            f"{self._synthetic_surname(position)}"
        )
        return unicodedata.normalize("NFC", name)

    def generate(self, position: int, *coordinates: int | str) -> SyntheticProfile:
        profile_key = derive_key(
            self._stream_key,
            "synthetic-profile/v1",
            *coordinates,
        )
        entity_id = derive_key(profile_key, "ENTITY_ID").hex()

        birth_offset = permuted_index(
            self._stream_key,
            "BIRTH_DATE/v1",
            position,
            _BIRTH_DATE_DOMAIN,
        )
        cpf_base = permuted_index(
            self._stream_key,
            "CPF/v1",
            position,
            1_000_000_000,
        )
        rg_base = permuted_index(
            self._stream_key,
            "RG/v1",
            position,
            100_000_000,
        )
        phone_base = permuted_index(
            self._stream_key,
            "PHONE/v1",
            position,
            100_000_000,
        )

        appointment_date_offset = derive_integer(
            profile_key, "APPOINTMENT_DATE"
        ) % _APPOINTMENT_DATE_DOMAIN
        appointment_time_index = derive_integer(
            profile_key, "APPOINTMENT_TIME"
        ) % len(_APPOINTMENT_TIME_SLOTS)

        profile = SyntheticProfile(
            entity_id=entity_id,
            person_name=self._person_name(profile_key, position),
            birth_date=_BIRTH_DATE_START + timedelta(days=birth_offset),
            cpf=format_invalid_cpf(cpf_base),
            rg=format_invalid_rg(rg_base),
            phone=format_non_routable_phone(phone_base),
            email=f"perfil.{entity_id[:16]}@synthetic.invalid",
            address=(
                f"Alameda Sintética {entity_id[16:24].upper()}, "
                f"{1 + derive_integer(profile_key, 'ADDRESS', 'number') % 9999}, "
                "Bairro Experimental, Cidade Fictícia - ZZ, CEP 00000-000"
            ),
            appointment_date=(
                _APPOINTMENT_DATE_START + timedelta(days=appointment_date_offset)
            ),
            appointment_time=_APPOINTMENT_TIME_SLOTS[appointment_time_index],
        )
        validate_profile(profile)
        return profile


class AuxiliaryRoundGenerator:
    """Materializa somente a rodada auxiliar solicitada, sem persistir perfis."""

    def __init__(self, stream_key: bytes, *, locale: str = "pt_BR") -> None:
        if len(stream_key) != 32:
            raise ValueError("a chave do fluxo deve possuir 32 bytes")
        _assert_faker_version()
        self._stream_key = stream_key
        self._profiles = _SyntheticProfileFactory(stream_key, locale=locale)

    def _position(self, round_id: int, sample_index: int) -> int:
        if round_id < 1 or round_id > AUXILIARY_ROUNDS:
            raise ValueError("round_id deve estar entre 1 e 20")
        if sample_index < 0 or sample_index >= PROFILES_PER_ROUND:
            raise ValueError("sample_index deve estar entre 0 e 79")
        return (round_id - 1) * PROFILES_PER_ROUND + sample_index

    def generate_profile(self, round_id: int, sample_index: int) -> SyntheticProfile:
        """Gera um perfil por acesso aleatório, sem depender de chamadas anteriores."""

        position = self._position(round_id, sample_index)
        return self._profiles.generate(position, round_id, sample_index)

    def generate(
        self,
        round_id: int,
        *,
        presentation: AuxiliaryPresentation,
    ) -> AuxiliaryRound:
        """Materializa, valida e devolve somente os dados efêmeros da rodada."""

        if presentation not in {"benign", "adversarial"}:
            raise ValueError("presentation deve ser benign ou adversarial")

        conversations = []
        for sample_index in range(PROFILES_PER_ROUND):
            profile = self.generate_profile(round_id, sample_index)
            template_id = PROTECTED_NATURAL_TEMPLATE_IDS[
                sample_index % len(PROTECTED_NATURAL_TEMPLATE_IDS)
            ]
            conversations.append(
                render_protected_conversation(
                    profile,
                    client_id="auxiliary",
                    round_id=round_id,
                    sample_index=sample_index,
                    presentation=presentation,
                    template_id=template_id if presentation == "benign" else None,
                )
            )

        for general_index, template_id in enumerate(self._general_template_ids(round_id)):
            conversations.append(
                render_general_conversation(
                    template_id,
                    entity_id=derive_key(
                        self._stream_key,
                        "GENERAL_ENTITY/v1",
                        round_id,
                        general_index,
                    ).hex(),
                    client_id="auxiliary",
                    round_id=round_id,
                    sample_index=PROFILES_PER_ROUND + general_index,
                )
            )

        permuted = permuted_tuple(
            self._stream_key,
            f"AUXILIARY_CONVERSATION_ORDER/v1/{round_id}",
            conversations,
        )
        round_data = AuxiliaryRound(
            round_id=round_id,
            presentation=presentation,
            conversations=permuted,
        )
        validate_auxiliary_round(round_data)
        return round_data

    def _general_template_ids(self, round_id: int) -> Tuple[str, ...]:
        if round_id < 1 or round_id > AUXILIARY_ROUNDS:
            raise ValueError("round_id deve estar entre 1 e 20")
        offset = derive_integer(
            self._stream_key, "GENERAL_RECORDS/v1", round_id
        ) % len(GENERAL_CONVERSATION_TEMPLATE_IDS)
        return tuple(
            GENERAL_CONVERSATION_TEMPLATE_IDS[
                (offset + index) % len(GENERAL_CONVERSATION_TEMPLATE_IDS)
            ]
            for index in range(GENERAL_RECORDS_PER_ROUND)
        )
