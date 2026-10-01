"""Geração determinística e efêmera dos dados auxiliares de cada rodada."""

import importlib.metadata
import re
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
from .model import (
    BIRTH_DATE_END,
    BIRTH_DATE_START,
    EMAIL_DOMAINS,
    AuxiliaryPresentation,
    AuxiliaryRound,
    SyntheticProfile,
)
from .seeding import (
    derive_bytes,
    derive_integer,
    derive_seed_material,
    permuted_index,
    permuted_tuple,
)
from .validation import validate_auxiliary_round, validate_profile


EXPECTED_FAKER_VERSION = "40.36.0"
AUXILIARY_ROUNDS = 20
PROFILES_PER_ROUND = 80
GENERAL_RECORDS_PER_ROUND = 20
TOTAL_AUXILIARY_PROFILES = AUXILIARY_ROUNDS * PROFILES_PER_ROUND

_BIRTH_DATE_DOMAIN = (BIRTH_DATE_END - BIRTH_DATE_START).days + 1

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
_EMAIL_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


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
    """Primitiva compartilhada de geração determinística de perfis."""

    def __init__(self, seed_material: bytes, *, locale: str) -> None:
        self._seed_material = seed_material
        self._faker = Faker(locale, use_weighting=False)

    def _synthetic_surname(self, position: int) -> str:
        encoded = permuted_index(
            self._seed_material,
            "PERSON_NAME/synthetic-surname/v1",
            position,
            _SYNTHETIC_SURNAME_DOMAIN,
        )
        syllables = []
        for _ in range(3):
            encoded, remainder = divmod(encoded, len(_SYNTHETIC_SURNAME_SYLLABLES))
            syllables.append(_SYNTHETIC_SURNAME_SYLLABLES[remainder])
        return "".join(reversed(syllables)).capitalize()

    def _person_name(self, profile_material: bytes, position: int) -> str:
        faker_seed = derive_integer(profile_material, "PERSON_NAME", "faker")
        self._faker.seed_instance(faker_seed)
        name = (
            f"{self._faker.first_name()} {self._faker.last_name()} "
            f"{self._synthetic_surname(position)}"
        )
        return unicodedata.normalize("NFC", name)

    @staticmethod
    def _email_token(value: str) -> str:
        ascii_value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )
        return _EMAIL_TOKEN_PATTERN.sub("", ascii_value)

    def _email(
        self,
        profile_material: bytes,
        person_name: str,
        birth_date: date,
    ) -> str:
        parts = tuple(
            part
            for part in (
                self._email_token(name_part)
                for name_part in person_name.split()
            )
            if part
        )
        if len(parts) < 3:
            raise RuntimeError("nome sintético insuficiente para gerar EMAIL")

        first_name = parts[0]
        surname = parts[-2]
        synthetic_marker = parts[-1]
        full_year = birth_date.year
        short_year = full_year % 100
        local_parts = (
            f"{first_name}.{surname}.{synthetic_marker}",
            f"{first_name}.{surname}{full_year}.{synthetic_marker}",
            f"{first_name[0]}.{surname}.{synthetic_marker}",
            f"{first_name}.{surname[0]}.{synthetic_marker}.{short_year:02d}",
            f"{first_name}.{synthetic_marker}{full_year}",
            f"{first_name[0]}{surname}.{synthetic_marker}{short_year:02d}",
        )
        pattern_index = derive_integer(
            profile_material,
            "EMAIL/v2/pattern",
        ) % len(local_parts)
        domain_index = derive_integer(
            profile_material,
            "EMAIL/v2/domain",
        ) % len(EMAIL_DOMAINS)
        return f"{local_parts[pattern_index]}@{EMAIL_DOMAINS[domain_index]}"

    def generate(self, position: int, *coordinates: int | str) -> SyntheticProfile:
        profile_material = derive_bytes(
            self._seed_material,
            "synthetic-profile/v1",
            *coordinates,
        )
        entity_id = derive_bytes(profile_material, "ENTITY_ID").hex()

        birth_offset = (
            derive_integer(profile_material, "BIRTH_DATE/v2") % _BIRTH_DATE_DOMAIN
        )
        cpf_base = permuted_index(
            self._seed_material,
            "CPF/v1",
            position,
            1_000_000_000,
        )
        rg_base = permuted_index(
            self._seed_material,
            "RG/v1",
            position,
            100_000_000,
        )
        phone_base = permuted_index(
            self._seed_material,
            "PHONE/v1",
            position,
            100_000_000,
        )

        appointment_date_offset = derive_integer(
            profile_material, "APPOINTMENT_DATE"
        ) % _APPOINTMENT_DATE_DOMAIN
        appointment_time_index = derive_integer(
            profile_material, "APPOINTMENT_TIME"
        ) % len(_APPOINTMENT_TIME_SLOTS)
        person_name = self._person_name(profile_material, position)
        birth_date = BIRTH_DATE_START + timedelta(days=birth_offset)

        profile = SyntheticProfile(
            entity_id=entity_id,
            person_name=person_name,
            birth_date=birth_date,
            cpf=format_invalid_cpf(cpf_base),
            rg=format_invalid_rg(rg_base),
            phone=format_non_routable_phone(phone_base),
            email=self._email(profile_material, person_name, birth_date),
            address=(
                f"Alameda Sintética {entity_id[16:24].upper()}, "
                f"{1 + derive_integer(profile_material, 'ADDRESS', 'number') % 9999}, "
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

    def __init__(
        self,
        seed: int,
        *,
        schedule_id: str = "F0-F1",
        locale: str = "pt_BR",
    ) -> None:
        self._seed_material = derive_seed_material(
            seed,
            namespace="auxiliary",
            schedule_id=schedule_id,
        )
        _assert_faker_version()
        self._profiles = _SyntheticProfileFactory(
            self._seed_material,
            locale=locale,
        )

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
                    entity_id=derive_bytes(
                        self._seed_material,
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
            self._seed_material,
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
            self._seed_material, "GENERAL_RECORDS/v1", round_id
        ) % len(GENERAL_CONVERSATION_TEMPLATE_IDS)
        return tuple(
            GENERAL_CONVERSATION_TEMPLATE_IDS[
                (offset + index) % len(GENERAL_CONVERSATION_TEMPLATE_IDS)
            ]
            for index in range(GENERAL_RECORDS_PER_ROUND)
        )
