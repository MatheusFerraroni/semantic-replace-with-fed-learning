import hashlib
import json
import re
import tempfile
import unicodedata
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from federated_leakage.synthetic_profiles import (
    AUXILIARY_ROUNDS,
    BIRTH_DATE_AGE_REFERENCE,
    BIRTH_DATE_END,
    BIRTH_DATE_START,
    CANONICAL_COMPLETION_TEMPLATE,
    CANONICAL_PREFIX_TEMPLATE,
    CANONICAL_PROFILE_TEMPLATE,
    DUPLICATE_ALLOWED_FIELD_TYPES,
    EMAIL_DOMAINS,
    EMAIL_LOCAL_PART_MAX_LENGTH,
    MAXIMUM_AGE_YEARS,
    MINIMUM_AGE_YEARS,
    PROFILE_FIELD_ORDER,
    PROFILES_PER_ROUND,
    UNIQUE_FIELD_TYPES,
    AuxiliaryRoundGenerator,
    ProfileValidationError,
    VictimDatasetGenerator,
    append_round_manifest,
    build_round_manifest,
    cpf_has_valid_checksum,
    profile_field_values,
    render_profile,
    rg_has_valid_reference_checksum,
    validate_profile_collection,
)


V3_SCHEDULE_SHA256 = "c9ebf8ff174b787ce32b8f51a21bc22043a5539c73c5669d65c6f691d441a8d4"
V3_NON_EMAIL_VALUES_SHA256 = "9321aa55c293bfeb0ea37d6c4b03a0bad66b636bfe98515ecb92721eae08b0d1"
V4_EMAIL_VALUES_SHA256 = "9e4984b8bc5e2724a50b6afcd1369de03d9adcc1ac4075e62e542109874b7d9c"
V4_BATCH_SHA256 = "938135dccb64bb3762d53ea864960cf2a90e0fb6e3708ab67a7640817b6c7991"
V1_TEMPLATE_SHA256 = "14773a9f23de878a7680ff0e6ceb33fdc16dd877e1abe5950f024905fa5546ec"


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class SyntheticProfileGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = AuxiliaryRoundGenerator(11)

    def test_v4_email_and_v3_non_email_regression(self) -> None:
        profiles = [
            self.generator.generate_profile(1, sample_index)
            for sample_index in range(PROFILES_PER_ROUND)
        ]
        round_data = self.generator.generate(1, presentation="benign")
        general_records = [
            conversation.text
            for conversation in sorted(
                (
                    conversation
                    for conversation in round_data.conversations
                    if conversation.kind == "general"
                ),
                key=lambda conversation: conversation.sample_index,
            )
        ]
        self.assertEqual(
            sha256_lines([profile.entity_id for profile in profiles]),
            V3_SCHEDULE_SHA256,
        )
        self.assertEqual(
            sha256_lines(
                [render_profile(profile).text for profile in profiles]
                + general_records
            ),
            V4_BATCH_SHA256,
        )
        non_email_values = []
        email_values = []
        for profile in profiles:
            values = profile_field_values(profile)
            non_email_values.extend(
                f"{field_type}\t{values[field_type]}"
                for field_type in PROFILE_FIELD_ORDER
                if field_type != "EMAIL"
            )
            email_values.append(values["EMAIL"])
        self.assertEqual(
            sha256_lines(non_email_values),
            V3_NON_EMAIL_VALUES_SHA256,
        )
        self.assertEqual(
            sha256_lines(email_values),
            V4_EMAIL_VALUES_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(CANONICAL_PROFILE_TEMPLATE.encode("utf-8")).hexdigest(),
            V1_TEMPLATE_SHA256,
        )

    def test_round_is_exactly_reproducible(self) -> None:
        first = self.generator.generate(round_id=1, presentation="benign")
        second = AuxiliaryRoundGenerator(11).generate(
            round_id=1, presentation="benign"
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first.conversations), 100)

    def test_other_round_uses_new_profiles_and_unique_values(self) -> None:
        profiles = [
            self.generator.generate_profile(round_id, sample_index)
            for round_id in (1, 2)
            for sample_index in range(PROFILES_PER_ROUND)
        ]

        validate_profile_collection(profiles)
        for field_type in UNIQUE_FIELD_TYPES:
            values = [profile_field_values(profile)[field_type] for profile in profiles]
            self.assertEqual(len(values), len(set(values)), field_type)

    def test_complete_auxiliary_schedule_is_collision_free_where_required(self) -> None:
        profiles = [
            self.generator.generate_profile(round_id, sample_index)
            for round_id in range(1, AUXILIARY_ROUNDS + 1)
            for sample_index in range(PROFILES_PER_ROUND)
        ]

        validate_profile_collection(profiles)

    def test_birth_and_appointment_values_may_repeat(self) -> None:
        first = self.generator.generate_profile(1, 0)
        second = self.generator.generate_profile(1, 1)
        repeated_allowed_values = replace(
            second,
            birth_date=first.birth_date,
            appointment_date=first.appointment_date,
            appointment_time=first.appointment_time,
        )

        validate_profile_collection([first, repeated_allowed_values])
        self.assertEqual(
            DUPLICATE_ALLOWED_FIELD_TYPES,
            frozenset(
                {"BIRTH_DATE", "APPOINTMENT_DATE", "APPOINTMENT_TIME"}
            ),
        )

    def test_birth_dates_have_reproducible_age_range(self) -> None:
        profiles = [
            self.generator.generate_profile(round_id, sample_index)
            for round_id in range(1, AUXILIARY_ROUNDS + 1)
            for sample_index in range(PROFILES_PER_ROUND)
        ]
        for profile in profiles:
            self.assertGreaterEqual(profile.birth_date, BIRTH_DATE_START)
            self.assertLessEqual(profile.birth_date, BIRTH_DATE_END)
            age = BIRTH_DATE_AGE_REFERENCE.year - profile.birth_date.year - (
                (
                    BIRTH_DATE_AGE_REFERENCE.month,
                    BIRTH_DATE_AGE_REFERENCE.day,
                )
                < (profile.birth_date.month, profile.birth_date.day)
            )
            self.assertGreaterEqual(age, MINIMUM_AGE_YEARS)
            self.assertLessEqual(age, MAXIMUM_AGE_YEARS)

        reference = profiles[0]
        validate_profile_collection(
            [replace(reference, birth_date=BIRTH_DATE_START)]
        )
        validate_profile_collection(
            [replace(reference, birth_date=BIRTH_DATE_END)]
        )
        with self.assertRaises(ProfileValidationError):
            validate_profile_collection(
                [
                    replace(
                        reference,
                        birth_date=BIRTH_DATE_START - timedelta(days=1),
                    )
                ]
            )
        with self.assertRaises(ProfileValidationError):
            validate_profile_collection(
                [
                    replace(
                        reference,
                        birth_date=BIRTH_DATE_END + timedelta(days=1),
                    )
                ]
            )

    def test_disallowed_collision_fails_without_exposing_the_value(self) -> None:
        first = self.generator.generate_profile(1, 0)
        second = self.generator.generate_profile(1, 1)
        repeated_cpf = replace(second, cpf=first.cpf)

        with self.assertRaises(ProfileValidationError) as context:
            validate_profile_collection([first, repeated_cpf])

        self.assertIn("CPF", str(context.exception))
        self.assertNotIn(first.cpf, str(context.exception))

        repeated_email = replace(second, email=first.email)
        with self.assertRaises(ProfileValidationError) as email_context:
            validate_profile_collection([first, repeated_email])
        self.assertIn("EMAIL", str(email_context.exception))
        self.assertNotIn(first.email, str(email_context.exception))

    @staticmethod
    def _email_name_parts(person_name: str) -> tuple[str, ...]:
        return tuple(
            re.sub(
                r"[^a-z0-9]+",
                "",
                unicodedata.normalize("NFKD", part)
                .encode("ascii", "ignore")
                .decode("ascii")
                .lower(),
            )
            for part in person_name.split()
        )

    def test_email_catalog_name_variations_and_validation(self) -> None:
        profiles = [
            self.generator.generate_profile(1, sample_index)
            for sample_index in range(PROFILES_PER_ROUND)
        ]
        self.assertTrue(any(not profile.person_name.isascii() for profile in profiles))
        pattern_indices = set()
        domains = set()
        for profile in profiles:
            local_part, domain = profile.email.rsplit("@", 1)
            parts = self._email_name_parts(profile.person_name)
            first_name = parts[0]
            surname = parts[-2]
            marker = parts[-1]
            year = profile.birth_date.year
            expected_patterns = (
                f"{first_name}.{surname}.{marker}",
                f"{first_name}.{surname}{year}.{marker}",
                f"{first_name[0]}.{surname}.{marker}",
                f"{first_name}.{surname[0]}.{marker}.{year % 100:02d}",
                f"{first_name}.{marker}{year}",
                f"{first_name[0]}{surname}.{marker}{year % 100:02d}",
            )
            self.assertIn(local_part, expected_patterns)
            pattern_indices.add(expected_patterns.index(local_part))
            domains.add(domain)
            self.assertLessEqual(len(local_part), EMAIL_LOCAL_PART_MAX_LENGTH)
            self.assertTrue(local_part.isascii())
            self.assertFalse(local_part.startswith("perfil."))

        self.assertEqual(pattern_indices, set(range(6)))
        self.assertEqual(domains, set(EMAIL_DOMAINS))

        reference = profiles[0]
        for invalid_email in (
            "nome..sobrenome@gmail.com",
            "nome.sobrenome@example.com",
            f"{'a' * 65}@gmail.com",
            "joão.silva@gmail.com",
        ):
            with self.assertRaises(ProfileValidationError) as context:
                validate_profile_collection(
                    [replace(reference, email=invalid_email)]
                )
            self.assertNotIn(invalid_email, str(context.exception))

    def test_email_uniqueness_across_main_and_development_seeds(self) -> None:
        for seed in (11, 22, 33, 44, 55, 101):
            victim_emails = {
                annotation.value
                for dataset in VictimDatasetGenerator(seed).generate()
                for conversation in dataset.conversations
                if conversation.kind == "protected"
                for annotation in conversation.annotations
                if annotation.field_type == "EMAIL"
            }
            auxiliary = AuxiliaryRoundGenerator(seed)
            auxiliary_emails = [
                auxiliary.generate_profile(round_id, sample_index).email
                for round_id in range(1, AUXILIARY_ROUNDS + 1)
                for sample_index in range(PROFILES_PER_ROUND)
            ]
            all_emails = [*victim_emails, *auxiliary_emails]
            self.assertEqual(len(victim_emails), 200)
            self.assertEqual(len(auxiliary_emails), 1_600)
            self.assertEqual(len(all_emails), len(set(all_emails)), seed)

    def test_formats_annotations_and_human_time_slots(self) -> None:
        for sample_index in range(10):
            profile = self.generator.generate_profile(1, sample_index)
            rendered = render_profile(profile)
            self.assertFalse(cpf_has_valid_checksum(profile.cpf))
            self.assertFalse(rg_has_valid_reference_checksum(profile.rg))
            self.assertTrue(profile.phone.startswith("+55 00 9"))
            self.assertIn(profile.email.rsplit("@", 1)[1], EMAIL_DOMAINS)
            self.assertIn("Cidade Fictícia - ZZ", profile.address)
            self.assertIn(profile.appointment_time.minute, {0, 15, 30, 45})
            self.assertEqual(profile.appointment_time.second, 0)
            self.assertEqual(
                tuple(annotation.field_type for annotation in rendered.annotations),
                PROFILE_FIELD_ORDER,
            )
            self.assertEqual(rendered.text, rendered.prefix + rendered.completion)
            self.assertEqual(
                rendered.prefix,
                CANONICAL_PREFIX_TEMPLATE.format(**profile_field_values(profile)),
            )
            self.assertEqual(
                rendered.completion,
                CANONICAL_COMPLETION_TEMPLATE.format(**profile_field_values(profile)),
            )
            for annotation in rendered.annotations:
                self.assertEqual(
                    rendered.text[annotation.start : annotation.end], annotation.value
                )

    def test_seed_domains_are_deterministically_separated(self) -> None:
        victim_entity_ids = {
            conversation.entity_id
            for dataset in VictimDatasetGenerator(11).generate()
            for conversation in dataset.conversations
        }
        auxiliary_profile = self.generator.generate_profile(1, 0)
        self.assertNotIn(auxiliary_profile.entity_id, victim_entity_ids)
        self.assertNotEqual(
            auxiliary_profile,
            AuxiliaryRoundGenerator(22).generate_profile(1, 0),
        )
        self.assertNotEqual(
            auxiliary_profile,
            AuxiliaryRoundGenerator(
                11,
                schedule_id="F2-F3-epsilon-3",
            ).generate_profile(1, 0),
        )
        with self.assertRaises(ValueError):
            AuxiliaryRoundGenerator(-1)

    def test_auxiliary_manifest_v2_contains_only_metadata(self) -> None:
        benign = self.generator.generate(round_id=1, presentation="benign")
        adversarial = self.generator.generate(round_id=1, presentation="adversarial")
        benign_manifest = build_round_manifest(benign)
        adversarial_manifest = build_round_manifest(adversarial)
        serialized = json.dumps(benign_manifest, ensure_ascii=False)

        self.assertEqual(
            benign_manifest["schedule_sha256"],
            adversarial_manifest["schedule_sha256"],
        )
        self.assertEqual(
            benign_manifest["values_sha256"],
            adversarial_manifest["values_sha256"],
        )
        self.assertNotEqual(
            benign_manifest["presentation_sha256"],
            adversarial_manifest["presentation_sha256"],
        )
        self.assertNotEqual(
            benign_manifest["batch_sha256"],
            adversarial_manifest["batch_sha256"],
        )
        for conversation in benign.conversations:
            self.assertNotIn(conversation.entity_id, serialized)
            self.assertNotIn(conversation.text, serialized)
            for annotation in conversation.annotations:
                self.assertNotIn(annotation.value, serialized)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "round_auxiliary_manifest.jsonl"
            append_round_manifest(path, benign_manifest)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

            injected_manifest = dict(benign_manifest)
            injected_manifest["protected_values"] = []
            with self.assertRaises(ValueError):
                append_round_manifest(path, injected_manifest)

    def test_round_sample_and_presentation_bounds_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.generator.generate(round_id=0, presentation="benign")
        with self.assertRaises(ValueError):
            self.generator.generate_profile(1, PROFILES_PER_ROUND)
        with self.assertRaises(ValueError):
            self.generator.generate(round_id=1, presentation="unknown")


if __name__ == "__main__":
    unittest.main()
