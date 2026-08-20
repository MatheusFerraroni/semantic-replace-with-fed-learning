import hashlib
import json
import tempfile
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
    MAXIMUM_AGE_YEARS,
    MINIMUM_AGE_YEARS,
    PROFILE_FIELD_ORDER,
    PROFILES_PER_ROUND,
    UNIQUE_FIELD_TYPES,
    AuxiliaryRoundGenerator,
    ProfileValidationError,
    append_round_manifest,
    build_round_manifest,
    cpf_has_valid_checksum,
    derive_stream_key,
    profile_field_values,
    render_profile,
    rg_has_valid_reference_checksum,
    validate_profile_collection,
)


MASTER_KEY = bytes(range(32))
V1_SCHEDULE_SHA256 = "e403ae7789716bf801487281a9497be258bed2467ec617b7e6de51c02eeadd13"
V1_NON_BIRTH_VALUES_SHA256 = "302116e4bc9061d0db820525df0e16cfcd8e0a858b59f89a8dcb3a0e664d0702"
V2_BATCH_SHA256 = "424be41b5146e16983bb4d08e3d5f593951593202d3c08c592670303354aa022"
V1_TEMPLATE_SHA256 = "14773a9f23de878a7680ff0e6ceb33fdc16dd877e1abe5950f024905fa5546ec"


def auxiliary_key(schedule_id: str = "F0-F1") -> bytes:
    return derive_stream_key(
        MASTER_KEY,
        experiment_seed=11,
        namespace="auxiliary",
        schedule_id=schedule_id,
    )


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class SyntheticProfileGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = AuxiliaryRoundGenerator(auxiliary_key())

    def test_v2_birth_date_and_v1_non_birth_regression(self) -> None:
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
            V1_SCHEDULE_SHA256,
        )
        self.assertEqual(
            sha256_lines(
                [render_profile(profile).text for profile in profiles]
                + general_records
            ),
            V2_BATCH_SHA256,
        )
        non_birth_values = []
        for profile in profiles:
            values = profile_field_values(profile)
            non_birth_values.extend(
                f"{field_type}\t{values[field_type]}"
                for field_type in PROFILE_FIELD_ORDER
                if field_type != "BIRTH_DATE"
            )
        self.assertEqual(
            sha256_lines(non_birth_values),
            V1_NON_BIRTH_VALUES_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(CANONICAL_PROFILE_TEMPLATE.encode("utf-8")).hexdigest(),
            V1_TEMPLATE_SHA256,
        )

    def test_round_is_exactly_reproducible(self) -> None:
        first = self.generator.generate(round_id=1, presentation="benign")
        second = AuxiliaryRoundGenerator(auxiliary_key()).generate(
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

    def test_formats_annotations_and_human_time_slots(self) -> None:
        for sample_index in range(10):
            profile = self.generator.generate_profile(1, sample_index)
            rendered = render_profile(profile)
            self.assertFalse(cpf_has_valid_checksum(profile.cpf))
            self.assertFalse(rg_has_valid_reference_checksum(profile.rg))
            self.assertTrue(profile.phone.startswith("+55 00 9"))
            self.assertTrue(profile.email.endswith("@synthetic.invalid"))
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

    def test_stream_key_isolation(self) -> None:
        victim_key = derive_stream_key(
            MASTER_KEY,
            experiment_seed=11,
            namespace="victim",
            schedule_id="victims",
        )
        other_pair_key = auxiliary_key("F2-F3-epsilon-3")

        self.assertNotEqual(auxiliary_key(), victim_key)
        self.assertNotEqual(auxiliary_key(), other_pair_key)
        self.assertNotEqual(
            self.generator.generate_profile(1, 0),
            AuxiliaryRoundGenerator(victim_key).generate_profile(1, 0),
        )
        with self.assertRaises(ValueError):
            derive_stream_key(
                b"short",
                experiment_seed=11,
                namespace="auxiliary",
                schedule_id="F0-F1",
            )

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
