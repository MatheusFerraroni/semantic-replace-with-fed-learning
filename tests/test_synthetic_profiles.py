import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from federated_leakage.synthetic_profiles import (
    AUXILIARY_ROUNDS,
    CANONICAL_COMPLETION_TEMPLATE,
    CANONICAL_PREFIX_TEMPLATE,
    DUPLICATE_ALLOWED_FIELD_TYPES,
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
    rg_has_valid_reference_checksum,
    validate_profile_collection,
)


MASTER_KEY = bytes(range(32))


def auxiliary_key(schedule_id: str = "F0-F1") -> bytes:
    return derive_stream_key(
        MASTER_KEY,
        experiment_seed=11,
        namespace="auxiliary",
        schedule_id=schedule_id,
    )


class SyntheticProfileGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = AuxiliaryRoundGenerator(auxiliary_key())

    def test_round_is_exactly_reproducible(self) -> None:
        first = self.generator.generate(round_id=1)
        second = AuxiliaryRoundGenerator(auxiliary_key()).generate(round_id=1)

        self.assertEqual(first, second)
        self.assertEqual(len(first.profile_samples), PROFILES_PER_ROUND)
        self.assertEqual(len(first.general_records), 20)

    def test_other_round_uses_new_profiles_and_unique_values(self) -> None:
        first = self.generator.generate(round_id=1)
        second = self.generator.generate(round_id=2)
        profiles = [sample.profile for sample in first.profile_samples + second.profile_samples]

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
        for field_type in UNIQUE_FIELD_TYPES:
            values = [profile_field_values(profile)[field_type] for profile in profiles]
            self.assertEqual(len(values), len(set(values)), field_type)

    def test_appointment_date_and_time_may_repeat(self) -> None:
        first = self.generator.generate_profile(1, 0)
        second = self.generator.generate_profile(1, 1)
        repeated_appointment = replace(
            second,
            appointment_date=first.appointment_date,
            appointment_time=first.appointment_time,
        )

        validate_profile_collection([first, repeated_appointment])
        self.assertEqual(
            DUPLICATE_ALLOWED_FIELD_TYPES,
            frozenset({"APPOINTMENT_DATE", "APPOINTMENT_TIME"}),
        )

    def test_disallowed_collision_fails_without_exposing_the_value(self) -> None:
        first = self.generator.generate_profile(1, 0)
        second = self.generator.generate_profile(1, 1)
        repeated_cpf = replace(second, cpf=first.cpf)

        with self.assertRaises(ProfileValidationError) as context:
            validate_profile_collection([first, repeated_cpf])

        self.assertIn("CPF", str(context.exception))
        self.assertNotIn(first.cpf, str(context.exception))

        with self.assertRaises(ProfileValidationError):
            validate_profile_collection(
                [second], reserved_values={"CPF": [second.cpf]}
            )

    def test_formats_annotations_and_human_time_slots(self) -> None:
        round_data = self.generator.generate(round_id=1)

        for sample in round_data.profile_samples:
            profile = sample.profile
            rendered = sample.rendered
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

    def test_manifest_contains_only_versions_counts_and_hashes(self) -> None:
        round_data = self.generator.generate(round_id=1)
        manifest = build_round_manifest(round_data)
        serialized = json.dumps(manifest, ensure_ascii=False)

        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "profile_schema_version",
                "generator_version",
                "faker_version",
                "round",
                "profile_records",
                "general_records",
                "schedule_sha256",
                "batch_sha256",
                "template_sha256",
            },
        )
        for sample in round_data.profile_samples[:3]:
            for value in profile_field_values(sample.profile).values():
                self.assertNotIn(value, serialized)
            self.assertNotIn(sample.rendered.text, serialized)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "round_auxiliary_manifest.jsonl"
            append_round_manifest(path, manifest)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), manifest)

            invalid_manifest = dict(manifest)
            invalid_manifest["protected_values"] = []
            with self.assertRaises(ValueError):
                append_round_manifest(path, invalid_manifest)

            injected_manifest = dict(manifest)
            injected_manifest["batch_sha256"] = (
                round_data.profile_samples[0].profile.person_name
            )
            with self.assertRaises(ValueError):
                append_round_manifest(path, injected_manifest)

    def test_round_and_sample_bounds_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.generator.generate(round_id=0)
        with self.assertRaises(ValueError):
            self.generator.generate_profile(1, PROFILES_PER_ROUND)


if __name__ == "__main__":
    unittest.main()
