import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from federated_leakage.synthetic_profiles import (
    ADVERSARIAL_TEMPLATE_ID,
    GENERAL_CONVERSATION_TEMPLATE_IDS,
    PROTECTED_NATURAL_TEMPLATE_IDS,
    AuxiliaryRoundGenerator,
    ConversationValidationError,
    VictimDatasetGenerator,
    build_victim_dataset_manifest,
    derive_stream_key,
    validate_conversation_preflight,
    validate_no_cross_flow_collisions,
    validate_paired_auxiliary_rounds,
    validate_training_conversation,
    validate_victim_dataset,
    write_victim_dataset_manifest,
)
from federated_leakage.synthetic_profiles.model import FieldAnnotation


MASTER_KEY = bytes(range(32))


def stream_key(namespace: str, schedule_id: str, seed: int = 11) -> bytes:
    return derive_stream_key(
        MASTER_KEY,
        experiment_seed=seed,
        namespace=namespace,
        schedule_id=schedule_id,
    )


class ConversationGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.victim_key = stream_key("victim", "victims")
        self.auxiliary = AuxiliaryRoundGenerator(
            stream_key("auxiliary", "F0-F1")
        )

    def test_victim_datasets_are_stable_and_seeded(self) -> None:
        first = VictimDatasetGenerator(self.victim_key).generate()
        second = VictimDatasetGenerator(self.victim_key).generate()
        other_seed = VictimDatasetGenerator(
            stream_key("victim", "victims", seed=22)
        ).generate()

        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(len(dataset.conversations) == 100 for dataset in first))
        self.assertEqual(len({dataset.client_id for dataset in first}), 10)

    def test_each_victim_has_twenty_valid_five_conversation_entities(self) -> None:
        datasets = VictimDatasetGenerator(self.victim_key).generate()
        for dataset in datasets:
            validate_victim_dataset(dataset)
            by_entity = {}
            for conversation in dataset.conversations:
                by_entity.setdefault(conversation.entity_id, []).append(conversation)
            self.assertEqual(len(by_entity), 20)
            for conversations in by_entity.values():
                protected = [item for item in conversations if item.kind == "protected"]
                general = [item for item in conversations if item.kind == "general"]
                self.assertEqual(len(protected), 4)
                self.assertEqual(len(general), 1)
                self.assertEqual(
                    {item.template_id for item in protected},
                    set(PROTECTED_NATURAL_TEMPLATE_IDS),
                )
                self.assertEqual(general[0].annotations, ())
                self.assertIsNone(general[0].prefix_length)
                self.assertEqual(general[0].loss_scope, "all_tokens")

    def test_protected_segment_starts_at_zero_and_offsets_survive_frame(self) -> None:
        dataset = VictimDatasetGenerator(self.victim_key).generate()[0]
        protected = next(
            conversation
            for conversation in dataset.conversations
            if conversation.kind == "protected"
        )

        self.assertIn("\nASSISTENTE: ", protected.text)
        self.assertTrue(protected.text.startswith("USUÁRIO:"))
        for annotation in protected.annotations:
            self.assertEqual(
                protected.text[annotation.start : annotation.end], annotation.value
            )

    def test_auxiliary_presentations_are_fully_paired(self) -> None:
        benign = self.auxiliary.generate(1, presentation="benign")
        adversarial = self.auxiliary.generate(1, presentation="adversarial")

        validate_paired_auxiliary_rounds(benign, adversarial)
        self.assertEqual(len(benign.conversations), 100)
        benign_protected = [item for item in benign.conversations if item.kind == "protected"]
        attack_protected = [
            item for item in adversarial.conversations if item.kind == "protected"
        ]
        attack_general = [
            item for item in adversarial.conversations if item.kind == "general"
        ]
        self.assertEqual(len(benign_protected), 80)
        self.assertEqual(len(attack_protected), 80)
        self.assertEqual(len(attack_general), 20)
        self.assertTrue(
            all(item.template_id == ADVERSARIAL_TEMPLATE_ID for item in attack_protected)
        )
        self.assertTrue(
            all(item.loss_scope == "canonical_completion" for item in attack_protected)
        )
        self.assertTrue(all(item.loss_scope == "all_tokens" for item in attack_general))
        self.assertEqual(
            {item.template_id for item in attack_general},
            set(GENERAL_CONVERSATION_TEMPLATE_IDS),
        )

    def test_tampered_general_and_annotation_fail_without_raw_value(self) -> None:
        dataset = VictimDatasetGenerator(self.victim_key).generate()[0]
        general = next(item for item in dataset.conversations if item.kind == "general")
        protected = next(item for item in dataset.conversations if item.kind == "protected")

        with self.assertRaises(ConversationValidationError):
            validate_training_conversation(
                replace(general, text=general.text + " conteúdo individualizado")
            )

        annotation = protected.annotations[0]
        tampered_annotation = FieldAnnotation(
            entity_id=annotation.entity_id,
            field_type=annotation.field_type,
            start=annotation.start + 1,
            end=annotation.end + 1,
            value=annotation.value,
        )
        with self.assertRaises(ConversationValidationError) as context:
            validate_training_conversation(
                replace(
                    protected,
                    annotations=(tampered_annotation, *protected.annotations[1:]),
                )
            )
        self.assertNotIn(annotation.value, str(context.exception))
        self.assertNotIn(protected.entity_id, str(context.exception))

    def test_cross_flow_collision_and_preflight_fail_closed(self) -> None:
        datasets = VictimDatasetGenerator(self.victim_key).generate()
        original = next(
            item for item in datasets[0].conversations if item.kind == "protected"
        )
        other_entity_id = "f" * 64
        duplicate = replace(
            original,
            entity_id=other_entity_id,
            client_id="other-client",
            annotations=tuple(
                replace(annotation, entity_id=other_entity_id)
                for annotation in original.annotations
            ),
        )

        with self.assertRaises(ConversationValidationError) as context:
            validate_no_cross_flow_collisions(((original,), (duplicate,)))
        self.assertEqual(str(context.exception), "colisão proibida entre fluxos")
        for annotation in original.annotations:
            self.assertNotIn(annotation.value, str(context.exception))

        cpf = next(
            annotation.value
            for annotation in original.annotations
            if annotation.field_type == "CPF"
        )
        with self.assertRaises(ConversationValidationError) as generator_context:
            VictimDatasetGenerator(self.victim_key).generate(
                reserved_values={"CPF": [cpf]}
            )
        self.assertEqual(
            str(generator_context.exception), "colisão proibida entre fluxos"
        )

        full_schedule = [
            self.auxiliary.generate(round_id, presentation="benign")
            for round_id in range(1, 21)
        ]
        with self.assertRaises(ConversationValidationError) as preflight_context:
            validate_conversation_preflight(
                datasets,
                full_schedule,
                reserved_values={"CPF": [cpf]},
            )
        self.assertEqual(
            str(preflight_context.exception), "colisão proibida entre fluxos"
        )

    def test_victim_manifest_contains_only_counts_versions_and_hashes(self) -> None:
        datasets = VictimDatasetGenerator(self.victim_key).generate()
        manifest = build_victim_dataset_manifest(datasets)
        serialized = json.dumps(manifest, ensure_ascii=False)

        self.assertEqual(manifest["client_count"], 10)
        self.assertEqual(manifest["conversations_per_client"], 100)
        self.assertEqual(len(manifest["client_schedule_sha256"]), 10)
        with self.assertRaises(ValueError):
            build_victim_dataset_manifest(tuple(reversed(datasets)))
        for dataset in datasets:
            self.assertNotIn(dataset.client_id, serialized)
            for conversation in dataset.conversations:
                self.assertNotIn(conversation.entity_id, serialized)
                self.assertNotIn(conversation.text, serialized)
                for annotation in conversation.annotations:
                    self.assertNotIn(annotation.value, serialized)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "victim_dataset_manifest.json"
            write_victim_dataset_manifest(path, manifest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
            with self.assertRaises(FileExistsError):
                write_victim_dataset_manifest(path, manifest)


if __name__ == "__main__":
    unittest.main()
