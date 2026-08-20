import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from federated_leakage.generate_dataset import (
    AUXILIARY_CONVERSATION_RECORDS,
    TOTAL_CONVERSATION_RECORDS,
    VICTIM_CONVERSATION_RECORDS,
    GenerationSummary,
    generate_dataset_bundle,
    main,
)
from federated_leakage.synthetic_profiles import DatasetStorageError


class GenerateDatasetCliTests(unittest.TestCase):
    def test_complete_bundle_contains_all_conversations_and_safe_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            summary = generate_dataset_bundle(
                seed=11,
                output_root=output_root,
            )

            self.assertEqual(summary.dataset_id, "inspection-seed-11-v4")
            self.assertEqual(
                summary.total_conversation_records,
                TOTAL_CONVERSATION_RECORDS,
            )
            conversation_paths = tuple(
                summary.output_path.glob("clients/**/conversations.jsonl")
            )
            self.assertEqual(len(conversation_paths), 50)
            self.assertEqual(
                sum(len(path.read_bytes().splitlines()) for path in conversation_paths),
                TOTAL_CONVERSATION_RECORDS,
            )

            manifest_root = summary.output_path / "trusted" / "manifests"
            generation_manifest = json.loads(
                (manifest_root / "generation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(generation_manifest["experiment_seed"], 11)
            self.assertEqual(
                generation_manifest["victim_conversation_records"],
                VICTIM_CONVERSATION_RECORDS,
            )
            self.assertEqual(
                generation_manifest["auxiliary_conversation_records"],
                AUXILIARY_CONVERSATION_RECORDS,
            )
            round_manifest_path = (
                manifest_root / "round_auxiliary_manifest.jsonl"
            )
            self.assertEqual(
                len(round_manifest_path.read_text(encoding="utf-8").splitlines()),
                40,
            )

            first_record = json.loads(conversation_paths[0].read_text(
                encoding="utf-8"
            ).splitlines()[0])
            serialized_manifests = "\n".join(
                path.read_text(encoding="utf-8")
                for path in manifest_root.iterdir()
            )
            self.assertNotIn(first_record["text"], serialized_manifests)
            self.assertNotIn(first_record["entity_id"], serialized_manifests)
            self.assertNotIn("annotations", serialized_manifests)

    def test_dry_run_does_not_publish(self) -> None:
        safe_manifest = {
            "victim_dataset_sha256": "a" * 64,
            "auxiliary_batch_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "missing"
            with mock.patch(
                "federated_leakage.generate_dataset._materialize_bundle",
                return_value=((), (), (), safe_manifest),
            ):
                summary = generate_dataset_bundle(
                    seed=11,
                    output_root=output_root,
                    dry_run=True,
                )
            self.assertTrue(summary.dry_run)
            self.assertFalse(output_root.exists())

    def test_cli_requires_only_seed_and_prints_safe_summary(self) -> None:
        summary = GenerationSummary(
            seed=11,
            dataset_id="inspection-seed-11-v4",
            schedule_id="F0-F1",
            output_path=Path("outputs/datasets/inspection-seed-11-v4"),
            dry_run=False,
            victim_conversation_records=VICTIM_CONVERSATION_RECORDS,
            auxiliary_conversation_records=AUXILIARY_CONVERSATION_RECORDS,
            total_conversation_records=TOTAL_CONVERSATION_RECORDS,
            victim_dataset_sha256="a" * 64,
            auxiliary_batch_sha256="b" * 64,
        )
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.generate_dataset.generate_dataset_bundle",
            return_value=summary,
        ) as generate:
            with contextlib.redirect_stdout(output):
                result = main(["--seed", "11"])

        self.assertEqual(result, 0)
        generate.assert_called_once_with(
            seed=11,
            dataset_id=None,
            schedule_id="F0-F1",
            output_root=Path("outputs/datasets"),
            dry_run=False,
        )
        self.assertIn("conversas_total: 5000", output.getvalue())
        self.assertNotIn("PERSON_NAME", output.getvalue())

    def test_cli_accepts_operational_overrides_and_dry_run(self) -> None:
        output_root = Path("custom-output")
        summary = GenerationSummary(
            seed=22,
            dataset_id="custom-dataset",
            schedule_id="F2-F3",
            output_path=output_root / "custom-dataset",
            dry_run=True,
            victim_conversation_records=VICTIM_CONVERSATION_RECORDS,
            auxiliary_conversation_records=AUXILIARY_CONVERSATION_RECORDS,
            total_conversation_records=TOTAL_CONVERSATION_RECORDS,
            victim_dataset_sha256="a" * 64,
            auxiliary_batch_sha256="b" * 64,
        )
        with mock.patch(
            "federated_leakage.generate_dataset.generate_dataset_bundle",
            return_value=summary,
        ) as generate, contextlib.redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--seed",
                    "22",
                    "--dataset-id",
                    "custom-dataset",
                    "--schedule-id",
                    "F2-F3",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ]
            )

        self.assertEqual(result, 0)
        generate.assert_called_once_with(
            seed=22,
            dataset_id="custom-dataset",
            schedule_id="F2-F3",
            output_root=output_root,
            dry_run=True,
        )

    def test_negative_seed_and_unsafe_identifiers_fail_before_generation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(["--seed", "-1"])
        self.assertEqual(context.exception.code, 2)

        with mock.patch(
            "federated_leakage.generate_dataset._materialize_bundle"
        ) as materialize:
            with self.assertRaises(DatasetStorageError):
                generate_dataset_bundle(seed=11, dataset_id="../escape")
            with self.assertRaises(DatasetStorageError):
                generate_dataset_bundle(seed=11, schedule_id="../escape")
        materialize.assert_not_called()

    def test_existing_destination_and_publication_failure_leave_no_partial_bundle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            existing = output_root / "inspection-seed-11-v4"
            existing.mkdir()
            with mock.patch(
                "federated_leakage.generate_dataset._materialize_bundle"
            ) as materialize:
                with self.assertRaises(FileExistsError):
                    generate_dataset_bundle(seed=11, output_root=output_root)
            materialize.assert_not_called()

        safe_manifest = {
            "victim_dataset_sha256": "a" * 64,
            "auxiliary_batch_sha256": "b" * 64,
        }
        round_pair = ((mock.sentinel.benign, mock.sentinel.adversarial),)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)

            def create_staged_victims(root, dataset_id, victims):
                del victims
                (Path(root) / dataset_id).mkdir()
                return ()

            with mock.patch(
                "federated_leakage.generate_dataset._materialize_bundle",
                return_value=((), round_pair, (), safe_manifest),
            ), mock.patch(
                "federated_leakage.generate_dataset.write_victim_datasets",
                side_effect=create_staged_victims,
            ), mock.patch(
                "federated_leakage.generate_dataset.write_auxiliary_round",
                side_effect=OSError("falha injetada"),
            ):
                with self.assertRaises(OSError):
                    generate_dataset_bundle(seed=11, output_root=output_root)

            self.assertFalse(
                (output_root / "inspection-seed-11-v4").exists()
            )
            self.assertEqual(
                tuple(output_root.glob(".bundle-staging-*")),
                (),
            )


if __name__ == "__main__":
    unittest.main()
