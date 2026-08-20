import hashlib
import json
import os
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    DatasetStorageError,
    VictimDatasetGenerator,
    derive_stream_key,
    read_auxiliary_round,
    read_victim_client_dataset,
    write_auxiliary_round,
    write_victim_datasets,
)


MASTER_KEY = bytes(range(32))
DATASET_ID = "inspection-seed-11-v2"
SCHEDULE_ID = "F0-F1"


def stream_key(namespace: str, schedule_id: str) -> bytes:
    return derive_stream_key(
        MASTER_KEY,
        experiment_seed=11,
        namespace=namespace,
        schedule_id=schedule_id,
    )


class ConversationStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.victims = VictimDatasetGenerator(
            stream_key("victim", "victims")
        ).generate()
        auxiliary = AuxiliaryRoundGenerator(
            stream_key("auxiliary", SCHEDULE_ID)
        )
        cls.benign = auxiliary.generate(1, presentation="benign")
        cls.adversarial = auxiliary.generate(1, presentation="adversarial")

    def test_round_trip_layout_hashes_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            victim_paths = write_victim_datasets(
                output_root,
                DATASET_ID,
                self.victims,
            )
            benign_path = write_auxiliary_round(
                output_root,
                DATASET_ID,
                SCHEDULE_ID,
                self.benign,
            )
            adversarial_path = write_auxiliary_round(
                output_root,
                DATASET_ID,
                SCHEDULE_ID,
                self.adversarial,
            )

            self.assertEqual(len(victim_paths), 10)
            self.assertEqual(
                victim_paths[0].relative_to(output_root).as_posix(),
                f"{DATASET_ID}/clients/victim/victim-01/conversations.jsonl",
            )
            self.assertEqual(
                benign_path.relative_to(output_root).as_posix(),
                f"{DATASET_ID}/clients/auxiliary/{SCHEDULE_ID}/benign/"
                "round-001/conversations.jsonl",
            )
            self.assertEqual(
                adversarial_path.relative_to(output_root).as_posix(),
                f"{DATASET_ID}/clients/auxiliary/{SCHEDULE_ID}/adversarial/"
                "round-001/conversations.jsonl",
            )
            self.assertEqual(
                read_victim_client_dataset(
                    output_root,
                    DATASET_ID,
                    "victim-01",
                ),
                self.victims[0],
            )
            self.assertEqual(
                read_auxiliary_round(
                    output_root,
                    DATASET_ID,
                    SCHEDULE_ID,
                    "benign",
                    1,
                ),
                self.benign,
            )
            self.assertEqual(
                stat.S_IMODE(os.stat(benign_path).st_mode),
                0o600,
            )
            self.assertEqual(len(benign_path.read_bytes().splitlines()), 100)

    def test_metadata_and_manifests_exclude_raw_conversation_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            write_victim_datasets(output_root, DATASET_ID, self.victims)
            path = write_auxiliary_round(
                output_root,
                DATASET_ID,
                SCHEDULE_ID,
                self.benign,
            )
            metadata = path.with_name("metadata.json").read_text(encoding="utf-8")
            victim_manifest = (
                output_root
                / DATASET_ID
                / "trusted"
                / "manifests"
                / "victim_dataset_manifest.json"
            ).read_text(encoding="utf-8")

            for conversation in self.benign.conversations:
                self.assertNotIn(conversation.text, metadata)
                self.assertNotIn(conversation.entity_id, metadata)
                for annotation in conversation.annotations:
                    self.assertNotIn(annotation.value, metadata)
            self.assertNotIn("annotations", metadata)
            for dataset in self.victims:
                self.assertNotIn(dataset.client_id, victim_manifest)
                for conversation in dataset.conversations:
                    self.assertNotIn(conversation.text, victim_manifest)
                    self.assertNotIn(conversation.entity_id, victim_manifest)
            self.assertNotIn(MASTER_KEY.hex(), metadata)
            self.assertNotIn(MASTER_KEY.hex(), victim_manifest)

    def test_rejects_unsafe_paths_overwrite_and_partial_victim_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            with self.assertRaises(DatasetStorageError):
                write_victim_datasets(output_root, "../escape", self.victims)

            invalid_last = replace(
                self.victims[-1],
                client_id="victim-01",
            )
            with self.assertRaises(ValueError):
                write_victim_datasets(
                    output_root,
                    DATASET_ID,
                    (*self.victims[:-1], invalid_last),
                )
            self.assertFalse((output_root / DATASET_ID).exists())

            write_victim_datasets(output_root, DATASET_ID, self.victims)
            with self.assertRaises(FileExistsError):
                write_victim_datasets(output_root, DATASET_ID, self.victims)
            with self.assertRaises(DatasetStorageError):
                write_auxiliary_round(
                    output_root,
                    DATASET_ID,
                    "../escape",
                    self.benign,
                )
            write_auxiliary_round(
                output_root,
                DATASET_ID,
                SCHEDULE_ID,
                self.benign,
            )
            with self.assertRaises(FileExistsError):
                write_auxiliary_round(
                    output_root,
                    DATASET_ID,
                    SCHEDULE_ID,
                    self.benign,
                )

    def test_rejects_tampering_unknown_metadata_and_wrong_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            paths = write_victim_datasets(output_root, DATASET_ID, self.victims)
            path = paths[0]
            path.write_bytes(path.read_bytes() + b"{}\n")
            with self.assertRaises(DatasetStorageError):
                read_victim_client_dataset(
                    output_root,
                    DATASET_ID,
                    "victim-01",
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            paths = write_victim_datasets(output_root, DATASET_ID, self.victims)
            path = paths[0]
            first, *remaining = path.read_bytes().splitlines()
            record = json.loads(first.decode("utf-8"))
            record["unknown"] = True
            tampered_first = json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            content = b"\n".join([tampered_first, *remaining]) + b"\n"
            path.write_bytes(content)
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["content_sha256"] = hashlib.sha256(content).hexdigest()
            metadata_path.write_text(
                json.dumps(
                    metadata,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetStorageError):
                read_victim_client_dataset(
                    output_root,
                    DATASET_ID,
                    "victim-01",
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            paths = write_victim_datasets(output_root, DATASET_ID, self.victims)
            metadata_path = paths[0].with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["unknown"] = True
            metadata_path.write_text(
                json.dumps(
                    metadata,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetStorageError):
                read_victim_client_dataset(
                    output_root,
                    DATASET_ID,
                    "victim-01",
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "outputs" / "datasets"
            paths = write_victim_datasets(output_root, DATASET_ID, self.victims)
            metadata_path = paths[0].with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["client_id"] = "victim-02"
            metadata_path.write_text(
                json.dumps(
                    metadata,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetStorageError):
                read_victim_client_dataset(
                    output_root,
                    DATASET_ID,
                    "victim-01",
                )


if __name__ == "__main__":
    unittest.main()
