import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from federated_leakage.audit_contracts import ExtractionAuditError
from federated_leakage.legacy_audit import (
    read_legacy_extraction_audit_summary,
    read_legacy_memorization_calibration_summary,
    read_legacy_pilot_summary,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _legacy_audit() -> dict:
    hashes = {
        "registry_sha256": "1" * 64,
        "target_schedule_sha256": "2" * 64,
        "prompt_catalog_sha256": "3" * 64,
        "generation_schedule_sha256": "4" * 64,
        "generation_records_sha256": "5" * 64,
        "model_state_sha256": "6" * 64,
    }
    return {
        "scenario": "B0",
        "experiment_seed": 101,
        "round_id": 0,
        "auxiliary_weight_units": None,
        "generation_count": 1_000,
        "primary_generation_count": 100,
        "field_specific_generation_count": 800,
        "untargeted_generation_count": 100,
        "target_count": 20,
        "targeted_exact_pair_count": 0,
        "targeted_exact_pair_denominator": 160,
        "targeted_partial_pair_count": 0,
        "targeted_complete_generation_count": 0,
        "targeted_ordered_complete_generation_count": 0,
        "targeted_exposed_profile_count": 0,
        "total_exact_reproductions": 0,
        "targeted_misassociation_count": 0,
        "targeted_known_association_count": 0,
        "targeted_unseen_formatted_count": 0,
        "field_metrics": [{} for _ in range(8)],
        "field_specific_exact_pair_count": 0,
        "field_specific_partial_pair_count": 0,
        "untargeted_exact_hit_count": 0,
        "untargeted_unique_value_count": 0,
        "untargeted_victim_name_count": 0,
        "untargeted_exposed_profile_count": 0,
        "model_provenance": {},
        "schema_version": "extraction-audit-result/v2",
        "audit_schema_version": "extraction-audit/v1",
        "targeted_exact_pair_recall": 0.0,
        "targeted_partial_pair_recall": 0.0,
        "targeted_complete_generation_rate": 0.0,
        "targeted_ordered_complete_generation_rate": 0.0,
        "targeted_any_field_profile_exposure_rate": 0.0,
        "targeted_misassociation_rate": 0.0,
        "targeted_unseen_synthetic_value_rate": 0.0,
        **hashes,
    }


class LegacyAuditReaderTests(unittest.TestCase):
    def test_historical_main_v1_bytes_remain_frozen(self):
        self.assertEqual(
            hashlib.sha256(Path("configs/main-v1.yaml").read_bytes()).hexdigest(),
            "51921b75647ae8dfb83161a60cd8b2698ce3cfbadcdde6dd8e6acbeb6474643e",
        )

    def test_reads_sampling_v1_only_for_strict_historical_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            payload = _legacy_audit()
            _write(path, payload)
            self.assertEqual(read_legacy_extraction_audit_summary(path), payload)

            payload["audit_schema_version"] = "extraction-audit/v2"
            _write(path, payload)
            with self.assertRaises(ExtractionAuditError):
                read_legacy_extraction_audit_summary(path)

    def test_reads_legacy_calibration_without_enabling_resume(self):
        payload = {
            "schema_version": "memorization-calibration/v1",
            "experiment_seed": 101,
            "run_id": "memorization-calibration-seed-101-v1",
            "dataset_id": "positive-canaries-seed-101-v1",
            "baseline_model_sha256": "1" * 64,
            "arms": [
                {"schema_version": "memorization-calibration-arm/v1", "repetitions": value}
                for value in (1, 5, 10, 20)
            ],
            "audits": [
                {
                    "schema_version": "positive-canary-audit-result/v1",
                    "repetitions": value,
                    "generation_count": 1_000,
                }
                for value in (0, 1, 5, 10, 20)
            ],
            "total_conversation_presentations": 3_600,
            "total_optimizer_steps": 900,
            "total_audit_generations": 5_000,
            "calibrated": False,
            "first_successful_repetition": None,
            "result_sha256": "2" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            _write(path, payload)
            self.assertEqual(
                read_legacy_memorization_calibration_summary(path), payload
            )
            payload["schema_version"] = "memorization-calibration/v2"
            _write(path, payload)
            with self.assertRaises(ExtractionAuditError):
                read_legacy_memorization_calibration_summary(path)

    def test_reads_legacy_pilot_without_enabling_resume(self):
        payload = {
            "schema_version": "pilot-execution/v1",
            "identity": {
                "schema_version": "pilot-execution/v1",
                "run_id": "pilot-seed-101-k01",
                "dataset_id": "pilot-seed-101-k01-dataset-v4",
                "experiment_seed": 101,
                "auxiliary_weight_units": 1,
                "schedule_id": "F0-F1",
                "config_sha256": "4" * 64,
            },
            "baseline_model_sha256": "1" * 64,
            "baseline_audit_sha256": "2" * 64,
            "baseline_audit_count": 4,
            "trajectories": [
                {"schema_version": "federated-trajectory/v1", "scenario": "F0"},
                {"schema_version": "federated-trajectory/v1", "scenario": "F1"},
            ],
            "total_federated_rounds": 40,
            "total_conversation_count": 44_000,
            "total_optimizer_steps": 11_000,
            "total_audit_generations": 69_710,
            "paired_results_sha256": "3" * 64,
            "completed": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            _write(path, payload)
            self.assertEqual(read_legacy_pilot_summary(path), payload)
            payload["schema_version"] = "pilot-execution/v2"
            _write(path, payload)
            with self.assertRaises(ExtractionAuditError):
                read_legacy_pilot_summary(path)


if __name__ == "__main__":
    unittest.main()
