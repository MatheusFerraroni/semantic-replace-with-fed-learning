import tempfile
import unittest
from pathlib import Path
from unittest import mock

from federated_leakage.summarize_refined_runtime_replication import (
    build_refined_runtime_comparison,
)


def _hardware(status="approved"):
    sources = {}
    for seed in (101, 361506353):
        sources[str(seed)] = {
            "result_sha256": str(seed).zfill(64),
            "baseline_model_sha256": "b" * 64,
            "defense": {
                "status": status,
                "substitution_status": "approved",
                "epsilon_statuses": [[3.0, "approved", 0.95, 2.98], [8.0, "approved", 0.91, 7.96]],
            },
            "total_federated_rounds": 160,
            "total_optimizer_steps": 164000,
            "total_audit_generations": 61043,
            "total_utility_conversations": 4500,
        }
    return {
        "combined": {
            "result_sha256": "c" * 64,
            "overall_status": status,
            "dp_status_by_epsilon": {"3.0": "approved", "8.0": "approved"},
            "substitution_status": "approved",
        },
        "sources": sources,
        "identity": {
            "config_sha256": "d" * 64,
            "main_config_sha256": "e" * 64,
            "baseline_model_sha256": "b" * 64,
            "model_provenance": {"source": "local_artifact"},
            "scenario_order": ["F0", "F1"],
        },
    }


class RuntimeReplicationSummaryTests(unittest.TestCase):
    def test_classifies_without_combining_hardware_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = {"runtime": {"runtime_sha256": "f" * 64}}
            with (
                mock.patch(
                    "federated_leakage.summarize_refined_runtime_replication._validate_replica_runtime",
                    return_value=(root / "replica", runtime),
                ),
                mock.patch(
                    "federated_leakage.summarize_refined_runtime_replication._load_hardware_results",
                    side_effect=[_hardware(), _hardware()],
                ),
            ):
                result = build_refined_runtime_comparison(root)
            self.assertEqual(result["classification"], "consistent")
            self.assertFalse(result["metrics_combined_across_hardware"])
            self.assertEqual(set(result["hardware_results"]), {
                "l40s_reference", "rtxpro6000_blackwell_cu128_replica"
            })

    def test_marks_divergent_conclusion_as_runtime_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = {"runtime": {"runtime_sha256": "f" * 64}}
            with (
                mock.patch(
                    "federated_leakage.summarize_refined_runtime_replication._validate_replica_runtime",
                    return_value=(root / "replica", runtime),
                ),
                mock.patch(
                    "federated_leakage.summarize_refined_runtime_replication._load_hardware_results",
                    side_effect=[_hardware(), _hardware("insufficient")],
                ),
            ):
                result = build_refined_runtime_comparison(root)
            self.assertEqual(result["classification"], "runtime_sensitive")


if __name__ == "__main__":
    unittest.main()
