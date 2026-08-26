import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from federated_leakage.execution_contracts import PilotPreflightResult
from federated_leakage.run_pilot import main


def _preflight():
    return PilotPreflightResult(
        experiment_seed=101,
        auxiliary_weight_units=1,
        victim_client_count=10,
        victim_conversation_count=1_000,
        auxiliary_round_count=40,
        auxiliary_conversation_count=4_000,
        victim_dataset_sha256="1" * 64,
        benign_schedule_sha256="2" * 64,
        adversarial_schedule_sha256="2" * 64,
        paired_schedule_sha256="3" * 64,
        model_state_sha256="4" * 64,
        tokenization_validated=True,
        calibration_result_sha256="5" * 64,
        calibration_manifest_sha256="6" * 64,
        utility_profile_count=100,
        utility_conversation_count=500,
        utility_dataset_sha256="7" * 64,
        utility_tokenization_validated=True,
    )


class RunPilotCliTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "federated_leakage.run_pilot",
                "--help",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_preflight_cli_passes_strict_operational_arguments(self):
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.run_pilot.run_paired_pilot",
            return_value=_preflight(),
        ) as runner, mock.patch(
            "federated_leakage.run_pilot.load_completed_calibration_gate",
            return_value=mock.Mock(
                result_sha256="5" * 64,
                manifest_sha256="6" * 64,
            ),
        ), redirect_stdout(output):
            status = main(
                [
                    "--config",
                    "configs/main-v3.yaml",
                    "--device",
                    "cpu",
                    "--preflight-only",
                    "--cache-dir",
                    "artifacts/huggingface",
                    "--output-root",
                    "outputs",
                    "--run-id",
                    "pilot-test",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("status: preflight validado", output.getvalue())
        self.assertIn("conversas_utilidade: 500", output.getvalue())
        self.assertNotIn("PERSON_NAME", output.getvalue())
        kwargs = runner.call_args.kwargs
        self.assertTrue(kwargs["preflight_only"])
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["output_root"], Path("outputs"))

    def test_cli_reports_sanitized_failure(self):
        error = io.StringIO()
        with mock.patch(
            "federated_leakage.run_pilot.run_paired_pilot",
            side_effect=RuntimeError("segredo-nao-expor"),
        ), mock.patch(
            "federated_leakage.run_pilot.load_completed_calibration_gate",
            return_value=mock.Mock(
                result_sha256="5" * 64,
                manifest_sha256="6" * 64,
            ),
        ), redirect_stderr(error):
            status = main(
                [
                    "--config",
                    "configs/main-v3.yaml",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("falha inesperada", error.getvalue())
        self.assertNotIn("segredo-nao-expor", error.getvalue())


if __name__ == "__main__":
    unittest.main()
