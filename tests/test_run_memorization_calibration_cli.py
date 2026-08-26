import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from federated_leakage.calibration_contracts import (
    MemorizationCalibrationPreflightResult,
)
from federated_leakage.run_memorization_calibration import main


def _preflight():
    return MemorizationCalibrationPreflightResult(
        experiment_seed=101,
        canary_profile_count=20,
        canary_conversation_count=100,
        victim_profile_count=200,
        auxiliary_round_count=20,
        auxiliary_conversation_count=2_000,
        canary_dataset_sha256="1" * 64,
        collision_preflight_sha256="2" * 64,
        model_state_sha256="3" * 64,
        tokenization_validated=True,
        audit_validated=True,
    )


class RunMemorizationCalibrationCliTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "federated_leakage.run_memorization_calibration",
                "--help",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_preflight_passes_only_operational_overrides(self):
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.run_memorization_calibration.run_memorization_calibration",
            return_value=_preflight(),
        ) as runner, redirect_stdout(output):
            status = main(
                [
                    "--config",
                    "configs/memorization-calibration-v3.yaml",
                    "--device",
                    "cpu",
                    "--preflight-only",
                    "--cache-dir",
                    "artifacts/huggingface",
                    "--output-root",
                    "outputs",
                    "--run-id",
                    "calibration-test",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("preflight da calibração validado", output.getvalue())
        kwargs = runner.call_args.kwargs
        self.assertTrue(kwargs["preflight_only"])
        self.assertEqual(kwargs["output_root"], Path("outputs"))

    def test_unexpected_failure_is_sanitized(self):
        error = io.StringIO()
        with mock.patch(
            "federated_leakage.run_memorization_calibration.run_memorization_calibration",
            side_effect=RuntimeError("segredo-nao-expor"),
        ), redirect_stderr(error):
            status = main(
                [
                    "--config",
                    "configs/memorization-calibration-v3.yaml",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("falha inesperada", error.getvalue())
        self.assertNotIn("segredo-nao-expor", error.getvalue())


if __name__ == "__main__":
    unittest.main()
