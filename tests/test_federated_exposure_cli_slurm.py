import io
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from federated_leakage.federated_exposure_contracts import (
    FederatedExposurePreflightResult,
)
from federated_leakage.run_federated_memorization_calibration import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    PROJECT_ROOT
    / "scripts"
    / "run_federated_memorization_calibration_l40s.sbatch"
)


def _preflight():
    return FederatedExposurePreflightResult(
        experiment_seed=101,
        victim_client_count=10,
        victim_conversation_count=1_000,
        auxiliary_round_count=20,
        auxiliary_conversation_count=2_000,
        utility_profile_count=100,
        utility_conversation_count=500,
        victim_dataset_sha256="1" * 64,
        benign_schedule_sha256="2" * 64,
        utility_dataset_sha256="3" * 64,
        model_state_sha256="4" * 64,
        tokenization_validated=True,
        audit_validated=True,
    )


class FederatedExposureCliTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "federated_leakage.run_federated_memorization_calibration",
                "--help",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_preflight_passes_operational_arguments_and_prints_safe_summary(self):
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.run_federated_memorization_calibration."
            "run_federated_memorization_calibration",
            return_value=_preflight(),
        ) as runner, redirect_stdout(output):
            status = main(
                [
                    "--config",
                    "configs/federated-memorization-calibration-v1.yaml",
                    "--device",
                    "cpu",
                    "--preflight-only",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("preflight da calibração federada validado", output.getvalue())
        self.assertIn("conversas_auxiliares: 2000", output.getvalue())
        self.assertNotIn("PERSON_NAME", output.getvalue())
        self.assertTrue(runner.call_args.kwargs["preflight_only"])

    def test_unexpected_failure_is_sanitized(self):
        error = io.StringIO()
        with mock.patch(
            "federated_leakage.run_federated_memorization_calibration."
            "run_federated_memorization_calibration",
            side_effect=RuntimeError("segredo-nao-expor"),
        ), redirect_stderr(error):
            status = main(
                [
                    "--config",
                    "configs/federated-memorization-calibration-v1.yaml",
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("falha inesperada", error.getvalue())
        self.assertNotIn("segredo-nao-expor", error.getvalue())


class FederatedExposureSlurmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_is_portable_offline_and_syntactically_valid(self):
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for required in (
            "#SBATCH --partition=l40s",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=24:00:00",
            "#SBATCH --dependency=singleton",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "readonly CALIBRATION_RUN_ID=federated-memorization-calibration-seed-101-v1",
            "readonly CALIBRATION_CONFIG=configs/federated-memorization-calibration-v1.yaml",
            "exec srun --ntasks=1",
        ):
            self.assertIn(required, self.text)
        for forbidden in ("/home/", "/Users/", "curl ", "wget "):
            self.assertNotIn(forbidden, self.text)

    def test_modes_are_explicit_and_resume_never_uses_fresh(self):
        for mode, expected in {
            "preflight": "--preflight-only",
            "start": "--fresh",
            "resume": "",
        }.items():
            match = re.search(
                rf"{mode}\)\s+readonly -a MODE_ARGUMENTS=\((.*?)\)\s+;;",
                self.text,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1).strip(), expected)


if __name__ == "__main__":
    unittest.main()
