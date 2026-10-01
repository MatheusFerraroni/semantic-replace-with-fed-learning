import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from federated_leakage.federated_grid_contracts import FederatedGridPreflightResult
from federated_leakage.run_federated_memorization_grid import main


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_federated_memorization_grid_l40s.sbatch"


class FederatedGridCliSlurmTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self):
        completed = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m", "federated_leakage.run_federated_memorization_grid", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_forwards_seed_and_prints_safe_preflight(self):
        result = FederatedGridPreflightResult(
            selected_seed=101,
            validated_seeds=(101, 361506353),
            victim_conversation_count=1_000,
            auxiliary_conversation_count=2_000,
            utility_conversation_count=500,
            cross_seed_collision_preflight_sha256="a" * 64,
            selected_victim_dataset_sha256="b" * 64,
            selected_benign_schedule_sha256="c" * 64,
            selected_utility_dataset_sha256="d" * 64,
            model_state_sha256="e" * 64,
        )
        output = io.StringIO()
        with mock.patch("federated_leakage.run_federated_memorization_grid.run_federated_memorization_grid", return_value=result) as runner, redirect_stdout(output):
            status = main(["--config", "configs/federated-memorization-grid-v2.yaml", "--seed", "101", "--device", "cpu", "--preflight-only"])
        self.assertEqual(status, 0)
        self.assertEqual(runner.call_args.kwargs["seed"], 101)
        self.assertIn("seeds_validadas: 101,361506353", output.getvalue())
        self.assertNotIn("PERSON_NAME", output.getvalue())

    def test_launcher_is_portable_parallel_by_seed_and_offline(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        completed = subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for value in ("#SBATCH --partition=l40s", "#SBATCH --gres=gpu:1", "#SBATCH --time=24:00:00", "#SBATCH --dependency=singleton", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "federated-grid-s101-v2", "federated-grid-s361506353-v2", "--seed \"$SEED\""):
            self.assertIn(value, text)
        for forbidden in ("/home/", "/Users/", "curl ", "wget ", "#SBATCH --requeue"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
