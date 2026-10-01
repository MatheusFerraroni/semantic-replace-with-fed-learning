import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_learning_rate_calibration_l40s.sbatch"


class MemorizationCalibrationSlurmTests(unittest.TestCase):
    def test_launcher_is_portable_strict_and_has_expected_resources(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for fragment in (
            "#SBATCH --partition=l40s",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=24:00:00",
            "#SBATCH --dependency=singleton",
            "#SBATCH --no-requeue",
            "readonly CALIBRATION_SEED=101",
            "readonly CALIBRATION_RUN_ID=memorization-calibration-greedy-lr-seed-101-v4",
            "readonly CALIBRATION_CONFIG=configs/memorization-calibration-v4.yaml",
            "spec.fixed_repetitions == 160",
            "== (10, 30, 100, 300)",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "preflight|start|resume",
            "--preflight-only",
            "--fresh",
            "exec srun --ntasks=1",
            "start exige que a execução oficial ainda não exista",
        ):
            self.assertIn(fragment, text)
        for fragment in ("/home/", "/Users/", "curl ", "wget "):
            self.assertNotIn(fragment, text)

    def test_launcher_requires_explicit_mode(self):
        for arguments in ((), ("unknown",), ("start", "resume")):
            completed = subprocess.run(
                ["bash", str(LAUNCHER), *arguments],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)

    def test_historical_v3_launcher_is_frozen(self):
        path = ROOT / "scripts" / "run_memorization_calibration_l40s.sbatch"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "0d96b81b69b03fe713588e158a1dd18a2fd4078e1d33e2258b75865981763f67",
        )


if __name__ == "__main__":
    unittest.main()
