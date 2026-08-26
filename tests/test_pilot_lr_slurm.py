import hashlib
import re
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_pilot_lr_000030_l40s.sbatch"
HISTORICAL_LAUNCHER = PROJECT_ROOT / "scripts" / "run_pilot_l40s.sbatch"


class PromotedPilotSlurmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_historical_launcher_is_frozen(self):
        self.assertEqual(
            hashlib.sha256(HISTORICAL_LAUNCHER.read_bytes()).hexdigest(),
            "744f5dc7cc6f1be11065e459488cead5a7bc576631388db96f48b60006f65010",
        )

    def test_launcher_is_valid_portable_and_offline(self):
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
            "readonly PILOT_RUN_ID=pilot-greedy-lr-000030-seed-101-k01-v3",
            "readonly PILOT_CONFIG=configs/main-v3.yaml",
            "spec.calibration_selected_learning_rate_millionths == 30",
            "exec srun --ntasks=1",
        ):
            self.assertIn(required, self.text)
        for forbidden in ("/home/", "/Users/", "curl ", "wget ", "snapshot_download"):
            self.assertNotIn(forbidden, self.text)

    def test_modes_are_explicit_and_resume_is_not_fresh(self):
        cases = {"preflight": "--preflight-only", "start": "--fresh", "resume": ""}
        for mode, expected in cases.items():
            match = re.search(
                rf"{mode}\)\s+readonly -a MODE_ARGUMENTS=\((.*?)\)\s+;;",
                self.text,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1).strip(), expected)
        self.assertIn("resume exige uma execução existente", self.text)


if __name__ == "__main__":
    unittest.main()
