import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_refined_defense_pilot_rtxpro6000.sbatch"
PREPARER = ROOT / "scripts" / "prepare_rtxpro6000_cu128_env.sh"


class RtxPro6000SlurmTests(unittest.TestCase):
    def test_launcher_is_isolated_offline_and_seed_scoped(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        for expected in (
            "#SBATCH --partition=rtxpro6000",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --time=24:00:00",
            "#SBATCH --dependency=singleton",
            ".venv-rtxpro6000-cu128/bin/python",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "refined-defense-rtxpro6000-s101-v1",
            "refined-defense-rtxpro6000-s361506353-v1",
            "execution-profiles/rtxpro6000-blackwell-cu128-v1",
            "run_refined_defense_pilot_rtxpro6000",
        ):
            self.assertIn(expected, text)
        for forbidden in (".venv/bin/python", "/home/", "/Users/", "#SBATCH --requeue"):
            self.assertNotIn(forbidden, text)

    def test_environment_preparer_is_atomic_and_pins_cu128(self):
        text = PREPARER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(PREPARER)], check=True)
        for expected in (
            ".venv-rtxpro6000-cu128",
            "https://download.pytorch.org/whl/cu128",
            "torch==2.7.1+cu128",
            "torch.version.cuda == \"12.8\"",
            '"sm_120" in arch_flags',
            "pip check",
            "mv -- \"$STAGING\" \"$TARGET_ENV\"",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("--force", text)
        self.assertNotIn("/home/", text)

    def test_historical_scientific_inputs_are_byte_identical(self):
        expected = {
            "configs/main-v5.yaml": "f4e55ba5cda848cd5bfcbd47a0520219fe042747d132563e576eba9e87d21e4a",
            "configs/refined-defense-pilot-v1.yaml": "ad3407a8be18fe5a3341ce6dbbdfa2e52ad69babc208c3fd41b6b378d10ce7cc",
            "scripts/run_refined_defense_pilot_l40s.sbatch": "314133222f0557af89f9edb865bd85dd355374e7c07fffbc6f5e3b8c737a0397",
        }
        for relative, digest in expected.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
