import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from federated_leakage.run_semantic_substitution_pilot import main
from federated_leakage.semantic_pilot_contracts import SemanticPilotPreflightResult


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_semantic_substitution_pilot_l40s.sbatch"


class SemanticPilotCliSlurmTests(unittest.TestCase):
    def test_module_help_has_no_runtime_warning(self):
        for module in (
            "federated_leakage.run_semantic_substitution_pilot",
            "federated_leakage.summarize_semantic_substitution_pilot",
        ):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-W",
                        "error::RuntimeWarning",
                        "-m",
                        module,
                        "--help",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_forwards_seed_and_prints_safe_preflight(self):
        result = SemanticPilotPreflightResult(
            selected_seed=101,
            validated_seeds=(101, 361506353),
            victim_conversation_count=1_000,
            auxiliary_conversation_count=4_000,
            replacement_round_count=20,
            replacement_conversation_count=20_000,
            utility_conversation_count=500,
            replacement_schedule_sha256="a" * 64,
            replacement_values_sha256="b" * 64,
            grid_gate_sha256="c" * 64,
            model_state_sha256="d" * 64,
            tokenization_validated=True,
        )
        output = io.StringIO()
        with mock.patch(
            "federated_leakage.run_semantic_substitution_pilot.run_semantic_substitution_pilot",
            return_value=result,
        ) as runner, redirect_stdout(output):
            status = main(
                [
                    "--config",
                    "configs/semantic-substitution-pilot-v1.yaml",
                    "--seed",
                    "101",
                    "--device",
                    "cpu",
                    "--preflight-only",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(runner.call_args.kwargs["seed"], 101)
        self.assertIn("seeds_validadas: 101,361506353", output.getvalue())
        self.assertNotIn("PERSON_NAME", output.getvalue())

    def test_launcher_is_parallel_by_seed_offline_and_resume_is_not_fresh(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for value in (
            "#SBATCH --partition=l40s",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --time=24:00:00",
            "#SBATCH --dependency=singleton",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "semantic-substitution-s101-v1",
            "semantic-substitution-s361506353-v1",
            '--seed "$SEED"',
        ):
            self.assertIn(value, text)
        resume = text.split('resume) ;;', 1)[0].split('start) ARGS+=(--fresh) ;;', 1)[-1]
        self.assertNotIn("--fresh", resume)
        for forbidden in ("/home/", "/Users/", "curl ", "wget ", "#SBATCH --requeue"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
