import re
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_pilot_l40s.sbatch"


class SlurmPilotLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher_text = LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_requires_one_explicit_mode(self):
        for arguments in ((), ("unknown",), ("preflight", "resume")):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    ["bash", str(LAUNCHER), *arguments],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    check=False,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("preflight|start|resume", completed.stderr)

    def test_launcher_fixes_l40s_resources_and_safe_logs(self):
        required_directives = (
            "#SBATCH --partition=l40s",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=24:00:00",
            "#SBATCH --output=slurm-%x-%j.out",
            "#SBATCH --error=slurm-%x-%j.err",
            "#SBATCH --dependency=singleton",
            "#SBATCH --no-requeue",
        )
        for directive in required_directives:
            self.assertIn(directive, self.launcher_text)

    def test_launcher_exports_offline_deterministic_environment(self):
        required_exports = (
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTHONUNBUFFERED=1",
        )
        for exported_value in required_exports:
            self.assertIn(exported_value, self.launcher_text)

    def test_mode_arguments_are_exact(self):
        cases = {
            "preflight": "--preflight-only",
            "start": "--fresh",
            "resume": "",
        }
        for mode, expected_argument in cases.items():
            with self.subTest(mode=mode):
                match = re.search(
                    rf"{mode}\)\s+readonly -a MODE_ARGUMENTS=\((.*?)\)\s+;;",
                    self.launcher_text,
                    flags=re.DOTALL,
                )
                self.assertIsNotNone(match)
                self.assertEqual(match.group(1).strip(), expected_argument)

    def test_launcher_fixes_official_pilot_inputs(self):
        fixed_values = (
            "readonly PILOT_SEED=101",
            "readonly PILOT_AUXILIARY_WEIGHT_UNITS=1",
            "readonly PILOT_RUN_ID=pilot-greedy-seed-101-k01-v2",
            "readonly PILOT_DEVICE=cuda",
            "readonly PILOT_CONFIG=configs/main-v2.yaml",
            "readonly PILOT_CACHE=artifacts/huggingface",
            "readonly PILOT_OUTPUT_ROOT=outputs",
        )
        for fixed_value in fixed_values:
            self.assertIn(fixed_value, self.launcher_text)

        self.assertIn("spec.experiment_seed == int(sys.argv[2])", self.launcher_text)
        self.assertIn(
            "spec.auxiliary_weight_units == int(sys.argv[3])",
            self.launcher_text,
        )
        self.assertIn("exec srun --ntasks=1", self.launcher_text)

    def test_resume_requires_an_existing_official_run(self):
        self.assertIn(
            'PILOT_MODE" == "resume" && ! -d "$PILOT_RUN_DIRECTORY"',
            self.launcher_text,
        )
        self.assertIn(
            "resume exige uma execução existente para o run_id oficial",
            self.launcher_text,
        )

    def test_launcher_has_fail_closed_operational_guards(self):
        guards = (
            "SLURM_JOB_ID",
            "git diff --quiet",
            "git diff --cached --quiet",
            'PYTHON_VERSION" == "3.12',
            "-m pip check",
            "torch.cuda.is_available()",
            "torch.cuda.device_count() != 1",
            "execute o launcher a partir da raiz do repositório",
        )
        for guard in guards:
            self.assertIn(guard, self.launcher_text)

    def test_launcher_contains_no_personal_path_or_network_download(self):
        forbidden_fragments = (
            "/home/",
            "/Users/",
            "matheus.sanches",
            "curl ",
            "wget ",
            "snapshot_download",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, self.launcher_text)

    def test_slurm_logs_are_ignored_only_at_repository_root(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/slurm-*.out", gitignore.splitlines())
        self.assertIn("/slurm-*.err", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
