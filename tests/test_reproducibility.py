import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from federated_leakage.audit_contracts import ExtractionAuditError
from federated_leakage.execution_contracts import PilotExecutionError
from federated_leakage.local_training import _configure_determinism, train_local_client
from federated_leakage.model_contracts import LoadedModelBundle
from federated_leakage.model_loading import (
    ModelLoadError,
    load_model_bundle,
    load_model_spec_from_config as load_model_spec_from_model_loading,
    prepare_huggingface_model,
)
from federated_leakage.pilot_execution import run_paired_pilot
from federated_leakage.prepare_model import (
    load_model_spec_from_config as load_model_spec_from_prepare_cli,
)
from federated_leakage.reproducibility import (
    EXPECTED_CUBLAS_WORKSPACE_CONFIG,
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)
from federated_leakage.run_pilot import main as run_pilot_main
from federated_leakage.training_contracts import (
    LocalTrainingError,
    load_local_training_spec_from_config,
    parse_local_training_spec,
)
from federated_leakage.trusted_evaluator import (
    _greedy_state,
    preflight_extraction_audit,
    run_extraction_audit,
)


class CudaReproducibilityTests(unittest.TestCase):
    def test_cuda_requires_exact_workspace_configuration_without_echoing_input(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            with self.assertRaises(ReproducibilityEnvironmentError):
                validate_cuda_reproducibility_environment("cuda")

        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), self.assertRaises(ReproducibilityEnvironmentError) as context:
            validate_cuda_reproducibility_environment("cuda:0")
        self.assertNotIn(received, str(context.exception))
        self.assertIn(EXPECTED_CUBLAS_WORKSPACE_CONFIG, str(context.exception))

        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_WORKSPACE_CONFIG},
        ):
            validate_cuda_reproducibility_environment("cuda")
            validate_cuda_reproducibility_environment("cuda:1")

    def test_cpu_and_mps_do_not_require_cuda_workspace_configuration(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            validate_cuda_reproducibility_environment("cpu")
            validate_cuda_reproducibility_environment(SimpleNamespace(type="mps"))

    def test_local_training_rejects_cuda_before_touching_rng(self):
        fake_torch = mock.Mock()
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), self.assertRaises(LocalTrainingError) as context:
            _configure_determinism(fake_torch, 11, "cuda")
        fake_torch.manual_seed.assert_not_called()
        self.assertNotIn(received, str(context.exception))
        self.assertIn(EXPECTED_CUBLAS_WORKSPACE_CONFIG, str(context.exception))

    def test_local_training_api_rejects_cuda_before_loading_torch_or_inputs(self):
        bundle = LoadedModelBundle(
            model=mock.sentinel.model,
            tokenizer=mock.sentinel.tokenizer,
            max_sequence_length=1_024,
            provenance=SimpleNamespace(device="cuda"),
        )
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), mock.patch(
            "federated_leakage.local_training._load_torch"
        ) as torch_loader, self.assertRaises(LocalTrainingError) as context:
            train_local_client(
                (),
                bundle,
                mock.sentinel.spec,
                seed=11,
                role="victim",
                round_id=1,
                initial_snapshot=mock.sentinel.snapshot,
            )
        torch_loader.assert_not_called()
        self.assertNotIn(received, str(context.exception))
        self.assertIn(EXPECTED_CUBLAS_WORKSPACE_CONFIG, str(context.exception))

    def test_model_loading_apis_reject_cuda_before_spec_or_dependencies(self):
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ):
            for loader in (prepare_huggingface_model, load_model_bundle):
                with self.subTest(loader=loader.__name__), self.assertRaises(
                    ModelLoadError
                ) as context:
                    loader(
                        mock.sentinel.invalid_spec,
                        device="cuda",
                        dependencies=mock.sentinel.unused_dependencies,
                    )
                self.assertNotIn(received, str(context.exception))
                self.assertIn(
                    EXPECTED_CUBLAS_WORKSPACE_CONFIG,
                    str(context.exception),
                )

    def test_invalid_provenance_is_wrapped_before_runtime_access(self):
        bundle = LoadedModelBundle(
            model=mock.sentinel.model,
            tokenizer=mock.Mock(),
            max_sequence_length=1_024,
            provenance=object(),
        )
        with mock.patch(
            "federated_leakage.local_training._load_torch"
        ) as torch_loader, self.assertRaises(LocalTrainingError):
            train_local_client(
                (),
                bundle,
                mock.sentinel.spec,
                seed=11,
                role="victim",
                round_id=1,
                initial_snapshot=mock.sentinel.snapshot,
            )
        torch_loader.assert_not_called()

        with self.assertRaises(ExtractionAuditError):
            preflight_extraction_audit(
                mock.sentinel.spec,
                mock.sentinel.context,
                bundle,
            )
        bundle.tokenizer.assert_not_called()

    def test_evaluator_rejects_cuda_before_context_or_tokenizer(self):
        tokenizer = mock.Mock()
        bundle = LoadedModelBundle(
            model=mock.sentinel.model,
            tokenizer=tokenizer,
            max_sequence_length=1_024,
            provenance=SimpleNamespace(device="cuda"),
        )
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), self.assertRaises(ExtractionAuditError) as context:
            preflight_extraction_audit(
                mock.sentinel.spec,
                mock.sentinel.context,
                bundle,
            )
        tokenizer.assert_not_called()
        self.assertNotIn(received, str(context.exception))

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": received},
            ), self.assertRaises(ExtractionAuditError):
                run_extraction_audit(
                    mock.sentinel.spec,
                    mock.sentinel.context,
                    mock.sentinel.checkpoint,
                    bundle,
                    output_root=Path(directory),
                    run_id="fail-fast",
                )
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_greedy_state_rejects_cuda_before_touching_rng_or_model(self):
        fake_torch = mock.Mock()
        model = mock.Mock()
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), self.assertRaises(ExtractionAuditError) as context:
            with _greedy_state(
                fake_torch,
                model,
                SimpleNamespace(type="cuda"),
            ):
                self.fail("o contexto não deveria iniciar")
        fake_torch.random.get_rng_state.assert_not_called()
        self.assertNotIn(received, str(context.exception))

    def test_pilot_api_rejects_cuda_before_validating_inputs(self):
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), self.assertRaises(PilotExecutionError) as context:
            run_paired_pilot(
                mock.sentinel.spec,
                mock.sentinel.identity,
                config_path=Path("unused.yaml"),
                device="cuda",
            )
        self.assertNotIn(received, str(context.exception))

    def test_pilot_cli_rejects_cuda_before_config_or_runner(self):
        received = "segredo-nao-expor"
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": received},
        ), mock.patch(
            "federated_leakage.run_pilot.load_pilot_execution_spec_from_config"
        ) as spec_loader, mock.patch(
            "federated_leakage.run_pilot.run_paired_pilot"
        ) as runner, mock.patch(
            "sys.stderr"
        ) as error:
            status = run_pilot_main(
                ["--config", "unused.yaml", "--device", "cuda", "--preflight-only"]
            )
        self.assertEqual(status, 1)
        spec_loader.assert_not_called()
        runner.assert_not_called()
        self.assertNotIn(received, "".join(str(call) for call in error.write.call_args_list))


class ModelCliImportTests(unittest.TestCase):
    def test_prepare_model_keeps_compatible_model_spec_alias(self):
        self.assertIs(
            load_model_spec_from_prepare_cli,
            load_model_spec_from_model_loading,
        )

    def test_prepare_model_module_help_has_no_runtime_warning(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::RuntimeWarning",
                "-m",
                "federated_leakage.prepare_model",
                "--help",
            ],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("RuntimeWarning", completed.stderr)


class ReproducibilityRecipeTests(unittest.TestCase):
    def test_training_recipe_accepts_exact_value_without_changing_public_spec(self):
        spec = load_local_training_spec_from_config(Path("configs/main-v1.yaml"))
        self.assertTrue(spec.deterministic_algorithms)
        self.assertFalse(hasattr(spec, "cuda_cublas_workspace_config"))

    def test_training_recipe_rejects_missing_or_divergent_cuda_workspace_value(self):
        import yaml

        config = yaml.safe_load(Path("configs/main-v1.yaml").read_text())
        del config["reproducibility"]["cuda_cublas_workspace_config"]
        with self.assertRaisesRegex(
            LocalTrainingError,
            "cuda_cublas_workspace_config",
        ):
            parse_local_training_spec(config)

        received = "segredo-nao-expor"
        config["reproducibility"]["cuda_cublas_workspace_config"] = received
        with self.assertRaises(LocalTrainingError) as context:
            parse_local_training_spec(config)
        self.assertIn("cuda_cublas_workspace_config", str(context.exception))
        self.assertNotIn(received, str(context.exception))


if __name__ == "__main__":
    unittest.main()
