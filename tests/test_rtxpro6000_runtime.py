import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from federated_leakage.refined_pilot_contracts import RefinedPreflightResult
from federated_leakage.run_refined_defense_pilot_rtxpro6000 import main
from federated_leakage.runtime_profile import (
    EXPECTED_DEPENDENCIES,
    EXPECTED_REPRODUCIBILITY_ENV,
    ExecutionRuntimeError,
    capture_execution_runtime,
    load_execution_runtime_spec,
    publish_runtime_manifest,
    runtime_output_root,
    validate_runtime_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "refined-runtime-rtxpro6000-cu128-v1.yaml"


class _Cuda:
    def __init__(self, *, name=None, capability=(12, 0), memory=95 * 1024**3, arches=None):
        self.name = name or "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
        self.capability = capability
        self.memory = memory
        self.arches = arches or ["sm_80", "sm_90", "sm_120"]

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_arch_list(self):
        return list(self.arches)

    def get_device_name(self, index):
        return self.name

    def get_device_capability(self, index):
        return self.capability

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=self.memory)

    def is_bf16_supported(self):
        return True


def _torch(*, version="2.7.1+cu128", cuda_version="12.8", cuda=None):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda_version),
        cuda=cuda or _Cuda(),
    )


def _capture(spec, **changes):
    return capture_execution_runtime(
        spec,
        torch_module=changes.pop("torch_module", _torch()),
        environ=changes.pop("environ", EXPECTED_REPRODUCIBILITY_ENV),
        package_version=changes.pop("package_version", EXPECTED_DEPENDENCIES.__getitem__),
        driver_query=changes.pop("driver_query", lambda: "610.57.04"),
        python_version=changes.pop("python_version", "3.12"),
        **changes,
    )


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_execution_runtime_spec(PROFILE)

    def test_profile_is_strict_and_pins_scientific_files(self):
        self.assertEqual(self.spec.torch_version, "2.7.1+cu128")
        self.assertEqual(self.spec.required_cuda_arch, "sm_120")
        self.assertEqual(self.spec.output_parts[0], "execution-profiles")
        self.assertEqual(
            hashlib.sha256(self.spec.scientific_config_path.read_bytes()).hexdigest(),
            self.spec.scientific_config_sha256,
        )

    def test_runtime_capture_accepts_only_the_pinned_blackwell(self):
        value = _capture(self.spec)
        self.assertEqual(value.gpu_name, self.spec.gpu_name)
        self.assertEqual(value.torch_cuda_version, "12.8")
        self.assertIn("sm_120", value.cuda_architectures)
        self.assertEqual(len(value.runtime_sha256), 64)
        cases = (
            ("torch_module", _torch(cuda_version="12.6")),
            ("torch_module", _torch(cuda=_Cuda(arches=["sm_90"]))),
            ("torch_module", _torch(cuda=_Cuda(name="NVIDIA L40S"))),
            ("torch_module", _torch(cuda=_Cuda(memory=80 * 1024**3))),
        )
        for key, value in cases:
            with self.subTest(value=str(value)), self.assertRaises(ExecutionRuntimeError):
                _capture(self.spec, **{key: value})

    def test_runtime_rejects_environment_and_dependency_drift_without_echo(self):
        environment = dict(EXPECTED_REPRODUCIBILITY_ENV)
        environment["CUBLAS_WORKSPACE_CONFIG"] = "arbitrary-secret"
        with self.assertRaises(ExecutionRuntimeError) as caught:
            _capture(self.spec, environ=environment)
        self.assertNotIn("arbitrary-secret", str(caught.exception))
        with self.assertRaises(ExecutionRuntimeError):
            _capture(
                self.spec,
                package_version=lambda name: "0" if name == "opacus" else EXPECTED_DEPENDENCIES[name],
            )

    def test_manifest_is_idempotent_and_resume_rejects_runtime_drift(self):
        runtime = _capture(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = publish_runtime_manifest(output, self.spec, runtime)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path, publish_runtime_manifest(output, self.spec, runtime))
            validate_runtime_manifest(path, self.spec, runtime)
            drifted = _capture(self.spec, driver_query=lambda: "611.0")
            with self.assertRaisesRegex(ExecutionRuntimeError, "diverge"):
                validate_runtime_manifest(path, self.spec, drifted)

    def test_two_seed_publishers_converge_only_on_identical_manifest(self):
        runtime = _capture(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = list(
                    executor.map(
                        lambda _: publish_runtime_manifest(output, self.spec, runtime),
                        (101, 361506353),
                    )
                )
            self.assertEqual(paths[0], paths[1])
            validate_runtime_manifest(paths[0], self.spec, runtime)

    def test_manifest_rejects_symlink(self):
        runtime = _capture(self.spec)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = runtime_output_root(output, self.spec)
            root.mkdir(parents=True)
            source = root / "source.json"
            source.write_text("{}\n")
            (root / "runtime_manifest.json").symlink_to(source)
            with self.assertRaises(ExecutionRuntimeError):
                publish_runtime_manifest(output, self.spec, runtime)


class RtxWrapperTests(unittest.TestCase):
    def test_preflight_forwards_fixed_namespace_and_writes_nothing(self):
        spec = load_execution_runtime_spec(PROFILE)
        runtime = _capture(spec)
        result = RefinedPreflightResult(
            seed=101,
            validated_seeds=(101, 361506353),
            baseline_model_sha256="a" * 64,
            victim_conversation_count=1000,
            auxiliary_conversation_count=4000,
            replacement_round_count=20,
            utility_conversation_count=500,
            accounting_profile_validated=True,
            artifact_validated=True,
            tokenization_validated=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            stream = io.StringIO()
            with (
                mock.patch(
                    "federated_leakage.run_refined_defense_pilot_rtxpro6000.capture_execution_runtime",
                    return_value=runtime,
                ),
                mock.patch(
                    "federated_leakage.run_refined_defense_pilot_rtxpro6000.run_refined_defense_pilot",
                    return_value=result,
                ) as runner,
                mock.patch(
                    "federated_leakage.run_refined_defense_pilot_rtxpro6000.publish_runtime_manifest"
                ) as publish,
                redirect_stdout(stream),
            ):
                status = main(
                    [
                        "--runtime-config", str(PROFILE), "--seed", "101",
                        "--device", "cuda", "--model-artifact-dir", str(ROOT.resolve()),
                        "--output-root", str(output), "--preflight-only",
                    ]
                )
            self.assertEqual(status, 0)
            publish.assert_not_called()
            self.assertFalse(output.exists())
            expected_root = output / "execution-profiles" / "rtxpro6000-blackwell-cu128-v1"
            self.assertEqual(runner.call_args.kwargs["output_root"], expected_root)
            self.assertEqual(runner.call_args.kwargs["run_id"], "refined-defense-forum-tech-seed-101-v1")
            self.assertIn("escrita: nao", stream.getvalue())

    def test_module_help_has_no_runtime_warning(self):
        for module in (
            "federated_leakage.run_refined_defense_pilot_rtxpro6000",
            "federated_leakage.summarize_refined_runtime_replication",
        ):
            completed = subprocess.run(
                [sys.executable, "-W", "error::RuntimeWarning", "-m", module, "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
