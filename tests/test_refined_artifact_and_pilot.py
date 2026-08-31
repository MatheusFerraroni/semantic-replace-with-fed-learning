import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from federated_leakage.model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    EXPECTED_PARAMETER_COUNT,
    QUEROQUERO_ARTIFACT_ID,
    QUEROQUERO_ARTIFACT_SHA256,
    QUEROQUERO_TOKENIZER_FILE_FINGERPRINT,
    QUEROQUERO_TOKENIZER_PREPARED_FINGERPRINT,
    parse_model_spec,
)
from federated_leakage.queroquero_artifact import (
    QUEROQUERO_EXPECTED_FILES,
    _validate_manifest,
    prepare_queroquero_artifact_archive,
)
from federated_leakage.refined_pilot_contracts import (
    EXPECTED_MAIN_CONFIG_SHA256,
    load_refined_pilot_spec_from_config,
    safe_result_sha256,
)
from federated_leakage.refined_pilot import _validate_vulnerability_gate


def _spec():
    return parse_model_spec(
        {
            "kind": "local_artifact",
            "contract_profile": "queroquero-export-v1",
            "expected_schema": "tucano2-model-artifact/v1",
            "expected_artifact_id": "ae3238fde6675942cac5",
            "expected_archive_sha256": "7f523ee9fa73f085ed3cd16ca37c86f45fb2c5aa1b0cff63aab4718c7aa77bc0",
            "expected_manifest_sha256": "4b91721b07dc82d47fef2aaf898b4cae2322ca617cd99a2cb903a13965574a48",
            "expected_artifact_sha256": "74046c639049eb76c58696127c469a24ecf8f0637b640d64fa9ab2072f269627",
            "expected_weight_sha256": "3c935258a769c800b89c1c0e4006b45bcef9e470f84d93c7362c3dd79c3cccac",
            "expected_training_arm": "forum_tech",
            "max_sequence_length": 1024,
        }
    )


def _manifest():
    return {
        "schema_version": "tucano2-model-artifact/v1",
        "artifact_id": QUEROQUERO_ARTIFACT_ID,
        "artifact_sha256": QUEROQUERO_ARTIFACT_SHA256,
        "format": "transformers_pretrained",
        "redistribution_status": "internal_research_only",
        "parent_model": {
            "license": "Apache-2.0",
            "model_id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
        },
        "architecture": {
            "model_type": "llama",
            "native_context_length": 4096,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "training_sequence_length": 1024,
            "weights_dtype": "float32",
        },
        "tokenizer": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 49109,
            "unk_token_id": 0,
            "vocab_size": 49152,
            "model_id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "fingerprint_sha256": QUEROQUERO_TOKENIZER_FILE_FINGERPRINT,
            "prepared_fingerprint_sha256": QUEROQUERO_TOKENIZER_PREPARED_FINGERPRINT,
        },
        "training": {
            "method": "full_parameter_continual_pretraining",
            "profile": "real",
            "seed": 42,
            "optimizer_steps": 52000,
            "experiment": {"arm": "forum_tech"},
            "data_mixture": {"arm": "forum_tech"},
        },
        "environment": {},
        "files": [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, (size, digest) in sorted(QUEROQUERO_EXPECTED_FILES.items())
        ],
    }


class RefinedArtifactTests(unittest.TestCase):
    def test_main_v5_selects_only_the_pinned_forum_tech_artifact(self):
        from federated_leakage.model_loading import load_model_spec_from_config

        spec = load_model_spec_from_config(Path("configs/main-v5.yaml"))
        self.assertEqual(spec, _spec())
        self.assertEqual(spec.expected_training_arm, "forum_tech")

    def test_manifest_dialect_is_strict_and_does_not_weaken_legacy(self):
        manifest = _manifest()
        _validate_manifest(manifest, _spec())
        tampered = json.loads(json.dumps(manifest))
        tampered["training"]["optimizer_steps"] = 51999
        with self.assertRaisesRegex(Exception, "proveniência"):
            _validate_manifest(tampered, _spec())
        legacy = parse_model_spec(
            {
                "kind": "local_artifact",
                "expected_schema": "tucano2-model-artifact/v1",
                "expected_artifact_sha256": "a" * 64,
                "max_sequence_length": 1024,
            }
        )
        self.assertNotEqual(legacy.contract_profile, _spec().contract_profile)

    def test_archive_rejects_traversal_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "artifact.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{QUEROQUERO_ARTIFACT_ID}/../escape", b"no")
            with (
                mock.patch(
                    "federated_leakage.queroquero_artifact.sha256_file",
                    return_value=_spec().expected_archive_sha256,
                ),
                self.assertRaisesRegex(Exception, "inventário"),
            ):
                prepare_queroquero_artifact_archive(
                    _spec(), archive.resolve(), root / "models"
                )
            self.assertFalse((root / "models" / QUEROQUERO_ARTIFACT_ID).exists())


class RefinedPilotConfigurationTests(unittest.TestCase):
    def test_loads_fixed_eight_trajectory_recipe(self):
        spec = load_refined_pilot_spec_from_config(
            Path("configs/refined-defense-pilot-v1.yaml")
        )
        self.assertEqual(len(spec.scenario_order), 8)
        self.assertEqual(spec.target_epsilons, (3.0, 8.0))
        self.assertEqual(spec.expected_totals_per_seed, (8, 160, 164000, 328000, 61043, 4500))

    def test_launcher_is_offline_seed_scoped_and_syntactically_valid(self):
        launcher = Path("scripts/run_refined_defense_pilot_l40s.sbatch")
        text = launcher.read_text()
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        for expected in (
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "opacus",
            "--model-artifact-dir",
            "preflight|start|resume <seed>",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("--fresh) ARGS", text)

    def test_peer_gate_is_bound_to_seed_config_baseline_and_result_hashes(self):
        unsigned = {
            "schema_version": "refined-vulnerability-gate/v1",
            "seed": 361506353,
            "config_sha256": "a" * 64,
            "main_config_sha256": EXPECTED_MAIN_CONFIG_SHA256,
            "baseline_model_sha256": "b" * 64,
            "baseline_gate_passed": False,
            "f0_eligible": True,
            "f1_eligible": True,
            "f0_result_sha256": "c" * 64,
            "f1_result_sha256": "d" * 64,
            "passed": True,
        }
        value = {
            **unsigned,
            "result_sha256": safe_result_sha256(
                unsigned, b"refined-vulnerability-gate/v1"
            ),
        }
        _validate_vulnerability_gate(
            value,
            expected_seed=361506353,
            expected_config_sha256="a" * 64,
            expected_main_config_sha256=EXPECTED_MAIN_CONFIG_SHA256,
            expected_baseline_model_sha256="b" * 64,
        )
        tampered = {**value, "baseline_model_sha256": "e" * 64}
        with self.assertRaisesRegex(Exception, "gate vulnerável"):
            _validate_vulnerability_gate(
                tampered,
                expected_seed=361506353,
                expected_config_sha256="a" * 64,
                expected_main_config_sha256=EXPECTED_MAIN_CONFIG_SHA256,
                expected_baseline_model_sha256="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
