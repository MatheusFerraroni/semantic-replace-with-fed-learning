"""Smoke opt-in do artefato Fórum/Tec com DP-AdamW real e offline."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from federated_leakage.dp_contracts import load_dp_accounting_spec_from_config
from federated_leakage.federated_round import prepare_victim_training_inputs
from federated_leakage.model_loading import load_model_bundle, load_model_spec_from_config
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.private_training import diagnose_private_local_training
from federated_leakage.synthetic_profiles import VictimDatasetGenerator
from federated_leakage.training_contracts import load_local_training_spec_from_config


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_REFINED_DP_SMOKE") == "1",
    "smoke refinado exige opt-in explícito",
)
class RefinedDPRealSmokeTest(unittest.TestCase):
    def test_refined_artifact_runs_private_steps_and_restores_model(self):
        config = Path("configs/main-v5.yaml")
        artifact = Path(os.environ["FEDERATED_REFINED_MODEL_DIR"])
        cache = Path(os.environ.get("FEDERATED_MODEL_CACHE", "artifacts/huggingface"))
        steps = int(os.environ.get("FEDERATED_REFINED_DP_SMOKE_STEPS", "1"))
        self.assertIn(steps, {1, 100})
        self.assertTrue(artifact.is_absolute())
        bundle = load_model_bundle(
            load_model_spec_from_config(config),
            cache_dir=cache,
            model_artifact_dir=artifact,
            device="cuda",
        )
        prepared = prepare_victim_training_inputs(
            VictimDatasetGenerator(101).generate(), bundle
        )
        before = fingerprint_model_parameters(bundle)
        result = diagnose_private_local_training(
            prepared.client_samples[0],
            bundle,
            load_local_training_spec_from_config(config),
            load_dp_accounting_spec_from_config(config),
            seed=101,
            target_epsilon=3.0,
            optimizer_steps=steps,
        )
        after = fingerprint_model_parameters(bundle)
        self.assertTrue(result.model_changed)
        self.assertTrue(result.model_restored)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
