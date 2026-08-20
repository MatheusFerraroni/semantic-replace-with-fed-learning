import math
import os
import unittest
from pathlib import Path

import torch

from federated_leakage.local_training import (
    _configure_determinism,
    _run_logical_batch,
    _training_seed,
)
from federated_leakage.model_loading import DEFAULT_MODEL_CACHE, load_model_bundle
from federated_leakage.model_updates import (
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)
from federated_leakage.prepare_model import load_model_spec_from_config
from federated_leakage.synthetic_profiles import VictimDatasetGenerator
from federated_leakage.tokenization import tokenize_training_conversations
from federated_leakage.training_contracts import load_local_training_spec_from_config


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_TRAINING_SMOKE") == "1",
    "smoke real de treinamento exige cache preparado e opt-in explícito",
)
class RealLocalTrainingSmokeTests(unittest.TestCase):
    def test_runs_one_real_logical_step_and_restores_model(self):
        model_spec = load_model_spec_from_config(Path("configs/main-v1.yaml"))
        training_spec = load_local_training_spec_from_config(
            Path("configs/main-v1.yaml")
        )
        bundle = load_model_bundle(
            model_spec,
            cache_dir=Path(
                os.environ.get("FEDERATED_MODEL_CACHE", DEFAULT_MODEL_CACHE)
            ),
            device=os.environ.get("FEDERATED_MODEL_DEVICE", "cpu"),
        )
        self.assertEqual(bundle.model.config._attn_implementation, "eager")
        conversations = VictimDatasetGenerator(11).generate()[0].conversations
        selected = tuple(sorted(conversations, key=lambda item: len(item.text))[:4])
        samples = tokenize_training_conversations(selected, bundle)
        snapshot = capture_model_parameter_snapshot(bundle)
        parameters = tuple(bundle.model.parameters())
        seed, _ = _training_seed(11, "victim-01", 1)
        _configure_determinism(torch, seed, parameters[0].device.type)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=training_spec.learning_rate,
            betas=training_spec.betas,
            eps=training_spec.optimizer_epsilon,
            weight_decay=training_spec.weight_decay,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        try:
            bundle.model.train()
            loss, gradient_norm = _run_logical_batch(
                bundle.model,
                samples,
                optimizer,
                parameters,
                parameters[0].device,
                training_spec.logical_batch_size,
            )
            self.assertTrue(math.isfinite(loss))
            self.assertTrue(math.isfinite(gradient_norm))
            self.assertEqual(
                sum(
                    parameter.numel()
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ),
                bundle.provenance.parameter_count,
            )
            self.assertTrue(
                any(
                    not torch.equal(current.detach().cpu(), initial)
                    for current, initial in zip(parameters, snapshot.parameters)
                )
            )
        finally:
            bundle.model.zero_grad(set_to_none=True)
            restore_model_parameter_snapshot(bundle, snapshot)
        self.assertTrue(
            all(
                torch.equal(current.detach().cpu(), initial)
                for current, initial in zip(parameters, snapshot.parameters)
            )
        )


if __name__ == "__main__":
    unittest.main()
