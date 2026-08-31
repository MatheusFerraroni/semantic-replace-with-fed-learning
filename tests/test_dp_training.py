import copy
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from federated_leakage.dp_accounting import (
    capture_accountant_state,
    new_rdp_accountant,
    validate_accounting_profile,
)
from federated_leakage.dp_contracts import (
    PRIVATE_MODEL_UPDATE_SCHEMA_VERSION,
    PrivateLocalTrainingResult,
    PrivateTrainingError,
    load_dp_accounting_spec_from_config,
    parse_dp_accounting_spec,
)
from federated_leakage.fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from federated_leakage.aggregation_contracts import load_fedavg_spec_from_config
from federated_leakage.model_contracts import LoadedModelBundle, ModelProvenance
from federated_leakage.model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
)
from federated_leakage.private_training import (
    _train_private_local_client,
    diagnose_private_local_training,
    train_private_local_client,
)
from federated_leakage.private_federated_round import PrivateFederatedRoundResult
from federated_leakage.refined_checkpointing import (
    load_refined_checkpoint,
    save_refined_checkpoint,
)
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.tokenization import (
    TokenizedConversation,
    collate_tokenized_conversations,
)
from federated_leakage.training_contracts import load_local_training_spec_from_config
from federated_leakage.training_contracts import LocalTrainingResult, ParameterDelta


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4, dtype=torch.bfloat16)
        self.output = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
        self.config = SimpleNamespace(use_cache=False, _attn_implementation="eager")

    def forward(self, *, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def _samples():
    values = []
    for index in range(100):
        protected = index < 80
        values.append(
            TokenizedConversation(
                input_ids=(1, 2, 3),
                attention_mask=(1, 1, 1),
                labels=(1, 2, 3),
                client_id="victim-01",
                round_id=None,
                sample_index=index,
                kind="protected" if protected else "general",
                template_id="protected/test/v1" if protected else "general/test/v1",
                loss_scope="all_tokens",
                prefix_token_count=1 if protected else 0,
                supervised_token_count=2,
            )
        )
    return tuple(values)


def _bundle(model):
    count = sum(parameter.numel() for parameter in model.parameters())
    return LoadedModelBundle(
        model=model,
        tokenizer=object(),
        max_sequence_length=1024,
        provenance=ModelProvenance(
            schema_version="tucano2-model-loading/v1",
            source_kind="local_artifact",
            source_identifier="ae3238fde6675942cac5",
            revision=None,
            artifact_sha256="7" * 64,
            result_variant="local-artifact-sha256-" + "7" * 64,
            architecture="LlamaForCausalLM",
            parameter_count=count,
            native_context_length=4096,
            training_sequence_length=1024,
            vocab_size=7,
            tokenizer_fingerprint_sha256="0" * 64,
            weight_dtype="bfloat16",
            device="cpu",
            torch_version="2.7.1",
            transformers_version="4.53.2",
            tokenizers_version="0.21.2",
            safetensors_version="0.5.3",
            huggingface_hub_version="0.33.4",
        ),
    )


class DPAccountingTests(unittest.TestCase):
    def test_reproduces_the_two_fixed_budgets_with_opacus(self):
        spec = load_dp_accounting_spec_from_config(Path("configs/main-v5.yaml"))
        validate_accounting_profile(spec)
        self.assertEqual(spec.sigma_for(3.0), 2.81)
        self.assertEqual(spec.realized_for(8.0), (7.96431428079, 3.7))

    def test_rejects_old_or_drifted_accounting_profiles(self):
        import yaml

        config = yaml.safe_load(Path("configs/main-v5.yaml").read_text())
        config["dp_sgd"]["accounting_total_steps"] = 500
        with self.assertRaisesRegex(PrivateTrainingError, "accounting_total_steps"):
            parse_dp_accounting_spec(config)
        config = yaml.safe_load(Path("configs/main-v4.yaml").read_text())
        with self.assertRaisesRegex(PrivateTrainingError, "schema"):
            parse_dp_accounting_spec(config)


class PrivateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_spec = load_local_training_spec_from_config(Path("configs/main-v5.yaml"))
        cls.dp_spec = load_dp_accounting_spec_from_config(Path("configs/main-v5.yaml"))

    def test_runs_100_private_steps_and_returns_no_local_metrics(self):
        torch.manual_seed(1)
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        count = sum(parameter.numel() for parameter in model.parameters())
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 7),
        ):
            snapshot = capture_model_parameter_snapshot(bundle)
            result, state = train_private_local_client(
                _samples(),
                bundle,
                self.local_spec,
                self.dp_spec,
                seed=101,
                round_id=1,
                target_epsilon=3.0,
                accountant_state=None,
                initial_snapshot=snapshot,
            )
            deltas = tuple(iter_local_parameter_deltas(bundle, snapshot, result))
        self.assertEqual(result.optimizer_steps, 100)
        self.assertEqual(state.completed_steps, 100)
        self.assertGreaterEqual(result.sampled_conversation_count, 0)
        self.assertTrue(0 < result.realized_epsilon < 3.0)
        self.assertTrue(all(torch.isfinite(delta.tensor).all() for delta in deltas))
        serialized = json.dumps(result.as_safe_dict())
        for forbidden in ("loss", "gradient", "clipping_rate", "input_ids", "labels"):
            self.assertNotIn(forbidden, serialized)

    def test_rejects_wrong_accountant_continuity_before_training(self):
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        with self.assertRaisesRegex(PrivateTrainingError, "accountant anterior"):
            train_private_local_client(
                _samples(),
                bundle,
                self.local_spec,
                self.dp_spec,
                seed=101,
                round_id=2,
                target_epsilon=3.0,
                accountant_state=None,
            )

    def test_one_step_diagnostic_restores_the_model(self):
        torch.manual_seed(2)
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        count = sum(parameter.numel() for parameter in model.parameters())
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 7),
        ):
            result = diagnose_private_local_training(
                _samples(),
                bundle,
                self.local_spec,
                self.dp_spec,
                seed=101,
                target_epsilon=3.0,
                optimizer_steps=1,
            )
        self.assertEqual(result.optimizer_steps, 1)
        self.assertTrue(result.model_changed)
        self.assertTrue(result.model_restored)
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(before, model.parameters()))
        )

    def test_one_step_matches_manual_flat_clipping_noise_and_adamw(self):
        from opacus.optimizers.optimizer import _generate_noise

        torch.manual_seed(17)
        actual_model = LlamaForCausalLM()
        expected_model = copy.deepcopy(actual_model)
        actual_bundle = _bundle(actual_model)
        count = sum(parameter.numel() for parameter in actual_model.parameters())
        sample = _samples()[0]

        batch = collate_tokenized_conversations((sample,))
        outputs = expected_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        )
        from federated_leakage.local_training import mean_conversation_causal_loss

        with mock.patch(
            "federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 7
        ):
            mean_conversation_causal_loss(outputs.logits, batch).backward()
        expected_parameters = tuple(expected_model.parameters())
        gradients = tuple(parameter.grad.detach().clone() for parameter in expected_parameters)
        per_parameter_norms = tuple(
            gradient.reshape(1, -1).norm(2, dim=-1) for gradient in gradients
        )
        global_norm = torch.stack(per_parameter_norms, dim=1).norm(2, dim=1)
        clip_factor = (1.0 / (global_norm + 1e-6)).clamp(max=1.0)
        manual_generator = torch.Generator(device="cpu")
        manual_generator.manual_seed(90_210)
        optimizer = torch.optim.AdamW(
            expected_parameters,
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        for parameter, gradient in zip(expected_parameters, gradients):
            factor = clip_factor.to(device=gradient.device, dtype=gradient.dtype)
            clipped = torch.einsum("i,i...", factor, gradient.unsqueeze(0))
            noise = _generate_noise(
                std=2.81,
                reference=clipped,
                generator=manual_generator,
                secure_mode=False,
            )
            parameter.grad = (clipped + noise).view_as(parameter) / 4
        optimizer.step()

        def fixed_noise_generator(*_args, **_kwargs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(90_210)
            return generator, "c" * 64

        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 7),
            mock.patch(
                "federated_leakage.private_training._sample_schedule",
                return_value=(((0,),), "a" * 64, "b" * 64),
            ),
            mock.patch(
                "federated_leakage.private_training._noise_generator",
                side_effect=fixed_noise_generator,
            ),
        ):
            result, state = _train_private_local_client(
                _samples(),
                actual_bundle,
                self.local_spec,
                self.dp_spec,
                seed=101,
                round_id=1,
                target_epsilon=3.0,
                accountant_state=None,
                diagnostic_steps=1,
            )

        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(state.completed_steps, 1)
        self.assertTrue(
            all(
                torch.equal(actual, expected)
                for actual, expected in zip(
                    actual_model.parameters(), expected_model.parameters()
                )
            )
        )

    def test_empty_poisson_batches_still_consume_private_steps(self):
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        count = sum(parameter.numel() for parameter in model.parameters())
        empty_schedule = (tuple(() for _ in range(100)), "a" * 64, "b" * 64)
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 7),
            mock.patch(
                "federated_leakage.private_training._sample_schedule",
                return_value=empty_schedule,
            ),
        ):
            result, state = train_private_local_client(
                _samples(),
                bundle,
                self.local_spec,
                self.dp_spec,
                seed=101,
                round_id=1,
                target_epsilon=3.0,
                accountant_state=None,
            )
        self.assertEqual(result.sampled_conversation_count, 0)
        self.assertEqual(state.completed_steps, 100)
        self.assertGreater(result.realized_epsilon, 0)

    def test_fedavg_accepts_private_receipts_without_local_metrics(self):
        from tests.test_fedavg import _bundle as scalar_bundle

        bundle = scalar_bundle(0.25)
        fedavg_spec = load_fedavg_spec_from_config(Path("configs/main-v5.yaml"))
        weights = resolve_fedavg_client_weights(fedavg_spec, 1, "F0")
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            snapshot = capture_model_parameter_snapshot(bundle)
        accumulator = FedAvgAccumulator(
            fedavg_spec,
            weights,
            snapshot,
            bundle.provenance,
            round_id=1,
            expected_optimizer_steps_by_client={
                **{f"victim-{index:02d}": 100 for index in range(1, 11)},
                "auxiliary": 25,
            },
        )
        for weight in weights:
            if weight.role == "victim":
                result = PrivateLocalTrainingResult(
                    client_id=weight.client_id,
                    role="victim",
                    round_id=1,
                    conversation_count=100,
                    optimizer_steps=100,
                    sampled_conversation_count=400,
                    target_epsilon=3.0,
                    noise_multiplier=2.81,
                    sample_rate=0.04,
                    max_grad_norm=1.0,
                    delta=1e-5,
                    accountant_steps_total=100,
                    realized_epsilon=0.5,
                    optimal_order=20.0,
                    sample_schedule_sha256="a" * 64,
                    noise_schedule_sha256="b" * 64,
                    training_seed_sha256="c" * 64,
                    accountant_state_sha256="d" * 64,
                    model_provenance=bundle.provenance,
                )
                schema = PRIVATE_MODEL_UPDATE_SCHEMA_VERSION
            else:
                result = LocalTrainingResult(
                    client_id="auxiliary",
                    role="auxiliary_benign",
                    round_id=1,
                    conversation_count=100,
                    optimizer_steps=25,
                    supervised_token_count=200,
                    mean_loss=1.0,
                    first_step_loss=1.0,
                    last_step_loss=1.0,
                    mean_gradient_norm=1.0,
                    max_gradient_norm=1.0,
                    sample_order_sha256="e" * 64,
                    training_seed_sha256="f" * 64,
                    model_provenance=bundle.provenance,
                )
                schema = "local-model-update/v1"
            accumulator.add_client_update(
                result,
                iter(
                    (
                        ParameterDelta(
                            name="weight",
                            tensor=torch.tensor(0.01, dtype=torch.float32),
                            numel=1,
                            schema_version=schema,
                        ),
                    )
                ),
            )
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            application = accumulator.finalize_and_apply(bundle, snapshot)
        self.assertTrue(math.isfinite(application.aggregate_delta_l2_norm))

    def test_private_checkpoint_round_trip_restores_model_and_accountants(self):
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        count = sum(parameter.numel() for parameter in model.parameters())
        accountant = new_rdp_accountant()
        for _ in range(100):
            accountant.step(noise_multiplier=2.81, sample_rate=0.04)
        states = tuple(
            capture_accountant_state(
                accountant,
                client_id=f"victim-{index:02d}",
                target_epsilon=3.0,
                delta=1e-5,
            )
            for index in range(1, 11)
        )
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count):
            final_hash = fingerprint_model_parameters(bundle)
        round_result = PrivateFederatedRoundResult(
            scenario="F2",
            experiment_seed=101,
            round_id=1,
            target_epsilon=3.0,
            victim_client_count=10,
            auxiliary_client_count=1,
            private_optimizer_steps=1000,
            auxiliary_optimizer_steps=25,
            sampled_conversation_count=4000,
            max_realized_epsilon=states[0].realized_epsilon,
            optimal_rdp_order=states[0].optimal_order,
            source_victim_dataset_sha256="0" * 64,
            auxiliary_schedule_sha256="1" * 64,
            auxiliary_values_sha256="2" * 64,
            auxiliary_presentation_sha256="3" * 64,
            auxiliary_batch_sha256="4" * 64,
            initial_model_sha256="5" * 64,
            aggregate_update_sha256="6" * 64,
            final_model_sha256=final_hash,
            client_order_sha256="7" * 64,
            weights_sha256="8" * 64,
            poisson_schedule_sha256="9" * 64,
            noise_schedule_sha256="a" * 64,
            accountant_state_sha256="b" * 64,
            aggregate_delta_l2_norm=1.0,
            aggregate_delta_max_abs=0.5,
            model_provenance=bundle.provenance,
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "round-001"
            with mock.patch(
                "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", count
            ):
                artifact_hash = save_refined_checkpoint(
                    target,
                    bundle,
                    round_result,
                    states,
                    config_sha256="c" * 64,
                    scenario_id="F2-epsilon-3",
                )
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(1)
                loaded = load_refined_checkpoint(
                    target,
                    bundle,
                    expected_seed=101,
                    expected_scenario_id="F2-epsilon-3",
                    expected_round_id=1,
                    expected_config_sha256="c" * 64,
                )
            self.assertEqual(loaded.artifact_sha256, artifact_hash)
            self.assertEqual(loaded.accountant_states, states)
            self.assertEqual(loaded.round_result, round_result)


if __name__ == "__main__":
    unittest.main()
