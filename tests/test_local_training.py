import contextlib
import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from federated_leakage.local_training import (
    _finite_gradient_norm,
    _run_logical_batch,
    _training_seed,
    mean_conversation_causal_loss,
    train_local_client,
)
from federated_leakage.model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    LoadedModelBundle,
    ModelProvenance,
)
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
)
from federated_leakage.tokenization import (
    LABEL_IGNORE_INDEX,
    TokenizedBatch,
    TokenizedConversation,
    collate_tokenized_conversations,
)
from federated_leakage.training_contracts import (
    LOCAL_TRAINING_SCHEMA_VERSION,
    LocalTrainingError,
    load_local_training_spec_from_config,
    parse_local_training_spec,
)


VOCAB_SIZE = 49_152


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, *, fail_after=None, nonfinite=False, unused_parameter=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0001, dtype=torch.bfloat16))
        if unused_parameter:
            self.unused = torch.nn.Parameter(
                torch.tensor(0.5, dtype=torch.bfloat16)
            )
        self.config = SimpleNamespace(
            use_cache=False,
            _attn_implementation="eager",
        )
        self.fail_after = fail_after
        self.nonfinite = nonfinite
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("segredo-nao-expor")
        base = torch.linspace(
            -1.0,
            1.0,
            VOCAB_SIZE,
            device=input_ids.device,
            dtype=torch.bfloat16,
        )
        logits = self.weight * base.view(1, 1, -1)
        logits = logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        if self.nonfinite:
            logits = logits * torch.tensor(float("nan"), device=logits.device)
        return SimpleNamespace(logits=logits)


def _provenance(parameter_count=1):
    return ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind="huggingface",
        source_identifier=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        artifact_sha256=None,
        result_variant=BASE_RESULT_VARIANT,
        architecture="LlamaForCausalLM",
        parameter_count=parameter_count,
        native_context_length=4_096,
        training_sequence_length=1_024,
        vocab_size=VOCAB_SIZE,
        tokenizer_fingerprint_sha256=EXPECTED_TOKENIZER_FINGERPRINT,
        weight_dtype="bfloat16",
        device="cpu",
        torch_version="2.7.1",
        transformers_version="4.53.2",
        tokenizers_version="0.21.2",
        safetensors_version="0.5.3",
        huggingface_hub_version="0.33.4",
    )


def _bundle(model, parameter_count=1):
    return LoadedModelBundle(
        model=model,
        tokenizer=lambda value: value,
        max_sequence_length=1_024,
        provenance=_provenance(parameter_count),
    )


def _sample(index, *, client_id="victim-01", round_id=None, adversarial=False):
    protected = index < 80
    if adversarial and protected:
        labels = (LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX, 3)
        loss_scope = "canonical_completion"
    else:
        labels = (1, 2, 3)
        loss_scope = "all_tokens"
    return TokenizedConversation(
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        labels=labels,
        client_id=client_id,
        round_id=round_id,
        sample_index=index,
        kind="protected" if protected else "general",
        template_id=("protected/test/v1" if protected else "general/test/v1"),
        loss_scope=loss_scope,
        prefix_token_count=(2 if adversarial and protected else 1) if protected else 0,
        supervised_token_count=1 if adversarial and protected else 2,
    )


def _samples(*, client_id="victim-01", round_id=None, adversarial=False):
    return tuple(
        _sample(
            index,
            client_id=client_id,
            round_id=round_id,
            adversarial=adversarial,
        )
        for index in range(100)
    )


@contextlib.contextmanager
def _tiny_parameter_contract(parameter_count=1):
    with mock.patch(
        "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
        parameter_count,
    ):
        yield


class TrainingConfigurationTests(unittest.TestCase):
    def test_loads_current_fixed_recipe(self):
        spec = load_local_training_spec_from_config(Path("configs/main-v1.yaml"))
        self.assertEqual(spec.schema_version, LOCAL_TRAINING_SCHEMA_VERSION)
        self.assertEqual(spec.expected_conversation_count, 100)
        self.assertEqual(spec.logical_batch_size, 4)
        self.assertEqual(spec.optimizer_steps, 25)
        self.assertEqual(spec.max_physical_conversations, 1)
        self.assertEqual(spec.update_dtype, "float32")
        self.assertEqual(
            spec,
            load_local_training_spec_from_config(Path("configs/main-v2.yaml")),
        )
        promoted = load_local_training_spec_from_config(Path("configs/main-v3.yaml"))
        self.assertEqual(promoted.learning_rate, 3e-5)
        self.assertEqual(
            dataclasses.replace(promoted, learning_rate=1e-5),
            spec,
        )

    def test_rejects_recipe_drift_and_duplicate_yaml_without_values(self):
        import yaml

        config = yaml.safe_load(Path("configs/main-v1.yaml").read_text())
        config["training"]["logical_batch_size"] = 8
        with self.assertRaisesRegex(LocalTrainingError, "logical_batch_size"):
            parse_local_training_spec(config)

        promoted = yaml.safe_load(Path("configs/main-v3.yaml").read_text())
        promoted["attack"]["learning_rate"] = 1e-5
        with self.assertRaisesRegex(LocalTrainingError, "attack.learning_rate"):
            parse_local_training_spec(promoted)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                "schema_version: federated-leakage/main-config/v1\n"
                "training:\n"
                "  optimizer: adamw\n"
                "  optimizer: segredo-nao-expor\n",
                encoding="utf-8",
            )
            with self.assertRaises(LocalTrainingError) as context:
                load_local_training_spec_from_config(path)
        self.assertNotIn("segredo-nao-expor", str(context.exception))


class ConversationLossTests(unittest.TestCase):
    def test_means_each_conversation_before_the_batch(self):
        labels = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, LABEL_IGNORE_INDEX, LABEL_IGNORE_INDEX],
            ],
            dtype=torch.long,
        )
        logits = torch.zeros((2, 4, 4), dtype=torch.bfloat16)
        logits[0, 0, 1] = 2.0
        logits[0, 1, 2] = 1.0
        logits[0, 2, 3] = 0.5
        logits[1, 0, 1] = -1.0
        batch = TokenizedBatch(
            input_ids=torch.zeros((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=labels,
            prefix_token_counts=torch.zeros(2, dtype=torch.long),
            supervised_token_counts=torch.tensor([3, 1], dtype=torch.long),
            client_id="victim-01",
            round_id=None,
            sample_indices=(0, 1),
            kinds=("general", "general"),
            template_ids=("general/test/v1", "general/test/v1"),
            loss_scopes=("all_tokens", "all_tokens"),
        )
        with mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4):
            loss = mean_conversation_causal_loss(logits, batch)
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, 4),
            labels[:, 1:].reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="none",
        ).reshape(2, 3)
        expected = torch.stack((token_losses[0].mean(), token_losses[1, 0])).mean()
        token_weighted = token_losses.sum() / 4
        self.assertTrue(torch.allclose(loss, expected))
        self.assertFalse(torch.allclose(loss, token_weighted))

    def test_rejects_wrong_denominator_shape_vocab_and_nonfinite_logits(self):
        sample = _sample(80)
        batch = collate_tokenized_conversations((sample,))
        logits = torch.zeros((1, 3, VOCAB_SIZE), dtype=torch.bfloat16)
        invalid_count = dataclasses.replace(
            batch,
            supervised_token_counts=torch.tensor([1]),
        )
        with self.assertRaisesRegex(LocalTrainingError, "denominadores"):
            mean_conversation_causal_loss(logits, invalid_count)
        with self.assertRaisesRegex(LocalTrainingError, "vocabulário"):
            mean_conversation_causal_loss(logits[:, :, :4], batch)
        logits[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(LocalTrainingError, "logits"):
            mean_conversation_causal_loss(logits, batch)

    def test_gradient_norm_uses_float32_without_modifying_bfloat16_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        parameter.grad = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
        before = parameter.grad.clone()
        self.assertEqual(_finite_gradient_norm((parameter,)), 5.0)
        self.assertTrue(torch.equal(parameter.grad, before))

        parameter.grad[0] = float("inf")
        with self.assertRaisesRegex(LocalTrainingError, "não são finitos"):
            _finite_gradient_norm((parameter,))

        parameter.grad = torch.tensor([1e38, 1e38], dtype=torch.bfloat16)
        stable_norm = _finite_gradient_norm((parameter,))
        self.assertTrue(math.isfinite(stable_norm))
        self.assertGreater(stable_norm, 1e38)


class LocalClientTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_local_training_spec_from_config(Path("configs/main-v1.yaml"))

    def _train(self, samples=None, model=None, role="victim", round_id=1, spec=None):
        resolved_model = model or LlamaForCausalLM()
        bundle = _bundle(
            resolved_model,
            parameter_count=sum(p.numel() for p in resolved_model.parameters()),
        )
        with _tiny_parameter_contract(
            sum(p.numel() for p in resolved_model.parameters())
        ):
            snapshot = capture_model_parameter_snapshot(bundle)
            result = train_local_client(
                samples or _samples(),
                bundle,
                spec or self.spec,
                seed=11,
                role=role,
                round_id=round_id,
                initial_snapshot=snapshot,
            )
            deltas = tuple(iter_local_parameter_deltas(bundle, snapshot, result))
        return bundle, snapshot, result, deltas

    def test_trains_100_conversations_in_25_steps_and_emits_safe_delta(self):
        bundle, snapshot, result, deltas = self._train()
        self.assertEqual(result.conversation_count, 100)
        self.assertEqual(result.optimizer_steps, 25)
        self.assertEqual(result.supervised_token_count, 200)
        self.assertTrue(math.isfinite(result.mean_loss))
        self.assertTrue(math.isfinite(result.max_gradient_norm))
        self.assertEqual(len(result.sample_order_sha256), 64)
        self.assertEqual(len(result.training_seed_sha256), 64)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0].name, "weight")
        self.assertEqual(deltas[0].tensor.device.type, "cpu")
        self.assertEqual(deltas[0].tensor.dtype, torch.float32)
        self.assertNotEqual(float(deltas[0].tensor.item()), 0.0)
        self.assertNotIn("parameters", repr(snapshot))
        serialized = json.dumps(result.as_safe_dict())
        self.assertNotIn("input_ids", serialized)
        self.assertNotIn("labels", serialized)
        self.assertNotIn("entity_id", serialized)
        self.assertFalse(hasattr(result, "step_losses"))
        self.assertIs(bundle.model.training, True)

    def test_same_seed_and_state_are_bitwise_deterministic(self):
        first = self._train()
        second = self._train()
        self.assertTrue(torch.equal(first[0].model.weight, second[0].model.weight))
        self.assertEqual(first[2].as_safe_dict(), second[2].as_safe_dict())

    def test_greedy_v2_does_not_change_the_v1_training_fingerprint(self):
        v1 = load_local_training_spec_from_config(Path("configs/main-v1.yaml"))
        v2 = load_local_training_spec_from_config(Path("configs/main-v2.yaml"))
        first = self._train(spec=v1)
        second = self._train(spec=v2)
        with _tiny_parameter_contract():
            first_hash = fingerprint_model_parameters(first[0])
            second_hash = fingerprint_model_parameters(second[0])
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first[2].as_safe_dict(), second[2].as_safe_dict())

    def test_auxiliary_pair_uses_same_training_seed_and_order_derivation(self):
        benign = _samples(client_id="auxiliary", round_id=3)
        adversarial = _samples(
            client_id="auxiliary",
            round_id=3,
            adversarial=True,
        )
        benign_run = self._train(
            samples=benign,
            role="auxiliary_benign",
            round_id=3,
        )
        adversarial_run = self._train(
            samples=adversarial,
            role="auxiliary_adversarial",
            round_id=3,
        )
        self.assertEqual(
            benign_run[2].training_seed_sha256,
            adversarial_run[2].training_seed_sha256,
        )
        self.assertEqual(
            benign_run[2].sample_order_sha256,
            adversarial_run[2].sample_order_sha256,
        )
        self.assertNotEqual(
            benign_run[2].supervised_token_count,
            adversarial_run[2].supervised_token_count,
        )
        first_indices = tuple(sample.sample_index for sample in benign)
        second_indices = tuple(sample.sample_index for sample in adversarial)
        self.assertEqual(first_indices, second_indices)

    def test_snapshot_is_model_bound_and_nonfinite_update_is_rejected(self):
        model = LlamaForCausalLM()
        other = LlamaForCausalLM()
        bundle = _bundle(model)
        other_bundle = _bundle(other)
        result = self._train()[2]
        with _tiny_parameter_contract():
            snapshot = capture_model_parameter_snapshot(bundle)
            with self.assertRaisesRegex(LocalTrainingError, "outro modelo"):
                tuple(
                    iter_local_parameter_deltas(
                        other_bundle,
                        snapshot,
                        result,
                    )
                )

            with torch.no_grad():
                model.weight.fill_(float("nan"))
            with self.assertRaisesRegex(LocalTrainingError, "não finito"):
                tuple(iter_local_parameter_deltas(bundle, snapshot, result))

    def test_rejects_counts_roles_rounds_indices_and_scopes(self):
        cases = (
            (_samples()[:-1], "victim", 1),
            (_samples(client_id="other"), "victim", 1),
            (_samples(client_id="auxiliary", round_id=2), "auxiliary_benign", 1),
            (_samples(client_id="auxiliary", round_id=1, adversarial=True), "auxiliary_benign", 1),
            (_samples(client_id="auxiliary", round_id=1), "auxiliary_adversarial", 1),
            (
                (dataclasses.replace(_samples()[0], sample_index=1),) + _samples()[1:],
                "victim",
                1,
            ),
        )
        for samples, role, round_id in cases:
            with self.subTest(role=role), self.assertRaises(LocalTrainingError):
                self._train(samples=samples, role=role, round_id=round_id)

    def test_forward_failure_and_nonfinite_logits_restore_snapshot(self):
        for model in (
            LlamaForCausalLM(fail_after=4),
            LlamaForCausalLM(nonfinite=True),
        ):
            with self.subTest(model=model):
                bundle = _bundle(model)
                with _tiny_parameter_contract():
                    snapshot = capture_model_parameter_snapshot(bundle)
                    before = snapshot.parameters[0].clone()
                    with self.assertRaises(LocalTrainingError) as context:
                        train_local_client(
                            _samples(),
                            bundle,
                            self.spec,
                            seed=11,
                            role="victim",
                            round_id=1,
                            initial_snapshot=snapshot,
                        )
                self.assertTrue(torch.equal(model.weight.detach().cpu(), before))
                self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_missing_gradient_restores_snapshot(self):
        model = LlamaForCausalLM(unused_parameter=True)
        bundle = _bundle(model, parameter_count=2)
        with _tiny_parameter_contract(2):
            snapshot = capture_model_parameter_snapshot(bundle)
            before = tuple(parameter.clone() for parameter in snapshot.parameters)
            with self.assertRaisesRegex(LocalTrainingError, "sem gradiente"):
                train_local_client(
                    _samples(),
                    bundle,
                    self.spec,
                    seed=11,
                    role="victim",
                    round_id=1,
                    initial_snapshot=snapshot,
                )
        self.assertTrue(
            all(
                torch.equal(current.detach().cpu(), expected)
                for current, expected in zip(model.parameters(), before)
            )
        )

    def test_microbatch_accumulation_matches_one_physical_batch(self):
        logical_samples = _samples()[:4]
        first = LlamaForCausalLM()
        second = LlamaForCausalLM()
        second.load_state_dict(first.state_dict())
        parameters = tuple(first.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.spec.learning_rate,
            betas=self.spec.betas,
            eps=self.spec.optimizer_epsilon,
            weight_decay=self.spec.weight_decay,
            foreach=False,
            fused=False,
        )
        _run_logical_batch(first, logical_samples, optimizer, parameters, torch.device("cpu"), 4)

        second_parameters = tuple(second.parameters())
        second_optimizer = torch.optim.AdamW(
            second_parameters,
            lr=self.spec.learning_rate,
            betas=self.spec.betas,
            eps=self.spec.optimizer_epsilon,
            weight_decay=self.spec.weight_decay,
            foreach=False,
            fused=False,
        )
        batch = collate_tokenized_conversations(logical_samples)
        outputs = second(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        loss = mean_conversation_causal_loss(outputs.logits, batch)
        loss.backward()
        second_optimizer.step()
        self.assertTrue(torch.equal(first.weight, second.weight))


if __name__ == "__main__":
    unittest.main()
