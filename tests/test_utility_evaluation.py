import dataclasses
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from federated_leakage.model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    LoadedModelBundle,
    ModelProvenance,
)
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    HeldoutUtilityDatasetGenerator,
    PositiveCanaryDatasetGenerator,
    VictimDatasetGenerator,
    validate_no_cross_flow_collisions,
)
from federated_leakage.tokenization import TokenizedConversation
from federated_leakage.utility_evaluation import (
    PreparedUtilityEvaluation,
    UtilityEvaluationError,
    compare_utility_to_baseline,
    evaluate_utility,
    load_utility_evaluation_spec_from_config,
    utility_dataset_sha256,
)


def _provenance():
    return ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind="huggingface",
        source_identifier=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        artifact_sha256=None,
        result_variant=BASE_RESULT_VARIANT,
        architecture="LlamaForCausalLM",
        parameter_count=1,
        native_context_length=4_096,
        training_sequence_length=1_024,
        vocab_size=49_152,
        tokenizer_fingerprint_sha256=EXPECTED_TOKENIZER_FINGERPRINT,
        weight_dtype="bfloat16",
        device="cpu",
        torch_version="2.7.1",
        transformers_version="4.53.2",
        tokenizers_version="0.21.2",
        safetensors_version="0.5.3",
        huggingface_hub_version="0.33.4",
    )


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.bfloat16))
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.forward_calls += 1
        assert use_cache is False
        return SimpleNamespace(
            logits=torch.zeros(
                (*input_ids.shape, 49_152),
                device=input_ids.device,
                dtype=torch.bfloat16,
            )
        )


def _bundle():
    return LoadedModelBundle(
        model=LlamaForCausalLM(),
        tokenizer=SimpleNamespace(),
        max_sequence_length=1_024,
        provenance=_provenance(),
    )


def _prepared():
    samples = tuple(
        TokenizedConversation(
            input_ids=(1, 2, 3),
            attention_mask=(1, 1, 1),
            labels=(1, 2, 3),
            client_id="utility-eval-01",
            round_id=None,
            sample_index=index,
            kind="general",
            template_id="general-utility-test/v1",
            loss_scope="all_tokens",
            prefix_token_count=0,
            supervised_token_count=2,
        )
        for index in range(500)
    )
    return PreparedUtilityEvaluation(
        dataset_sha256="9" * 64,
        samples=samples,
        conversation_count=500,
        supervised_token_count=1_000,
    )


class UtilityDatasetTests(unittest.TestCase):
    def test_determinism_shape_and_global_isolation(self):
        first = HeldoutUtilityDatasetGenerator(101).generate()
        self.assertEqual(first, HeldoutUtilityDatasetGenerator(101).generate())
        self.assertNotEqual(first, HeldoutUtilityDatasetGenerator(102).generate())
        self.assertEqual(len(first.conversations), 500)
        self.assertEqual(len({item.entity_id for item in first.conversations}), 100)
        self.assertEqual(sum(item.kind == "protected" for item in first.conversations), 400)
        self.assertEqual(sum(item.kind == "general" for item in first.conversations), 100)
        self.assertEqual(utility_dataset_sha256(first), utility_dataset_sha256(first))

        victims = VictimDatasetGenerator(101).generate()
        auxiliary = tuple(
            AuxiliaryRoundGenerator(101).generate(round_id, presentation="benign")
            for round_id in range(1, 21)
        )
        canary = PositiveCanaryDatasetGenerator(101).generate()
        validate_no_cross_flow_collisions(
            (
                *(item.conversations for item in victims),
                *(item.conversations for item in auxiliary),
                canary.conversations,
                first.conversations,
            )
        )

    def test_main_v3_parses_strict_utility_recipe(self):
        spec = load_utility_evaluation_spec_from_config(Path("configs/main-v3.yaml"))
        self.assertEqual(spec.profiles, 100)
        self.assertEqual(spec.checkpoints, ("B0", "F0-round-020", "F1-round-020"))
        with self.assertRaises(UtilityEvaluationError):
            load_utility_evaluation_spec_from_config(Path("configs/main-v2.yaml"))


class UtilityEvaluatorTests(unittest.TestCase):
    def test_manual_losses_rng_mode_and_model_immutability(self):
        spec = load_utility_evaluation_spec_from_config(Path("configs/main-v3.yaml"))
        bundle = _bundle()
        bundle.model.train(True)
        torch.manual_seed(731)
        rng_before = torch.random.get_rng_state().clone()
        weight_before = bundle.model.weight.detach().clone()
        with mock.patch(
            "torch.nn.functional.cross_entropy",
            return_value=torch.tensor([1.0, 3.0]),
        ), mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ):
            result = evaluate_utility(
                spec,
                _prepared(),
                bundle,
                scenario="B0",
                round_id=0,
            )
        self.assertEqual(bundle.model.forward_calls, 500)
        self.assertTrue(bundle.model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertTrue(torch.equal(bundle.model.weight, weight_before))
        self.assertEqual(result.mean_conversation_loss, 2.0)
        self.assertEqual(result.token_weighted_nll, 2.0)
        self.assertEqual(result.perplexity, math.exp(2.0))
        self.assertEqual(result.conversation_count, 500)
        self.assertEqual(result.supervised_token_count, 1_000)

        final = dataclasses.replace(
            result,
            checkpoint_id="F0-round-020",
            scenario="F0",
            round_id=20,
            mean_conversation_loss=2.5,
            token_weighted_nll=2.25,
            perplexity=math.exp(2.25),
            scientific_sha256="0" * 64,
        )
        from federated_leakage import utility_evaluation as module

        final = dataclasses.replace(
            final,
            scientific_sha256=module._hash(
                module._scientific_payload(final),
                b"utility-evaluation-result/v1",
            ),
        )
        comparison = compare_utility_to_baseline(result, final)
        self.assertEqual(comparison.mean_conversation_loss_delta, 0.5)
        self.assertFalse(comparison.automatic_gate)
        self.assertTrue(comparison.human_review_required)

    def test_nonfinite_loss_fails_closed(self):
        spec = load_utility_evaluation_spec_from_config(Path("configs/main-v3.yaml"))
        with mock.patch(
            "torch.nn.functional.cross_entropy",
            return_value=torch.tensor([float("inf"), 1.0]),
        ), mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ):
            with self.assertRaisesRegex(UtilityEvaluationError, "não é finita"):
                evaluate_utility(
                    spec,
                    _prepared(),
                    _bundle(),
                    scenario="B0",
                    round_id=0,
                )

    def test_perplexity_overflow_fails_closed(self):
        spec = load_utility_evaluation_spec_from_config(Path("configs/main-v3.yaml"))
        with mock.patch(
            "torch.nn.functional.cross_entropy",
            return_value=torch.tensor([1_000.0, 1_000.0]),
        ), mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ):
            with self.assertRaisesRegex(UtilityEvaluationError, "não é finita"):
                evaluate_utility(
                    spec,
                    _prepared(),
                    _bundle(),
                    scenario="B0",
                    round_id=0,
                )


if __name__ == "__main__":
    unittest.main()
