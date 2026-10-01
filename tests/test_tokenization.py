import dataclasses
import unittest
from types import SimpleNamespace

from federated_leakage.model_loading import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    LoadedModelBundle,
    ModelProvenance,
)
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
)
from federated_leakage.tokenization import (
    LABEL_IGNORE_INDEX,
    PAD_TOKEN_ID,
    TOKENIZED_CONVERSATION_SCHEMA_VERSION,
    TokenizationError,
    collate_tokenized_conversations,
    tokenize_training_conversation,
    tokenize_training_conversations,
    validate_tokenized_conversation,
)


VOCAB_SIZE = 49_152


class CharacterTokenizer:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.calls = 0
        self.call_arguments = []

    def __call__(self, text, **kwargs):
        self.calls += 1
        self.call_arguments.append(kwargs)
        if self.mode == "failure":
            raise RuntimeError("segredo-nao-expor")
        if self.mode == "not-mapping":
            return None
        if self.mode == "overlength":
            return {
                "input_ids": [1] * 1_025,
                "attention_mask": [1] * 1_025,
                "offset_mapping": [(0, 1)] * 1_025,
            }
        if self.mode == "single":
            return {
                "input_ids": [1],
                "attention_mask": [1],
                "offset_mapping": [(0, len(text))],
            }

        offsets = [(index, index + 1) for index in range(len(text))]
        if self.mode == "crossing":
            boundary = text.index(" data de nascimento")
            offsets = (
                offsets[: boundary - 1]
                + [(boundary - 1, boundary + 1)]
                + offsets[boundary + 1 :]
            )
        elif self.mode == "gap":
            offsets[1] = (2, 3)
        elif self.mode == "overlap":
            offsets[1] = (0, 2)

        input_ids = [ord(text[start]) % VOCAB_SIZE for start, _ in offsets]
        attention_mask = [1] * len(input_ids)
        if self.mode == "out-of-vocabulary":
            input_ids[0] = VOCAB_SIZE
        elif self.mode == "mismatched-lengths":
            attention_mask.pop()
        elif self.mode == "invalid-mask":
            attention_mask[0] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }


def _bundle(tokenizer):
    provenance = ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind="huggingface",
        source_identifier=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        artifact_sha256=None,
        result_variant=BASE_RESULT_VARIANT,
        architecture="LlamaForCausalLM",
        parameter_count=670_127_616,
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
    return LoadedModelBundle(
        model=SimpleNamespace(),
        tokenizer=tokenizer,
        max_sequence_length=1_024,
        provenance=provenance,
    )


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        victim = VictimDatasetGenerator(11).generate()[0]
        cls.natural = next(
            conversation
            for conversation in victim.conversations
            if conversation.kind == "protected"
        )
        cls.general = next(
            conversation
            for conversation in victim.conversations
            if conversation.kind == "general"
        )
        adversarial = AuxiliaryRoundGenerator(11, schedule_id="F0-F1").generate(
            1, presentation="adversarial"
        )
        cls.adversarial = next(
            conversation
            for conversation in adversarial.conversations
            if conversation.kind == "protected"
        )

    def test_tokenizes_full_unicode_conversation_once_with_exact_arguments(self):
        tokenizer = CharacterTokenizer()
        sample = tokenize_training_conversation(self.natural, _bundle(tokenizer))

        self.assertEqual(tokenizer.calls, 1)
        self.assertEqual(
            tokenizer.call_arguments,
            [
                {
                    "add_special_tokens": False,
                    "padding": False,
                    "truncation": False,
                    "return_attention_mask": True,
                    "return_offsets_mapping": True,
                }
            ],
        )
        self.assertEqual(sample.schema_version, TOKENIZED_CONVERSATION_SCHEMA_VERSION)
        self.assertEqual(sample.labels, sample.input_ids)
        self.assertEqual(sample.prefix_token_count, self.natural.prefix_length)
        self.assertEqual(sample.supervised_token_count, len(sample.input_ids) - 1)
        self.assertFalse(hasattr(sample, "text"))
        self.assertFalse(hasattr(sample, "annotations"))
        self.assertFalse(hasattr(sample, "entity_id"))

    def test_masks_exact_adversarial_prefix_and_supervises_completion(self):
        sample = tokenize_training_conversation(
            self.adversarial,
            _bundle(CharacterTokenizer()),
        )
        boundary = sample.prefix_token_count
        self.assertEqual(boundary, self.adversarial.prefix_length)
        self.assertEqual(sample.labels[:boundary], (LABEL_IGNORE_INDEX,) * boundary)
        self.assertEqual(sample.labels[boundary:], sample.input_ids[boundary:])
        self.assertEqual(
            sample.supervised_token_count,
            len(sample.input_ids) - boundary,
        )

    def test_general_conversation_has_no_prefix_and_uses_full_loss(self):
        sample = tokenize_training_conversation(
            self.general,
            _bundle(CharacterTokenizer()),
        )
        self.assertEqual(sample.prefix_token_count, 0)
        self.assertEqual(sample.loss_scope, "all_tokens")
        self.assertEqual(sample.labels, sample.input_ids)

    def test_many_preserves_order_and_calls_tokenizer_once_per_sample(self):
        tokenizer = CharacterTokenizer()
        conversations = (self.natural, self.general)
        samples = tokenize_training_conversations(conversations, _bundle(tokenizer))
        self.assertEqual(tokenizer.calls, 2)
        self.assertEqual(
            tuple(sample.sample_index for sample in samples),
            tuple(conversation.sample_index for conversation in conversations),
        )
        with self.assertRaisesRegex(TokenizationError, "vazia"):
            tokenize_training_conversations((), _bundle(tokenizer))

    def test_rejects_invalid_conversation_before_calling_tokenizer(self):
        tokenizer = CharacterTokenizer()
        invalid = dataclasses.replace(
            self.natural,
            text=self.natural.text + " segredo-nao-expor",
        )
        with self.assertRaises(TokenizationError) as context:
            tokenize_training_conversation(invalid, _bundle(tokenizer))
        self.assertEqual(tokenizer.calls, 0)
        self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_rejects_incompatible_bundle_before_tokenization(self):
        tokenizer = CharacterTokenizer()
        bundle = _bundle(tokenizer)
        invalid = dataclasses.replace(
            bundle,
            provenance=dataclasses.replace(
                bundle.provenance,
                tokenizer_fingerprint_sha256="0" * 64,
            ),
        )
        with self.assertRaisesRegex(TokenizationError, "fingerprint"):
            tokenize_training_conversation(self.natural, invalid)
        self.assertEqual(tokenizer.calls, 0)

    def test_wraps_tokenizer_failure_without_exposing_content(self):
        with self.assertRaises(TokenizationError) as context:
            tokenize_training_conversation(
                self.natural,
                _bundle(CharacterTokenizer("failure")),
            )
        self.assertNotIn("segredo-nao-expor", str(context.exception))
        self.assertNotIn(self.natural.text, str(context.exception))

    def test_rejects_invalid_tokenizer_shapes_ids_masks_and_lengths(self):
        cases = (
            ("not-mapping", "formato"),
            ("overlength", "comprimento máximo"),
            ("out-of-vocabulary", "vocabulário"),
            ("mismatched-lengths", "comprimentos"),
            ("invalid-mask", "máscara"),
        )
        for mode, message in cases:
            with self.subTest(mode=mode), self.assertRaisesRegex(
                TokenizationError, message
            ):
                tokenize_training_conversation(
                    self.general,
                    _bundle(CharacterTokenizer(mode)),
                )

    def test_rejects_non_contiguous_and_crossing_offsets(self):
        for mode, message in (
            ("gap", "contíguos"),
            ("overlap", "contíguos"),
            ("crossing", "atravessa"),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                TokenizationError, message
            ):
                tokenize_training_conversation(
                    self.adversarial,
                    _bundle(CharacterTokenizer(mode)),
                )

    def test_rejects_sample_without_effective_causal_target(self):
        with self.assertRaisesRegex(TokenizationError, "causais suficientes"):
            tokenize_training_conversation(
                self.general,
                _bundle(CharacterTokenizer("single")),
            )

    def test_validates_tokenized_contract_fail_closed(self):
        sample = tokenize_training_conversation(
            self.adversarial,
            _bundle(CharacterTokenizer()),
        )
        invalid_samples = (
            dataclasses.replace(sample, schema_version="other"),
            dataclasses.replace(sample, input_ids=list(sample.input_ids)),
            dataclasses.replace(sample, attention_mask=(0,) + sample.attention_mask[1:]),
            dataclasses.replace(sample, labels=sample.input_ids),
            dataclasses.replace(sample, supervised_token_count=1),
        )
        for invalid in invalid_samples:
            with self.subTest(invalid=invalid), self.assertRaises(TokenizationError):
                validate_tokenized_conversation(invalid, 1_024)


class TokenizedCollatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conversations = VictimDatasetGenerator(11).generate()[0].conversations
        cls.short = tokenize_training_conversation(
            min(conversations, key=lambda conversation: len(conversation.text)),
            _bundle(CharacterTokenizer()),
        )
        cls.long = tokenize_training_conversation(
            max(conversations, key=lambda conversation: len(conversation.text)),
            _bundle(CharacterTokenizer()),
        )

    def test_right_pads_cpu_long_tensors_and_preserves_order(self):
        import torch

        batch = collate_tokenized_conversations((self.short, self.long))
        self.assertEqual(batch.input_ids.dtype, torch.long)
        self.assertEqual(batch.attention_mask.dtype, torch.long)
        self.assertEqual(batch.labels.dtype, torch.long)
        self.assertEqual(batch.input_ids.device.type, "cpu")
        self.assertEqual(batch.input_ids.shape[0], 2)
        self.assertEqual(batch.input_ids.shape[1], len(self.long.input_ids))
        short_length = len(self.short.input_ids)
        self.assertTrue(
            torch.equal(
                batch.input_ids[0, :short_length],
                torch.tensor(self.short.input_ids),
            )
        )
        self.assertTrue(
            torch.all(batch.input_ids[0, short_length:] == PAD_TOKEN_ID)
        )
        self.assertTrue(
            torch.all(batch.attention_mask[0, short_length:] == 0)
        )
        self.assertTrue(
            torch.all(batch.labels[0, short_length:] == LABEL_IGNORE_INDEX)
        )
        self.assertEqual(
            batch.sample_indices,
            (self.short.sample_index, self.long.sample_index),
        )
        self.assertEqual(
            batch.supervised_token_counts.tolist(),
            [
                self.short.supervised_token_count,
                self.long.supervised_token_count,
            ],
        )

    def test_rejects_empty_mixed_or_duplicate_batches(self):
        with self.assertRaisesRegex(TokenizationError, "vazio"):
            collate_tokenized_conversations(())

        cases = (
            dataclasses.replace(self.long, client_id="victim-02"),
            dataclasses.replace(self.long, round_id=1),
            dataclasses.replace(self.long, sample_index=self.short.sample_index),
        )
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(TokenizationError):
                collate_tokenized_conversations((self.short, changed))


if __name__ == "__main__":
    unittest.main()
