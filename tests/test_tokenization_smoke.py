import os
import unittest
from collections import Counter
from pathlib import Path

from federated_leakage.model_loading import DEFAULT_MODEL_CACHE, load_model_bundle
from federated_leakage.prepare_model import load_model_spec_from_config
from federated_leakage.synthetic_profiles import (
    AUXILIARY_ROUNDS,
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
)
from federated_leakage.tokenization import (
    collate_tokenized_conversations,
    tokenize_training_conversations,
)


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_TOKENIZATION_SMOKE") == "1",
    "smoke real de tokenização exige cache preparado e opt-in explícito",
)
class RealTokenizationSmokeTests(unittest.TestCase):
    def test_tokenizes_complete_seed_11_bundle_offline(self):
        spec = load_model_spec_from_config(Path("configs/main-v1.yaml"))
        bundle = load_model_bundle(
            spec,
            cache_dir=Path(os.environ.get("FEDERATED_MODEL_CACHE", DEFAULT_MODEL_CACHE)),
            device=os.environ.get("FEDERATED_MODEL_DEVICE", "cpu"),
        )

        scope_counts = Counter()
        total = 0
        maximum_length = 0
        first_victim_samples = None
        for dataset in VictimDatasetGenerator(11).generate():
            samples = tokenize_training_conversations(
                dataset.conversations,
                bundle,
            )
            if first_victim_samples is None:
                first_victim_samples = samples
            total += len(samples)
            scope_counts.update(sample.loss_scope for sample in samples)
            maximum_length = max(
                maximum_length,
                max(len(sample.input_ids) for sample in samples),
            )

        generator = AuxiliaryRoundGenerator(11, schedule_id="F0-F1")
        for round_id in range(1, AUXILIARY_ROUNDS + 1):
            for presentation in ("benign", "adversarial"):
                round_data = generator.generate(
                    round_id,
                    presentation=presentation,
                )
                samples = tokenize_training_conversations(
                    round_data.conversations,
                    bundle,
                )
                total += len(samples)
                scope_counts.update(sample.loss_scope for sample in samples)
                maximum_length = max(
                    maximum_length,
                    max(len(sample.input_ids) for sample in samples),
                )

        self.assertEqual(total, 5_000)
        self.assertEqual(
            scope_counts,
            Counter({"all_tokens": 3_400, "canonical_completion": 1_600}),
        )
        self.assertLessEqual(maximum_length, bundle.max_sequence_length)
        self.assertIsNotNone(first_victim_samples)
        batch = collate_tokenized_conversations(first_victim_samples[:4])
        self.assertEqual(tuple(batch.input_ids.shape)[0], 4)
        self.assertEqual(batch.client_id, "victim-01")


if __name__ == "__main__":
    unittest.main()
