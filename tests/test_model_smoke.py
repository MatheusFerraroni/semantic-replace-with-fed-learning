import os
import unittest
from pathlib import Path

from federated_leakage.model_loading import (
    DEFAULT_MODEL_CACHE,
    EXPECTED_PARAMETER_COUNT,
    load_model_bundle,
)
from federated_leakage.prepare_model import load_model_spec_from_config


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_MODEL_SMOKE") == "1",
    "smoke real exige cache preparado e opt-in explícito",
)
class RealModelSmokeTests(unittest.TestCase):
    def test_loads_pinned_tucano_offline(self):
        spec = load_model_spec_from_config(Path("configs/main-v1.yaml"))
        bundle = load_model_bundle(
            spec,
            cache_dir=Path(os.environ.get("FEDERATED_MODEL_CACHE", DEFAULT_MODEL_CACHE)),
            device=os.environ.get("FEDERATED_MODEL_DEVICE", "cpu"),
        )
        self.assertEqual(bundle.provenance.parameter_count, EXPECTED_PARAMETER_COUNT)
        self.assertEqual(len(bundle.tokenizer), 49152)
        self.assertEqual(bundle.provenance.native_context_length, 4096)


if __name__ == "__main__":
    unittest.main()
