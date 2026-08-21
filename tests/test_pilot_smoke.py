import os
import unittest
from pathlib import Path

from federated_leakage.execution_contracts import (
    PilotPreflightResult,
    build_pilot_run_identity,
    load_pilot_execution_spec_from_config,
)
from federated_leakage.pilot_execution import run_paired_pilot


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_PILOT_PREFLIGHT_SMOKE") == "1",
    "smoke real do piloto exige cache preparado e opt-in explícito",
)
class RealPilotPreflightSmokeTests(unittest.TestCase):
    def test_validates_real_model_tokenizer_and_all_synthetic_schedules_offline(self):
        config = Path("configs/main-v1.yaml")
        spec = load_pilot_execution_spec_from_config(config)
        identity = build_pilot_run_identity(spec, run_id="pilot-preflight-smoke")
        result = run_paired_pilot(
            spec,
            identity,
            config_path=config,
            cache_dir=Path("artifacts/huggingface"),
            device=os.environ.get("FEDERATED_PILOT_SMOKE_DEVICE", "cpu"),
            preflight_only=True,
        )
        self.assertIsInstance(result, PilotPreflightResult)
        self.assertTrue(result.tokenization_validated)
        self.assertEqual(result.victim_conversation_count, 1_000)
        self.assertEqual(result.auxiliary_conversation_count, 4_000)
        self.assertIsNotNone(result.model_state_sha256)


if __name__ == "__main__":
    unittest.main()
