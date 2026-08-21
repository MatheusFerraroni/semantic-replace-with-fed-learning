import os
import unittest
from pathlib import Path

from federated_leakage.audit_contracts import load_extraction_audit_spec_from_config
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.model_loading import DEFAULT_MODEL_CACHE, load_model_bundle
from federated_leakage.prepare_model import load_model_spec_from_config
from federated_leakage.synthetic_profiles import VictimDatasetGenerator
from federated_leakage.trusted_evaluator import (
    _generate_query,
    _query_schedule,
    preflight_extraction_audit,
    prepare_trusted_evaluator,
)


@unittest.skipUnless(
    os.environ.get("FEDERATED_RUN_AUDIT_SMOKE") == "1",
    "smoke real da auditoria exige cache preparado e opt-in explícito",
)
class RealExtractionAuditSmokeTests(unittest.TestCase):
    def test_generates_one_query_of_each_mode_without_mutating_model(self):
        model_spec = load_model_spec_from_config(Path("configs/main-v1.yaml"))
        audit_spec = load_extraction_audit_spec_from_config(
            Path("configs/main-v1.yaml")
        )
        bundle = load_model_bundle(
            model_spec,
            cache_dir=Path(
                os.environ.get("FEDERATED_MODEL_CACHE", DEFAULT_MODEL_CACHE)
            ),
            device=os.environ.get("FEDERATED_MODEL_DEVICE", "cpu"),
        )
        context = prepare_trusted_evaluator(
            VictimDatasetGenerator(11).generate(), 11
        )
        preflight_extraction_audit(audit_spec, context, bundle)
        queries = _query_schedule(audit_spec, context)
        selected = (
            queries[0],
            queries[100],
            queries[900],
        )
        initial_mode = bundle.model.training
        initial_hash = fingerprint_model_parameters(bundle)
        records = tuple(
            _generate_query(audit_spec, bundle, query) for query in selected
        )
        self.assertEqual(tuple(record.mode for record in records), (
            "primary",
            "field_specific",
            "untargeted",
        ))
        self.assertTrue(all(isinstance(record.generated_text, str) for record in records))
        self.assertEqual(fingerprint_model_parameters(bundle), initial_hash)
        self.assertEqual(bundle.model.training, initial_mode)


if __name__ == "__main__":
    unittest.main()
