import dataclasses
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from federated_leakage import utility_evaluation as utility_module
from federated_leakage.execution_storage import utility_result_from_safe_payload
from federated_leakage.refined_pilot_contracts import (
    SCENARIO_IDS,
    RefinedDefenseResult,
    RefinedPilotError,
    RefinedPilotResult,
    RefinedTrajectoryResult,
    default_run_id,
    safe_result_sha256,
)
from federated_leakage.summarize_refined_defense_pilot import build_refined_combined_result
from federated_leakage.semantic_pilot_storage import canonical_json_bytes
from tests.test_execution import _utility_result


def _signed(result, domain):
    payload = result.as_safe_dict()
    payload.pop("result_sha256")
    return dataclasses.replace(result, result_sha256=safe_result_sha256(payload, domain))


def _utility(seed=101, scenario="F0", ulps=0):
    draft = _utility_result(scenario=scenario)
    perplexity = draft.perplexity
    direction = math.inf if ulps > 0 else -math.inf
    for _ in range(abs(ulps)):
        perplexity = math.nextafter(perplexity, direction)
    draft = dataclasses.replace(draft, experiment_seed=seed, perplexity=perplexity)
    return dataclasses.replace(
        draft,
        scientific_sha256=utility_module._hash(
            utility_module._scientific_payload(draft), b"utility-evaluation-result/v1"
        ),
    )


def _pilot_result(seed):
    trajectories = []
    for scenario_id in SCENARIO_IDS:
        scenario = scenario_id.split("-", 1)[0]
        private = scenario in {"F2", "F3"}
        defended = scenario not in {"F0", "F1"}
        epsilon = float(scenario_id.rsplit("-", 1)[1]) if private else None
        trajectory = RefinedTrajectoryResult(
            scenario_id=scenario_id,
            seed=seed,
            completed_rounds=20,
            optimizer_steps=20_500,
            non_private_conversation_presentations=0 if private else 82_000,
            private_sampled_conversation_count=80_000 if private else None,
            target_epsilon=epsilon,
            max_realized_epsilon=(
                {3.0: 2.98777705562, 8.0: 7.96431428079}[epsilon] if private else None
            ),
            baseline_model_sha256="b" * 64,
            final_model_sha256="0" * 64,
            original_exact_pair_count=5 if defended else 100,
            original_complete_profile_count=0 if defended else 5,
            distinctive_exact_pair_count=3 if defended else 50,
            distinctive_exposed_entity_count=3 if defended else 25,
            distinctive_field_type_count=2,
            audit_result_sha256="a" * 64,
            utility=_utility(seed, scenario, ulps=int(seed == 101 and scenario == "F0")),
            result_sha256="0" * 64,
        )
        trajectories.append(_signed(trajectory, b"refined-defense-trajectory-result/v1"))
    defense = RefinedDefenseResult(
        seed=seed,
        baseline_gate_passed=False,
        vulnerability_gate_passed=True,
        epsilon_statuses=((3.0, "approved", 0.95, 0.95), (8.0, "approved", 0.95, 0.95)),
        substitution_status="approved",
        f4_reduction=0.95,
        f5_reduction=0.95,
        status="approved",
        result_sha256="0" * 64,
    )
    result = RefinedPilotResult(
        run_id=default_run_id(seed),
        seed=seed,
        baseline_model_sha256="b" * 64,
        trajectories=tuple(trajectories),
        defense=_signed(defense, b"refined-defense-result/v1"),
        total_federated_rounds=160,
        total_optimizer_steps=164_000,
        non_private_conversation_presentations=328_000,
        private_sampled_conversation_count=320_000,
        total_audit_generations=61_043,
        total_utility_conversations=4_500,
        result_sha256="0" * 64,
    )
    return _signed(result, b"refined-defense-pilot-result/v1")


def _write_sources(root):
    originals = {}
    for seed in (101, 361506353):
        path = root / "runs" / default_run_id(seed) / "completed.json"
        path.parent.mkdir(parents=True)
        raw = canonical_json_bytes(_pilot_result(seed).as_safe_dict())
        path.write_bytes(raw)
        originals[path] = raw
    return originals


class UtilityPayloadRegressionTests(unittest.TestCase):
    def test_accepts_two_ulps_in_both_directions_without_changing_values_or_hash(self):
        for ulps in (-2, -1, 0, 1, 2):
            with self.subTest(ulps=ulps):
                original = _utility(ulps=ulps)
                payload = json.loads(json.dumps(original.as_safe_dict()))
                restored = utility_result_from_safe_payload(payload)
                self.assertEqual(restored, original)
                self.assertEqual(restored.as_safe_dict(), payload)
                self.assertIs(utility_module.validate_utility_evaluation_result(restored), restored)

    def test_rejects_larger_differences_even_with_matching_hash(self):
        for ulps in (-3, 3, 100):
            with self.subTest(ulps=ulps), self.assertRaisesRegex(
                utility_module.UtilityEvaluationError, "diverge do contrato"
            ):
                utility_module.validate_utility_evaluation_result(_utility(ulps=ulps))

    def test_small_tampering_still_fails_exact_hash_validation(self):
        original = _utility()
        tampered = dataclasses.replace(original, perplexity=math.nextafter(original.perplexity, math.inf))
        with self.assertRaisesRegex(utility_module.UtilityEvaluationError, "diverge do contrato"):
            utility_module.validate_utility_evaluation_result(tampered)

    def test_nonfinite_perplexities_still_fail(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(utility_module.UtilityEvaluationError):
                utility_module.validate_utility_evaluation_result(
                    dataclasses.replace(_utility(), perplexity=value)
                )


class RefinedSummaryRegressionTests(unittest.TestCase):
    def test_real_readers_combine_both_seeds_and_preserve_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = _write_sources(root)
            result = build_refined_combined_result(root)
            self.assertEqual(result["dp_status_by_epsilon"], {"3.0": "approved", "8.0": "approved"})
            self.assertEqual(result["substitution_status"], "approved")
            self.assertEqual(result["overall_status"], "approved")
            self.assertEqual(result["total_federated_rounds"], 320)
            self.assertEqual(result["total_optimizer_steps"], 328_000)
            self.assertEqual(result["total_audit_generations"], 122_086)
            self.assertEqual(result["total_utility_conversations"], 9_000)
            self.assertEqual(result["total_private_sampled_conversation_count"], 640_000)
            self.assertEqual(result["source_result_sha256_by_seed"], {
                str(seed): _pilot_result(seed).result_sha256 for seed in (101, 361506353)
            })
            target = root / "runs" / "refined-defense-forum-tech-combined-v1" / "combined.json"
            combined_bytes = target.read_bytes()
            self.assertEqual(json.loads(combined_bytes), result)
            self.assertEqual(build_refined_combined_result(root), result)
            self.assertEqual(target.read_bytes(), combined_bytes)
            for path, raw in originals.items():
                self.assertEqual(path.read_bytes(), raw)

    def test_accepts_list_and_tuple_containers_at_both_levels(self):
        expected_hash = None
        for outer in (list, tuple):
            for inner in (list, tuple):
                with self.subTest(outer=outer, inner=inner), tempfile.TemporaryDirectory() as directory:
                    source = _pilot_result(101).as_safe_dict()
                    source["defense"]["epsilon_statuses"] = outer(
                        inner(entry) for entry in source["defense"]["epsilon_statuses"]
                    )
                    with mock.patch("federated_leakage.summarize_refined_defense_pilot._source", return_value=source):
                        result = build_refined_combined_result(Path(directory))
                    if expected_hash is None:
                        expected_hash = result["result_sha256"]
                    self.assertEqual(result["result_sha256"], expected_hash)

    def test_rejects_malformed_dp_sequences_without_publishing(self):
        entry = (3.0, "approved", 0.95, 0.95)
        invalid = (None, {}, (), (entry,), (entry, entry, entry),
                   ("private-marker", entry), (entry[:3], entry), (entry + (0,), entry),
                   (entry, entry))
        for entries in invalid:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = _pilot_result(101).as_safe_dict()
                source["defense"]["epsilon_statuses"] = entries
                with mock.patch("federated_leakage.summarize_refined_defense_pilot._source", return_value=source):
                    with self.assertRaises(RefinedPilotError) as context:
                        build_refined_combined_result(root)
                self.assertNotIn("private-marker", str(context.exception))
                self.assertFalse((root / "runs").exists())

    def test_tampered_payloads_fail_without_publishing_or_rewriting(self):
        for mutation in ("utility", "trajectory_hash", "result_hash", "extra_key"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                originals = _write_sources(root)
                path = next(iter(originals))
                payload = json.loads(originals[path])
                if mutation == "utility":
                    value = payload["trajectories"][0]["utility"]
                    value["perplexity"] = math.nextafter(value["perplexity"], math.inf)
                elif mutation == "trajectory_hash":
                    payload["trajectories"][0]["result_sha256"] = "f" * 64
                elif mutation == "result_hash":
                    payload["result_sha256"] = "f" * 64
                else:
                    payload["trajectories"][0]["private-marker"] = "private-marker"
                changed = canonical_json_bytes(payload)
                path.write_bytes(changed)
                with self.assertRaises(RefinedPilotError) as context:
                    build_refined_combined_result(root)
                self.assertNotIn("private-marker", str(context.exception))
                self.assertEqual(path.read_bytes(), changed)
                self.assertFalse((root / "runs" / "refined-defense-forum-tech-combined-v1").exists())

    def test_cli_consolidates_without_runtime_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = _write_sources(root)
            process = subprocess.run(
                [sys.executable, "-W", "error::RuntimeWarning", "-m",
                 "federated_leakage.summarize_refined_defense_pilot", "--output-root", str(root)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stderr, "")
            self.assertIn("status_geral: approved", process.stdout)
            self.assertNotIn(str(root), process.stdout)
            for path, raw in originals.items():
                self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
