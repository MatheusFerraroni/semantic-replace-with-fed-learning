import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from federated_leakage.federated_grid_calibration import preflight_federated_memorization_grid
from federated_leakage.federated_grid_calibration import run_federated_memorization_grid
from federated_leakage.federated_grid_checkpointing import load_grid_checkpoint, save_grid_checkpoint
from federated_leakage.federated_grid_contracts import (
    FederatedGridArmResult,
    FederatedGridAuditResult,
    FederatedGridError,
    FederatedGridRoundResult,
    GridArmSpec,
    grid_arm_id,
    load_federated_grid_spec_from_config,
)
from federated_leakage.federated_grid_round import run_federated_grid_round
from federated_leakage.federated_grid_summary import build_federated_grid_combined_result
from federated_leakage.federated_round import prepare_auxiliary_training_input, prepare_victim_training_inputs
from federated_leakage.local_training import train_local_client_for_federated_grid
from federated_leakage.model_updates import capture_model_parameter_snapshot
from federated_leakage.synthetic_profiles import AuxiliaryRoundGenerator, VictimDatasetGenerator
from federated_leakage.training_contracts import LocalTrainingError, LocalTrainingResult

from tests.test_execution import _utility_result
import federated_leakage.utility_evaluation as utility_module
from tests.test_fedavg import _bundle as _fedavg_bundle
from tests.test_local_training import LlamaForCausalLM, _bundle, _samples, _tiny_parameter_contract


def _utility_for_seed(*, seed, scenario, model_hash):
    draft = dataclasses.replace(
        _utility_result(scenario=scenario, model_hash=model_hash),
        experiment_seed=seed,
    )
    return dataclasses.replace(
        draft,
        scientific_sha256=utility_module._hash(
            utility_module._scientific_payload(draft),
            b"utility-evaluation-result/v1",
        ),
    )


class FederatedGridConfigurationTests(unittest.TestCase):
    def test_loads_fixed_two_seed_grid_and_cross_seed_preflight(self):
        spec = load_federated_grid_spec_from_config(Path("configs/federated-memorization-grid-v2.yaml"))
        self.assertEqual(spec.experiment_seeds, (101, 361506353))
        self.assertEqual(len(spec.arms), 6)
        self.assertEqual(spec.expected_per_seed, (6, 120, 1_132_000, 283_000, 12_607, 3_500))
        self.assertEqual(spec.expected_combined, (12, 240, 2_264_000, 566_000, 25_214, 7_000))
        first, _, _ = preflight_federated_memorization_grid(spec, selected_seed=101)
        second, _, _ = preflight_federated_memorization_grid(spec, selected_seed=361506353)
        self.assertEqual(first.cross_seed_collision_preflight_sha256, second.cross_seed_collision_preflight_sha256)
        self.assertNotEqual(first.selected_victim_dataset_sha256, second.selected_victim_dataset_sha256)

    def test_rejects_drift_duplicate_yaml_and_v1_schema(self):
        source = Path("configs/federated-memorization-grid-v2.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main-v3.yaml").write_bytes(Path("configs/main-v3.yaml").read_bytes())
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            payload["victim_grid"]["repetition_multipliers"] = [4, 8, 32]
            target = root / source.name
            target.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(FederatedGridError):
                load_federated_grid_spec_from_config(target)
            target.write_text("schema_version: federated-memorization-grid/v2\nexperiment_seeds: [101]\nexperiment_seeds: [segredo-nao-expor]\n", encoding="utf-8")
            with self.assertRaises(FederatedGridError) as context:
                load_federated_grid_spec_from_config(target)
            self.assertNotIn("segredo-nao-expor", str(context.exception))


class FederatedGridTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from federated_leakage.training_contracts import load_local_training_spec_from_config
        cls.spec = load_local_training_spec_from_config(Path("configs/main-v3.yaml"))

    def test_explicit_lr_keeps_same_seed_and_reuses_one_optimizer(self):
        seed_hashes = []
        order_hashes = []
        for learning_rate in (3e-5, 1e-4):
            model = LlamaForCausalLM()
            bundle = _bundle(model)
            with _tiny_parameter_contract():
                snapshot = capture_model_parameter_snapshot(bundle)
                with mock.patch("federated_leakage.local_training._create_adamw_optimizer_for_learning_rate", wraps=__import__("federated_leakage.local_training", fromlist=["_create_adamw_optimizer_for_learning_rate"])._create_adamw_optimizer_for_learning_rate) as optimizer:
                    result = train_local_client_for_federated_grid(
                        _samples(), bundle, self.spec, seed=101, round_id=3,
                        initial_snapshot=snapshot, repetition_multiplier=4,
                        learning_rate=learning_rate,
                    )
            self.assertEqual(optimizer.call_count, 1)
            self.assertEqual(result.optimizer_steps, 100)
            self.assertEqual(model.calls, 400)
            seed_hashes.append(result.training_seed_sha256)
            order_hashes.append(result.sample_order_sha256)
        self.assertEqual(seed_hashes[0], seed_hashes[1])
        self.assertEqual(order_hashes[0], order_hashes[1])

    def test_rejects_unversioned_grid_values(self):
        bundle = _bundle(LlamaForCausalLM())
        with _tiny_parameter_contract():
            snapshot = capture_model_parameter_snapshot(bundle)
            with self.assertRaises(LocalTrainingError):
                train_local_client_for_federated_grid(_samples(), bundle, self.spec, seed=101, round_id=1, initial_snapshot=snapshot, repetition_multiplier=2, learning_rate=3e-5)
            with self.assertRaises(LocalTrainingError):
                train_local_client_for_federated_grid(_samples(), bundle, self.spec, seed=101, round_id=1, initial_snapshot=snapshot, repetition_multiplier=4, learning_rate=5e-5)


class FederatedGridRoundTests(unittest.TestCase):
    def test_only_victims_receive_grid_lr_and_repetitions(self):
        from federated_leakage.aggregation_contracts import load_fedavg_spec_from_config
        from federated_leakage.training_contracts import load_local_training_spec_from_config

        bundle = _fedavg_bundle()
        victims = VictimDatasetGenerator(101).generate()
        auxiliary = AuxiliaryRoundGenerator(101).generate(1, presentation="benign")
        victim_inputs = prepare_victim_training_inputs(victims, bundle)
        auxiliary_input = prepare_auxiliary_training_input(auxiliary, bundle)
        local_spec = load_local_training_spec_from_config(Path("configs/main-v3.yaml"))
        fedavg_spec = load_fedavg_spec_from_config(Path("configs/main-v3.yaml"))
        calls = []

        def train(samples, model_bundle, recipe, **kwargs):
            role = kwargs.get("role", "victim")
            multiplier = kwargs.get("repetition_multiplier", 1)
            calls.append((role, multiplier, kwargs.get("learning_rate")))
            with torch.no_grad():
                model_bundle.model.weight.add_(0.01)
            return LocalTrainingResult(
                client_id=samples[0].client_id,
                role=role,
                round_id=kwargs["round_id"],
                conversation_count=100,
                optimizer_steps=25 * multiplier,
                supervised_token_count=200 * multiplier,
                mean_loss=1.0,
                first_step_loss=1.0,
                last_step_loss=1.0,
                mean_gradient_norm=0.5,
                max_gradient_norm=0.5,
                sample_order_sha256=("a" if role == "victim" else "b") * 64,
                training_seed_sha256="c" * 64,
                model_provenance=model_bundle.provenance,
            )

        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1), mock.patch("federated_leakage.federated_grid_round.train_local_client_for_federated_grid", side_effect=train) as victims_train, mock.patch("federated_leakage.federated_grid_round.train_local_client", side_effect=train) as auxiliary_train:
            result = run_federated_grid_round(
                victim_inputs, auxiliary_input, bundle, local_spec, fedavg_spec,
                seed=101, round_id=1,
                arm=GridArmSpec(arm_id=grid_arm_id(100, 8), victim_learning_rate_millionths=100, victim_repetition_multiplier=8),
            )
        self.assertEqual(victims_train.call_count, 10)
        self.assertEqual(auxiliary_train.call_count, 1)
        self.assertTrue(all(value == ("victim", 8, 1e-4) for value in calls[:10]))
        self.assertEqual(calls[-1], ("auxiliary_benign", 1, None))
        self.assertEqual(result.optimizer_steps, 2_025)
        self.assertEqual(result.conversation_presentations, 8_100)

    def test_v2_checkpoint_round_trip_rejects_v1_identity(self):
        from federated_leakage.model_fingerprint import fingerprint_model_parameters

        bundle = _fedavg_bundle()
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
        result = FederatedGridRoundResult(
            experiment_seed=101,
            arm_id=grid_arm_id(30, 4),
            victim_learning_rate_millionths=30,
            auxiliary_learning_rate_millionths=30,
            victim_repetition_multiplier=4,
            round_id=1,
            conversation_presentations=4_100,
            optimizer_steps=1_025,
            victim_optimizer_steps=1_000,
            auxiliary_optimizer_steps=25,
            mean_client_loss=1.0,
            mean_victim_loss=1.0,
            auxiliary_loss=1.0,
            aggregate_delta_l2_norm=0.1,
            aggregate_delta_max_abs=0.01,
            victim_dataset_sha256="1" * 64,
            auxiliary_schedule_sha256="2" * 64,
            auxiliary_values_sha256="3" * 64,
            initial_model_sha256="4" * 64,
            aggregate_update_sha256="5" * 64,
            final_model_sha256=model_hash,
            client_order_sha256="6" * 64,
            weights_sha256="7" * 64,
            sample_order_schedule_sha256="8" * 64,
            training_seed_schedule_sha256="9" * 64,
            model_provenance=bundle.provenance,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            path = Path(directory) / "round-001"
            artifact = save_grid_checkpoint(path, bundle, result, grid_config_sha256="a" * 64)
            with torch.no_grad():
                bundle.model.weight.add_(1)
            loaded = load_grid_checkpoint(
                path,
                bundle,
                expected_seed=101,
                expected_arm_id=result.arm_id,
                expected_learning_rate_millionths=30,
                expected_multiplier=4,
                expected_round_id=1,
                expected_config_sha256="a" * 64,
            )
            self.assertEqual(loaded.round_result, result)
            self.assertEqual(loaded.artifact_sha256, artifact)
            with self.assertRaises(FederatedGridError):
                load_grid_checkpoint(
                    path,
                    bundle,
                    expected_seed=101,
                    expected_arm_id="victim-repetitions-004",
                    expected_learning_rate_millionths=30,
                    expected_multiplier=4,
                    expected_round_id=1,
                    expected_config_sha256="a" * 64,
                )


class FederatedGridOrchestrationTests(unittest.TestCase):
    def test_publishes_exact_per_seed_totals_with_six_independent_arms(self):
        from federated_leakage.execution_contracts import PILOT_BASELINE_MODEL_SHA256
        from federated_leakage.federated_grid_contracts import FederatedGridPreflightResult

        spec = load_federated_grid_spec_from_config(Path("configs/federated-memorization-grid-v2.yaml"))
        seed = 361506353
        victims = VictimDatasetGenerator(seed).generate()
        utility_dataset = SimpleNamespace(conversations=tuple())
        preflight = FederatedGridPreflightResult(
            selected_seed=seed,
            validated_seeds=(101, 361506353),
            victim_conversation_count=1_000,
            auxiliary_conversation_count=2_000,
            utility_conversation_count=500,
            cross_seed_collision_preflight_sha256="1" * 64,
            selected_victim_dataset_sha256=spec.hashes_for_seed(seed).victim_dataset_sha256,
            selected_benign_schedule_sha256=spec.hashes_for_seed(seed).benign_schedule_sha256,
            selected_utility_dataset_sha256=spec.hashes_for_seed(seed).utility_dataset_sha256,
        )
        bundle = _fedavg_bundle()
        baseline_utility = _utility_for_seed(seed=seed, scenario="B0", model_hash=PILOT_BASELINE_MODEL_SHA256)
        baseline_audit = FederatedGridAuditResult(
            experiment_seed=seed,
            arm_id=None,
            victim_learning_rate_millionths=0,
            victim_repetition_multiplier=0,
            extraction_result_sha256="2" * 64,
            target_count=200,
            distinctive_exact_pair_count=0,
            distinctive_exposed_entity_count=0,
            distinctive_exact_pairs_by_type=(("CPF", 0), ("RG", 0), ("PHONE", 0), ("EMAIL", 0), ("ADDRESS", 0)),
            distinctive_field_type_count=0,
            gate_passed=False,
            model_state_sha256=PILOT_BASELINE_MODEL_SHA256,
        )

        def arm_result(*, arm, **kwargs):
            index = spec.arms.index(arm) + 1
            model_hash = f"{index:x}" * 64
            passed = index >= 3
            breakdown = (("CPF", 30 if passed else 5), ("RG", 25 if passed else 0), ("PHONE", 0), ("EMAIL", 0), ("ADDRESS", 0))
            audit = FederatedGridAuditResult(
                experiment_seed=seed,
                arm_id=arm.arm_id,
                victim_learning_rate_millionths=arm.victim_learning_rate_millionths,
                victim_repetition_multiplier=arm.victim_repetition_multiplier,
                extraction_result_sha256=f"{index:x}" * 64,
                target_count=200,
                distinctive_exact_pair_count=sum(value for _, value in breakdown),
                distinctive_exposed_entity_count=30 if passed else 5,
                distinctive_exact_pairs_by_type=breakdown,
                distinctive_field_type_count=sum(value > 0 for _, value in breakdown),
                gate_passed=passed,
                model_state_sha256=model_hash,
            )
            utility = _utility_for_seed(seed=seed, scenario="F0", model_hash=model_hash)
            return FederatedGridArmResult(
                experiment_seed=seed,
                arm_id=arm.arm_id,
                victim_learning_rate_millionths=arm.victim_learning_rate_millionths,
                auxiliary_learning_rate_millionths=30,
                victim_repetition_multiplier=arm.victim_repetition_multiplier,
                completed_rounds=20,
                conversation_presentations=20 * (1_000 * arm.victim_repetition_multiplier + 100),
                optimizer_steps=20 * (250 * arm.victim_repetition_multiplier + 25),
                baseline_model_sha256=PILOT_BASELINE_MODEL_SHA256,
                final_model_sha256=model_hash,
                round_result_sha256="a" * 64,
                audit=audit,
                utility=utility,
                checkpoint_artifact_sha256="b" * 64,
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch("federated_leakage.federated_grid_calibration._validate_reference_v1", return_value=(SimpleNamespace(), {})), mock.patch("federated_leakage.federated_grid_calibration.preflight_federated_memorization_grid", return_value=(preflight, victims, utility_dataset)), mock.patch("federated_leakage.federated_grid_calibration.fingerprint_model_parameters", return_value=PILOT_BASELINE_MODEL_SHA256), mock.patch("federated_leakage.federated_grid_calibration.prepare_victim_training_inputs", return_value=SimpleNamespace()), mock.patch("federated_leakage.federated_grid_calibration.prepare_utility_evaluation", return_value=SimpleNamespace(dataset_sha256="9" * 64)), mock.patch("federated_leakage.federated_grid_calibration.prepare_trusted_evaluator", return_value=SimpleNamespace()), mock.patch("federated_leakage.federated_grid_calibration.preflight_extraction_audit"), mock.patch("federated_leakage.federated_grid_calibration._ensure_victims", return_value=victims), mock.patch("federated_leakage.federated_grid_calibration._run_or_reuse_audit", return_value=baseline_audit), mock.patch("federated_leakage.federated_grid_calibration._run_or_reuse_utility", return_value=baseline_utility), mock.patch("federated_leakage.federated_grid_calibration._run_arm", side_effect=arm_result), mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            result = run_federated_memorization_grid(
                spec,
                seed=seed,
                config_path=Path("configs/federated-memorization-grid-v2.yaml"),
                output_root=Path(directory),
                device="cpu",
                model_loader=lambda: bundle,
            )
        self.assertEqual(result.total_federated_rounds, 120)
        self.assertEqual(result.total_conversation_presentations, 1_132_000)
        self.assertEqual(result.total_optimizer_steps, 283_000)
        self.assertEqual(result.total_audit_generations, 12_607)
        self.assertEqual(result.total_utility_conversations, 3_500)
        self.assertEqual(result.first_successful_arm, spec.arms[2].arm_id)


class FederatedGridSummaryTests(unittest.TestCase):
    def test_classifies_without_averaging_away_seed_instability(self):
        spec = load_federated_grid_spec_from_config(Path("configs/federated-memorization-grid-v2.yaml"))

        def seed_result(seed, pass_pattern):
            arms = []
            for index, arm in enumerate(spec.arms):
                passed = pass_pattern[index]
                arms.append(SimpleNamespace(
                    arm_id=arm.arm_id,
                    experiment_seed=seed,
                    audit=SimpleNamespace(gate_passed=passed, distinctive_exact_pair_count=60 if passed else 20, distinctive_exposed_entity_count=30 if passed else 10),
                    utility=_utility_result(scenario="F0", model_hash=str((index + 1) % 10) * 64),
                ))
            return SimpleNamespace(
                experiment_seed=seed,
                baseline_gate_passed=False,
                result_sha256=("a" if seed == 101 else "b") * 64,
                arms=tuple(arms),
                total_federated_rounds=120,
                total_conversation_presentations=1_132_000,
                total_optimizer_steps=283_000,
                total_audit_generations=12_607,
                total_utility_conversations=3_500,
            )

        first = seed_result(101, (True, True, False, False, False, False))
        second = seed_result(361506353, (True, False, False, True, False, False))
        with tempfile.TemporaryDirectory() as directory, mock.patch("federated_leakage.federated_grid_summary.read_safe_json", return_value={}), mock.patch("federated_leakage.federated_grid_summary.grid_seed_result_from_payload", side_effect=(first, second)):
            result = build_federated_grid_combined_result(spec, output_root=Path(directory))
            self.assertEqual(result.classifications[0].classification, "robust")
            self.assertEqual(result.classifications[1].classification, "unstable")
            self.assertEqual(result.classifications[2].classification, "insufficient")
            self.assertEqual(result.first_robust_arm, spec.arms[0].arm_id)
            self.assertTrue((Path(directory) / "runs" / "federated-memorization-grid-v2" / "combined.json").is_file())


if __name__ == "__main__":
    unittest.main()
