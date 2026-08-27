import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from federated_leakage.fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from federated_leakage.federated_exposure_calibration import (
    _ensure_victim_artifacts,
    _validate_reference_pilot,
    preflight_federated_memorization_calibration,
    run_federated_memorization_calibration,
)
from federated_leakage.federated_exposure_checkpointing import (
    load_exposure_checkpoint,
    save_exposure_checkpoint,
)
from federated_leakage.federated_exposure_contracts import (
    ExposureArmSpec,
    FederatedExposureError,
    FederatedExposureArmResult,
    FederatedExposureAuditResult,
    FederatedExposureRoundResult,
    load_federated_exposure_spec_from_config,
)
from federated_leakage.federated_exposure_round import (
    run_federated_exposure_round,
)
from federated_leakage.federated_round import (
    prepare_auxiliary_training_input,
    prepare_victim_training_inputs,
)
from federated_leakage.local_training import (
    train_local_client,
    train_local_client_with_repetitions_for_calibration,
)
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.model_updates import capture_model_parameter_snapshot
from federated_leakage.training_contracts import ParameterDelta
from federated_leakage.training_contracts import LocalTrainingResult
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
)

from tests.test_fedavg import _bundle as _fedavg_bundle
from tests.test_fedavg import _local_result
from tests.test_local_training import (
    LlamaForCausalLM,
    _bundle,
    _samples,
    _tiny_parameter_contract,
)
from tests.test_execution import _utility_result


class FederatedExposureConfigurationTests(unittest.TestCase):
    def test_loads_strict_recipe_and_materializes_expected_data(self):
        spec = load_federated_exposure_spec_from_config(
            Path("configs/federated-memorization-calibration-v1.yaml")
        )
        self.assertEqual(
            tuple(arm.victim_repetition_multiplier for arm in spec.arms),
            (1, 2, 4),
        )
        self.assertEqual(spec.expected_total_conversation_presentations, 146_000)
        self.assertEqual(spec.expected_total_optimizer_steps, 36_500)
        result = preflight_federated_memorization_calibration(spec)
        self.assertEqual(result.victim_conversation_count, 1_000)
        self.assertEqual(result.auxiliary_round_count, 20)
        self.assertEqual(result.auxiliary_conversation_count, 2_000)
        self.assertEqual(result.utility_conversation_count, 500)
        self.assertEqual(
            result.victim_dataset_sha256,
            spec.expected_victim_dataset_sha256,
        )
        self.assertEqual(
            result.benign_schedule_sha256,
            spec.expected_benign_schedule_sha256,
        )

    def test_rejects_drift_duplicate_yaml_and_wrong_main_hash(self):
        source = Path("configs/federated-memorization-calibration-v1.yaml")
        main = Path("configs/main-v3.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main-v3.yaml").write_bytes(main.read_bytes())
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            payload["victim_repetition_multipliers"] = [1, 2, 8]
            changed = root / source.name
            changed.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(FederatedExposureError):
                load_federated_exposure_spec_from_config(changed)

            changed.write_text(
                "schema_version: federated-memorization-calibration/v1\n"
                "experiment_seed: 101\n"
                "experiment_seed: segredo-nao-expor\n",
                encoding="utf-8",
            )
            with self.assertRaises(FederatedExposureError) as context:
                load_federated_exposure_spec_from_config(changed)
            self.assertNotIn("segredo-nao-expor", str(context.exception))

            (root / "main-v3.yaml").write_text("alterado\n", encoding="utf-8")
            changed.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(FederatedExposureError, "hash"):
                load_federated_exposure_spec_from_config(changed)

    def test_rejects_symlinked_reference_and_does_not_recreate_completed_data(self):
        spec = load_federated_exposure_spec_from_config(
            Path("configs/federated-memorization-calibration-v1.yaml")
        )
        victims = VictimDatasetGenerator(101).generate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            runs.mkdir()
            real = root / "reference-real"
            real.mkdir()
            (runs / spec.reference_pilot_run_id).symlink_to(
                real,
                target_is_directory=True,
            )
            with self.assertRaises(FederatedExposureError):
                _validate_reference_pilot(root, spec)

            paths = SimpleNamespace(dataset_root=root / "datasets")
            paths.dataset_root.mkdir()
            with self.assertRaises(FederatedExposureError):
                _ensure_victim_artifacts(
                    paths,
                    spec,
                    victims,
                    allow_create=False,
                )
            self.assertFalse((paths.dataset_root / spec.dataset_id).exists())


class RepeatedVictimTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from federated_leakage.training_contracts import (
            load_local_training_spec_from_config,
        )

        cls.spec = load_local_training_spec_from_config(Path("configs/main-v3.yaml"))

    def test_reuses_one_adamw_and_preserves_seed_across_multipliers(self):
        model = LlamaForCausalLM()
        bundle = _bundle(model)
        with _tiny_parameter_contract():
            snapshot = capture_model_parameter_snapshot(bundle)
            repeated = train_local_client_with_repetitions_for_calibration(
                _samples(),
                bundle,
                self.spec,
                seed=101,
                round_id=3,
                initial_snapshot=snapshot,
                repetition_multiplier=2,
            )
        self.assertEqual(repeated.optimizer_steps, 50)
        self.assertEqual(repeated.supervised_token_count, 400)
        self.assertEqual(model.calls, 200)

        official_model = LlamaForCausalLM()
        official_bundle = _bundle(official_model)
        with _tiny_parameter_contract():
            official_snapshot = capture_model_parameter_snapshot(official_bundle)
            official = train_local_client(
                _samples(),
                official_bundle,
                self.spec,
                seed=101,
                role="victim",
                round_id=3,
                initial_snapshot=official_snapshot,
            )
        self.assertEqual(repeated.training_seed_sha256, official.training_seed_sha256)

    def test_failure_restores_snapshot_and_rejects_unversioned_multiplier(self):
        model = LlamaForCausalLM(fail_after=110)
        bundle = _bundle(model)
        with _tiny_parameter_contract():
            snapshot = capture_model_parameter_snapshot(bundle)
            before = snapshot.parameters[0].clone()
            with self.assertRaises(Exception) as context:
                train_local_client_with_repetitions_for_calibration(
                    _samples(),
                    bundle,
                    self.spec,
                    seed=101,
                    round_id=1,
                    initial_snapshot=snapshot,
                    repetition_multiplier=2,
                )
            self.assertTrue(torch.equal(bundle.model.weight.detach().cpu(), before))
        self.assertNotIn("segredo-nao-expor", str(context.exception))
        with self.assertRaises(Exception):
            train_local_client_with_repetitions_for_calibration(
                _samples(),
                bundle,
                self.spec,
                seed=101,
                round_id=1,
                initial_snapshot=snapshot,
                repetition_multiplier=8,
            )


class ExposureAccumulatorAndCheckpointTests(unittest.TestCase):
    def test_explicit_step_contract_changes_only_expected_clients(self):
        from federated_leakage.aggregation_contracts import (
            load_fedavg_spec_from_config,
        )

        spec = load_fedavg_spec_from_config(Path("configs/main-v3.yaml"))
        bundle = _fedavg_bundle()
        with mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ):
            snapshot = capture_model_parameter_snapshot(bundle)
        weights = resolve_fedavg_client_weights(spec, 1, "F0")
        expected = {
            **{weight.client_id: 50 for weight in weights[:-1]},
            "auxiliary": 25,
        }
        accumulator = FedAvgAccumulator(
            spec,
            weights,
            snapshot,
            bundle.provenance,
            round_id=1,
            expected_optimizer_steps_by_client=expected,
        )
        first = dataclasses.replace(
            _local_result(weights[0], bundle.provenance),
            optimizer_steps=50,
        )
        accumulator.add_client_update(
            first,
            iter(
                (
                    ParameterDelta(
                        name="weight",
                        tensor=torch.tensor(0.1, dtype=torch.float32),
                        numel=1,
                    ),
                )
            ),
        )
        self.assertEqual(accumulator.state, "accepting")

        default = FedAvgAccumulator(
            spec,
            weights,
            snapshot,
            bundle.provenance,
            round_id=1,
        )
        with self.assertRaises(Exception):
            default.add_client_update(first, iter(()))

    def test_checkpoint_round_trip_and_tamper_rejection(self):
        bundle = _fedavg_bundle()
        with mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ):
            model_hash = fingerprint_model_parameters(bundle)
        result = FederatedExposureRoundResult(
            arm_id="victim-repetitions-001",
            victim_repetition_multiplier=1,
            round_id=1,
            conversation_presentations=1_100,
            optimizer_steps=275,
            victim_optimizer_steps=250,
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round-001"
            config_hash = "a" * 64
            with mock.patch(
                "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
                1,
            ):
                artifact = save_exposure_checkpoint(
                    path,
                    bundle,
                    result,
                    calibration_config_sha256=config_hash,
                )
                with torch.no_grad():
                    bundle.model.weight.add_(1)
                loaded = load_exposure_checkpoint(
                    path,
                    bundle,
                    expected_arm_id=result.arm_id,
                    expected_multiplier=1,
                    expected_round_id=1,
                    expected_config_sha256=config_hash,
                )
            self.assertEqual(loaded.round_result, result)
            self.assertEqual(loaded.artifact_sha256, artifact)
            (path / "extra.bin").write_bytes(b"proibido")
            with self.assertRaises(FederatedExposureError):
                load_exposure_checkpoint(
                    path,
                    bundle,
                    expected_arm_id=result.arm_id,
                    expected_multiplier=1,
                    expected_round_id=1,
                    expected_config_sha256=config_hash,
                )


class FederatedExposureRoundTests(unittest.TestCase):
    def test_only_victims_repeat_and_auxiliary_keeps_twenty_five_steps(self):
        from federated_leakage.aggregation_contracts import (
            load_fedavg_spec_from_config,
        )
        from federated_leakage.training_contracts import (
            load_local_training_spec_from_config,
        )

        bundle = _fedavg_bundle()
        victims = VictimDatasetGenerator(101).generate()
        auxiliary = AuxiliaryRoundGenerator(101).generate(
            1,
            presentation="benign",
        )
        victim_inputs = prepare_victim_training_inputs(victims, bundle)
        auxiliary_input = prepare_auxiliary_training_input(auxiliary, bundle)
        local_spec = load_local_training_spec_from_config(Path("configs/main-v3.yaml"))
        fedavg_spec = load_fedavg_spec_from_config(Path("configs/main-v3.yaml"))

        calls = []

        def train(samples, model_bundle, spec, **kwargs):
            client_id = samples[0].client_id
            role = kwargs.get("role", "victim")
            multiplier = kwargs.get("repetition_multiplier", 1)
            calls.append((client_id, role, multiplier))
            with torch.no_grad():
                model_bundle.model.weight.add_(0.01)
            return LocalTrainingResult(
                client_id=client_id,
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

        with mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT",
            1,
        ), mock.patch(
            "federated_leakage.federated_exposure_round."
            "train_local_client_with_repetitions_for_calibration",
            side_effect=train,
        ) as repeated, mock.patch(
            "federated_leakage.federated_exposure_round.train_local_client",
            side_effect=train,
        ) as official:
            result = run_federated_exposure_round(
                victim_inputs,
                auxiliary_input,
                bundle,
                local_spec,
                fedavg_spec,
                seed=101,
                round_id=1,
                arm=ExposureArmSpec(
                    arm_id="victim-repetitions-002",
                    victim_repetition_multiplier=2,
                ),
            )
        self.assertEqual(repeated.call_count, 10)
        self.assertEqual(official.call_count, 1)
        self.assertEqual(result.victim_optimizer_steps, 500)
        self.assertEqual(result.auxiliary_optimizer_steps, 25)
        self.assertEqual(result.optimizer_steps, 525)
        self.assertEqual(result.conversation_presentations, 2_100)
        self.assertEqual(calls[-1], ("auxiliary", "auxiliary_benign", 1))


class FederatedExposureOrchestrationTests(unittest.TestCase):
    def test_runs_three_independent_arms_and_publishes_protocol_totals(self):
        from federated_leakage.execution_contracts import (
            PILOT_BASELINE_MODEL_SHA256,
        )

        spec = load_federated_exposure_spec_from_config(
            Path("configs/federated-memorization-calibration-v1.yaml")
        )
        preflight = preflight_federated_memorization_calibration(spec)
        bundle = _fedavg_bundle()
        baseline_utility = _utility_result(
            scenario="B0",
            model_hash=PILOT_BASELINE_MODEL_SHA256,
        )
        baseline_audit = FederatedExposureAuditResult(
            arm_id=None,
            victim_repetition_multiplier=0,
            extraction_result_sha256="a" * 64,
            target_count=200,
            distinctive_exact_pair_count=0,
            distinctive_exact_pair_denominator=1_000,
            distinctive_exposed_entity_count=0,
            distinctive_entity_denominator=200,
            calibrated_at_checkpoint=False,
            model_state_sha256=PILOT_BASELINE_MODEL_SHA256,
        )

        def arm_result(*, arm, **kwargs):
            multiplier = arm.victim_repetition_multiplier
            model_hash = str(multiplier) * 64
            audit = FederatedExposureAuditResult(
                arm_id=arm.arm_id,
                victim_repetition_multiplier=multiplier,
                extraction_result_sha256=("b" if multiplier == 1 else "c")
                * 64,
                target_count=200,
                distinctive_exact_pair_count=multiplier * 5,
                distinctive_exact_pair_denominator=1_000,
                distinctive_exposed_entity_count=multiplier + 3,
                distinctive_entity_denominator=200,
                calibrated_at_checkpoint=multiplier >= 2,
                model_state_sha256=model_hash,
            )
            return FederatedExposureArmResult(
                arm_id=arm.arm_id,
                victim_repetition_multiplier=multiplier,
                completed_rounds=20,
                conversation_presentations=20 * (1_000 * multiplier + 100),
                optimizer_steps=20 * (250 * multiplier + 25),
                baseline_model_sha256=PILOT_BASELINE_MODEL_SHA256,
                final_model_sha256=model_hash,
                round_result_sha256="d" * 64,
                audit=audit,
                utility=_utility_result(scenario="F0", model_hash=model_hash),
                checkpoint_artifact_sha256="e" * 64,
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "_validate_reference_pilot",
            return_value={},
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "_materialize_data_preflight",
            return_value=(preflight, ()),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "utility_dataset_sha256",
            return_value=spec.expected_utility_dataset_sha256,
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "load_extraction_audit_spec_from_config",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "load_local_training_spec_from_config",
            return_value=SimpleNamespace(learning_rate=3e-5),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "load_fedavg_spec_from_config",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "load_utility_evaluation_spec_from_config",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "fingerprint_model_parameters",
            return_value=PILOT_BASELINE_MODEL_SHA256,
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "prepare_victim_training_inputs",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "prepare_utility_evaluation",
            return_value=SimpleNamespace(
                dataset_sha256=spec.expected_utility_dataset_sha256
            ),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "prepare_trusted_evaluator",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "preflight_extraction_audit",
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "_ensure_victim_artifacts",
            side_effect=lambda paths, spec, victims: tuple(victims),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "capture_model_parameter_snapshot",
            return_value=object(),
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "restore_model_parameter_snapshot",
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "_run_or_reuse_audit",
            return_value=baseline_audit,
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration."
            "_run_or_reuse_utility",
            return_value=baseline_utility,
        ), mock.patch(
            "federated_leakage.federated_exposure_calibration._run_arm",
            side_effect=arm_result,
        ) as run_arm:
            result = run_federated_memorization_calibration(
                spec,
                config_path=Path(
                    "configs/federated-memorization-calibration-v1.yaml"
                ),
                output_root=Path(directory),
                device="cpu",
                fresh=True,
                model_loader=lambda: bundle,
            )
            completed_exists = (
                Path(directory) / "runs" / result.run_id / "completed.json"
            ).is_file()
        self.assertEqual(run_arm.call_count, 3)
        self.assertEqual(result.total_federated_rounds, 60)
        self.assertEqual(result.total_conversation_presentations, 146_000)
        self.assertEqual(result.total_optimizer_steps, 36_500)
        self.assertEqual(result.total_audit_generations, 7_204)
        self.assertEqual(result.total_utility_conversations, 2_000)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.first_successful_multiplier, 2)
        self.assertTrue(completed_exists)


if __name__ == "__main__":
    unittest.main()
