import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from federated_leakage.aggregation_contracts import FedAvgRoundResult
from federated_leakage.audit_contracts import (
    ExtractionAuditResult,
    FieldAuditMetric,
    TARGET_FIELD_TYPES,
)
from federated_leakage.checkpointing import (
    build_federated_checkpoint_metadata,
    load_federated_checkpoint,
    save_federated_checkpoint,
)
from federated_leakage.execution_contracts import (
    LoadedFederatedCheckpoint,
    PilotExecutionError,
    PilotPreflightResult,
    build_pilot_run_identity,
    load_pilot_execution_spec_from_config,
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
from federated_leakage.pilot_execution import (
    run_paired_pilot,
    validate_paired_federated_trajectory_results,
)


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
    def __init__(self, value=0.25):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.bfloat16))


def _bundle(value=0.25):
    return LoadedModelBundle(
        model=LlamaForCausalLM(value),
        tokenizer=SimpleNamespace(),
        max_sequence_length=1_024,
        provenance=_provenance(),
    )


def _round_result(
    scenario="F0",
    round_id=1,
    final_hash="f" * 64,
    initial_hash=None,
):
    return FedAvgRoundResult(
        scenario=scenario,
        experiment_seed=101,
        round_id=round_id,
        auxiliary_weight_units=1,
        victim_client_count=10,
        auxiliary_client_count=1,
        conversation_count=1_100,
        optimizer_steps=275,
        supervised_token_count=10_000,
        mean_client_loss=1.0,
        mean_victim_loss=1.0,
        auxiliary_loss=1.0,
        mean_client_gradient_norm=0.5,
        max_client_gradient_norm=0.75,
        aggregate_delta_l2_norm=0.1,
        aggregate_delta_max_abs=0.01,
        client_order_sha256="1" * 64,
        weights_sha256="2" * 64,
        sample_order_schedule_sha256="3" * 64,
        training_seed_schedule_sha256="4" * 64,
        victim_dataset_sha256="5" * 64,
        auxiliary_schedule_sha256="6" * 64,
        auxiliary_values_sha256="7" * 64,
        auxiliary_presentation_sha256=("8" if scenario == "F0" else "9") * 64,
        auxiliary_batch_sha256=("a" if scenario == "F0" else "b") * 64,
        initial_model_sha256=(
            initial_hash
            if initial_hash is not None
            else f"{round_id - 1:064x}"[-64:]
        ),
        aggregate_update_sha256=("c" if scenario == "F0" else "d") * 64,
        final_model_sha256=final_hash,
        model_provenance=_provenance(),
    )


def _audit_result(
    *,
    scenario="F0",
    round_id=1,
    target_count=20,
    model_hash="f" * 64,
):
    primary = target_count
    fields = target_count * 8
    return ExtractionAuditResult(
        scenario=scenario,
        experiment_seed=101,
        round_id=round_id,
        auxiliary_weight_units=None if scenario == "B0" else 1,
        generation_count=primary + fields + 1,
        primary_generation_count=primary,
        field_specific_generation_count=fields,
        untargeted_generation_count=1,
        target_count=target_count,
        targeted_exact_pair_count=0,
        targeted_exact_pair_denominator=target_count * 8,
        targeted_partial_pair_count=0,
        targeted_complete_generation_count=0,
        targeted_ordered_complete_generation_count=0,
        targeted_exposed_profile_count=0,
        total_exact_reproductions=0,
        targeted_misassociation_count=0,
        targeted_known_association_count=0,
        targeted_unseen_formatted_count=0,
        field_metrics=tuple(
            FieldAuditMetric(
                field_type=field_type,
                exact_count=0,
                partial_count=0,
                denominator=target_count,
            )
            for field_type in TARGET_FIELD_TYPES
        ),
        field_specific_exact_pair_count=0,
        field_specific_partial_pair_count=0,
        untargeted_exact_hit_count=0,
        untargeted_unique_value_count=0,
        untargeted_victim_name_count=0,
        untargeted_exposed_profile_count=0,
        registry_sha256="e" * 64,
        target_schedule_sha256=f"{target_count:064x}"[-64:],
        prompt_catalog_sha256="1" * 64,
        generation_schedule_sha256=f"{target_count + 1:064x}"[-64:],
        generation_records_sha256="2" * 64,
        model_state_sha256=model_hash,
        model_provenance=_provenance(),
    )


class ExecutionContractTests(unittest.TestCase):
    def test_loads_fixed_pilot_and_rejects_drift_and_unsafe_id(self):
        spec = load_pilot_execution_spec_from_config(Path("configs/main-v2.yaml"))
        self.assertEqual(spec.experiment_seed, 101)
        self.assertEqual(spec.auxiliary_weight_units, 1)
        self.assertEqual(spec.expected_generation_count, 12_992)
        identity = build_pilot_run_identity(
            spec,
            calibration_result_sha256="c" * 64,
            calibration_manifest_sha256="d" * 64,
        )
        self.assertEqual(identity.run_id, "pilot-greedy-seed-101-k01-v2")
        self.assertTrue(identity.dataset_id.endswith("-v4"))
        with self.assertRaises(PilotExecutionError):
            build_pilot_run_identity(
                spec,
                run_id="../escape",
                calibration_result_sha256="c" * 64,
                calibration_manifest_sha256="d" * 64,
            )

        config = yaml.safe_load(Path("configs/main-v2.yaml").read_text())
        config["pilot"]["auxiliary_weight_units"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(PilotExecutionError, "auxiliary_weight_units"):
                load_pilot_execution_spec_from_config(path)

    def test_rejects_unknown_and_duplicate_pilot_configuration_keys(self):
        config = yaml.safe_load(Path("configs/main-v2.yaml").read_text())
        config["pilot"]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.yaml"
            unknown.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(PilotExecutionError, "desconhecidas"):
                load_pilot_execution_spec_from_config(unknown)

            duplicate = Path(directory) / "duplicate.yaml"
            duplicate.write_text(
                Path("configs/main-v2.yaml").read_text(encoding="utf-8")
                + "\npilot:\n  required_before_main_campaign: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(PilotExecutionError):
                load_pilot_execution_spec_from_config(duplicate)


class CheckpointTests(unittest.TestCase):
    def test_safetensors_round_trip_rng_and_tampering(self):
        bundle = _bundle(0.5)
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
            result = _round_result(final_hash=model_hash)
            audit = _audit_result(model_hash=model_hash)
            metadata = build_federated_checkpoint_metadata(
                round_result=result,
                audits=(audit,),
                config_sha256="3" * 64,
                baseline_model_sha256="4" * 64,
                baseline_audit_sha256="6" * 64,
                canonical_template_sha256="5" * 64,
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "round-001"
                torch.manual_seed(991)
                saved_rng = torch.random.get_rng_state().clone()
                saved = save_federated_checkpoint(path, bundle, metadata, result)
                self.assertTrue((path / "model.safetensors").is_file())
                bundle.model.weight.data.fill_(2.0)
                torch.manual_seed(77)
                loaded = load_federated_checkpoint(
                    path,
                    bundle,
                    expected_scenario="F0",
                    expected_round_id=1,
                    expected_config_sha256="3" * 64,
                    expected_victim_dataset_sha256="5" * 64,
                    expected_baseline_model_sha256="4" * 64,
                    expected_baseline_audit_sha256="6" * 64,
                )
                self.assertEqual(loaded.artifact_sha256, saved.artifact_sha256)
                self.assertEqual(fingerprint_model_parameters(bundle), model_hash)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), saved_rng))

                metadata_path = path / "metadata.json"
                original_metadata = metadata_path.read_bytes()
                payload = json.loads(metadata_path.read_text())
                payload["schema_version"] = "federated-checkpoint/v1"
                metadata_path.write_text(
                    json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        path,
                        bundle,
                        expected_scenario="F0",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )
                metadata_path.write_bytes(original_metadata)
                payload = json.loads(original_metadata)
                payload["checkpoint"]["experiment_seed"] = 999
                metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                before = bundle.model.weight.detach().clone()
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        path,
                        bundle,
                        expected_scenario="F0",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )
                self.assertTrue(torch.equal(bundle.model.weight, before))

    def test_rejects_overwrite_extra_file_symlink_and_wrong_scenario(self):
        bundle = _bundle(0.5)
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
            result = _round_result(final_hash=model_hash)
            metadata = build_federated_checkpoint_metadata(
                round_result=result,
                audits=(_audit_result(model_hash=model_hash),),
                config_sha256="3" * 64,
                baseline_model_sha256="4" * 64,
                baseline_audit_sha256="6" * 64,
                canonical_template_sha256="5" * 64,
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "round-001"
                with self.assertRaises(PilotExecutionError):
                    save_federated_checkpoint(
                        root / "nested" / ".." / "escaped",
                        bundle,
                        metadata,
                        result,
                    )
                save_federated_checkpoint(path, bundle, metadata, result)
                with self.assertRaises(FileExistsError):
                    save_federated_checkpoint(path, bundle, metadata, result)
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        path,
                        bundle,
                        expected_scenario="F1",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )

                extra = path / "extra.txt"
                extra.write_text("extra", encoding="utf-8")
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        path,
                        bundle,
                        expected_scenario="F0",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )
                extra.unlink()

                linked = root / "linked-checkpoint"
                linked.symlink_to(path, target_is_directory=True)
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        linked,
                        bundle,
                        expected_scenario="F0",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )
                with self.assertRaises(PilotExecutionError):
                    load_federated_checkpoint(
                        root / "nested" / ".." / "round-001",
                        bundle,
                        expected_scenario="F0",
                        expected_round_id=1,
                        expected_config_sha256="3" * 64,
                        expected_victim_dataset_sha256="5" * 64,
                        expected_baseline_model_sha256="4" * 64,
                        expected_baseline_audit_sha256="6" * 64,
                    )

                invalid_bundle = _bundle(0.5)
                invalid_bundle.model.weight.data = (
                    invalid_bundle.model.weight.data.float()
                )
                with self.assertRaises(PilotExecutionError):
                    save_federated_checkpoint(
                        root / "invalid-dtype",
                        invalid_bundle,
                        metadata,
                        result,
                    )


class PairedPilotOrchestrationTests(unittest.TestCase):
    def test_preflight_only_validates_model_without_publishing_outputs(self):
        spec = load_pilot_execution_spec_from_config(Path("configs/main-v2.yaml"))
        identity = build_pilot_run_identity(
            spec,
            run_id="preflight-test",
            calibration_result_sha256="c" * 64,
            calibration_manifest_sha256="d" * 64,
        )
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.pilot_execution.fingerprint_model_parameters",
            return_value="0" * 64,
        ), mock.patch(
            "federated_leakage.pilot_execution.prepare_victim_training_inputs",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.pilot_execution.prepare_trusted_evaluator",
            side_effect=lambda *args, target_count, **kwargs: SimpleNamespace(
                target_budget=SimpleNamespace(target_count=target_count)
            ),
        ), mock.patch(
            "federated_leakage.pilot_execution.preflight_extraction_audit"
        ) as audit_preflight, mock.patch(
            "federated_leakage.pilot_execution.load_completed_calibration_gate",
            return_value=SimpleNamespace(
                result_sha256="c" * 64,
                manifest_sha256="d" * 64,
                baseline_model_sha256="0" * 64,
                model_provenance=bundle.provenance,
            ),
        ):
            result = run_paired_pilot(
                spec,
                identity,
                config_path=Path("configs/main-v2.yaml"),
                output_root=Path(directory),
                device="cpu",
                preflight_only=True,
                model_loader=lambda: bundle,
            )
            self.assertIsInstance(result, PilotPreflightResult)
            self.assertTrue(result.tokenization_validated)
            self.assertEqual(audit_preflight.call_count, 4)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_runs_full_simulated_pilot_and_revalidates_completed_run(self):
        spec = load_pilot_execution_spec_from_config(Path("configs/main-v2.yaml"))
        identity = build_pilot_run_identity(
            spec,
            run_id="test-pilot",
            calibration_result_sha256="c" * 64,
            calibration_manifest_sha256="d" * 64,
        )
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return _bundle()

        baseline_hash = "0" * 64
        round_calls = []
        audit_calls = []
        audit_invocations = []
        audit_cache = {}
        fail_f1_round_two = True

        def fake_round(*args, scenario, round_id, **kwargs):
            round_calls.append((scenario, round_id))
            final = ("a" if scenario == "F0" else "b") + f"{round_id:063x}"
            initial = (
                baseline_hash
                if round_id == 1
                else ("a" if scenario == "F0" else "b")
                + f"{round_id - 1:063x}"
            )
            return _round_result(
                scenario,
                round_id,
                final[-64:],
                initial[-64:],
            )

        def fake_audit(spec_arg, context, checkpoint, bundle, **kwargs):
            key = (
                checkpoint.scenario,
                checkpoint.round_id,
                context.target_budget.target_count,
            )
            audit_invocations.append(key)
            if key not in audit_cache:
                audit_calls.append(key)
                audit_cache[key] = _audit_result(
                    scenario=checkpoint.scenario,
                    round_id=checkpoint.round_id,
                    target_count=context.target_budget.target_count,
                    model_hash=checkpoint.expected_model_sha256,
                )
            return audit_cache[key]

        def fake_save(path, bundle, metadata, round_result):
            if (
                fail_f1_round_two
                and round_result.scenario == "F1"
                and round_result.round_id == 2
            ):
                raise OSError("falha-injetada")
            path.mkdir(parents=True)
            return LoadedFederatedCheckpoint(
                metadata=metadata,
                round_result_payload=round_result.as_safe_dict(),
                artifact_sha256=f"{round_result.round_id:064x}"[-64:],
            )

        def fake_load(path, bundle, **kwargs):
            scenario = kwargs["expected_scenario"]
            round_id = kwargs["expected_round_id"]
            final = ("a" if scenario == "F0" else "b") + f"{round_id:063x}"
            initial = (
                baseline_hash
                if round_id == 1
                else ("a" if scenario == "F0" else "b")
                + f"{round_id - 1:063x}"
            )
            result = _round_result(
                scenario,
                round_id,
                final[-64:],
                initial[-64:],
            )
            targets = (1, 5, 20, 200) if round_id == 20 else (20,)
            audits = tuple(
                _audit_result(
                    scenario=scenario,
                    round_id=round_id,
                    target_count=count,
                    model_hash=result.final_model_sha256,
                )
                for count in targets
            )
            metadata = build_federated_checkpoint_metadata(
                round_result=result,
                audits=audits,
                config_sha256=spec.config_sha256,
                baseline_model_sha256=baseline_hash,
                baseline_audit_sha256=kwargs["expected_baseline_audit_sha256"],
                canonical_template_sha256="5" * 64,
            )
            return LoadedFederatedCheckpoint(
                metadata=metadata,
                round_result_payload=result.as_safe_dict(),
                artifact_sha256=f"{round_id:064x}"[-64:],
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.pilot_execution.fingerprint_model_parameters",
            return_value=baseline_hash,
        ), mock.patch(
            "federated_leakage.pilot_execution.prepare_victim_training_inputs",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.pilot_execution.prepare_auxiliary_training_input",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.pilot_execution.preflight_extraction_audit"
        ), mock.patch(
            "federated_leakage.pilot_execution.run_non_private_federated_round",
            side_effect=fake_round,
        ), mock.patch(
            "federated_leakage.pilot_execution.run_extraction_audit",
            side_effect=fake_audit,
        ), mock.patch(
            "federated_leakage.pilot_execution.save_federated_checkpoint",
            side_effect=fake_save,
        ), mock.patch(
            "federated_leakage.pilot_execution.load_federated_checkpoint",
            side_effect=fake_load,
        ), mock.patch(
            "federated_leakage.pilot_execution.capture_model_parameter_snapshot",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.pilot_execution.restore_model_parameter_snapshot"
        ), mock.patch(
            "federated_leakage.pilot_execution._revalidate_completed_audit",
            side_effect=lambda paths, spec, contexts, result: result,
        ), mock.patch(
            "federated_leakage.pilot_execution.load_completed_calibration_gate",
            return_value=SimpleNamespace(
                result_sha256="c" * 64,
                manifest_sha256="d" * 64,
                baseline_model_sha256=baseline_hash,
                model_provenance=_provenance(),
            ),
        ):
            with self.assertRaises(PilotExecutionError):
                run_paired_pilot(
                    spec,
                    identity,
                    config_path=Path("configs/main-v2.yaml"),
                    output_root=Path(directory),
                    device="cpu",
                    model_loader=loader,
                )
            self.assertFalse(
                (Path(directory) / "runs/test-pilot/completed.json").exists()
            )
            f0_state = json.loads(
                (
                    Path(directory)
                    / "runs/test-pilot/trajectories/F0-k01/state.json"
                ).read_text(encoding="utf-8")
            )
            f1_state = json.loads(
                (
                    Path(directory)
                    / "runs/test-pilot/trajectories/F1-k01/state.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(f0_state["completed_round"], 20)
            self.assertEqual(f1_state["completed_round"], 1)
            self.assertIn(("F1", 2, 20), audit_cache)

            fail_f1_round_two = False

            result = run_paired_pilot(
                spec,
                identity,
                config_path=Path("configs/main-v2.yaml"),
                output_root=Path(directory),
                device="cpu",
                model_loader=loader,
            )
            self.assertEqual(result.total_federated_rounds, 40)
            self.assertEqual(result.total_conversation_count, 44_000)
            self.assertEqual(result.total_optimizer_steps, 11_000)
            self.assertEqual(result.total_audit_generations, 12_992)
            self.assertEqual(len(round_calls), 41)
            self.assertEqual(round_calls.count(("F1", 2)), 2)
            self.assertEqual(round_calls.count(("F0", 1)), 1)
            self.assertEqual(round_calls.count(("F0", 20)), 1)
            self.assertEqual(round_calls.count(("F1", 1)), 1)
            self.assertEqual(len(audit_calls), 50)
            self.assertEqual(audit_calls.count(("F1", 2, 20)), 1)
            self.assertEqual(audit_invocations.count(("F1", 2, 20)), 2)
            validate_paired_federated_trajectory_results(*result.trajectories)
            self.assertNotEqual(
                result.trajectories[0].round_results[1].initial_model_sha256,
                result.trajectories[1].round_results[1].initial_model_sha256,
            )
            with self.assertRaises(PilotExecutionError):
                validate_paired_federated_trajectory_results(
                    result.trajectories[0],
                    dataclasses.replace(
                        result.trajectories[1],
                        baseline_model_sha256="f" * 64,
                    ),
                )
            invalid_f1_rounds = list(result.trajectories[1].round_results)
            invalid_f1_rounds[1] = dataclasses.replace(
                invalid_f1_rounds[1],
                initial_model_sha256="e" * 64,
            )
            with self.assertRaisesRegex(
                PilotExecutionError,
                "rodada 2",
            ):
                validate_paired_federated_trajectory_results(
                    result.trajectories[0],
                    dataclasses.replace(
                        result.trajectories[1],
                        round_results=tuple(invalid_f1_rounds),
                    ),
                )
            invalid_f0_rounds = list(result.trajectories[0].round_results)
            invalid_f0_rounds[0] = dataclasses.replace(
                invalid_f0_rounds[0],
                initial_model_sha256="d" * 64,
            )
            with self.assertRaisesRegex(
                PilotExecutionError,
                "rodada 1",
            ):
                validate_paired_federated_trajectory_results(
                    dataclasses.replace(
                        result.trajectories[0],
                        round_results=tuple(invalid_f0_rounds),
                    ),
                    result.trajectories[1],
                )
            self.assertTrue(
                (Path(directory) / "runs/test-pilot/completed.json").is_file()
            )
            self.assertEqual(
                len(
                    (
                        Path(directory)
                        / "runs/test-pilot/trajectories/F0-k01/training_metrics.jsonl"
                    ).read_text().splitlines()
                ),
                20,
            )
            safe_metadata = "\n".join(
                path.read_text(encoding="utf-8")
                for path in (Path(directory) / "runs/test-pilot").rglob("*")
                if path.is_file()
            )
            for forbidden in (
                "USUÁRIO:",
                "PERSON_NAME",
                '"annotations"',
                '"entity_id"',
                '"input_ids"',
                '"labels"',
            ):
                self.assertNotIn(forbidden, safe_metadata)

            repeated = run_paired_pilot(
                spec,
                identity,
                config_path=Path("configs/main-v2.yaml"),
                output_root=Path(directory),
                device="cpu",
                model_loader=loader,
            )
            self.assertEqual(repeated.as_safe_dict(), result.as_safe_dict())
            self.assertEqual(len(round_calls), 41)


if __name__ == "__main__":
    unittest.main()
