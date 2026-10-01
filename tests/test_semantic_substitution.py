import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch

from federated_leakage.audit_contracts import AuditGenerationRecord
from federated_leakage.semantic_audit import (
    _queries,
    _score,
    prepare_semantic_audit_targets,
)
from federated_leakage.semantic_pilot_contracts import (
    SemanticPilotPreflightResult,
    SemanticPilotError,
    load_semantic_pilot_spec_from_config,
    safe_result_sha256,
    semantic_combined_from_payload,
    semantic_trajectory_from_payload,
)
from federated_leakage.semantic_pilot import _gate, run_semantic_substitution_pilot
from federated_leakage.semantic_checkpointing import (
    load_semantic_checkpoint,
    save_semantic_checkpoint,
)
from federated_leakage.semantic_round import (
    run_semantic_federated_round,
    validate_paired_semantic_round_results,
)
from federated_leakage.semantic_substitution import (
    REPLACEMENT_SCHEDULE_VERSION,
    RotatingVictimSubstitutionGenerator,
    SemanticSubstitutionError,
    prepare_substituted_victim_training_inputs,
)
from federated_leakage.synthetic_profiles import (
    PROFILE_FIELD_ORDER,
    UNIQUE_FIELD_TYPES,
    VictimDatasetGenerator,
    profile_field_values,
)
from federated_leakage.trusted_evaluator import prepare_trusted_evaluator
from federated_leakage.audit_contracts import load_extraction_audit_spec_from_config
from federated_leakage.aggregation_contracts import load_fedavg_spec_from_config
from federated_leakage.federated_round import prepare_auxiliary_training_input
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.training_contracts import (
    LocalTrainingResult,
    load_local_training_spec_from_config,
)
from tests.test_fedavg import _bundle as _fedavg_bundle
from tests.test_execution import _utility_result


class SemanticReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.victims = VictimDatasetGenerator(101).generate()

    def test_replaces_all_nine_fields_rotates_and_is_deterministic(self):
        self.assertEqual(REPLACEMENT_SCHEDULE_VERSION, "rotating-profile/v3")
        first = RotatingVictimSubstitutionGenerator(101)
        second = RotatingVictimSubstitutionGenerator(101)
        round_one = first.generate_round(self.victims, 1)
        repeated = second.generate_round(self.victims, 1)
        round_two = first.generate_round(self.victims, 2)
        self.assertEqual(round_one, repeated)
        self.assertNotEqual(round_one.values_sha256, round_two.values_sha256)
        self.assertEqual(len(round_one.entries), 200)
        self.assertEqual(sum(len(item.dataset.conversations) for item in round_one.clients), 1_000)

        originals = {
            conversation.entity_id: {
                annotation.field_type: annotation.value
                for annotation in conversation.annotations
            }
            for dataset in self.victims
            for conversation in dataset.conversations
            if conversation.kind == "protected"
        }
        previous = {
            entry.source_entity_id: profile_field_values(entry.replacement_profile)
            for entry in round_one.entries
        }
        for entry in round_two.entries:
            current = profile_field_values(entry.replacement_profile)
            self.assertEqual(tuple(current), PROFILE_FIELD_ORDER)
            self.assertTrue(
                all(current[field] != originals[entry.source_entity_id][field] for field in PROFILE_FIELD_ORDER)
            )
            self.assertTrue(
                all(current[field] != previous[entry.source_entity_id][field] for field in PROFILE_FIELD_ORDER)
            )

    def test_preserves_general_conversations_and_technical_identity(self):
        round_data = RotatingVictimSubstitutionGenerator(101).generate_round(
            self.victims, 3
        )
        for original, substituted in zip(self.victims, round_data.datasets):
            self.assertEqual(original.client_id, substituted.client_id)
            for before, after in zip(original.conversations, substituted.conversations):
                self.assertEqual(before.sample_index, after.sample_index)
                self.assertEqual(before.entity_id, after.entity_id)
                self.assertEqual(before.template_id, after.template_id)
                if before.kind == "general":
                    self.assertEqual(before, after)
                else:
                    self.assertNotEqual(before.text, after.text)
                    self.assertEqual(
                        tuple(annotation.field_type for annotation in after.annotations),
                        PROFILE_FIELD_ORDER,
                    )

    def test_false_collisions_are_safe_metadata_not_validation_failures(self):
        round_data = RotatingVictimSubstitutionGenerator(101).generate_round(
            self.victims, 1
        )
        counts = dict(round_data.collision_counts)
        self.assertEqual(set(counts), set(PROFILE_FIELD_ORDER))
        self.assertTrue(all(value >= 0 for value in counts.values()))
        self.assertLessEqual(round_data.ambiguous_name_count, counts["PERSON_NAME"])
        safe = round_data.as_safe_dict()
        self.assertNotIn(round_data.entries[0].source_entity_id, repr(round_data))
        for field_type in UNIQUE_FIELD_TYPES:
            value = profile_field_values(round_data.entries[0].replacement_profile)[field_type]
            self.assertNotIn(value, repr(safe))

    def test_rejects_invalid_round_without_exposing_values(self):
        with self.assertRaises(SemanticSubstitutionError):
            RotatingVictimSubstitutionGenerator(101).generate_round(self.victims, 0)


class SemanticAuditScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.victims = VictimDatasetGenerator(101).generate()
        cls.context20 = prepare_trusted_evaluator(cls.victims, 101, target_count=20)
        cls.rounds = tuple(
            RotatingVictimSubstitutionGenerator(101).generate_round(cls.victims, value)
            for value in range(1, 21)
        )
        cls.spec = load_extraction_audit_spec_from_config(Path("configs/main-v4.yaml"))

    def test_current_and_historical_schedules_have_exact_counts(self):
        current = prepare_semantic_audit_targets(
            self.context20, self.rounds, view="current_alias"
        )
        historical = prepare_semantic_audit_targets(
            self.context20, self.rounds, view="historical_alias"
        )
        self.assertEqual(len(current), 20)
        self.assertEqual(len(historical), 380)
        self.assertEqual(len(_queries(self.spec, current)), 181)
        self.assertEqual(len(_queries(self.spec, historical)), 3_421)

    def test_scoring_deduplicates_alias_pairs_and_marks_ambiguity(self):
        targets = list(
            prepare_semantic_audit_targets(
                self.context20, self.rounds[:1], view="current_alias"
            )
        )
        targets[1] = dataclasses.replace(targets[1], query_name=targets[0].query_name)
        queries = _queries(self.spec, targets)
        records = []
        for query in queries:
            generated = ""
            if query.mode == "primary" and query.target_index in {0, 1}:
                target = targets[query.target_index]
                generated = " ".join(value for field, value in target.expected_fields if field != "PERSON_NAME")
            records.append(
                AuditGenerationRecord(
                    query_index=query.query_index,
                    mode=query.mode,
                    target_index=query.target_index,
                    target_entity_id=query.target_entity_id,
                    field_type=query.field_type,
                    max_new_tokens=query.max_new_tokens,
                    finish_reason="max_tokens",
                    prompt=query.prompt,
                    generated_text=generated,
                )
            )
        result = _score(
            self.spec,
            scenario="F4",
            round_id=1,
            view="current_alias",
            targets=targets,
            records=records,
            target_registry_sha256="a" * 64,
            generation_schedule_sha256="b" * 64,
            model_state_sha256="c" * 64,
            cross_replacement_rounds=self.rounds[:1],
        )
        self.assertEqual(result.ambiguous_alias_count, 1)
        self.assertEqual(result.ambiguous_target_count, 2)
        self.assertEqual(result.distinctive_exposed_entity_count, 0)
        self.assertGreater(result.distinctive_exact_pair_count, 0)


class SemanticPilotConfigurationTests(unittest.TestCase):
    def test_loads_fixed_recipe_and_preserves_historical_configs(self):
        spec = load_semantic_pilot_spec_from_config(
            Path("configs/semantic-substitution-pilot-v1.yaml")
        )
        self.assertEqual(spec.experiment_seeds, (101, 361506353))
        self.assertEqual(spec.scenario_order, ("F0", "F1", "F4", "F5"))
        self.assertEqual(spec.expected_per_seed, (4, 80, 328_000, 82_000, 40_083, 2_500))
        self.assertEqual(spec.victim_learning_rate_millionths, 100)
        self.assertEqual(spec.victim_repetition_multiplier, 4)

    def test_rejects_drift_and_duplicate_yaml(self):
        source = Path("configs/semantic-substitution-pilot-v1.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main-v4.yaml").write_bytes(Path("configs/main-v4.yaml").read_bytes())
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            payload["victim_repetition_multiplier"] = 8
            target = root / source.name
            target.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(SemanticPilotError):
                load_semantic_pilot_spec_from_config(target)
            target.write_text(
                "schema_version: semantic-substitution-pilot/v1\n"
                "experiment_seeds: [101]\n"
                "experiment_seeds: [segredo-nao-expor]\n",
                encoding="utf-8",
            )
            with self.assertRaises(SemanticPilotError) as context:
                load_semantic_pilot_spec_from_config(target)
            self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_gate_requires_both_comparators_and_ninety_percent_reduction(self):
        spec = load_semantic_pilot_spec_from_config(
            Path("configs/semantic-substitution-pilot-v1.yaml")
        )

        def trajectory(scenario, original_pairs, *, complete=0, pairs=100):
            return SimpleNamespace(
                scenario=scenario,
                original_audit_exact_pairs=original_pairs,
                original_audit_complete_profiles=complete,
                distinctive_exact_pair_count=pairs,
                distinctive_exposed_entity_count=50,
                distinctive_field_type_count=3,
            )

        trajectories = (
            trajectory("F0", 100),
            trajectory("F1", 100),
            trajectory("F4", 10),
            trajectory("F5", 10),
        )
        approved = _gate(
            spec,
            seed=101,
            baseline_gate_passed=False,
            trajectories=trajectories,
        )
        self.assertEqual(approved.status, "approved")
        self.assertEqual(approved.f4_original_exact_pair_reduction, 0.9)

        inconclusive = _gate(
            spec,
            seed=101,
            baseline_gate_passed=False,
            trajectories=(
                trajectories[0],
                trajectory("F1", 100, pairs=49),
                trajectories[2],
                trajectories[3],
            ),
        )
        self.assertEqual(inconclusive.status, "inconclusive")

        failed = _gate(
            spec,
            seed=101,
            baseline_gate_passed=True,
            trajectories=trajectories,
        )
        self.assertEqual(failed.status, "failed")

    def test_combined_contract_requires_both_fixed_seeds(self):
        unsigned = {
            "schema_version": "semantic-substitution-combined/v1",
            "source_result_sha256_by_seed": {
                "101": "a" * 64,
                "361506353": "b" * 64,
            },
            "status_by_seed": {"101": "approved", "361506353": "approved"},
            "combined_status": "approved",
            "require_both_seeds": True,
            "total_trajectories": 8,
            "total_federated_rounds": 160,
            "total_conversation_presentations": 656_000,
            "total_optimizer_steps": 164_000,
            "total_audit_generations": 80_166,
            "total_utility_conversations": 5_000,
        }
        result = semantic_combined_from_payload(
            {
                **unsigned,
                "result_sha256": safe_result_sha256(
                    unsigned, b"semantic-substitution-combined-result/v1"
                ),
            }
        )
        self.assertEqual(result.combined_status, "approved")
        with self.assertRaises(SemanticPilotError):
            semantic_combined_from_payload(
                {
                    **unsigned,
                    "source_result_sha256_by_seed": {"101": "a" * 64},
                    "result_sha256": "c" * 64,
                }
            )


class SemanticPilotOrchestrationTests(unittest.TestCase):
    def test_full_runner_publishes_four_independent_trajectories_and_totals(self):
        from federated_leakage.execution_contracts import PILOT_BASELINE_MODEL_SHA256
        from unittest import mock

        spec = load_semantic_pilot_spec_from_config(
            Path("configs/semantic-substitution-pilot-v1.yaml")
        )
        victims = VictimDatasetGenerator(101).generate()
        preflight = SemanticPilotPreflightResult(
            selected_seed=101,
            validated_seeds=(101, 361506353),
            victim_conversation_count=1_000,
            auxiliary_conversation_count=4_000,
            replacement_round_count=20,
            replacement_conversation_count=20_000,
            utility_conversation_count=500,
            replacement_schedule_sha256="1" * 64,
            replacement_values_sha256="2" * 64,
            grid_gate_sha256=spec.grid_combined_result_sha256,
        )

        def trajectory(*, scenario, **kwargs):
            model_hash = str({"F0": 3, "F1": 4, "F4": 5, "F5": 6}[scenario]) * 64
            utility = _utility_result(scenario=scenario, model_hash=model_hash)
            unsigned = {
                "schema_version": "semantic-substitution-trajectory/v1",
                "scenario": scenario,
                "experiment_seed": 101,
                "completed_rounds": 20,
                "conversation_presentations": 82_000,
                "optimizer_steps": 20_500,
                "baseline_model_sha256": PILOT_BASELINE_MODEL_SHA256,
                "final_model_sha256": model_hash,
                "round_result_sha256": "7" * 64,
                "original_audit_exact_pairs": 100 if scenario in {"F0", "F1"} else 5,
                "original_audit_complete_profiles": 0,
                "distinctive_exact_pair_count": 100 if scenario in {"F0", "F1"} else 5,
                "distinctive_exposed_entity_count": 50 if scenario in {"F0", "F1"} else 5,
                "distinctive_field_type_count": 3 if scenario in {"F0", "F1"} else 1,
                "original_audit_result_sha256": "8" * 64,
                "alias_audit_result_sha256": None if scenario in {"F0", "F1"} else "9" * 64,
                "historical_audit_result_sha256": None if scenario in {"F0", "F1"} else "a" * 64,
                "utility": utility.as_safe_dict(),
            }
            return semantic_trajectory_from_payload(
                {
                    **unsigned,
                    "result_sha256": safe_result_sha256(
                        unsigned, b"semantic-substitution-trajectory-result/v1"
                    ),
                }
            )

        baseline_audit = SimpleNamespace(as_safe_dict=lambda: {"safe": True})
        baseline_utility = _utility_result(
            scenario="B0", model_hash=PILOT_BASELINE_MODEL_SHA256
        )
        bundle_loader = lambda: _fedavg_bundle()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.semantic_pilot._grid_reference",
            return_value=(SimpleNamespace(), SimpleNamespace()),
        ), mock.patch(
            "federated_leakage.semantic_pilot.preflight_semantic_substitution_pilot",
            return_value=(preflight, victims, SimpleNamespace()),
        ), mock.patch(
            "federated_leakage.semantic_pilot.fingerprint_model_parameters",
            return_value=PILOT_BASELINE_MODEL_SHA256,
        ), mock.patch(
            "federated_leakage.semantic_pilot.prepare_victim_training_inputs",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.semantic_pilot.prepare_utility_evaluation",
            return_value=SimpleNamespace(dataset_sha256="b" * 64),
        ), mock.patch(
            "federated_leakage.semantic_pilot.prepare_trusted_evaluator",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.semantic_pilot.preflight_extraction_audit"
        ), mock.patch(
            "federated_leakage.semantic_pilot.prepare_substituted_victim_training_inputs",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "federated_leakage.semantic_pilot.RotatingVictimSubstitutionGenerator"
        ) as generator, mock.patch(
            "federated_leakage.semantic_pilot._standard_audit",
            return_value=(baseline_audit, 0, 0, (("CPF", 0),)),
        ), mock.patch(
            "federated_leakage.semantic_pilot._utility",
            return_value=baseline_utility,
        ), mock.patch(
            "federated_leakage.semantic_pilot._run_trajectory",
            side_effect=trajectory,
        ) as run_trajectory, mock.patch(
            "federated_leakage.semantic_pilot._validate_pairs"
        ), mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1
        ):
            generator.return_value.generate_round.return_value = SimpleNamespace(
                datasets=victims
            )
            result = run_semantic_substitution_pilot(
                spec,
                seed=101,
                config_path=Path("configs/semantic-substitution-pilot-v1.yaml"),
                output_root=Path(directory),
                device="cpu",
                fresh=True,
                model_loader=bundle_loader,
            )
        self.assertEqual(run_trajectory.call_count, 4)
        self.assertEqual(
            tuple(call.kwargs["scenario"] for call in run_trajectory.call_args_list),
            ("F0", "F1", "F4", "F5"),
        )
        self.assertEqual(result.gate.status, "approved")
        self.assertEqual(result.total_federated_rounds, 80)
        self.assertEqual(result.total_conversation_presentations, 328_000)
        self.assertEqual(result.total_optimizer_steps, 82_000)
        self.assertEqual(result.total_audit_generations, 40_083)
        self.assertEqual(result.total_utility_conversations, 2_500)


class SemanticFederatedRoundTests(unittest.TestCase):
    def test_only_victims_receive_repetitions_and_f4_f5_share_replacements(self):
        victims = VictimDatasetGenerator(101).generate()
        replacement = RotatingVictimSubstitutionGenerator(101).generate_round(
            victims, 1
        )
        local_spec = load_local_training_spec_from_config(Path("configs/main-v4.yaml"))
        fedavg_spec = load_fedavg_spec_from_config(Path("configs/main-v4.yaml"))
        results = []
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

        from unittest import mock

        for scenario, presentation in (("F4", "benign"), ("F5", "adversarial")):
            bundle = _fedavg_bundle()
            victim_inputs = prepare_substituted_victim_training_inputs(
                replacement, bundle
            )
            auxiliary = prepare_auxiliary_training_input(
                __import__(
                    "federated_leakage.synthetic_profiles",
                    fromlist=["AuxiliaryRoundGenerator"],
                ).AuxiliaryRoundGenerator(101).generate(
                    1, presentation=presentation
                ),
                bundle,
            )
            with mock.patch(
                "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1
            ), mock.patch(
                "federated_leakage.semantic_round.train_local_client_for_federated_grid",
                side_effect=train,
            ) as victim_train, mock.patch(
                "federated_leakage.semantic_round.train_local_client",
                side_effect=train,
            ) as auxiliary_train:
                result = run_semantic_federated_round(
                    victim_inputs,
                    auxiliary,
                    bundle,
                    local_spec,
                    fedavg_spec,
                    seed=101,
                    scenario=scenario,
                    round_id=1,
                    source_victim_dataset_sha256="d" * 64,
                )
            self.assertEqual(victim_train.call_count, 10)
            self.assertEqual(auxiliary_train.call_count, 1)
            self.assertEqual(result.optimizer_steps, 1_025)
            self.assertEqual(result.conversation_presentations, 4_100)
            results.append(result)
        self.assertTrue(all(value == ("victim", 4, 1e-4) for value in calls[:10]))
        self.assertEqual(calls[10], ("auxiliary_benign", 1, None))
        self.assertEqual(calls[21], ("auxiliary_adversarial", 1, None))
        validate_paired_semantic_round_results(
            results[0],
            results[1],
            expected_benign_initial_model_sha256=results[0].initial_model_sha256,
            expected_adversarial_initial_model_sha256=results[1].initial_model_sha256,
        )

    def test_semantic_checkpoint_round_trip_is_strict(self):
        victims = VictimDatasetGenerator(101).generate()
        replacement = RotatingVictimSubstitutionGenerator(101).generate_round(
            victims, 1
        )
        local_spec = load_local_training_spec_from_config(Path("configs/main-v4.yaml"))
        fedavg_spec = load_fedavg_spec_from_config(Path("configs/main-v4.yaml"))
        bundle = _fedavg_bundle()
        victim_inputs = prepare_substituted_victim_training_inputs(replacement, bundle)
        auxiliary = prepare_auxiliary_training_input(
            __import__(
                "federated_leakage.synthetic_profiles",
                fromlist=["AuxiliaryRoundGenerator"],
            ).AuxiliaryRoundGenerator(101).generate(1, presentation="benign"),
            bundle,
        )
        from unittest import mock

        def train(samples, model_bundle, recipe, **kwargs):
            role = kwargs.get("role", "victim")
            multiplier = kwargs.get("repetition_multiplier", 1)
            with torch.no_grad():
                model_bundle.model.weight.add_(0.01)
            return LocalTrainingResult(
                client_id=samples[0].client_id,
                role=role,
                round_id=1,
                conversation_count=100,
                optimizer_steps=25 * multiplier,
                supervised_token_count=200 * multiplier,
                mean_loss=1.0,
                first_step_loss=1.0,
                last_step_loss=1.0,
                mean_gradient_norm=1.0,
                max_gradient_norm=1.0,
                sample_order_sha256="a" * 64,
                training_seed_sha256="b" * 64,
                model_provenance=model_bundle.provenance,
            )

        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1), mock.patch(
            "federated_leakage.semantic_round.train_local_client_for_federated_grid",
            side_effect=train,
        ), mock.patch(
            "federated_leakage.semantic_round.train_local_client", side_effect=train
        ):
            result = run_semantic_federated_round(
                victim_inputs,
                auxiliary,
                bundle,
                local_spec,
                fedavg_spec,
                seed=101,
                scenario="F4",
                round_id=1,
                source_victim_dataset_sha256="d" * 64,
            )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1
        ):
            target = Path(directory) / "round-001"
            artifact = save_semantic_checkpoint(
                target, bundle, result, config_sha256="e" * 64
            )
            with torch.no_grad():
                bundle.model.weight.add_(1)
            loaded = load_semantic_checkpoint(
                target,
                bundle,
                expected_seed=101,
                expected_scenario="F4",
                expected_round_id=1,
                expected_config_sha256="e" * 64,
            )
            self.assertEqual(loaded.artifact_sha256, artifact)
            self.assertEqual(loaded.round_result, result)
            self.assertEqual(fingerprint_model_parameters(bundle), result.final_model_sha256)
            (target / "extra.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SemanticPilotError):
                load_semantic_checkpoint(
                    target,
                    bundle,
                    expected_seed=101,
                    expected_scenario="F4",
                    expected_round_id=1,
                    expected_config_sha256="e" * 64,
                )


if __name__ == "__main__":
    unittest.main()
