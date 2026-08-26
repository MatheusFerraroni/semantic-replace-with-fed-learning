import dataclasses
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from federated_leakage.audit_contracts import (
    AuditCheckpoint,
    AuditGenerationRecord,
    ExtractionAuditError,
    load_extraction_audit_spec_from_config,
    parse_extraction_audit_spec,
)
from federated_leakage.audit_storage import prepare_audit_journal
from federated_leakage.model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    LoadedModelBundle,
    ModelProvenance,
)
from federated_leakage.model_fingerprint import fingerprint_model_parameters
from federated_leakage.synthetic_profiles import VictimDatasetGenerator
from federated_leakage.synthetic_profiles.rendering import (
    CANONICAL_COMPLETION_TEMPLATE,
    CANONICAL_PREFIX_TEMPLATE,
)
from federated_leakage.trusted_evaluator import (
    _generate_query,
    _query_schedule,
    prepare_trusted_evaluator,
    run_extraction_audit,
    score_extraction_audit,
    validate_paired_extraction_audit_results,
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


class CompactTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        spans = tuple(re.finditer(r"\s+|\S+", text))
        ids = [10 + (sum(ord(character) for character in span.group()) % 1000) for span in spans]
        result = {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [span.span() for span in spans]
        return result

    def decode(self, values, **kwargs):
        return "" if not values else "texto-gerado"


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.bfloat16))
        self.calls = []
        self.generation_config = SimpleNamespace(
            temperature=0.1,
            top_p=0.95,
            top_k=50,
        )
        self.inherited_sampling_states = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.inherited_sampling_states.append(
            (
                self.generation_config.temperature,
                self.generation_config.top_p,
                self.generation_config.top_k,
            )
        )
        return kwargs["input_ids"].clone()


def _bundle():
    return LoadedModelBundle(
        model=LlamaForCausalLM(),
        tokenizer=CompactTokenizer(),
        max_sequence_length=1_024,
        provenance=_provenance(),
    )


def _context(seed=11, target_count=20):
    return prepare_trusted_evaluator(
        VictimDatasetGenerator(seed).generate(),
        seed,
        target_count=target_count,
    )


def _checkpoint(context, bundle, *, scenario="B0", round_id=0, k=None):
    with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
        state_hash = fingerprint_model_parameters(bundle)
    return AuditCheckpoint(
        scenario=scenario,
        experiment_seed=context.experiment_seed,
        round_id=round_id,
        auxiliary_weight_units=k,
        expected_model_sha256=state_hash,
        model_provenance=bundle.provenance,
    )


def _blank_records(spec, context):
    return tuple(
        AuditGenerationRecord(
            query_index=query.query_index,
            mode=query.mode,
            target_index=query.target_index,
            target_entity_id=query.target_entity_id,
            field_type=query.field_type,
            max_new_tokens=query.max_new_tokens,
            finish_reason="max_tokens",
            prompt=query.prompt,
            generated_text="",
        )
        for query in _query_schedule(spec, context)
    )


class AuditConfigurationAndContextTests(unittest.TestCase):
    def test_loads_strict_recipe_and_selects_two_targets_per_client(self):
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        self.assertEqual(spec.expected_generation_count, 181)
        first = _context(11)
        second = _context(11)
        changed = _context(22)
        self.assertEqual(first, second)
        self.assertNotEqual(first.target_schedule_sha256, changed.target_schedule_sha256)
        self.assertEqual(
            tuple(record.client_id for record in first.targets),
            tuple(f"victim-{index:02d}" for index in range(1, 11) for _ in range(2)),
        )
        self.assertNotIn(first.targets[0].value("PERSON_NAME"), repr(first))

    def test_target_budgets_are_nested_balanced_and_have_exact_counts(self):
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        contexts = tuple(_context(101, count) for count in (1, 5, 20, 200))
        target_sets = tuple(
            {record.entity_id for record in context.targets} for context in contexts
        )
        self.assertTrue(
            all(first < second for first, second in zip(target_sets, target_sets[1:]))
        )
        self.assertEqual(
            tuple(len(_query_schedule(spec, context)) for context in contexts),
            (10, 46, 181, 1_801),
        )
        twenty = contexts[2]
        self.assertEqual(
            tuple(
                sum(record.client_id == f"victim-{index:02d}" for record in twenty.targets)
                for index in range(1, 11)
            ),
            (2,) * 10,
        )
        shared = contexts[0].targets[0]
        schedules = tuple(_query_schedule(spec, context) for context in contexts)
        shared_prompts = []
        for context, schedule in zip(contexts, schedules):
            index = context.targets.index(shared)
            shared_prompts.append(
                tuple(
                    query.prompt
                    for query in schedule
                    if query.mode == "primary" and query.target_index == index
                )
            )
        self.assertEqual(len(set(shared_prompts)), 1)

    def test_rejects_recipe_drift_duplicate_yaml_and_invalid_dataset(self):
        with self.assertRaisesRegex(ExtractionAuditError, "legada"):
            load_extraction_audit_spec_from_config(Path("configs/main-v1.yaml"))
        config = yaml.safe_load(Path("configs/main-v2.yaml").read_text())
        self.assertNotIn("temperature", config["audit"]["generation"])
        self.assertNotIn("top_p", config["audit"]["generation"])
        self.assertNotIn("top_k", config["audit"]["generation"])
        config["audit"]["generation"]["top_k"] = 49
        with self.assertRaisesRegex(ExtractionAuditError, "chaves"):
            parse_extraction_audit_spec(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                "audit:\n  owner: trusted_evaluator\n  owner: segredo-nao-expor\n",
                encoding="utf-8",
            )
            with self.assertRaises(ExtractionAuditError) as caught:
                load_extraction_audit_spec_from_config(path)
        self.assertNotIn("segredo-nao-expor", str(caught.exception))
        with self.assertRaises(ExtractionAuditError):
            prepare_trusted_evaluator(VictimDatasetGenerator(11).generate()[:9], 11)


class AuditScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        cls.context = _context()
        cls.bundle = _bundle()
        cls.checkpoint = _checkpoint(cls.context, cls.bundle)

    def test_scores_exact_ordered_wrong_unseen_field_and_untargeted(self):
        records = list(_blank_records(self.spec, self.context))
        target = self.context.targets[0]
        values = dict(target.field_values)
        records[0] = dataclasses.replace(
            records[0], generated_text=CANONICAL_COMPLETION_TEMPLATE.format(**values)
        )
        other_cpf = self.context.targets[0].value("CPF")
        records[1] = dataclasses.replace(
            records[1],
            generated_text=f" CPF {other_cpf}; CPF 123.456.789-01",
        )
        field_start = 20
        records[field_start] = dataclasses.replace(
            records[field_start], generated_text=" " + values["BIRTH_DATE"]
        )
        untargeted_start = 180
        full = CANONICAL_PREFIX_TEMPLATE.format(**values) + CANONICAL_COMPLETION_TEMPLATE.format(**values)
        records[untargeted_start] = dataclasses.replace(
            records[untargeted_start], generated_text=full
        )

        result = score_extraction_audit(
            self.spec, self.context, self.checkpoint, records
        )
        self.assertEqual(result.generation_count, 181)
        self.assertEqual(result.targeted_exact_pair_count, 8)
        self.assertEqual(result.targeted_complete_generation_count, 1)
        self.assertEqual(result.targeted_ordered_complete_generation_count, 1)
        self.assertEqual(result.targeted_exposed_profile_count, 1)
        self.assertEqual(result.total_exact_reproductions, 8)
        self.assertEqual(result.targeted_misassociation_count, 1)
        self.assertGreaterEqual(result.targeted_unseen_formatted_count, 1)
        self.assertEqual(result.field_specific_exact_pair_count, 1)
        self.assertGreaterEqual(result.untargeted_victim_name_count, 1)
        self.assertGreaterEqual(result.untargeted_exposed_profile_count, 1)
        safe = json.dumps(result.as_safe_dict(), ensure_ascii=False)
        self.assertNotIn(values["PERSON_NAME"], safe)
        self.assertNotIn(values["CPF"], safe)

    def test_pair_validation_ignores_model_hash_but_rejects_schedule(self):
        result = score_extraction_audit(
            self.spec, self.context, self.checkpoint, _blank_records(self.spec, self.context)
        )
        benign = dataclasses.replace(
            result, scenario="F0", round_id=1, auxiliary_weight_units=3
        )
        adversarial = dataclasses.replace(
            result,
            scenario="F1",
            round_id=1,
            auxiliary_weight_units=3,
            model_state_sha256="f" * 64,
        )
        validate_paired_extraction_audit_results(benign, adversarial)
        with self.assertRaises(ExtractionAuditError):
            validate_paired_extraction_audit_results(
                benign,
                dataclasses.replace(adversarial, generation_schedule_sha256="0" * 64),
            )

    def test_partial_similarity_does_not_become_an_exact_hit(self):
        records = list(_blank_records(self.spec, self.context))
        expected = self.context.targets[0].value("APPOINTMENT_TIME")
        hour = int(expected[:2])
        replacement_hour = hour - 1 if hour == 18 else hour + 1
        candidate = f"{replacement_hour:02d}{expected[2:]}"
        appointment_time_query = 20 + 7
        records[appointment_time_query] = dataclasses.replace(
            records[appointment_time_query], generated_text=" " + candidate
        )
        result = score_extraction_audit(
            self.spec, self.context, self.checkpoint, records
        )
        self.assertEqual(result.field_specific_exact_pair_count, 0)
        self.assertEqual(result.field_specific_partial_pair_count, 1)


class AuditStorageAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        self.context = _context()
        self.bundle = _bundle()
        self.checkpoint = _checkpoint(self.context, self.bundle)

    def test_journal_resumes_recovers_terminal_partial_and_rejects_tampering(self):
        query = _query_schedule(self.spec, self.context)[0]
        record = AuditGenerationRecord(
            query_index=0,
            mode=query.mode,
            target_index=query.target_index,
            target_entity_id=query.target_entity_id,
            field_type=query.field_type,
            max_new_tokens=query.max_new_tokens,
            finish_reason="max_tokens",
            prompt=query.prompt,
            generated_text="parcial",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = prepare_audit_journal(
                output_root=root,
                run_id="run-11",
                spec=self.spec,
                context=self.context,
                checkpoint=self.checkpoint,
                generation_schedule_sha256="a" * 64,
                resume=True,
            )
            journal.append(record)
            path = root / "run-11/evaluator/private/audits/B0-targets-020-round-000.incomplete/extraction_results.jsonl"
            with path.open("ab") as output:
                output.write(b'{"partial":')
            resumed = prepare_audit_journal(
                output_root=root,
                run_id="run-11",
                spec=self.spec,
                context=self.context,
                checkpoint=self.checkpoint,
                generation_schedule_sha256="a" * 64,
                resume=True,
            )
            self.assertEqual(resumed.records, (record,))
            metadata = path.with_name("metadata.json")
            payload = json.loads(metadata.read_text())
            payload["expected_generation_count"] = 180
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ExtractionAuditError):
                prepare_audit_journal(
                    output_root=root,
                    run_id="run-11",
                    spec=self.spec,
                    context=self.context,
                    checkpoint=self.checkpoint,
                    generation_schedule_sha256="a" * 64,
                    resume=True,
                )

    def test_large_target_budget_journal_uses_its_own_expected_count(self):
        context = _context(target_count=200)
        checkpoint = _checkpoint(context, self.bundle)
        queries = _query_schedule(self.spec, context)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.audit_storage.os.fsync"
        ):
            journal = prepare_audit_journal(
                output_root=Path(directory),
                run_id="run-200",
                spec=self.spec,
                context=context,
                checkpoint=checkpoint,
                generation_schedule_sha256="a" * 64,
                resume=True,
            )
            for query in queries[:182]:
                journal.append(
                    AuditGenerationRecord(
                        query_index=query.query_index,
                        mode=query.mode,
                        target_index=query.target_index,
                        target_entity_id=query.target_entity_id,
                        field_type=query.field_type,
                        max_new_tokens=query.max_new_tokens,
                        finish_reason="max_tokens",
                        prompt=query.prompt,
                        generated_text="",
                    )
                )
            resumed = prepare_audit_journal(
                output_root=Path(directory),
                run_id="run-200",
                spec=self.spec,
                context=context,
                checkpoint=checkpoint,
                generation_schedule_sha256="a" * 64,
                resume=True,
            )
            self.assertEqual(len(resumed.records), 182)

    def test_runs_all_greedy_queries_and_publishes_safe_summary(self):
        rng_before = torch.random.get_rng_state().clone()
        self.bundle.model.train()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1
        ), mock.patch("federated_leakage.audit_storage.os.fsync"):
            result = run_extraction_audit(
                self.spec,
                self.context,
                self.checkpoint,
                self.bundle,
                output_root=Path(directory),
                run_id="run-11",
            )
            summary = Path(directory) / "run-11/evaluator/summaries/B0-targets-020-round-000.json"
            private = Path(directory) / "run-11/evaluator/private/audits/B0-targets-020-round-000/extraction_results.jsonl"
            self.assertTrue(summary.is_file())
            self.assertTrue(private.is_file())
            self.assertEqual(len(private.read_text().splitlines()), 181)
            self.assertEqual(len(self.bundle.model.calls), 181)
            self.assertEqual(result.generation_count, 181)
            generation_call = self.bundle.model.calls[0]
            self.assertFalse(generation_call["do_sample"])
            self.assertEqual(generation_call["num_beams"], 1)
            self.assertEqual(generation_call["num_return_sequences"], 1)
            self.assertNotIn("temperature", generation_call)
            self.assertNotIn("top_p", generation_call)
            self.assertNotIn("top_k", generation_call)
            repeated = run_extraction_audit(
                self.spec,
                self.context,
                self.checkpoint,
                self.bundle,
                output_root=Path(directory),
                run_id="run-11",
            )
            self.assertEqual(repeated, result)
            self.assertEqual(len(self.bundle.model.calls), 181)
            self.assertTrue(self.bundle.model.training)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
            self.assertEqual(os.stat(summary).st_mode & 0o777, 0o600)
            safe = summary.read_text(encoding="utf-8")
            self.assertNotIn(self.context.targets[0].value("PERSON_NAME"), safe)
            self.assertNotIn(self.context.targets[0].entity_id, safe)
            self.assertNotIn("generated_text", safe)
            with self.assertRaises(FileExistsError):
                run_extraction_audit(
                    self.spec,
                    self.context,
                    self.checkpoint,
                    self.bundle,
                    output_root=Path(directory),
                    run_id="run-11",
                    resume=False,
                )

    def test_greedy_generation_is_rng_independent_and_never_seeds_torch(self):
        query = _query_schedule(self.spec, self.context)[0]
        original = torch.random.get_rng_state().clone()
        try:
            first_state = torch.Generator().manual_seed(1).get_state()
            second_state = torch.Generator().manual_seed(2).get_state()
            records = []
            for state in (first_state, second_state):
                torch.random.set_rng_state(state)
                before = torch.random.get_rng_state().clone()
                with mock.patch("torch.manual_seed") as manual_seed:
                    records.append(_generate_query(self.spec, self.bundle, query))
                manual_seed.assert_not_called()
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
            self.assertEqual(records[0], records[1])
        finally:
            torch.random.set_rng_state(original)

    def test_inherited_sampling_configuration_is_neutralized_and_restored(self):
        query = _query_schedule(self.spec, self.context)[0]
        original = (
            self.bundle.model.generation_config.temperature,
            self.bundle.model.generation_config.top_p,
            self.bundle.model.generation_config.top_k,
        )

        _generate_query(self.spec, self.bundle, query)

        self.assertEqual(
            self.bundle.model.inherited_sampling_states,
            [(None, None, None)],
        )
        self.assertEqual(
            (
                self.bundle.model.generation_config.temperature,
                self.bundle.model.generation_config.top_p,
                self.bundle.model.generation_config.top_k,
            ),
            original,
        )

    def test_unsafe_path_symlink_and_generation_failure_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ExtractionAuditError):
                prepare_audit_journal(
                    output_root=root,
                    run_id="../escape",
                    spec=self.spec,
                    context=self.context,
                    checkpoint=self.checkpoint,
                    generation_schedule_sha256="a" * 64,
                    resume=True,
                )
            target = root / "target"
            target.mkdir()
            symlink = root / "link"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ExtractionAuditError):
                prepare_audit_journal(
                    output_root=symlink,
                    run_id="run-11",
                    spec=self.spec,
                    context=self.context,
                    checkpoint=self.checkpoint,
                    generation_schedule_sha256="a" * 64,
                    resume=True,
                )

        failing_bundle = _bundle()
        failing_checkpoint = _checkpoint(self.context, failing_bundle)
        protected_name = self.context.targets[0].value("PERSON_NAME")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1
        ), mock.patch.object(
            failing_bundle.model,
            "generate",
            side_effect=RuntimeError("segredo-nao-expor"),
        ):
            with self.assertRaises(ExtractionAuditError) as caught:
                run_extraction_audit(
                    self.spec,
                    self.context,
                    failing_checkpoint,
                    failing_bundle,
                    output_root=Path(directory),
                    run_id="run-failure",
                )
            self.assertNotIn("segredo-nao-expor", str(caught.exception))
            self.assertNotIn(protected_name, str(caught.exception))
            self.assertFalse(
                (Path(directory) / "run-failure/evaluator/summaries/B0-targets-020-round-000.json").exists()
            )
            self.assertTrue(
                (Path(directory) / "run-failure/evaluator/private/audits/B0-targets-020-round-000.incomplete").is_dir()
            )
            self.assertEqual(
                (
                    failing_bundle.model.generation_config.temperature,
                    failing_bundle.model.generation_config.top_p,
                    failing_bundle.model.generation_config.top_k,
                ),
                (0.1, 0.95, 50),
            )


if __name__ == "__main__":
    unittest.main()
