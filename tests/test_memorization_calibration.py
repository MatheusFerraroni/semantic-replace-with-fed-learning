import dataclasses
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from federated_leakage.audit_contracts import (
    AuditGenerationRecord,
    ExtractionAuditError,
    load_extraction_audit_spec_from_config,
)
from federated_leakage.calibration_checkpointing import (
    load_calibration_checkpoint,
    save_calibration_checkpoint,
)
from federated_leakage.calibration_gate import load_completed_calibration_gate
from federated_leakage.calibration_contracts import (
    MemorizationCalibrationError,
    PositiveCanaryAuditCheckpoint,
    load_memorization_calibration_spec_from_config,
)
from federated_leakage.calibration_training import (
    train_memorization_calibration_arm,
)
from federated_leakage.canary_audit import (
    _query_schedule,
    preflight_positive_canary_audit,
    prepare_positive_canary_evaluator,
    run_positive_canary_audit,
    score_positive_canary_audit,
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
from federated_leakage.model_updates import capture_model_parameter_snapshot
from federated_leakage.execution_contracts import (
    PilotExecutionError,
    load_pilot_execution_spec_from_config,
)
from federated_leakage.memorization_calibration import (
    _calibration_outcome,
    run_memorization_calibration,
)
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    DatasetStorageError,
    PositiveCanaryDatasetGenerator,
    VictimDatasetGenerator,
    validate_no_cross_flow_collisions,
    validate_positive_canary_dataset,
    read_positive_canary_dataset,
    write_positive_canary_dataset,
)
from federated_leakage.synthetic_profiles.validation import ConversationValidationError
from federated_leakage.tokenization import TokenizedConversation
from federated_leakage.training_contracts import load_local_training_spec_from_config


def _provenance():
    return ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind="huggingface",
        source_identifier=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        artifact_sha256=None,
        result_variant=BASE_RESULT_VARIANT,
        architecture="LlamaForCausalLM",
        parameter_count=1,
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
    def __init__(self, value=0.01):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.bfloat16))
        self.config = SimpleNamespace(use_cache=False, _attn_implementation="eager")

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        base = torch.arange(4, dtype=torch.bfloat16, device=input_ids.device)
        logits = self.weight * base.view(1, 1, 4)
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], input_ids.shape[1], 4))

    def generate(self, **kwargs):
        return kwargs["input_ids"].clone()


class CompactTokenizer:
    def __call__(self, text, **kwargs):
        spans = tuple(re.finditer(r"\s+|\S+", text))
        result = {
            "input_ids": [index % 4 for index, _ in enumerate(spans)],
            "attention_mask": [1] * len(spans),
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [span.span() for span in spans]
        return result

    def decode(self, values, **kwargs):
        return ""


def _bundle(value=0.001):
    return LoadedModelBundle(
        model=LlamaForCausalLM(value),
        tokenizer=CompactTokenizer(),
        max_sequence_length=1_024,
        provenance=_provenance(),
    )


def _samples():
    return tuple(
        TokenizedConversation(
            input_ids=(0, 1, 2),
            attention_mask=(1, 1, 1),
            labels=(0, 1, 2),
            client_id="positive-canary-01",
            round_id=None,
            sample_index=index,
            kind="protected" if index < 80 else "general",
            template_id="protected/test/v1" if index < 80 else "general/test/v1",
            loss_scope="all_tokens",
            prefix_token_count=1 if index < 80 else 0,
            supervised_token_count=2,
        )
        for index in range(100)
    )


class CanaryGenerationAndConfigTests(unittest.TestCase):
    def test_fixed_config_and_complete_disjoint_bundle(self):
        spec = load_memorization_calibration_spec_from_config(
            Path("configs/memorization-calibration-v3.yaml")
        )
        self.assertEqual(spec.repetitions, (20, 40, 80, 160))
        with self.assertRaises(MemorizationCalibrationError):
            load_memorization_calibration_spec_from_config(
                Path("configs/memorization-calibration-v2.yaml")
            )
        first = PositiveCanaryDatasetGenerator(101).generate()
        second = PositiveCanaryDatasetGenerator(101).generate()
        self.assertEqual(first, second)
        self.assertNotEqual(first, PositiveCanaryDatasetGenerator(102).generate())
        validate_positive_canary_dataset(first)
        self.assertEqual(len(first.conversations), 100)
        self.assertEqual(sum(item.kind == "protected" for item in first.conversations), 80)
        self.assertTrue(all(item.loss_scope == "all_tokens" for item in first.conversations))
        victims = VictimDatasetGenerator(101).generate()
        rounds = tuple(
            AuxiliaryRoundGenerator(101).generate(index, presentation="benign")
            for index in range(1, 21)
        )
        validate_no_cross_flow_collisions(
            (
                *(item.conversations for item in victims),
                *(item.conversations for item in rounds),
                first.conversations,
            )
        )
        protected_value = next(
            item.annotations[0] for item in first.conversations if item.kind == "protected"
        )
        with self.assertRaises(ConversationValidationError) as collision:
            PositiveCanaryDatasetGenerator(101).generate(
                reserved_values={
                    protected_value.field_type: (protected_value.value,),
                }
            )
        self.assertNotIn(protected_value.value, str(collision.exception))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_positive_canary_dataset(root, spec.dataset_id, first)
            self.assertEqual(
                path,
                root
                / spec.dataset_id
                / "clients"
                / "positive-canary-01"
                / "conversations.jsonl",
            )
            self.assertEqual(
                read_positive_canary_dataset(root, spec.dataset_id), first
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(DatasetStorageError, "raiz"):
                write_positive_canary_dataset(linked, spec.dataset_id, first)

    def test_main_config_hash_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main-v2.yaml").write_text("changed: true\n", encoding="utf-8")
            text = Path("configs/memorization-calibration-v3.yaml").read_text()
            (root / "calibration.yaml").write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(MemorizationCalibrationError, "hash"):
                load_memorization_calibration_spec_from_config(root / "calibration.yaml")


class CanaryTrainingAndCheckpointTests(unittest.TestCase):
    def test_baseline_gate_blocks_promotion_even_when_an_arm_passes(self):
        audits = (
            SimpleNamespace(repetitions=0, calibrated_at_checkpoint=True),
            SimpleNamespace(repetitions=20, calibrated_at_checkpoint=True),
            SimpleNamespace(repetitions=40, calibrated_at_checkpoint=False),
        )
        self.assertEqual(_calibration_outcome(audits), (True, False, 20))

    def test_arm_keeps_one_recipe_and_checkpoint_round_trips(self):
        bundle = _bundle()
        spec = load_local_training_spec_from_config(Path("configs/main-v2.yaml"))
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4),
        ):
            baseline = capture_model_parameter_snapshot(bundle)
            result = train_memorization_calibration_arm(
                _samples(), bundle, spec, seed=101, repetitions=20,
                baseline_snapshot=baseline,
            )
            self.assertEqual(result.optimizer_steps, 500)
            self.assertEqual(result.conversation_presentations, 2_000)
            self.assertEqual(
                result.final_model_sha256,
                "b43a014d750466635f4bc196122a3bc8383ce72b6d264cf4fda9b2e3ef1431b6",
            )
            self.assertEqual(
                result.sample_order_sha256,
                "0ac3131528143b36ce498d66816714c9212e1b9d1a49643536e55becbe36a8e6",
            )
            self.assertEqual(
                result.training_seed_sha256,
                "0be6c4e2c6ac3b3a68178bbda2d2daf14bf83dadd58a289c8d3129e937cb391e",
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "repetitions-020" / "checkpoint"
                artifact = save_calibration_checkpoint(
                    path,
                    bundle,
                    result,
                    main_config_sha256="5" * 64,
                    dataset_sha256="6" * 64,
                )
                bundle.model.weight.data.fill_(2.0)
                loaded, loaded_artifact = load_calibration_checkpoint(
                    path,
                    bundle,
                    expected_repetitions=20,
                    expected_main_config_sha256="5" * 64,
                    expected_dataset_sha256="6" * 64,
                )
                self.assertEqual(loaded, result)
                self.assertEqual(artifact, loaded_artifact)
                self.assertEqual(fingerprint_model_parameters(bundle), result.final_model_sha256)

    def test_larger_arm_has_the_exact_training_prefix_of_the_smaller_arm(self):
        local_spec = load_local_training_spec_from_config(Path("configs/main-v2.yaml"))
        twenty_bundle = _bundle()
        forty_bundle = _bundle()
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4),
        ):
            twenty_snapshot = capture_model_parameter_snapshot(twenty_bundle)
            twenty_result = train_memorization_calibration_arm(
                _samples(),
                twenty_bundle,
                local_spec,
                seed=101,
                repetitions=20,
                baseline_snapshot=twenty_snapshot,
            )
            twenty_weight = twenty_bundle.model.weight.detach().clone()
            forty_snapshot = capture_model_parameter_snapshot(forty_bundle)
            from federated_leakage import calibration_training

            original_batch = calibration_training._run_logical_batch
            prefix_weights = []

            def track_batch(*args, **kwargs):
                outcome = original_batch(*args, **kwargs)
                if len(prefix_weights) < 500:
                    prefix_weights.append(forty_bundle.model.weight.detach().clone())
                return outcome

            with mock.patch(
                "federated_leakage.calibration_training._run_logical_batch",
                side_effect=track_batch,
            ):
                forty_result = train_memorization_calibration_arm(
                    _samples(),
                    forty_bundle,
                    local_spec,
                    seed=101,
                    repetitions=40,
                    baseline_snapshot=forty_snapshot,
                )
            self.assertTrue(torch.equal(twenty_weight, prefix_weights[-1]))
            self.assertEqual(
                twenty_result.training_seed_sha256,
                forty_result.training_seed_sha256,
            )
            self.assertEqual(
                twenty_result.sample_order_sha256,
                forty_result.sample_order_sha256,
            )

    def test_full_small_run_is_idempotent_and_resumes_confirmed_arms(self):
        calibration_spec = load_memorization_calibration_spec_from_config(
            Path("configs/memorization-calibration-v3.yaml")
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4),
        ):
            first = run_memorization_calibration(
                calibration_spec,
                output_root=Path(directory),
                device="cpu",
                fresh=True,
                bundle_loader=_bundle,
            )
            self.assertEqual(first.total_optimizer_steps, 7_500)
            self.assertEqual(first.total_conversation_presentations, 30_000)
            self.assertEqual(first.total_audit_generations, 905)
            self.assertFalse(first.baseline_gate_passed)
            self.assertFalse(first.calibrated)
            self.assertIsNone(first.first_successful_repetition)
            for repetitions in (20, 40, 80, 160):
                checkpoint = (
                    Path(directory)
                    / "runs"
                    / calibration_spec.default_run_id
                    / "arms"
                    / f"repetitions-{repetitions:03d}"
                    / "checkpoint"
                    / "model.safetensors"
                )
                self.assertTrue(checkpoint.is_file())
            with mock.patch(
                "federated_leakage.memorization_calibration.train_memorization_calibration_arm",
                side_effect=AssertionError("braço confirmado foi retreinado"),
            ):
                resumed = run_memorization_calibration(
                    calibration_spec,
                    output_root=Path(directory),
                    device="cpu",
                    bundle_loader=_bundle,
                )
            self.assertEqual(first.result_sha256, resumed.result_sha256)

            completed_path = (
                Path(directory)
                / "runs"
                / calibration_spec.default_run_id
                / "completed.json"
            )
            payload = json.loads(completed_path.read_text(encoding="utf-8"))
            successful = payload["audits"][-1]
            for metric in successful["field_metrics"]:
                if metric["field_type"] in {"CPF", "RG"}:
                    metric["primary_exact_count"] = 5
                    metric["primary_partial_count"] = 5
            successful.update(
                {
                    "targeted_exact_pair_count": 10,
                    "targeted_partial_pair_count": 10,
                    "distinctive_exact_pair_count": 10,
                    "distinctive_exposed_entity_count": 5,
                    "calibrated_at_checkpoint": True,
                }
            )
            payload["calibrated"] = True
            payload["first_successful_repetition"] = 160
            without_hash = dict(payload)
            without_hash.pop("result_sha256")
            canonical = (
                json.dumps(
                    without_hash,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            payload["result_sha256"] = hashlib.sha256(
                b"memorization-calibration-result/v3\0" + canonical
            ).hexdigest()
            completed_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            pilot_spec = load_pilot_execution_spec_from_config(
                Path("configs/main-v2.yaml")
            )
            with self.assertRaises(PilotExecutionError):
                load_completed_calibration_gate(Path(directory), pilot_spec)

            v2_repetitions = (1, 5, 10, 20)
            v2_arms = []
            for repetitions, arm in zip(v2_repetitions, first.arms):
                item = arm.as_safe_dict()
                supervised_per_repetition = (
                    item["supervised_token_presentations"] // arm.repetitions
                )
                item.update(
                    {
                        "schema_version": "memorization-calibration-arm/v1",
                        "repetitions": repetitions,
                        "conversation_presentations": repetitions * 100,
                        "optimizer_steps": repetitions * 25,
                        "supervised_token_presentations": (
                            repetitions * supervised_per_repetition
                        ),
                    }
                )
                v2_arms.append(item)
            v2_audits = []
            for repetitions, audit in zip((0, *v2_repetitions), first.audits):
                item = audit.as_safe_dict()
                item.update(
                    {
                        "schema_version": "positive-canary-audit-result/v2",
                        "repetitions": repetitions,
                        "checkpoint_id": (
                            "baseline"
                            if repetitions == 0
                            else f"repetitions-{repetitions:03d}"
                        ),
                    }
                )
                v2_audits.append(item)
            successful_v2 = v2_audits[-1]
            for metric in successful_v2["field_metrics"]:
                if metric["field_type"] in {"CPF", "RG"}:
                    metric["primary_exact_count"] = 5
                    metric["primary_partial_count"] = 5
            successful_v2.update(
                {
                    "targeted_exact_pair_count": 10,
                    "targeted_partial_pair_count": 10,
                    "distinctive_exact_pair_count": 10,
                    "distinctive_exposed_entity_count": 5,
                    "calibrated_at_checkpoint": True,
                }
            )
            v2_payload = {
                "schema_version": "memorization-calibration/v2",
                "experiment_seed": 101,
                "run_id": pilot_spec.calibration_run_id,
                "dataset_id": calibration_spec.dataset_id,
                "baseline_model_sha256": first.baseline_model_sha256,
                "arms": v2_arms,
                "audits": v2_audits,
                "total_conversation_presentations": 3_600,
                "total_optimizer_steps": 900,
                "total_audit_generations": 905,
                "baseline_gate_passed": False,
                "calibrated": True,
                "first_successful_repetition": 20,
            }
            v2_canonical = (
                json.dumps(
                    v2_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            v2_payload["result_sha256"] = hashlib.sha256(
                b"memorization-calibration-result/v2\0" + v2_canonical
            ).hexdigest()
            v2_run = Path(directory) / "runs" / pilot_spec.calibration_run_id
            v2_run.mkdir(mode=0o700)
            v2_manifest = {
                "schema_version": "memorization-calibration/v2",
                "run_id": pilot_spec.calibration_run_id,
                "experiment_seed": 101,
                "dataset_id": calibration_spec.dataset_id,
                "client_id": calibration_spec.client_id,
                "repetitions": list(v2_repetitions),
                "main_config_sha256": pilot_spec.config_sha256,
                "canary_dataset_sha256": (
                    pilot_spec.calibration_canary_dataset_sha256
                ),
                "collision_preflight_sha256": (
                    pilot_spec.calibration_collision_preflight_sha256
                ),
                "model_provenance": first.audits[0].model_provenance.as_safe_dict(),
                "decoding_strategy": "tokenwise_greedy_argmax/v1",
                "rng_used": False,
            }
            for path, value in (
                (v2_run / "run_manifest.json", v2_manifest),
                (v2_run / "completed.json", v2_payload),
            ):
                path.write_text(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            gate = load_completed_calibration_gate(Path(directory), pilot_spec)
            self.assertEqual(gate.result_sha256, v2_payload["result_sha256"])
            self.assertEqual(gate.audit_model_sha256[-1], successful_v2["model_state_sha256"])

    def test_interrupted_arm_restarts_from_baseline_without_repeating_confirmed_arm(self):
        calibration_spec = load_memorization_calibration_spec_from_config(
            Path("configs/memorization-calibration-v3.yaml")
        )

        def fail_second_arm(*args, **kwargs):
            if kwargs["repetitions"] == 40:
                raise MemorizationCalibrationError("falha injetada no braço")
            return train_memorization_calibration_arm(*args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4),
        ):
            root = Path(directory)
            with mock.patch(
                "federated_leakage.memorization_calibration.train_memorization_calibration_arm",
                side_effect=fail_second_arm,
            ):
                with self.assertRaisesRegex(MemorizationCalibrationError, "injetada"):
                    run_memorization_calibration(
                        calibration_spec,
                        output_root=root,
                        device="cpu",
                        fresh=True,
                        bundle_loader=_bundle,
                    )
            confirmed = (
                root
                / "runs"
                / calibration_spec.default_run_id
                / "arms"
                / "repetitions-020"
                / "completed.json"
            )
            self.assertTrue(confirmed.is_file())
            with mock.patch(
                "federated_leakage.memorization_calibration.train_memorization_calibration_arm",
                wraps=train_memorization_calibration_arm,
            ) as trainer:
                result = run_memorization_calibration(
                    calibration_spec,
                    output_root=root,
                    device="cpu",
                    bundle_loader=_bundle,
                )
            self.assertEqual(
                [call.kwargs["repetitions"] for call in trainer.call_args_list],
                [40, 80, 160],
            )
            self.assertEqual(result.total_optimizer_steps, 7_500)

    def test_cuda_contract_fails_before_training_or_orchestration(self):
        cuda_bundle = dataclasses.replace(
            _bundle(),
            provenance=dataclasses.replace(_provenance(), device="cuda"),
        )
        local_spec = load_local_training_spec_from_config(Path("configs/main-v2.yaml"))
        calibration_spec = load_memorization_calibration_spec_from_config(
            Path("configs/memorization-calibration-v3.yaml")
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(MemorizationCalibrationError, "CUBLAS"):
                train_memorization_calibration_arm(
                    (), cuda_bundle, local_spec, seed=101, repetitions=20,
                    baseline_snapshot=mock.Mock(),
                )
            with mock.patch(
                "federated_leakage.memorization_calibration._materialize_data_preflight"
            ) as data_preflight:
                with self.assertRaisesRegex(MemorizationCalibrationError, "CUBLAS"):
                    run_memorization_calibration(
                        calibration_spec,
                        device="cuda",
                        preflight_only=True,
                        bundle_loader=lambda: cuda_bundle,
                    )
                data_preflight.assert_not_called()


class CanaryAuditScoringTests(unittest.TestCase):
    def test_incomplete_canary_audit_repairs_only_terminal_partial_and_resumes(self):
        dataset = PositiveCanaryDatasetGenerator(101).generate()
        context = prepare_positive_canary_evaluator(dataset, 101)
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        bundle = _bundle()
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
            checkpoint = PositiveCanaryAuditCheckpoint(
                checkpoint_id="baseline",
                repetitions=0,
                experiment_seed=101,
                expected_model_sha256=model_hash,
                model_provenance=bundle.provenance,
            )
            from federated_leakage import canary_audit

            original_generate = canary_audit._generate_query

            def fail_after_ten(specification, loaded_bundle, query):
                if query.query_index == 10:
                    raise ExtractionAuditError("falha injetada")
                return original_generate(specification, loaded_bundle, query)

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch(
                    "federated_leakage.canary_audit._generate_query",
                    side_effect=fail_after_ten,
                ):
                    with self.assertRaisesRegex(MemorizationCalibrationError, "injetada"):
                        run_positive_canary_audit(
                            spec, context, checkpoint, bundle, output_root=root
                        )
                journal = (
                    root
                    / "private"
                    / "audits"
                    / ".baseline.incomplete"
                    / "extraction_results.jsonl"
                )
                with journal.open("ab") as output:
                    output.write(b'{"terminal":"partial"')
                result = run_positive_canary_audit(
                    spec, context, checkpoint, bundle, output_root=root
                )
                self.assertEqual(result.generation_count, 181)
                self.assertFalse((journal.parent).exists())

    def test_completed_private_audit_rejects_tampering_and_extra_files(self):
        dataset = PositiveCanaryDatasetGenerator(101).generate()
        context = prepare_positive_canary_evaluator(dataset, 101)
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        bundle = _bundle()
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
            checkpoint = PositiveCanaryAuditCheckpoint(
                checkpoint_id="baseline",
                repetitions=0,
                experiment_seed=101,
                expected_model_sha256=model_hash,
                model_provenance=bundle.provenance,
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = run_positive_canary_audit(
                    spec,
                    context,
                    checkpoint,
                    bundle,
                    output_root=root,
                )
                self.assertEqual(result.generation_count, 181)
                summary_raw = (root / "summaries" / "baseline.json").read_text()
                for protected in context.registry:
                    self.assertNotIn(protected.entity_id, summary_raw)
                    for _, value in protected.field_values:
                        self.assertNotIn(value, summary_raw)
                private = root / "private" / "audits" / "baseline"
                metadata = private / "metadata.json"
                original_metadata = metadata.read_bytes()
                metadata.write_bytes(original_metadata.replace(b'"repetitions":0', b'"repetitions":1'))
                with self.assertRaisesRegex(MemorizationCalibrationError, "identidade"):
                    run_positive_canary_audit(
                        spec, context, checkpoint, bundle, output_root=root
                    )
                metadata.write_bytes(original_metadata)
                extra = private / "unexpected.txt"
                extra.write_text("unexpected", encoding="utf-8")
                with self.assertRaisesRegex(MemorizationCalibrationError, "arquivos"):
                    run_positive_canary_audit(
                        spec, context, checkpoint, bundle, output_root=root
                    )
                extra.unlink()
                journal = private / "extraction_results.jsonl"
                original_journal = journal.read_bytes()
                journal.write_bytes(original_journal[:-1])
                with self.assertRaisesRegex(MemorizationCalibrationError, "truncado"):
                    run_positive_canary_audit(
                        spec, context, checkpoint, bundle, output_root=root
                    )

    def test_distinctive_threshold_uses_pairs_and_entities(self):
        dataset = PositiveCanaryDatasetGenerator(101).generate()
        context = prepare_positive_canary_evaluator(dataset, 101)
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        bundle = _bundle()
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            model_hash = fingerprint_model_parameters(bundle)
        checkpoint = PositiveCanaryAuditCheckpoint(
            checkpoint_id="repetitions-020",
            repetitions=20,
            experiment_seed=101,
            expected_model_sha256=model_hash,
            model_provenance=bundle.provenance,
        )
        records = [
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
        ]
        for target_index in range(5):
            target = context.registry[target_index]
            records[target_index] = dataclasses.replace(
                records[target_index],
                generated_text=f" {target.value('CPF')} {target.value('EMAIL')}",
            )
        result = score_positive_canary_audit(spec, context, checkpoint, records)
        self.assertEqual(result.distinctive_exact_pair_count, 10)
        self.assertEqual(result.distinctive_exposed_entity_count, 5)
        self.assertTrue(result.calibrated_at_checkpoint)
        safe = json.dumps(result.as_safe_dict(), ensure_ascii=False)
        self.assertNotIn(context.registry[0].value("CPF"), safe)
        self.assertNotIn(context.registry[0].entity_id, safe)

    def test_cuda_contract_fails_before_canary_context_validation(self):
        spec = load_extraction_audit_spec_from_config(Path("configs/main-v2.yaml"))
        cuda_bundle = dataclasses.replace(
            _bundle(),
            provenance=dataclasses.replace(_provenance(), device="cuda"),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(MemorizationCalibrationError, "CUBLAS"):
                preflight_positive_canary_audit(spec, object(), cuda_bundle)


if __name__ == "__main__":
    unittest.main()
