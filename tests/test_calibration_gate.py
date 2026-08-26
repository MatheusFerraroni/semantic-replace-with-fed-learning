import dataclasses
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from federated_leakage.calibration_contracts import (
    CALIBRATION_LEARNING_RATE_MILLIONTHS,
    EXPECTED_ANCHOR_MODEL_SHA256,
    CanaryFieldMetric,
    MemorizationCalibrationArmResult,
    PositiveCanaryAuditResult,
    learning_rate_arm_id,
)
from federated_leakage.calibration_gate import load_completed_calibration_gate
from federated_leakage.execution_contracts import (
    PilotExecutionError,
    load_pilot_execution_spec_from_config,
)
from federated_leakage.model_contracts import ModelProvenance


def _canonical(payload):
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _provenance():
    return ModelProvenance(
        schema_version="tucano2-model-loading/v1",
        source_kind="huggingface",
        source_identifier="Polygl0t/Tucano2-0.6B-Base",
        revision="dad97dc864a8f9a1d240fb9351d098f3af9511d7",
        artifact_sha256=None,
        result_variant="upstream_baseline",
        architecture="LlamaForCausalLM",
        parameter_count=670_127_616,
        native_context_length=4_096,
        training_sequence_length=1_024,
        vocab_size=49_152,
        tokenizer_fingerprint_sha256="0" * 64,
        weight_dtype="bfloat16",
        device="cuda",
        torch_version="2.7.1",
        transformers_version="4.53.2",
        tokenizers_version="0.21.2",
        safetensors_version="0.5.3",
        huggingface_hub_version="0.33.4",
    )


def _field_metrics(successful):
    fields = (
        "BIRTH_DATE",
        "CPF",
        "RG",
        "PHONE",
        "EMAIL",
        "ADDRESS",
        "APPOINTMENT_DATE",
        "APPOINTMENT_TIME",
    )
    return tuple(
        CanaryFieldMetric(
            field_type=field,
            primary_exact_count=(
                20
                if successful
                and field in {"CPF", "RG", "PHONE", "EMAIL", "ADDRESS"}
                else 0
            ),
            primary_partial_count=(
                20
                if successful
                and field in {"CPF", "RG", "PHONE", "EMAIL", "ADDRESS"}
                else 0
            ),
            field_specific_exact_count=0,
            field_specific_partial_count=0,
            untargeted_exact_count=0,
            denominator=20,
        )
        for field in fields
    )


def _audit(*, learning_rate, model_hash, successful):
    baseline = learning_rate is None
    metrics = _field_metrics(successful)
    exact = sum(item.primary_exact_count for item in metrics)
    return PositiveCanaryAuditResult(
        checkpoint_id="baseline" if baseline else learning_rate_arm_id(learning_rate),
        arm_id=None if baseline else learning_rate_arm_id(learning_rate),
        learning_rate_millionths=learning_rate,
        repetitions=0 if baseline else 160,
        generation_count=181,
        primary_generation_count=20,
        field_specific_generation_count=160,
        untargeted_generation_count=1,
        targeted_exact_pair_count=exact,
        targeted_exact_pair_denominator=160,
        targeted_partial_pair_count=exact,
        distinctive_exact_pair_count=exact,
        distinctive_exact_pair_denominator=100,
        repeatable_exact_pair_count=0,
        repeatable_exact_pair_denominator=60,
        distinctive_exposed_entity_count=20 if successful else 0,
        targeted_complete_generation_count=0,
        targeted_ordered_complete_generation_count=0,
        targeted_misassociation_count=0,
        targeted_unseen_formatted_count=0,
        field_specific_exact_pair_count=0,
        field_specific_partial_pair_count=0,
        untargeted_exact_hit_count=0,
        untargeted_unique_value_count=0,
        untargeted_canary_name_count=0,
        untargeted_exposed_profile_count=0,
        field_metrics=metrics,
        calibrated_at_checkpoint=successful,
        registry_sha256="7" * 64,
        target_schedule_sha256="8" * 64,
        generation_schedule_sha256="9" * 64,
        generation_records_sha256=hashlib.sha256(
            str(learning_rate).encode("ascii")
        ).hexdigest(),
        model_state_sha256=model_hash,
        model_provenance=_provenance(),
    )


def _write_gate(root):
    spec = load_pilot_execution_spec_from_config(Path("configs/main-v3.yaml"))
    baseline = "b" * 64
    arms = []
    audits = [_audit(learning_rate=None, model_hash=baseline, successful=False)]
    for index, learning_rate in enumerate(CALIBRATION_LEARNING_RATE_MILLIONTHS, 1):
        final = f"{index:x}" * 64
        arms.append(
            MemorizationCalibrationArmResult(
                arm_id=learning_rate_arm_id(learning_rate),
                learning_rate_millionths=learning_rate,
                repetitions=160,
                conversation_presentations=16_000,
                optimizer_steps=4_000,
                supervised_token_presentations=1,
                mean_loss=1.0,
                first_step_loss=2.0,
                last_step_loss=0.5,
                mean_gradient_norm=1.0,
                max_gradient_norm=2.0,
                sample_order_sha256="c" * 64,
                training_seed_sha256="d" * 64,
                initial_model_sha256=baseline,
                final_model_sha256=final,
                model_provenance=_provenance(),
            )
        )
        audits.append(
            _audit(
                learning_rate=learning_rate,
                model_hash=final,
                successful=learning_rate == 30,
            )
        )
    payload = {
        "schema_version": "memorization-calibration/v4",
        "experiment_seed": 101,
        "run_id": spec.calibration_run_id,
        "dataset_id": "positive-canaries-seed-101-v1",
        "baseline_model_sha256": baseline,
        "arms": [item.as_safe_dict() for item in arms],
        "audits": [item.as_safe_dict() for item in audits],
        "total_conversation_presentations": 64_000,
        "total_optimizer_steps": 16_000,
        "total_audit_generations": 905,
        "baseline_gate_passed": False,
        "calibrated": True,
        "first_successful_arm_id": "lr-000030",
        "first_successful_learning_rate_millionths": 30,
    }
    payload["result_sha256"] = hashlib.sha256(
        b"memorization-calibration-result/v4\0" + _canonical(payload)
    ).hexdigest()
    manifest = {
        "schema_version": "memorization-calibration/v4",
        "run_id": spec.calibration_run_id,
        "experiment_seed": 101,
        "dataset_id": "positive-canaries-seed-101-v1",
        "client_id": "positive-canary-01",
        "fixed_repetitions": 160,
        "learning_rate_arms": [
            {
                "arm_id": learning_rate_arm_id(value),
                "learning_rate_millionths": value,
            }
            for value in CALIBRATION_LEARNING_RATE_MILLIONTHS
        ],
        "expected_anchor_model_sha256": EXPECTED_ANCHOR_MODEL_SHA256,
        "main_config_sha256": spec.calibration_main_config_sha256,
        "canary_dataset_sha256": spec.calibration_canary_dataset_sha256,
        "collision_preflight_sha256": spec.calibration_collision_preflight_sha256,
        "model_provenance": _provenance().as_safe_dict(),
        "decoding_strategy": spec.calibration_decoding_strategy,
        "rng_used": False,
    }
    run_root = Path(root) / "runs" / spec.calibration_run_id
    run_root.mkdir(parents=True)
    (run_root / "run_manifest.json").write_bytes(_canonical(manifest))
    (run_root / "completed.json").write_bytes(_canonical(payload))
    return dataclasses.replace(
        spec,
        calibration_result_sha256=payload["result_sha256"],
        calibration_baseline_model_sha256=baseline,
    ), run_root


class CalibrationGateTests(unittest.TestCase):
    def test_accepts_only_the_complete_selected_v4_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, _ = _write_gate(directory)
            gate = load_completed_calibration_gate(Path(directory), spec)
            self.assertEqual(gate.selected_arm_id, "lr-000030")
            self.assertEqual(gate.selected_learning_rate_millionths, 30)
            self.assertEqual(gate.baseline_model_sha256, "b" * 64)

    def test_rejects_manifest_result_schedule_private_content_and_symlink(self):
        mutations = (
            (
                "run_manifest.json",
                lambda value: value.update(expected_anchor_model_sha256="e" * 64),
            ),
            (
                "completed.json",
                lambda value: value.update(baseline_gate_passed=True),
            ),
            (
                "completed.json",
                lambda value: value["audits"][2].update(
                    generation_schedule_sha256="e" * 64
                ),
            ),
            ("completed.json", lambda value: value.update(prompt="conteudo-privado")),
        )
        for filename, mutate in mutations:
            with self.subTest(filename=filename, mutation=mutate):
                with tempfile.TemporaryDirectory() as directory:
                    spec, run_root = _write_gate(directory)
                    path = run_root / filename
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    mutate(payload)
                    path.write_bytes(_canonical(payload))
                    with self.assertRaises(PilotExecutionError):
                        load_completed_calibration_gate(Path(directory), spec)

        with tempfile.TemporaryDirectory() as directory:
            spec, run_root = _write_gate(directory)
            manifest = run_root / "run_manifest.json"
            target = run_root / "manifest-target.json"
            target.write_bytes(manifest.read_bytes())
            manifest.unlink()
            os.symlink(target, manifest)
            with self.assertRaises(PilotExecutionError):
                load_completed_calibration_gate(Path(directory), spec)


if __name__ == "__main__":
    unittest.main()
