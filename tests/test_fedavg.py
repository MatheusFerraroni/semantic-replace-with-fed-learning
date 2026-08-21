import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from federated_leakage.aggregation_contracts import (
    FEDAVG_AGGREGATION_SCHEMA_VERSION,
    FEDERATED_ROUND_SCHEMA_VERSION,
    FedAvgError,
    FedAvgRoundResult,
    load_fedavg_spec_from_config,
    parse_fedavg_spec,
)
from federated_leakage.fedavg import (
    FedAvgAccumulator,
    resolve_fedavg_client_weights,
)
from federated_leakage.federated_round import (
    prepare_auxiliary_training_input,
    prepare_victim_training_inputs,
    run_non_private_federated_round,
    validate_paired_federated_round_results,
)
from federated_leakage.local_training import (
    train_local_client as actual_train_local_client,
)
from federated_leakage.model_contracts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BASE_RESULT_VARIANT,
    EXPECTED_TOKENIZER_FINGERPRINT,
    LoadedModelBundle,
    ModelProvenance,
)
from federated_leakage.model_updates import capture_model_parameter_snapshot
from federated_leakage.synthetic_profiles import (
    AuxiliaryRoundGenerator,
    VictimDatasetGenerator,
)
from federated_leakage.training_contracts import (
    LOCAL_MODEL_UPDATE_SCHEMA_VERSION,
    LOCAL_TRAINING_SCHEMA_VERSION,
    LocalTrainingError,
    LocalTrainingResult,
    ParameterDelta,
    load_local_training_spec_from_config,
)


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, initial=0.25):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(initial, dtype=torch.bfloat16)
        )
        self.config = SimpleNamespace(
            use_cache=False,
            _attn_implementation="eager",
        )

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        base = torch.linspace(
            -1.0,
            1.0,
            4,
            device=input_ids.device,
            dtype=torch.bfloat16,
        )
        logits = self.weight * base.view(1, 1, -1)
        return SimpleNamespace(
            logits=logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        )


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [1] * len(text),
            "attention_mask": [1] * len(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


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


def _bundle(initial=0.0001):
    return LoadedModelBundle(
        model=LlamaForCausalLM(initial),
        tokenizer=CharacterTokenizer(),
        max_sequence_length=1_024,
        provenance=_provenance(),
    )


def _local_result(weight, provenance, round_id=1):
    return LocalTrainingResult(
        client_id=weight.client_id,
        role=weight.role,
        round_id=round_id,
        conversation_count=100,
        optimizer_steps=25,
        supervised_token_count=200,
        mean_loss=1.0,
        first_step_loss=1.1,
        last_step_loss=0.9,
        mean_gradient_norm=0.5,
        max_gradient_norm=0.75,
        sample_order_sha256="a" * 64,
        training_seed_sha256="b" * 64,
        model_provenance=provenance,
    )


class FedAvgConfigurationTests(unittest.TestCase):
    def test_loads_recipe_and_resolves_all_exact_weights(self):
        spec = load_fedavg_spec_from_config(Path("configs/main-v1.yaml"))
        self.assertEqual(spec.schema_version, FEDAVG_AGGREGATION_SCHEMA_VERSION)
        self.assertEqual(spec.round_schema_version, FEDERATED_ROUND_SCHEMA_VERSION)
        for scenario in ("F0", "F1"):
            for k in range(1, 11):
                weights = resolve_fedavg_client_weights(spec, k, scenario)
                self.assertEqual(len(weights), 11)
                self.assertEqual(
                    sum(weight.numerator_units for weight in weights),
                    weights[0].denominator_units,
                )
                self.assertLessEqual(weights[-1].value, 0.5)
                self.assertEqual(weights[0].value, 1 / (10 + k))
                self.assertEqual(weights[-1].value, k / (10 + k))

    def test_rejects_drift_k_and_duplicate_yaml_without_exposing_value(self):
        config = yaml.safe_load(Path("configs/main-v1.yaml").read_text())
        config["federated"]["aggregation_dtype"] = "float64"
        with self.assertRaisesRegex(FedAvgError, "aggregation_dtype"):
            parse_fedavg_spec(config)

        spec = load_fedavg_spec_from_config(Path("configs/main-v1.yaml"))
        for invalid in (0, 11, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(FedAvgError):
                resolve_fedavg_client_weights(spec, invalid, "F0")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                "schema_version: federated-leakage/main-config/v1\n"
                "federated:\n"
                "  aggregation: fedavg\n"
                "  aggregation: segredo-nao-expor\n",
                encoding="utf-8",
            )
            with self.assertRaises(FedAvgError) as context:
                load_fedavg_spec_from_config(path)
        self.assertNotIn("segredo-nao-expor", str(context.exception))


class FedAvgAccumulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_fedavg_spec_from_config(Path("configs/main-v1.yaml"))

    def _accumulator(self, *, scenario="F0", k=3, initial=0.25):
        bundle = _bundle(initial)
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            snapshot = capture_model_parameter_snapshot(bundle)
        weights = resolve_fedavg_client_weights(self.spec, k, scenario)
        accumulator = FedAvgAccumulator(
            self.spec,
            weights,
            snapshot,
            bundle.provenance,
            round_id=1,
        )
        return bundle, snapshot, weights, accumulator

    def test_matches_closed_form_and_cannot_be_reused(self):
        bundle, snapshot, weights, accumulator = self._accumulator(k=3)
        deltas = []
        for index, weight in enumerate(weights, start=1):
            value = index / 100
            deltas.append(value)
            accumulator.add_client_update(
                _local_result(weight, bundle.provenance),
                iter(
                    (
                        ParameterDelta(
                            name="weight",
                            tensor=torch.tensor(value, dtype=torch.float32),
                            numel=1,
                        ),
                    )
                ),
            )
        expected_delta = sum(
            value * weight.value for value, weight in zip(deltas, weights)
        )
        expected = torch.tensor(
            float(snapshot.parameters[0]) + expected_delta,
            dtype=torch.bfloat16,
        )
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            application = accumulator.finalize_and_apply(bundle, snapshot)
        self.assertTrue(torch.equal(bundle.model.weight.detach().cpu(), expected))
        self.assertAlmostEqual(
            application.aggregate_delta_l2_norm,
            expected_delta,
            places=6,
        )
        self.assertEqual(accumulator.state, "applied")
        with self.assertRaises(FedAvgError):
            accumulator.add_client_update(
                _local_result(weights[0], bundle.provenance), iter(())
            )

    def test_rejects_invalid_stream_and_discards_partial_sum(self):
        bundle, _, weights, accumulator = self._accumulator()
        bad_streams = (
            iter(()),
            iter(
                (
                    ParameterDelta(
                        name="other",
                        tensor=torch.tensor(1.0, dtype=torch.float32),
                        numel=1,
                    ),
                )
            ),
            iter(
                (
                    ParameterDelta(
                        name="weight",
                        tensor=torch.tensor(float("nan"), dtype=torch.float32),
                        numel=1,
                    ),
                )
            ),
            iter(
                (
                    ParameterDelta(
                        name="weight",
                        tensor=torch.tensor(1.0, dtype=torch.float64),
                        numel=1,
                    ),
                )
            ),
            iter(
                (
                    ParameterDelta(
                        name="weight",
                        tensor=torch.tensor(1.0, dtype=torch.float32),
                        numel=1,
                    ),
                    ParameterDelta(
                        name="weight",
                        tensor=torch.tensor(2.0, dtype=torch.float32),
                        numel=1,
                    ),
                )
            ),
        )
        for stream in bad_streams:
            with self.subTest(stream=stream):
                _, _, case_weights, case = self._accumulator()
                with self.assertRaises(FedAvgError):
                    case.add_client_update(
                        _local_result(case_weights[0], bundle.provenance),
                        stream,
                    )
                self.assertEqual(case.state, "invalid")

        with self.assertRaises(FedAvgError):
            accumulator.add_client_update(
                dataclasses.replace(
                    _local_result(weights[0], bundle.provenance),
                    client_id="victim-02",
                ),
                iter(()),
            )
        self.assertEqual(accumulator.state, "invalid")

    def test_incomplete_round_and_apply_failure_restore_global_model(self):
        bundle, snapshot, weights, accumulator = self._accumulator()
        before = snapshot.parameters[0].clone()
        accumulator.add_client_update(
            _local_result(weights[0], bundle.provenance),
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
        with mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1):
            with self.assertRaisesRegex(FedAvgError, "todos os clientes"):
                accumulator.finalize_and_apply(bundle, snapshot)
        self.assertTrue(torch.equal(bundle.model.weight.detach().cpu(), before))

        bundle, snapshot, weights, accumulator = self._accumulator()
        for weight in weights:
            accumulator.add_client_update(
                _local_result(weight, bundle.provenance),
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
        before = snapshot.parameters[0].clone()
        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch(
                "federated_leakage.fedavg._fingerprint_named_tensors",
                side_effect=("b" * 64, RuntimeError("segredo-nao-expor")),
            ),
            self.assertRaises(FedAvgError) as context,
        ):
            accumulator.finalize_and_apply(bundle, snapshot)
        self.assertTrue(torch.equal(bundle.model.weight.detach().cpu(), before))
        self.assertNotIn("segredo-nao-expor", str(context.exception))


class FederatedRoundIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fedavg_spec = load_fedavg_spec_from_config(
            Path("configs/main-v1.yaml")
        )
        cls.local_spec = load_local_training_spec_from_config(
            Path("configs/main-v1.yaml")
        )
        cls.victim_datasets = VictimDatasetGenerator(11).generate()
        generator = AuxiliaryRoundGenerator(11)
        cls.benign_round = generator.generate(1, presentation="benign")
        cls.adversarial_round = generator.generate(1, presentation="adversarial")

    def _run_pair(self):
        benign_bundle = _bundle()
        adversarial_bundle = _bundle()
        victims = prepare_victim_training_inputs(
            self.victim_datasets, benign_bundle
        )
        benign_input = prepare_auxiliary_training_input(
            self.benign_round, benign_bundle
        )
        adversarial_input = prepare_auxiliary_training_input(
            self.adversarial_round, adversarial_bundle
        )
        patches = (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch("federated_leakage.local_training.EXPECTED_VOCAB_SIZE", 4),
        )
        starting_parameters = []

        def recording_train(*args, **kwargs):
            starting_parameters.append(
                args[1].model.weight.detach().cpu().clone()
            )
            return actual_train_local_client(*args, **kwargs)

        with (
            patches[0],
            patches[1],
            mock.patch(
                "federated_leakage.federated_round.train_local_client",
                side_effect=recording_train,
            ),
        ):
            benign = run_non_private_federated_round(
                victims,
                benign_input,
                benign_bundle,
                self.local_spec,
                self.fedavg_spec,
                seed=11,
                scenario="F0",
                round_id=1,
                auxiliary_weight_units=3,
            )
            adversarial = run_non_private_federated_round(
                victims,
                adversarial_input,
                adversarial_bundle,
                self.local_spec,
                self.fedavg_spec,
                seed=11,
                scenario="F1",
                round_id=1,
                auxiliary_weight_units=3,
            )
        return (
            benign_bundle,
            adversarial_bundle,
            victims,
            benign,
            adversarial,
            tuple(starting_parameters),
        )

    def test_runs_all_clients_and_validates_f0_f1_pair(self):
        _, _, victims, benign, adversarial, starts = self._run_pair()
        validate_paired_federated_round_results(benign, adversarial)
        self.assertEqual(benign.conversation_count, 1_100)
        self.assertEqual(benign.optimizer_steps, 275)
        self.assertEqual(benign.victim_dataset_sha256, victims.dataset_sha256)
        self.assertEqual(benign.initial_model_sha256, adversarial.initial_model_sha256)
        self.assertGreater(benign.aggregate_delta_l2_norm, 0.0)
        self.assertNotEqual(benign.initial_model_sha256, benign.final_model_sha256)
        self.assertNotEqual(
            adversarial.initial_model_sha256,
            adversarial.final_model_sha256,
        )
        self.assertEqual(len(starts), 22)
        self.assertTrue(all(torch.equal(starts[0], value) for value in starts))
        self.assertNotEqual(
            benign.auxiliary_presentation_sha256,
            adversarial.auxiliary_presentation_sha256,
        )
        self.assertNotEqual(
            benign.supervised_token_count,
            adversarial.supervised_token_count,
        )
        safe = json.dumps(benign.as_safe_dict(), ensure_ascii=False)
        for forbidden in (
            '"input_ids"',
            '"labels"',
            '"entity_id"',
            '"annotations"',
            '"text"',
        ):
            self.assertNotIn(forbidden, safe)
        self.assertNotIn("client_samples", repr(victims))

    def test_training_failure_restores_the_round_snapshot(self):
        bundle = _bundle()
        victims = prepare_victim_training_inputs(self.victim_datasets, bundle)
        auxiliary = prepare_auxiliary_training_input(self.benign_round, bundle)
        before = bundle.model.weight.detach().cpu().clone()

        def fail_training(*args, **kwargs):
            with torch.no_grad():
                args[1].model.weight.add_(1.0)
            raise LocalTrainingError("segredo-nao-expor")

        with (
            mock.patch("federated_leakage.model_updates.EXPECTED_PARAMETER_COUNT", 1),
            mock.patch(
                "federated_leakage.federated_round.train_local_client",
                side_effect=fail_training,
            ),
            self.assertRaises(FedAvgError) as context,
        ):
            run_non_private_federated_round(
                victims,
                auxiliary,
                bundle,
                self.local_spec,
                self.fedavg_spec,
                seed=11,
                scenario="F0",
                round_id=1,
                auxiliary_weight_units=1,
            )
        self.assertTrue(torch.equal(bundle.model.weight.detach().cpu(), before))
        self.assertNotIn("segredo-nao-expor", str(context.exception))

    def test_pair_rejects_different_initial_model_or_k(self):
        provenance = _provenance()
        base = FedAvgRoundResult(
            scenario="F0",
            experiment_seed=11,
            round_id=1,
            auxiliary_weight_units=1,
            victim_client_count=10,
            auxiliary_client_count=1,
            conversation_count=1_100,
            optimizer_steps=275,
            supervised_token_count=1,
            mean_client_loss=1.0,
            mean_victim_loss=1.0,
            auxiliary_loss=1.0,
            mean_client_gradient_norm=1.0,
            max_client_gradient_norm=1.0,
            aggregate_delta_l2_norm=1.0,
            aggregate_delta_max_abs=1.0,
            client_order_sha256="a" * 64,
            weights_sha256="b" * 64,
            sample_order_schedule_sha256="c" * 64,
            training_seed_schedule_sha256="d" * 64,
            victim_dataset_sha256="e" * 64,
            auxiliary_schedule_sha256="f" * 64,
            auxiliary_values_sha256="0" * 64,
            auxiliary_presentation_sha256="1" * 64,
            auxiliary_batch_sha256="2" * 64,
            initial_model_sha256="3" * 64,
            aggregate_update_sha256="4" * 64,
            final_model_sha256="5" * 64,
            model_provenance=provenance,
        )
        adversarial = dataclasses.replace(
            base,
            scenario="F1",
            auxiliary_presentation_sha256="6" * 64,
            auxiliary_batch_sha256="7" * 64,
            initial_model_sha256="8" * 64,
        )
        with self.assertRaisesRegex(FedAvgError, "pareamento"):
            validate_paired_federated_round_results(base, adversarial)

    def test_preparation_and_round_failures_hide_conversation_content(self):
        tampered = dataclasses.replace(
            self.victim_datasets[0].conversations[0],
            text="segredo-nao-expor",
        )
        dataset = dataclasses.replace(
            self.victim_datasets[0],
            conversations=(tampered,) + self.victim_datasets[0].conversations[1:],
        )
        with self.assertRaises(FedAvgError) as context:
            prepare_victim_training_inputs(
                (dataset,) + self.victim_datasets[1:],
                _bundle(),
            )
        self.assertNotIn("segredo-nao-expor", str(context.exception))


if __name__ == "__main__":
    unittest.main()
