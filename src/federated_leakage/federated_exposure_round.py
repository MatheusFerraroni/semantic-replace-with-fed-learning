"""Rodada F0 exclusiva da calibração de exposição local das vítimas."""

from __future__ import annotations

import math

from .aggregation_contracts import FedAvgSpec, validate_fedavg_spec
from .fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from .federated_exposure_contracts import (
    FEDERATED_EXPOSURE_ROUND_SCHEMA_VERSION,
    ExposureArmSpec,
    FederatedExposureError,
    FederatedExposureRoundResult,
    arm_id,
)
from .federated_round import (
    PreparedAuxiliaryTrainingInput,
    PreparedVictimTrainingInputs,
    _round_hashes,
    _validate_auxiliary_input,
    _validate_victim_inputs,
)
from .local_training import (
    train_local_client,
    train_local_client_with_repetitions_for_calibration,
)
from .model_contracts import LoadedModelBundle
from .model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
    restore_model_parameter_snapshot,
)
from .training_contracts import (
    LocalTrainingResult,
    LocalTrainingSpec,
    validate_local_training_spec,
)


def validate_exposure_round_result(
    result: object,
) -> FederatedExposureRoundResult:
    if not isinstance(result, FederatedExposureRoundResult):
        raise FederatedExposureError("resultado da rodada de exposição é inválido")
    multiplier = result.victim_repetition_multiplier
    hashes = (
        result.victim_dataset_sha256,
        result.auxiliary_schedule_sha256,
        result.auxiliary_values_sha256,
        result.initial_model_sha256,
        result.aggregate_update_sha256,
        result.final_model_sha256,
        result.client_order_sha256,
        result.weights_sha256,
        result.sample_order_schedule_sha256,
        result.training_seed_schedule_sha256,
    )
    if (
        result.schema_version != FEDERATED_EXPOSURE_ROUND_SCHEMA_VERSION
        or result.arm_id != arm_id(multiplier)
        or not 1 <= result.round_id <= 20
        or result.conversation_presentations != 1_000 * multiplier + 100
        or result.optimizer_steps != 250 * multiplier + 25
        or result.victim_optimizer_steps != 250 * multiplier
        or result.auxiliary_optimizer_steps != 25
        or not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in hashes
        )
        or any(
            not math.isfinite(value)
            for value in (
                result.mean_client_loss,
                result.mean_victim_loss,
                result.auxiliary_loss,
                result.aggregate_delta_l2_norm,
                result.aggregate_delta_max_abs,
            )
        )
    ):
        raise FederatedExposureError("resultado da rodada diverge do contrato")
    return result


def run_federated_exposure_round(
    victim_inputs: PreparedVictimTrainingInputs,
    auxiliary_input: PreparedAuxiliaryTrainingInput,
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    seed: int,
    round_id: int,
    arm: ExposureArmSpec,
) -> FederatedExposureRoundResult:
    """Treina dez vítimas repetidas, um auxiliar 1× e aplica FedAvg atômico."""

    victims = _validate_victim_inputs(victim_inputs)
    auxiliary = _validate_auxiliary_input(auxiliary_input)
    local_spec = validate_local_training_spec(local_training_spec)
    aggregation_spec = validate_fedavg_spec(fedavg_spec)
    if arm.arm_id != arm_id(arm.victim_repetition_multiplier):
        raise FederatedExposureError("braço da rodada é inválido")
    if auxiliary.presentation != "benign" or auxiliary.round_id != round_id:
        raise FederatedExposureError("auxiliar da calibração não é F0 pareado")
    if not isinstance(model_bundle, LoadedModelBundle):
        raise FederatedExposureError("bundle da rodada é inválido")

    weights = resolve_fedavg_client_weights(aggregation_spec, 1, "F0")
    multiplier = arm.victim_repetition_multiplier
    expected_steps = {
        **{f"victim-{index:02d}": 25 * multiplier for index in range(1, 11)},
        "auxiliary": 25,
    }
    previous_mode = bool(model_bundle.model.training)
    snapshot = capture_model_parameter_snapshot(model_bundle)
    accumulator = FedAvgAccumulator(
        aggregation_spec,
        weights,
        snapshot,
        model_bundle.provenance,
        round_id=round_id,
        expected_optimizer_steps_by_client=expected_steps,
    )
    results: list[LocalTrainingResult] = []
    try:
        for index, (weight, samples) in enumerate(
            zip(weights, (*victims.client_samples, auxiliary.samples))
        ):
            restore_model_parameter_snapshot(model_bundle, snapshot)
            if index < 10 and multiplier > 1:
                result = train_local_client_with_repetitions_for_calibration(
                    samples,
                    model_bundle,
                    local_spec,
                    seed=seed,
                    round_id=round_id,
                    initial_snapshot=snapshot,
                    repetition_multiplier=multiplier,
                )
            else:
                result = train_local_client(
                    samples,
                    model_bundle,
                    local_spec,
                    seed=seed,
                    role=weight.role,
                    round_id=round_id,
                    initial_snapshot=snapshot,
                )
            accumulator.add_client_update(
                result,
                iter_local_parameter_deltas(model_bundle, snapshot, result),
            )
            results.append(result)
            restore_model_parameter_snapshot(model_bundle, snapshot)
        application = accumulator.finalize_and_apply(model_bundle, snapshot)
        model_bundle.model.train(previous_mode)
    except Exception as error:
        accumulator.abort()
        try:
            restore_model_parameter_snapshot(model_bundle, snapshot)
            model_bundle.model.train(previous_mode)
        except Exception as restore_error:
            raise FederatedExposureError(
                "rodada falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, FederatedExposureError):
            raise
        raise FederatedExposureError("treinamento da rodada de exposição falhou") from error

    client_hash, weight_hash, sample_hash, seed_hash = _round_hashes(
        tuple(results), weights
    )
    victim_results = results[:-1]
    result = FederatedExposureRoundResult(
        arm_id=arm.arm_id,
        victim_repetition_multiplier=multiplier,
        round_id=round_id,
        conversation_presentations=1_000 * multiplier + 100,
        optimizer_steps=sum(item.optimizer_steps for item in results),
        victim_optimizer_steps=sum(item.optimizer_steps for item in victim_results),
        auxiliary_optimizer_steps=results[-1].optimizer_steps,
        mean_client_loss=sum(item.mean_loss for item in results) / len(results),
        mean_victim_loss=sum(item.mean_loss for item in victim_results)
        / len(victim_results),
        auxiliary_loss=results[-1].mean_loss,
        aggregate_delta_l2_norm=application.aggregate_delta_l2_norm,
        aggregate_delta_max_abs=application.aggregate_delta_max_abs,
        victim_dataset_sha256=victims.dataset_sha256,
        auxiliary_schedule_sha256=auxiliary.schedule_sha256,
        auxiliary_values_sha256=auxiliary.values_sha256,
        initial_model_sha256=application.initial_model_sha256,
        aggregate_update_sha256=application.aggregate_update_sha256,
        final_model_sha256=application.final_model_sha256,
        client_order_sha256=client_hash,
        weights_sha256=weight_hash,
        sample_order_schedule_sha256=sample_hash,
        training_seed_schedule_sha256=seed_hash,
        model_provenance=model_bundle.provenance,
    )
    return validate_exposure_round_result(result)


__all__ = ["run_federated_exposure_round", "validate_exposure_round_result"]
