"""Rodada F0 da grade federada de intensidade."""

from __future__ import annotations

from .aggregation_contracts import FedAvgSpec, validate_fedavg_spec
from .fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from .federated_grid_contracts import (
    FederatedGridError,
    FederatedGridRoundResult,
    GridArmSpec,
    grid_arm_id,
    validate_grid_round_result,
)
from .federated_round import (
    PreparedAuxiliaryTrainingInput,
    PreparedVictimTrainingInputs,
    _round_hashes,
    _validate_auxiliary_input,
    _validate_victim_inputs,
)
from .local_training import train_local_client, train_local_client_for_federated_grid
from .model_contracts import LoadedModelBundle
from .model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
    restore_model_parameter_snapshot,
)
from .training_contracts import LocalTrainingResult, LocalTrainingSpec, validate_local_training_spec


def run_federated_grid_round(
    victim_inputs: PreparedVictimTrainingInputs,
    auxiliary_input: PreparedAuxiliaryTrainingInput,
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    seed: int,
    round_id: int,
    arm: GridArmSpec,
) -> FederatedGridRoundResult:
    """Treina vítimas com a dose do braço e mantém o auxiliar oficial em 3e-5."""

    victims = _validate_victim_inputs(victim_inputs)
    auxiliary = _validate_auxiliary_input(auxiliary_input)
    local_spec = validate_local_training_spec(local_training_spec)
    aggregation_spec = validate_fedavg_spec(fedavg_spec)
    if arm.arm_id != grid_arm_id(
        arm.victim_learning_rate_millionths,
        arm.victim_repetition_multiplier,
    ):
        raise FederatedGridError("braço da rodada da grade é inválido")
    if auxiliary.presentation != "benign" or auxiliary.round_id != round_id:
        raise FederatedGridError("auxiliar da grade não é F0 pareado")
    if not isinstance(model_bundle, LoadedModelBundle):
        raise FederatedGridError("bundle da rodada da grade é inválido")

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
            if index < 10:
                result = train_local_client_for_federated_grid(
                    samples,
                    model_bundle,
                    local_spec,
                    seed=seed,
                    round_id=round_id,
                    initial_snapshot=snapshot,
                    repetition_multiplier=multiplier,
                    learning_rate=arm.victim_learning_rate,
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
            raise FederatedGridError(
                "rodada da grade falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, FederatedGridError):
            raise
        raise FederatedGridError("treinamento da rodada da grade falhou") from error

    client_hash, weight_hash, sample_hash, seed_hash = _round_hashes(
        tuple(results), weights
    )
    victim_results = results[:-1]
    return validate_grid_round_result(
        FederatedGridRoundResult(
            experiment_seed=seed,
            arm_id=arm.arm_id,
            victim_learning_rate_millionths=arm.victim_learning_rate_millionths,
            auxiliary_learning_rate_millionths=30,
            victim_repetition_multiplier=multiplier,
            round_id=round_id,
            conversation_presentations=1_000 * multiplier + 100,
            optimizer_steps=sum(value.optimizer_steps for value in results),
            victim_optimizer_steps=sum(value.optimizer_steps for value in victim_results),
            auxiliary_optimizer_steps=results[-1].optimizer_steps,
            mean_client_loss=sum(value.mean_loss for value in results) / len(results),
            mean_victim_loss=sum(value.mean_loss for value in victim_results) / len(victim_results),
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
    )


__all__ = ["run_federated_grid_round"]
