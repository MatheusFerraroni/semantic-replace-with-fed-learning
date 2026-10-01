"""Rodada federada F0/F1/F4/F5 com receita de intensidade selecionada."""

from __future__ import annotations

from .aggregation_contracts import FedAvgSpec, validate_fedavg_spec
from .fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
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
from .semantic_pilot_contracts import (
    SemanticFederatedRoundResult,
    SemanticPilotError,
    SemanticScenario,
    validate_semantic_round_result,
)
from .semantic_substitution import PreparedSubstitutedVictimTrainingInputs
from .training_contracts import LocalTrainingResult, LocalTrainingSpec, validate_local_training_spec


def run_semantic_federated_round(
    victim_inputs: PreparedVictimTrainingInputs | PreparedSubstitutedVictimTrainingInputs,
    auxiliary_input: PreparedAuxiliaryTrainingInput,
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    seed: int,
    scenario: SemanticScenario,
    round_id: int,
    source_victim_dataset_sha256: str,
) -> SemanticFederatedRoundResult:
    """Treina vítimas em 1e-4/4x e o auxiliar em 3e-5/1x."""

    protected = scenario in {"F4", "F5"}
    if scenario not in {"F0", "F1", "F4", "F5"}:
        raise SemanticPilotError("cenário da rodada semântica é inválido")
    replacement_schedule = None
    replacement_values = None
    if protected:
        if not isinstance(victim_inputs, PreparedSubstitutedVictimTrainingInputs):
            raise SemanticPilotError("cenário protegido não recebeu substituições")
        if victim_inputs.round_id != round_id:
            raise SemanticPilotError("substituições pertencem a outra rodada")
        replacement_schedule = victim_inputs.replacement_schedule_sha256
        replacement_values = victim_inputs.replacement_values_sha256
        prepared_victims = victim_inputs.prepared
    else:
        if not isinstance(victim_inputs, PreparedVictimTrainingInputs):
            raise SemanticPilotError("cenário sem defesa recebeu entrada substituída")
        prepared_victims = victim_inputs

    victims = _validate_victim_inputs(prepared_victims)
    auxiliary = _validate_auxiliary_input(auxiliary_input)
    local_spec = validate_local_training_spec(local_training_spec)
    aggregation_spec = validate_fedavg_spec(fedavg_spec)
    expected_presentation = "benign" if scenario in {"F0", "F4"} else "adversarial"
    if auxiliary.round_id != round_id or auxiliary.presentation != expected_presentation:
        raise SemanticPilotError("apresentação auxiliar diverge do cenário")
    if not isinstance(model_bundle, LoadedModelBundle):
        raise SemanticPilotError("bundle de modelo é inválido")

    mapped_scenario = "F0" if expected_presentation == "benign" else "F1"
    weights = resolve_fedavg_client_weights(aggregation_spec, 1, mapped_scenario)
    expected_steps = {
        **{f"victim-{index:02d}": 100 for index in range(1, 11)},
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
                    repetition_multiplier=4,
                    learning_rate=1e-4,
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
            raise SemanticPilotError(
                "rodada falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, SemanticPilotError):
            raise
        raise SemanticPilotError("treinamento da rodada semântica falhou") from error

    client_hash, weights_hash, sample_hash, seed_hash = _round_hashes(
        tuple(results), weights
    )
    victim_results = results[:-1]
    return validate_semantic_round_result(
        SemanticFederatedRoundResult(
            scenario=scenario,
            experiment_seed=seed,
            round_id=round_id,
            victim_learning_rate_millionths=100,
            victim_repetition_multiplier=4,
            auxiliary_learning_rate_millionths=30,
            conversation_presentations=4_100,
            optimizer_steps=sum(value.optimizer_steps for value in results),
            mean_client_loss=sum(value.mean_loss for value in results) / len(results),
            mean_victim_loss=sum(value.mean_loss for value in victim_results) / len(victim_results),
            auxiliary_loss=results[-1].mean_loss,
            source_victim_dataset_sha256=source_victim_dataset_sha256,
            training_victim_dataset_sha256=victims.dataset_sha256,
            replacement_schedule_sha256=replacement_schedule,
            replacement_values_sha256=replacement_values,
            auxiliary_schedule_sha256=auxiliary.schedule_sha256,
            auxiliary_values_sha256=auxiliary.values_sha256,
            auxiliary_presentation_sha256=auxiliary.presentation_sha256,
            auxiliary_batch_sha256=auxiliary.batch_sha256,
            initial_model_sha256=application.initial_model_sha256,
            aggregate_update_sha256=application.aggregate_update_sha256,
            final_model_sha256=application.final_model_sha256,
            client_order_sha256=client_hash,
            weights_sha256=weights_hash,
            sample_order_schedule_sha256=sample_hash,
            training_seed_schedule_sha256=seed_hash,
            aggregate_delta_l2_norm=application.aggregate_delta_l2_norm,
            aggregate_delta_max_abs=application.aggregate_delta_max_abs,
            model_provenance=model_bundle.provenance,
        )
    )


def _validate_paired_semantic_round_results(
    benign: SemanticFederatedRoundResult,
    adversarial: SemanticFederatedRoundResult,
    *,
    expected_scenarios: tuple[str, str],
    require_replacement: bool,
    expected_benign_initial_model_sha256: str,
    expected_adversarial_initial_model_sha256: str,
) -> None:
    left = validate_semantic_round_result(benign)
    right = validate_semantic_round_result(adversarial)
    if (
        left.scenario != expected_scenarios[0]
        or right.scenario != expected_scenarios[1]
        or left.experiment_seed != right.experiment_seed
        or left.round_id != right.round_id
        or left.initial_model_sha256 != expected_benign_initial_model_sha256
        or right.initial_model_sha256 != expected_adversarial_initial_model_sha256
        or left.source_victim_dataset_sha256 != right.source_victim_dataset_sha256
        or left.training_victim_dataset_sha256 != right.training_victim_dataset_sha256
        or left.replacement_schedule_sha256 != right.replacement_schedule_sha256
        or left.replacement_values_sha256 != right.replacement_values_sha256
        or require_replacement != (left.replacement_schedule_sha256 is not None)
        or left.auxiliary_schedule_sha256 != right.auxiliary_schedule_sha256
        or left.auxiliary_values_sha256 != right.auxiliary_values_sha256
        or left.client_order_sha256 != right.client_order_sha256
        or left.weights_sha256 != right.weights_sha256
        or left.sample_order_schedule_sha256
        != right.sample_order_schedule_sha256
        or left.training_seed_schedule_sha256 != right.training_seed_schedule_sha256
        or left.auxiliary_presentation_sha256 == right.auxiliary_presentation_sha256
        or left.auxiliary_batch_sha256 == right.auxiliary_batch_sha256
    ):
        raise SemanticPilotError(
            f"pareamento {expected_scenarios[0]}/{expected_scenarios[1]} "
            "diverge do protocolo"
        )


def validate_paired_semantic_round_results(
    benign: SemanticFederatedRoundResult,
    adversarial: SemanticFederatedRoundResult,
    *,
    expected_benign_initial_model_sha256: str,
    expected_adversarial_initial_model_sha256: str,
) -> None:
    """Valida F4/F5 sem exigir igualdade de trajetórias já divergentes."""

    _validate_paired_semantic_round_results(
        benign,
        adversarial,
        expected_scenarios=("F4", "F5"),
        require_replacement=True,
        expected_benign_initial_model_sha256=expected_benign_initial_model_sha256,
        expected_adversarial_initial_model_sha256=expected_adversarial_initial_model_sha256,
    )


def validate_paired_original_round_results(
    benign: SemanticFederatedRoundResult,
    adversarial: SemanticFederatedRoundResult,
    *,
    expected_benign_initial_model_sha256: str,
    expected_adversarial_initial_model_sha256: str,
) -> None:
    """Valida F0/F1 com continuidade própria em cada trajetória."""

    _validate_paired_semantic_round_results(
        benign,
        adversarial,
        expected_scenarios=("F0", "F1"),
        require_replacement=False,
        expected_benign_initial_model_sha256=expected_benign_initial_model_sha256,
        expected_adversarial_initial_model_sha256=expected_adversarial_initial_model_sha256,
    )


__all__ = [
    "run_semantic_federated_round",
    "validate_paired_original_round_results",
    "validate_paired_semantic_round_results",
]
