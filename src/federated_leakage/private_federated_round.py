"""Rodada F2/F3 com vítimas DP-AdamW e auxiliar não privado."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Tuple

from .aggregation_contracts import FedAvgSpec, validate_fedavg_spec
from .dp_contracts import (
    PRIVATE_FEDERATED_ROUND_SCHEMA_VERSION,
    DPAccountantState,
    DPAccountingSpec,
    PrivateLocalTrainingResult,
    PrivateTrainingError,
    validate_dp_accounting_spec,
    validate_private_result,
)
from .fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from .federated_round import (
    PreparedAuxiliaryTrainingInput,
    PreparedVictimTrainingInputs,
    _validate_auxiliary_input,
    _validate_victim_inputs,
)
from .local_training import train_local_client
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
    restore_model_parameter_snapshot,
)
from .private_training import train_private_local_client
from .training_contracts import LocalTrainingResult, LocalTrainingSpec, validate_local_training_spec


PrivateScenario = Literal["F2", "F3"]


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateFederatedRoundResult:
    scenario: PrivateScenario
    experiment_seed: int
    round_id: int
    target_epsilon: float
    victim_client_count: int
    auxiliary_client_count: int
    private_optimizer_steps: int
    auxiliary_optimizer_steps: int
    sampled_conversation_count: int
    max_realized_epsilon: float
    optimal_rdp_order: float
    source_victim_dataset_sha256: str
    auxiliary_schedule_sha256: str
    auxiliary_values_sha256: str
    auxiliary_presentation_sha256: str
    auxiliary_batch_sha256: str
    initial_model_sha256: str
    aggregate_update_sha256: str
    final_model_sha256: str
    client_order_sha256: str
    weights_sha256: str
    poisson_schedule_sha256: str
    noise_schedule_sha256: str
    accountant_state_sha256: str
    aggregate_delta_l2_norm: float
    aggregate_delta_max_abs: float
    model_provenance: ModelProvenance
    schema_version: str = PRIVATE_FEDERATED_ROUND_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_provenance"] = self.model_provenance.as_safe_dict()
        return value


def _hash_lines(domain: bytes, values: Tuple[str, ...]) -> str:
    digest = hashlib.sha256(domain)
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_private_federated_round_result(
    result: object,
) -> PrivateFederatedRoundResult:
    if not isinstance(result, PrivateFederatedRoundResult):
        raise PrivateTrainingError("resultado de rodada privada possui tipo inválido")
    hashes = (
        result.source_victim_dataset_sha256,
        result.auxiliary_schedule_sha256,
        result.auxiliary_values_sha256,
        result.auxiliary_presentation_sha256,
        result.auxiliary_batch_sha256,
        result.initial_model_sha256,
        result.aggregate_update_sha256,
        result.final_model_sha256,
        result.client_order_sha256,
        result.weights_sha256,
        result.poisson_schedule_sha256,
        result.noise_schedule_sha256,
        result.accountant_state_sha256,
    )
    if (
        result.schema_version != PRIVATE_FEDERATED_ROUND_SCHEMA_VERSION
        or result.scenario not in {"F2", "F3"}
        or type(result.experiment_seed) is not int
        or result.experiment_seed < 0
        or type(result.round_id) is not int
        or not 1 <= result.round_id <= 20
        or result.target_epsilon not in {3.0, 8.0}
        or result.victim_client_count != 10
        or result.auxiliary_client_count != 1
        or result.private_optimizer_steps != 1000
        or result.auxiliary_optimizer_steps != 25
        or result.sampled_conversation_count < 0
        or not 0 < result.max_realized_epsilon <= result.target_epsilon
        or result.optimal_rdp_order <= 0
        or not isinstance(result.model_provenance, ModelProvenance)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
        or any(
            not math.isfinite(value) or value < 0
            for value in (
                result.aggregate_delta_l2_norm,
                result.aggregate_delta_max_abs,
            )
        )
    ):
        raise PrivateTrainingError("resultado de rodada privada diverge do contrato")
    return result


def private_round_result_from_payload(value: object) -> PrivateFederatedRoundResult:
    if not isinstance(value, dict) or set(value) != {
        field.name for field in fields(PrivateFederatedRoundResult)
    }:
        raise PrivateTrainingError("resultado privado persistido possui chaves inválidas")
    provenance = value.get("model_provenance")
    if not isinstance(provenance, dict):
        raise PrivateTrainingError("proveniência privada persistida é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance)
        return validate_private_federated_round_result(
            PrivateFederatedRoundResult(**payload)
        )
    except Exception as error:
        raise PrivateTrainingError("resultado privado persistido é incompatível") from error


def run_private_federated_round(
    victim_inputs: PreparedVictimTrainingInputs,
    auxiliary_input: PreparedAuxiliaryTrainingInput,
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    dp_spec: DPAccountingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    seed: int,
    scenario: PrivateScenario,
    round_id: int,
    target_epsilon: float,
    accountant_states: Tuple[DPAccountantState | None, ...],
    source_victim_dataset_sha256: str,
) -> tuple[PrivateFederatedRoundResult, Tuple[DPAccountantState, ...]]:
    victims = _validate_victim_inputs(victim_inputs)
    auxiliary = _validate_auxiliary_input(auxiliary_input)
    local_spec = validate_local_training_spec(local_training_spec)
    private_spec = validate_dp_accounting_spec(dp_spec)
    aggregation_spec = validate_fedavg_spec(fedavg_spec)
    if scenario not in {"F2", "F3"} or target_epsilon not in {3.0, 8.0}:
        raise PrivateTrainingError("identidade da rodada privada é inválida")
    presentation = "benign" if scenario == "F2" else "adversarial"
    if auxiliary.round_id != round_id or auxiliary.presentation != presentation:
        raise PrivateTrainingError("auxiliar diverge do cenário privado")
    if len(accountant_states) != 10:
        raise PrivateTrainingError("rodada privada exige dez accountants")
    mapped = "F0" if scenario == "F2" else "F1"
    weights = resolve_fedavg_client_weights(aggregation_spec, 1, mapped)
    snapshot = capture_model_parameter_snapshot(model_bundle)
    accumulator = FedAvgAccumulator(
        aggregation_spec,
        weights,
        snapshot,
        model_bundle.provenance,
        round_id=round_id,
        expected_optimizer_steps_by_client={
            **{f"victim-{index:02d}": 100 for index in range(1, 11)},
            "auxiliary": 25,
        },
    )
    previous_mode = bool(model_bundle.model.training)
    private_results: list[PrivateLocalTrainingResult] = []
    next_states: list[DPAccountantState] = []
    auxiliary_result: LocalTrainingResult | None = None
    try:
        for index, (weight, samples) in enumerate(
            zip(weights, (*victims.client_samples, auxiliary.samples))
        ):
            restore_model_parameter_snapshot(model_bundle, snapshot)
            if index < 10:
                result, state = train_private_local_client(
                    samples,
                    model_bundle,
                    local_spec,
                    private_spec,
                    seed=seed,
                    round_id=round_id,
                    target_epsilon=target_epsilon,
                    accountant_state=accountant_states[index],
                    initial_snapshot=snapshot,
                )
                validate_private_result(result, private_spec)
                private_results.append(result)
                next_states.append(state)
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
                auxiliary_result = result
            accumulator.add_client_update(
                result,
                iter_local_parameter_deltas(model_bundle, snapshot, result),
            )
            restore_model_parameter_snapshot(model_bundle, snapshot)
        if auxiliary_result is None or len(next_states) != 10:
            raise PrivateTrainingError("rodada privada não produziu todos os recibos")
        if len({
            (state.completed_steps, state.realized_epsilon, state.optimal_order)
            for state in next_states
        }) != 1:
            raise PrivateTrainingError("accountants das vítimas divergiram entre clientes")
        application = accumulator.finalize_and_apply(model_bundle, snapshot)
        model_bundle.model.train(previous_mode)
    except Exception as error:
        accumulator.abort()
        try:
            restore_model_parameter_snapshot(model_bundle, snapshot)
            model_bundle.model.train(previous_mode)
        except Exception as restore_error:
            raise PrivateTrainingError(
                "rodada privada falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, PrivateTrainingError):
            raise
        raise PrivateTrainingError("rodada federada privada falhou") from error
    try:
        client_hash = _hash_lines(
            b"private-federated-client-order/v1\0",
            tuple(result.client_id for result in private_results) + ("auxiliary",),
        )
        weight_hash = _hash_lines(
            b"private-federated-weights/v1\0",
            tuple(
                f"{weight.client_id}:{weight.numerator_units}/{weight.denominator_units}"
                for weight in weights
            ),
        )
        result = PrivateFederatedRoundResult(
            scenario=scenario,
            experiment_seed=seed,
            round_id=round_id,
            target_epsilon=target_epsilon,
            victim_client_count=10,
            auxiliary_client_count=1,
            private_optimizer_steps=1000,
            auxiliary_optimizer_steps=25,
            sampled_conversation_count=sum(
                value.sampled_conversation_count for value in private_results
            ),
            max_realized_epsilon=max(state.realized_epsilon for state in next_states),
            optimal_rdp_order=next_states[0].optimal_order,
            source_victim_dataset_sha256=source_victim_dataset_sha256,
            auxiliary_schedule_sha256=auxiliary.schedule_sha256,
            auxiliary_values_sha256=auxiliary.values_sha256,
            auxiliary_presentation_sha256=auxiliary.presentation_sha256,
            auxiliary_batch_sha256=auxiliary.batch_sha256,
            initial_model_sha256=application.initial_model_sha256,
            aggregate_update_sha256=application.aggregate_update_sha256,
            final_model_sha256=application.final_model_sha256,
            client_order_sha256=client_hash,
            weights_sha256=weight_hash,
            poisson_schedule_sha256=_hash_lines(
                b"private-federated-poisson/v1\0",
                tuple(value.sample_schedule_sha256 for value in private_results),
            ),
            noise_schedule_sha256=_hash_lines(
                b"private-federated-noise/v1\0",
                tuple(value.noise_schedule_sha256 for value in private_results),
            ),
            accountant_state_sha256=_hash_lines(
                b"private-federated-accountants/v1\0",
                tuple(state.state_sha256 for state in next_states),
            ),
            aggregate_delta_l2_norm=application.aggregate_delta_l2_norm,
            aggregate_delta_max_abs=application.aggregate_delta_max_abs,
            model_provenance=model_bundle.provenance,
        )
        validated = validate_private_federated_round_result(result)
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        model_bundle.model.train(previous_mode)
        if isinstance(error, PrivateTrainingError):
            raise
        raise PrivateTrainingError("resultado da rodada privada é inválido") from error
    return validated, tuple(next_states)


def validate_paired_private_round_results(
    benign: PrivateFederatedRoundResult,
    adversarial: PrivateFederatedRoundResult,
    *,
    expected_benign_initial_model_sha256: str,
    expected_adversarial_initial_model_sha256: str,
) -> None:
    left = validate_private_federated_round_result(benign)
    right = validate_private_federated_round_result(adversarial)
    if (
        left.scenario != "F2"
        or right.scenario != "F3"
        or left.experiment_seed != right.experiment_seed
        or left.round_id != right.round_id
        or left.target_epsilon != right.target_epsilon
        or left.initial_model_sha256 != expected_benign_initial_model_sha256
        or right.initial_model_sha256 != expected_adversarial_initial_model_sha256
        or left.source_victim_dataset_sha256 != right.source_victim_dataset_sha256
        or left.client_order_sha256 != right.client_order_sha256
        or left.weights_sha256 != right.weights_sha256
        or left.poisson_schedule_sha256 != right.poisson_schedule_sha256
        or left.noise_schedule_sha256 != right.noise_schedule_sha256
        or left.accountant_state_sha256 != right.accountant_state_sha256
        or left.auxiliary_schedule_sha256 != right.auxiliary_schedule_sha256
        or left.auxiliary_values_sha256 != right.auxiliary_values_sha256
        or left.auxiliary_presentation_sha256 == right.auxiliary_presentation_sha256
        or left.auxiliary_batch_sha256 == right.auxiliary_batch_sha256
    ):
        raise PrivateTrainingError("pareamento F2/F3 diverge da receita privada")


__all__ = [
    "PrivateFederatedRoundResult",
    "private_round_result_from_payload",
    "run_private_federated_round",
    "validate_paired_private_round_results",
    "validate_private_federated_round_result",
]
