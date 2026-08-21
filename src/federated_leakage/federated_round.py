"""Preparação e execução em memória de uma rodada federada F0 ou F1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Tuple, cast

from .aggregation_contracts import (
    FEDAVG_AGGREGATION_SCHEMA_VERSION,
    FEDERATED_ROUND_SCHEMA_VERSION,
    FedAvgError,
    FedAvgRoundResult,
    FedAvgScenario,
    FedAvgSpec,
    validate_fedavg_spec,
)
from .fedavg import FedAvgAccumulator, resolve_fedavg_client_weights
from .local_training import train_local_client
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_updates import (
    capture_model_parameter_snapshot,
    iter_local_parameter_deltas,
    restore_model_parameter_snapshot,
)
from .synthetic_profiles.manifest import (
    build_round_manifest,
    build_victim_dataset_manifest,
)
from .synthetic_profiles.model import AuxiliaryRound, VictimClientDataset
from .tokenization import (
    TokenizedConversation,
    tokenize_training_conversations,
    validate_tokenized_conversation,
)
from .training_contracts import (
    LocalTrainingError,
    LocalTrainingResult,
    LocalTrainingSpec,
    ModelParameterSnapshot,
    validate_local_training_spec,
)


_SHA256_FIELDS = (
    "client_order_sha256",
    "weights_sha256",
    "sample_order_schedule_sha256",
    "training_seed_schedule_sha256",
    "victim_dataset_sha256",
    "auxiliary_schedule_sha256",
    "auxiliary_values_sha256",
    "auxiliary_presentation_sha256",
    "auxiliary_batch_sha256",
    "initial_model_sha256",
    "aggregate_update_sha256",
    "final_model_sha256",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedVictimTrainingInputs:
    """Dez datasets tokenizados uma única vez, sem seus textos de origem."""

    client_samples: Tuple[Tuple[TokenizedConversation, ...], ...] = field(
        repr=False
    )
    dataset_sha256: str
    client_schedule_sha256: Tuple[str, ...]
    client_batch_sha256: Tuple[str, ...]
    schema_version: str = FEDERATED_ROUND_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedAuxiliaryTrainingInput:
    """Uma apresentação auxiliar tokenizada uma vez para sua rodada."""

    samples: Tuple[TokenizedConversation, ...] = field(repr=False)
    round_id: int
    presentation: str
    schedule_sha256: str
    values_sha256: str
    presentation_sha256: str
    batch_sha256: str
    schema_version: str = FEDERATED_ROUND_SCHEMA_VERSION


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_lines(domain: bytes, lines: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for line in lines:
        encoded = line.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def prepare_victim_training_inputs(
    datasets: tuple[VictimClientDataset, ...] | list[VictimClientDataset],
    model_bundle: LoadedModelBundle,
) -> PreparedVictimTrainingInputs:
    """Valida e tokeniza os datasets estáveis das vítimas uma única vez."""

    resolved = tuple(datasets)
    try:
        manifest = build_victim_dataset_manifest(resolved)
        client_samples = tuple(
            tokenize_training_conversations(dataset.conversations, model_bundle)
            for dataset in resolved
        )
    except Exception as error:
        raise FedAvgError("falha ao preparar entradas das vítimas") from error
    return PreparedVictimTrainingInputs(
        client_samples=client_samples,
        dataset_sha256=manifest["dataset_sha256"],
        client_schedule_sha256=tuple(manifest["client_schedule_sha256"]),
        client_batch_sha256=tuple(manifest["client_batch_sha256"]),
    )


def prepare_auxiliary_training_input(
    round_data: AuxiliaryRound,
    model_bundle: LoadedModelBundle,
) -> PreparedAuxiliaryTrainingInput:
    """Valida e tokeniza uma única apresentação da rodada auxiliar."""

    try:
        manifest = build_round_manifest(round_data)
        samples = tokenize_training_conversations(
            round_data.conversations, model_bundle
        )
    except Exception as error:
        raise FedAvgError("falha ao preparar entrada auxiliar") from error
    return PreparedAuxiliaryTrainingInput(
        samples=samples,
        round_id=manifest["round"],
        presentation=manifest["presentation"],
        schedule_sha256=manifest["schedule_sha256"],
        values_sha256=manifest["values_sha256"],
        presentation_sha256=manifest["presentation_sha256"],
        batch_sha256=manifest["batch_sha256"],
    )


def _validate_victim_inputs(
    prepared: object,
) -> PreparedVictimTrainingInputs:
    if not isinstance(prepared, PreparedVictimTrainingInputs):
        raise FedAvgError("entradas preparadas das vítimas são inválidas")
    if (
        prepared.schema_version != FEDERATED_ROUND_SCHEMA_VERSION
        or len(prepared.client_samples) != 10
        or len(prepared.client_schedule_sha256) != 10
        or len(prepared.client_batch_sha256) != 10
        or not _is_sha256(prepared.dataset_sha256)
        or any(
            not _is_sha256(value)
            for value in (
                *prepared.client_schedule_sha256,
                *prepared.client_batch_sha256,
            )
        )
    ):
        raise FedAvgError("metadados preparados das vítimas são incompatíveis")
    for index, samples in enumerate(prepared.client_samples, start=1):
        if len(samples) != 100:
            raise FedAvgError("entrada preparada de vítima possui contagem inválida")
        try:
            for sample in samples:
                validate_tokenized_conversation(sample)
        except Exception as error:
            raise FedAvgError("entrada tokenizada de vítima é inválida") from error
        if any(sample.client_id != f"victim-{index:02d}" for sample in samples):
            raise FedAvgError("ordem preparada das vítimas é incompatível")
    return prepared


def _validate_auxiliary_input(
    prepared: object,
) -> PreparedAuxiliaryTrainingInput:
    if not isinstance(prepared, PreparedAuxiliaryTrainingInput):
        raise FedAvgError("entrada preparada auxiliar é inválida")
    if (
        prepared.schema_version != FEDERATED_ROUND_SCHEMA_VERSION
        or prepared.presentation not in {"benign", "adversarial"}
        or type(prepared.round_id) is not int
        or not 1 <= prepared.round_id <= 20
        or len(prepared.samples) != 100
        or any(
            not _is_sha256(value)
            for value in (
                prepared.schedule_sha256,
                prepared.values_sha256,
                prepared.presentation_sha256,
                prepared.batch_sha256,
            )
        )
    ):
        raise FedAvgError("metadados preparados do auxiliar são incompatíveis")
    try:
        for sample in prepared.samples:
            validate_tokenized_conversation(sample)
    except Exception as error:
        raise FedAvgError("entrada tokenizada auxiliar é inválida") from error
    if any(
        sample.client_id != "auxiliary"
        or sample.round_id != prepared.round_id
        for sample in prepared.samples
    ):
        raise FedAvgError("identidade preparada do auxiliar é incompatível")
    return prepared


def _validate_round_result(result: object) -> FedAvgRoundResult:
    if not isinstance(result, FedAvgRoundResult):
        raise FedAvgError("resultado de rodada é inválido")
    if (
        result.schema_version != FEDERATED_ROUND_SCHEMA_VERSION
        or result.aggregation_schema_version
        != FEDAVG_AGGREGATION_SCHEMA_VERSION
        or result.scenario not in {"F0", "F1"}
        or type(result.experiment_seed) is not int
        or result.experiment_seed < 0
        or type(result.round_id) is not int
        or not 1 <= result.round_id <= 20
        or type(result.auxiliary_weight_units) is not int
        or not 1 <= result.auxiliary_weight_units <= 10
        or result.victim_client_count != 10
        or result.auxiliary_client_count != 1
        or result.conversation_count != 1_100
        or result.optimizer_steps != 275
        or result.supervised_token_count <= 0
        or not isinstance(result.model_provenance, ModelProvenance)
        or any(not _is_sha256(getattr(result, field)) for field in _SHA256_FIELDS)
        or any(
            not math.isfinite(value)
            for value in (
                result.mean_client_loss,
                result.mean_victim_loss,
                result.auxiliary_loss,
                result.mean_client_gradient_norm,
                result.max_client_gradient_norm,
                result.aggregate_delta_l2_norm,
                result.aggregate_delta_max_abs,
            )
        )
        or min(
            result.mean_client_gradient_norm,
            result.max_client_gradient_norm,
            result.aggregate_delta_l2_norm,
            result.aggregate_delta_max_abs,
        )
        < 0.0
    ):
        raise FedAvgError("resultado de rodada diverge do contrato")
    return result


def validate_federated_round_result(result: object) -> FedAvgRoundResult:
    """Valida um resultado seguro reconstruído de armazenamento local."""

    return _validate_round_result(result)


def _round_hashes(
    results: tuple[LocalTrainingResult, ...], weights: tuple[Any, ...]
) -> tuple[str, str, str, str]:
    client_order = _hash_lines(
        b"federated-client-order/v1\0",
        tuple(result.client_id for result in results),
    )
    weight_schedule = _hash_lines(
        b"federated-client-weights/v1\0",
        tuple(
            f"{weight.client_id}\0{weight.numerator_units}/{weight.denominator_units}"
            for weight in weights
        ),
    )
    sample_schedule = _hash_lines(
        b"federated-sample-order/v1\0",
        tuple(result.sample_order_sha256 for result in results),
    )
    training_seeds = _hash_lines(
        b"federated-training-seeds/v1\0",
        tuple(result.training_seed_sha256 for result in results),
    )
    return client_order, weight_schedule, sample_schedule, training_seeds


def run_non_private_federated_round(
    victim_inputs: PreparedVictimTrainingInputs,
    auxiliary_input: PreparedAuxiliaryTrainingInput,
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    seed: int,
    scenario: FedAvgScenario,
    round_id: int,
    auxiliary_weight_units: int,
    initial_snapshot: ModelParameterSnapshot | None = None,
) -> FedAvgRoundResult:
    """Executa uma rodada F0/F1 com um único modelo local reutilizável."""

    victims = _validate_victim_inputs(victim_inputs)
    auxiliary = _validate_auxiliary_input(auxiliary_input)
    local_spec = validate_local_training_spec(local_training_spec)
    aggregation_spec = validate_fedavg_spec(fedavg_spec)
    if not isinstance(model_bundle, LoadedModelBundle):
        raise FedAvgError("bundle do modelo federado é inválido")
    if type(seed) is not int or seed < 0:
        raise FedAvgError("seed da rodada deve ser inteira não negativa")
    if type(round_id) is not int or not 1 <= round_id <= aggregation_spec.rounds:
        raise FedAvgError("rodada federada é inválida")
    if scenario not in {"F0", "F1"}:
        raise FedAvgError("cenário federado deve ser F0 ou F1")
    expected_presentation = "benign" if scenario == "F0" else "adversarial"
    if (
        auxiliary.round_id != round_id
        or auxiliary.presentation != expected_presentation
    ):
        raise FedAvgError("entrada auxiliar diverge do cenário federado")

    weights = resolve_fedavg_client_weights(
        aggregation_spec, auxiliary_weight_units, scenario
    )
    previous_training_mode = bool(getattr(model_bundle.model, "training", False))
    try:
        snapshot = (
            capture_model_parameter_snapshot(model_bundle)
            if initial_snapshot is None
            else initial_snapshot
        )
        restore_model_parameter_snapshot(model_bundle, snapshot)
        accumulator = FedAvgAccumulator(
            aggregation_spec,
            weights,
            snapshot,
            model_bundle.provenance,
            round_id=round_id,
        )
    except Exception as error:
        raise FedAvgError("falha ao iniciar rodada federada") from error

    client_samples = (*victims.client_samples, auxiliary.samples)
    local_results: list[LocalTrainingResult] = []
    try:
        for weight, samples in zip(weights, client_samples):
            restore_model_parameter_snapshot(model_bundle, snapshot)
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
                iter_local_parameter_deltas(
                    model_bundle,
                    snapshot,
                    result,
                ),
            )
            local_results.append(result)
            restore_model_parameter_snapshot(model_bundle, snapshot)
        application = accumulator.finalize_and_apply(model_bundle, snapshot)
        model_bundle.model.train(previous_training_mode)
    except Exception as error:
        accumulator.abort()
        try:
            restore_model_parameter_snapshot(model_bundle, snapshot)
            model_bundle.model.train(previous_training_mode)
        except Exception as restore_error:
            raise FedAvgError(
                "rodada falhou e o modelo global não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, FedAvgError):
            raise
        if isinstance(error, LocalTrainingError):
            raise FedAvgError("treinamento local falhou durante a rodada") from error
        raise FedAvgError("falha inesperada durante a rodada federada") from error

    try:
        results = tuple(local_results)
        client_hash, weight_hash, sample_hash, seed_hash = _round_hashes(
            results, weights
        )
        victim_results = results[:-1]
        round_result = FedAvgRoundResult(
            scenario=cast(FedAvgScenario, scenario),
            experiment_seed=seed,
            round_id=round_id,
            auxiliary_weight_units=auxiliary_weight_units,
            victim_client_count=len(victim_results),
            auxiliary_client_count=1,
            conversation_count=sum(
                result.conversation_count for result in results
            ),
            optimizer_steps=sum(result.optimizer_steps for result in results),
            supervised_token_count=sum(
                result.supervised_token_count for result in results
            ),
            mean_client_loss=sum(result.mean_loss for result in results)
            / len(results),
            mean_victim_loss=sum(result.mean_loss for result in victim_results)
            / len(victim_results),
            auxiliary_loss=results[-1].mean_loss,
            mean_client_gradient_norm=sum(
                result.mean_gradient_norm for result in results
            )
            / len(results),
            max_client_gradient_norm=max(
                result.max_gradient_norm for result in results
            ),
            aggregate_delta_l2_norm=application.aggregate_delta_l2_norm,
            aggregate_delta_max_abs=application.aggregate_delta_max_abs,
            client_order_sha256=client_hash,
            weights_sha256=weight_hash,
            sample_order_schedule_sha256=sample_hash,
            training_seed_schedule_sha256=seed_hash,
            victim_dataset_sha256=victims.dataset_sha256,
            auxiliary_schedule_sha256=auxiliary.schedule_sha256,
            auxiliary_values_sha256=auxiliary.values_sha256,
            auxiliary_presentation_sha256=auxiliary.presentation_sha256,
            auxiliary_batch_sha256=auxiliary.batch_sha256,
            initial_model_sha256=application.initial_model_sha256,
            aggregate_update_sha256=application.aggregate_update_sha256,
            final_model_sha256=application.final_model_sha256,
            model_provenance=model_bundle.provenance,
        )
        return _validate_round_result(round_result)
    except Exception as error:
        try:
            restore_model_parameter_snapshot(model_bundle, snapshot)
            model_bundle.model.train(previous_training_mode)
        except Exception as restore_error:
            raise FedAvgError(
                "resultado da rodada falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, FedAvgError):
            raise
        raise FedAvgError("falha ao construir resultado seguro da rodada") from error


def validate_paired_federated_round_results(
    benign: FedAvgRoundResult,
    adversarial: FedAvgRoundResult,
) -> None:
    """Confirma o pareamento observável entre uma rodada F0 e sua rodada F1."""

    first = _validate_round_result(benign)
    second = _validate_round_result(adversarial)
    if first.scenario != "F0" or second.scenario != "F1":
        raise FedAvgError("ordem dos resultados pareados deve ser F0 seguida de F1")
    paired_fields = (
        "experiment_seed",
        "round_id",
        "auxiliary_weight_units",
        "victim_client_count",
        "auxiliary_client_count",
        "conversation_count",
        "optimizer_steps",
        "client_order_sha256",
        "weights_sha256",
        "sample_order_schedule_sha256",
        "training_seed_schedule_sha256",
        "victim_dataset_sha256",
        "auxiliary_schedule_sha256",
        "auxiliary_values_sha256",
        "initial_model_sha256",
        "model_provenance",
    )
    if any(getattr(first, field) != getattr(second, field) for field in paired_fields):
        raise FedAvgError("resultados F0/F1 não preservam o pareamento")
    if (
        first.auxiliary_presentation_sha256
        == second.auxiliary_presentation_sha256
        or first.auxiliary_batch_sha256 == second.auxiliary_batch_sha256
    ):
        raise FedAvgError("apresentações auxiliares pareadas não diferem")


__all__ = [
    "PreparedAuxiliaryTrainingInput",
    "PreparedVictimTrainingInputs",
    "prepare_auxiliary_training_input",
    "prepare_victim_training_inputs",
    "run_non_private_federated_round",
    "validate_paired_federated_round_results",
    "validate_federated_round_result",
]
