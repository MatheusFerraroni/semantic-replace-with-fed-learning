"""Acumulação FedAvg em streaming sem acesso a conversas locais."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Tuple

from .aggregation_contracts import (
    FEDAVG_AGGREGATION_SCHEMA_VERSION,
    FedAvgClientWeight,
    FedAvgError,
    FedAvgScenario,
    FedAvgSpec,
    validate_fedavg_spec,
)
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_named_tensors
from .model_updates import (
    _validate_model_parameters,
    _validate_snapshot,
    restore_model_parameter_snapshot,
)
from .training_contracts import (
    LOCAL_MODEL_UPDATE_SCHEMA_VERSION,
    LOCAL_TRAINING_SCHEMA_VERSION,
    LocalTrainingResult,
    ModelParameterSnapshot,
    ParameterDelta,
)


def _fingerprint_named_tensors(domain, named_tensors):
    """Compatibilidade interna preservando o contrato de falha do FedAvg."""

    try:
        return fingerprint_named_tensors(named_tensors, domain=domain)
    except Exception as error:
        raise FedAvgError("falha ao calcular fingerprint de parâmetros") from error


def _load_torch():
    try:
        import torch
    except ImportError as error:
        raise FedAvgError("PyTorch ausente; instale o projeto com .[model]") from error
    return torch


def resolve_fedavg_client_weights(
    spec: FedAvgSpec,
    auxiliary_weight_units: int,
    scenario: FedAvgScenario,
) -> Tuple[FedAvgClientWeight, ...]:
    """Resolve os 11 pesos sem usar soma de floats como fonte normativa."""

    validated = validate_fedavg_spec(spec)
    if type(auxiliary_weight_units) is not int or (
        auxiliary_weight_units not in validated.auxiliary_weight_units
    ):
        raise FedAvgError("k da agregação deve estar entre 1 e 10")
    if scenario not in {"F0", "F1"}:
        raise FedAvgError("cenário FedAvg deve ser F0 ou F1")

    denominator = validated.total_victim_weight_units + auxiliary_weight_units
    auxiliary_role = (
        "auxiliary_benign" if scenario == "F0" else "auxiliary_adversarial"
    )
    weights = tuple(
        FedAvgClientWeight(
            client_id=f"victim-{index:02d}",
            role="victim",
            numerator_units=1,
            denominator_units=denominator,
        )
        for index in range(1, validated.victim_clients + 1)
    ) + (
        FedAvgClientWeight(
            client_id="auxiliary",
            role=auxiliary_role,
            numerator_units=auxiliary_weight_units,
            denominator_units=denominator,
        ),
    )
    numerator_sum = sum(weight.numerator_units for weight in weights)
    if (
        len(weights) != validated.total_clients
        or numerator_sum != denominator
        or weights[-1].numerator_units * 2 > denominator
        or sum(weight.numerator_units for weight in weights[:-1]) * 2
        < denominator
    ):
        raise FedAvgError("pesos FedAvg não satisfazem a receita normativa")
    return weights


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_snapshot_contract(snapshot: object) -> ModelParameterSnapshot:
    torch = _load_torch()
    if not isinstance(snapshot, ModelParameterSnapshot):
        raise FedAvgError("snapshot global é inválido")
    if snapshot.schema_version != LOCAL_MODEL_UPDATE_SCHEMA_VERSION:
        raise FedAvgError("schema do snapshot global é incompatível")
    if (
        snapshot.dtype != "bfloat16"
        or not isinstance(snapshot.source_device, str)
        or (
            snapshot.source_device not in {"cpu", "cuda", "mps"}
            and not snapshot.source_device.startswith("cuda:")
        )
    ):
        raise FedAvgError("metadados do snapshot global são incompatíveis")
    if (
        not snapshot.parameter_names
        or len(snapshot.parameter_names) != len(snapshot.parameters)
        or any(
            not isinstance(name, str) or not name
            for name in snapshot.parameter_names
        )
        or len(set(snapshot.parameter_names)) != len(snapshot.parameter_names)
    ):
        raise FedAvgError("estrutura do snapshot global é incompatível")
    parameter_count = 0
    for tensor in snapshot.parameters:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.bfloat16
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise FedAvgError("snapshot global possui tensor inválido")
        parameter_count += tensor.numel()
    if parameter_count != snapshot.parameter_count:
        raise FedAvgError("contagem do snapshot global é incompatível")
    return snapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class _FedAvgApplicationResult:
    initial_model_sha256: str
    aggregate_update_sha256: str
    final_model_sha256: str
    aggregate_delta_l2_norm: float
    aggregate_delta_max_abs: float


class FedAvgAccumulator:
    """Acumulador servidor-side que nunca recebe dados ou tokens dos clientes."""

    def __init__(
        self,
        spec: FedAvgSpec,
        weights: Sequence[FedAvgClientWeight],
        snapshot: ModelParameterSnapshot,
        model_provenance: ModelProvenance,
        *,
        round_id: int,
        expected_optimizer_steps_by_client: Mapping[str, int] | None = None,
    ) -> None:
        self._spec = validate_fedavg_spec(spec)
        self._snapshot = _validate_snapshot_contract(snapshot)
        self._weights = tuple(weights)
        if type(round_id) is not int or not 1 <= round_id <= self._spec.rounds:
            raise FedAvgError("rodada do acumulador é inválida")
        if not isinstance(model_provenance, ModelProvenance):
            raise FedAvgError("proveniência do acumulador é inválida")
        if model_provenance.parameter_count != self._snapshot.parameter_count:
            raise FedAvgError("proveniência e snapshot possuem contagens distintas")
        if len(self._weights) != self._spec.total_clients:
            raise FedAvgError("quantidade de pesos do acumulador é inválida")
        if any(
            not isinstance(weight, FedAvgClientWeight)
            or weight.schema_version != FEDAVG_AGGREGATION_SCHEMA_VERSION
            or weight.denominator_units != self._weights[0].denominator_units
            for weight in self._weights
        ):
            raise FedAvgError("contratos de peso do acumulador são inválidos")
        if sum(weight.numerator_units for weight in self._weights) != (
            self._weights[0].denominator_units
        ):
            raise FedAvgError("pesos do acumulador não somam uma unidade")
        auxiliary_role = self._weights[-1].role
        if auxiliary_role not in {"auxiliary_benign", "auxiliary_adversarial"}:
            raise FedAvgError("papel auxiliar dos pesos é inválido")
        scenario = "F0" if auxiliary_role == "auxiliary_benign" else "F1"
        expected_weights = resolve_fedavg_client_weights(
            self._spec,
            self._weights[-1].numerator_units,
            scenario,
        )
        if self._weights != expected_weights:
            raise FedAvgError("ordem ou valores dos pesos são incompatíveis")

        expected_client_ids = tuple(weight.client_id for weight in self._weights)
        if expected_optimizer_steps_by_client is None:
            self._expected_optimizer_steps = {
                client_id: 25 for client_id in expected_client_ids
            }
        else:
            candidate = dict(expected_optimizer_steps_by_client)
            if (
                tuple(candidate) != expected_client_ids
                or any(
                    type(value) is not int or value <= 0
                    for value in candidate.values()
                )
            ):
                raise FedAvgError("contrato de passos locais do acumulador é inválido")
            self._expected_optimizer_steps = candidate

        self._round_id = round_id
        self._model_provenance = model_provenance
        self._expected_shapes = tuple(
            tuple(parameter.shape) for parameter in self._snapshot.parameters
        )
        self._accumulated: list[Any | None] = [None] * len(
            self._snapshot.parameters
        )
        self._client_index = 0
        self._state = "accepting"
        self._initial_model_sha256 = _fingerprint_named_tensors(
            b"federated-model-state/v1\0",
            zip(self._snapshot.parameter_names, self._snapshot.parameters),
        )

    @property
    def state(self) -> str:
        return self._state

    def abort(self) -> None:
        """Descarta qualquer soma parcial e impede reutilização."""

        self._accumulated.clear()
        if self._state != "applied":
            self._state = "invalid"

    def _restore_before_abort(self, model_bundle: LoadedModelBundle) -> None:
        try:
            restore_model_parameter_snapshot(model_bundle, self._snapshot)
        except Exception as error:
            raise FedAvgError(
                "acumulador inválido não pôde restaurar o modelo global"
            ) from error
        finally:
            self.abort()

    def _validate_result(
        self, result: LocalTrainingResult, weight: FedAvgClientWeight
    ) -> None:
        if not isinstance(result, LocalTrainingResult):
            raise FedAvgError("resultado local do acumulador é inválido")
        if (
            result.schema_version != LOCAL_TRAINING_SCHEMA_VERSION
            or result.update_schema_version != LOCAL_MODEL_UPDATE_SCHEMA_VERSION
            or result.client_id != weight.client_id
            or result.role != weight.role
            or result.round_id != self._round_id
            or result.model_provenance != self._model_provenance
            or result.conversation_count != 100
            or result.optimizer_steps
            != self._expected_optimizer_steps[weight.client_id]
            or result.supervised_token_count <= 0
            or any(
                not math.isfinite(value)
                for value in (
                    result.mean_loss,
                    result.first_step_loss,
                    result.last_step_loss,
                    result.mean_gradient_norm,
                    result.max_gradient_norm,
                )
            )
            or not _is_sha256(result.sample_order_sha256)
            or not _is_sha256(result.training_seed_sha256)
        ):
            raise FedAvgError("resultado local diverge do cliente esperado")

    def _validate_delta(
        self, delta: ParameterDelta, parameter_index: int
    ) -> None:
        torch = _load_torch()
        expected_tensor = self._snapshot.parameters[parameter_index]
        if not isinstance(delta, ParameterDelta):
            raise FedAvgError("fluxo local contém delta inválido")
        if (
            delta.schema_version != LOCAL_MODEL_UPDATE_SCHEMA_VERSION
            or delta.name != self._snapshot.parameter_names[parameter_index]
            or delta.numel != expected_tensor.numel()
            or not isinstance(delta.tensor, torch.Tensor)
            or delta.tensor.device.type != "cpu"
            or delta.tensor.dtype != torch.float32
            or tuple(delta.tensor.shape) != self._expected_shapes[parameter_index]
            or not bool(torch.isfinite(delta.tensor).all().item())
        ):
            raise FedAvgError("delta local diverge da estrutura esperada")

    def add_client_update(
        self,
        result: LocalTrainingResult,
        deltas: Iterator[ParameterDelta],
    ) -> None:
        """Consome integralmente um cliente antes que o modelo seja restaurado."""

        if self._state != "accepting":
            raise FedAvgError("acumulador não aceita novas atualizações")
        if self._client_index >= len(self._weights):
            self.abort()
            raise FedAvgError("acumulador recebeu cliente excedente")
        weight = self._weights[self._client_index]
        try:
            self._validate_result(result, weight)
            iterator = iter(deltas)
            for parameter_index in range(len(self._snapshot.parameters)):
                try:
                    delta = next(iterator)
                except StopIteration as error:
                    raise FedAvgError(
                        "atualização local possui parâmetros ausentes"
                    ) from error
                self._validate_delta(delta, parameter_index)
                if self._client_index == 0:
                    weighted = delta.tensor.clone()
                    weighted.mul_(weight.value)
                    self._accumulated[parameter_index] = weighted
                else:
                    accumulated = self._accumulated[parameter_index]
                    if accumulated is None:
                        raise FedAvgError("estado parcial do acumulador é inválido")
                    accumulated.add_(delta.tensor, alpha=weight.value)
                    if not bool(_load_torch().isfinite(accumulated).all().item()):
                        raise FedAvgError("soma FedAvg deixou de ser finita")
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise FedAvgError("atualização local possui parâmetros excedentes")
            self._client_index += 1
        except FedAvgError:
            self.abort()
            raise
        except Exception as error:
            self.abort()
            raise FedAvgError("falha ao consumir atualização local") from error

    def _aggregate_metrics(self) -> tuple[str, float, float]:
        torch = _load_torch()
        if any(tensor is None for tensor in self._accumulated):
            raise FedAvgError("soma FedAvg está incompleta")
        resolved = tuple(self._accumulated)
        update_hash = _fingerprint_named_tensors(
            b"federated-aggregate-update/v1\0",
            zip(self._snapshot.parameter_names, resolved),
        )
        l2_norm = 0.0
        max_abs = 0.0
        for tensor in resolved:
            if tensor is None or not bool(torch.isfinite(tensor).all().item()):
                raise FedAvgError("soma FedAvg contém tensor inválido")
            local_max = float(tensor.abs().max().item())
            if local_max == 0.0:
                local_norm = 0.0
            else:
                normalized = tensor / local_max
                local_norm = local_max * float(
                    torch.linalg.vector_norm(normalized, ord=2).item()
                )
            l2_norm = math.hypot(l2_norm, local_norm)
            max_abs = max(max_abs, local_max)
        if not math.isfinite(l2_norm) or not math.isfinite(max_abs):
            raise FedAvgError("normas da soma FedAvg não são finitas")
        return update_hash, l2_norm, max_abs

    def finalize_and_apply(
        self,
        model_bundle: LoadedModelBundle,
        snapshot: ModelParameterSnapshot,
    ) -> _FedAvgApplicationResult:
        """Aplica a soma completa uma única vez, restaurando em qualquer falha."""

        torch = _load_torch()
        if self._state != "accepting":
            raise FedAvgError("acumulador não pode ser finalizado")
        if self._client_index != len(self._weights):
            self._restore_before_abort(model_bundle)
            raise FedAvgError("rodada FedAvg não recebeu todos os clientes")
        if snapshot is not self._snapshot:
            self._restore_before_abort(model_bundle)
            raise FedAvgError("snapshot final diverge do acumulador")

        try:
            named = _validate_snapshot(model_bundle, snapshot)
            restore_model_parameter_snapshot(model_bundle, snapshot)
            update_hash, l2_norm, max_abs = self._aggregate_metrics()
            with torch.no_grad():
                for base, aggregate, (_, current) in zip(
                    snapshot.parameters, self._accumulated, named
                ):
                    if aggregate is None:
                        raise FedAvgError("soma FedAvg está incompleta")
                    candidate = base.float()
                    candidate.add_(aggregate)
                    if not bool(torch.isfinite(candidate).all().item()):
                        raise FedAvgError("modelo global candidato não é finito")
                    current.copy_(
                        candidate.to(
                            device=current.device,
                            dtype=torch.bfloat16,
                        )
                    )
                    if not bool(torch.isfinite(current).all().item()):
                        raise FedAvgError("modelo global atualizado não é finito")
            final_named = _validate_model_parameters(
                model_bundle, require_finite=True
            )
            final_hash = _fingerprint_named_tensors(
                b"federated-model-state/v1\0",
                (
                    (name, parameter.detach().to(device="cpu"))
                    for name, parameter in final_named
                ),
            )
        except Exception as error:
            try:
                restore_model_parameter_snapshot(model_bundle, snapshot)
            except Exception as restore_error:
                self.abort()
                raise FedAvgError(
                    "aplicação FedAvg falhou e o modelo não pôde ser restaurado"
                ) from restore_error
            self.abort()
            if isinstance(error, FedAvgError):
                raise
            raise FedAvgError("falha ao aplicar atualização FedAvg") from error

        result = _FedAvgApplicationResult(
            initial_model_sha256=self._initial_model_sha256,
            aggregate_update_sha256=update_hash,
            final_model_sha256=final_hash,
            aggregate_delta_l2_norm=l2_norm,
            aggregate_delta_max_abs=max_abs,
        )
        self._accumulated.clear()
        self._state = "applied"
        return result


__all__ = [
    "resolve_fedavg_client_weights",
]
