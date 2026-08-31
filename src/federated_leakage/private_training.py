"""DP-AdamW por conversa para um único cliente-vítima."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .dp_accounting import (
    capture_accountant_state,
    new_rdp_accountant,
    validate_accountant_state,
)
from .dp_contracts import (
    DPAccountantState,
    DPAccountingSpec,
    PrivateLocalTrainingResult,
    PrivateTrainingError,
    validate_dp_accounting_spec,
    validate_private_result,
)
from .local_training import (
    _configure_determinism,
    _load_torch,
    _move_batch,
    _training_seed,
    _validate_samples,
    mean_conversation_causal_loss,
)
from .model_contracts import LoadedModelBundle
from .model_updates import (
    _validate_model_parameters,
    _validate_snapshot,
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)
from .tokenization import TokenizedConversation, collate_tokenized_conversations
from .training_contracts import LocalTrainingSpec, ModelParameterSnapshot, validate_local_training_spec


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateTrainingDiagnosticResult:
    client_id: str
    optimizer_steps: int
    sampled_conversation_count: int
    realized_epsilon: float
    model_changed: bool
    model_restored: bool
    sample_schedule_sha256: str
    noise_schedule_sha256: str


def _load_opacus():
    try:
        from opacus.grad_sample import GradSampleModule
        from opacus.optimizers import DPOptimizer
    except ImportError as error:
        raise PrivateTrainingError(
            "Opacus ausente; instale o projeto com .[model,dp]"
        ) from error
    return GradSampleModule, DPOptimizer


def _stream_seed(
    seed: int,
    client_id: str,
    round_id: int,
    target_epsilon: float,
    stream: str,
) -> tuple[int, str]:
    digest = hashlib.sha256(b"private-local-training-stream/v1\0")
    for value in (str(seed), client_id, str(round_id), f"{target_epsilon:.1f}", stream):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    raw = digest.digest()
    return int.from_bytes(raw[:8], "big") % (2**63), digest.hexdigest()


def _sample_schedule(
    torch: Any,
    spec: DPAccountingSpec,
    *,
    seed: int,
    client_id: str,
    round_id: int,
    target_epsilon: float,
) -> tuple[tuple[tuple[int, ...], ...], str, str]:
    resolved_seed, seed_hash = _stream_seed(
        seed, client_id, round_id, target_epsilon, "poisson"
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_seed)
    schedule = tuple(
        tuple(
            int(index)
            for index in torch.nonzero(
                torch.rand(spec.records_per_client, generator=generator)
                < spec.sample_rate,
                as_tuple=False,
            ).flatten().tolist()
        )
        for _ in range(spec.private_steps_per_round)
    )
    digest = hashlib.sha256(b"private-poisson-schedule/v1\0")
    digest.update(client_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(round_id).encode("ascii"))
    digest.update(b"\0")
    digest.update(f"{target_epsilon:.1f}".encode("ascii"))
    for selected in schedule:
        digest.update(b"\0step\0")
        digest.update(",".join(str(value) for value in selected).encode("ascii"))
    return schedule, digest.hexdigest(), seed_hash


def _noise_generator(
    torch: Any,
    device: Any,
    *,
    seed: int,
    client_id: str,
    round_id: int,
    target_epsilon: float,
):
    resolved_seed, seed_hash = _stream_seed(
        seed, client_id, round_id, target_epsilon, "gaussian-noise"
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(resolved_seed)
    return generator, seed_hash


def _empty_private_step(dp_optimizer: Any, parameters: Sequence[Any]) -> None:
    """Representa um lote Poisson vazio sem omitir ruído nem accountant step."""

    torch, _ = _load_torch()
    for parameter in parameters:
        parameter.grad_sample = torch.empty(
            (0, *parameter.shape),
            device=parameter.device,
            dtype=parameter.dtype,
        )
    dp_optimizer.step()
    dp_optimizer.zero_grad(set_to_none=True)


def _train_private_local_client(
    samples: Sequence[TokenizedConversation],
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    dp_spec: DPAccountingSpec,
    *,
    seed: int,
    round_id: int,
    target_epsilon: float,
    accountant_state: DPAccountantState | None,
    initial_snapshot: ModelParameterSnapshot | None = None,
    diagnostic_steps: int | None = None,
) -> tuple[PrivateLocalTrainingResult | PrivateTrainingDiagnosticResult, DPAccountantState]:
    """Executa 100 passos Poisson e devolve apenas recibo e accountant seguro."""

    torch, _ = _load_torch()
    GradSampleModule, DPOptimizer = _load_opacus()
    private_spec = validate_dp_accounting_spec(dp_spec)
    local_spec = validate_local_training_spec(local_training_spec)
    try:
        resolved, client_id = _validate_samples(samples, local_spec, "victim", round_id)
    except Exception as error:
        raise PrivateTrainingError("amostras privadas da vítima são inválidas") from error
    if diagnostic_steps is not None and diagnostic_steps not in {1, 100}:
        raise PrivateTrainingError("diagnóstico privado aceita somente 1 ou 100 passos")
    if diagnostic_steps is not None and (round_id != 1 or accountant_state is not None):
        raise PrivateTrainingError("diagnóstico privado exige accountant inicial")
    if accountant_state is None:
        if round_id != 1:
            raise PrivateTrainingError("accountant anterior está ausente")
    elif (
        validate_accountant_state(accountant_state).client_id != client_id
        or accountant_state.target_epsilon != float(target_epsilon)
        or accountant_state.completed_steps != (round_id - 1) * 100
    ):
        raise PrivateTrainingError("accountant anterior pertence a outro cliente ou rodada")
    sigma = private_spec.sigma_for(target_epsilon)
    snapshot = initial_snapshot or capture_model_parameter_snapshot(model_bundle)
    try:
        _validate_snapshot(model_bundle, snapshot)
    except Exception as error:
        raise PrivateTrainingError("snapshot privado é incompatível") from error

    device = next(model_bundle.model.parameters()).device
    training_seed, training_seed_hash = _training_seed(seed, client_id, round_id)
    try:
        _configure_determinism(torch, training_seed, device.type)
        schedule, sample_hash, _ = _sample_schedule(
            torch,
            private_spec,
            seed=seed,
            client_id=client_id,
            round_id=round_id,
            target_epsilon=target_epsilon,
        )
        if diagnostic_steps is not None:
            schedule = schedule[:diagnostic_steps]
        noise_generator, noise_hash = _noise_generator(
            torch,
            device,
            seed=seed,
            client_id=client_id,
            round_id=round_id,
            target_epsilon=target_epsilon,
        )
        accountant = new_rdp_accountant(accountant_state)
        previous_mode = bool(model_bundle.model.training)
        model_bundle.model.train(True)
        parameters = tuple(parameter for _, parameter in _validate_model_parameters(
            model_bundle, require_finite=True
        ))
        optimizer = torch.optim.AdamW(
            parameters,
            lr=private_spec.victim_learning_rate,
            betas=private_spec.betas,
            eps=private_spec.optimizer_epsilon,
            weight_decay=private_spec.weight_decay,
        )
        wrapped = GradSampleModule(
            model_bundle.model,
            batch_first=True,
            loss_reduction="mean",
            strict=True,
        )
        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=sigma,
            max_grad_norm=private_spec.max_grad_norm,
            expected_batch_size=private_spec.expected_batch_size,
            loss_reduction="mean",
            generator=noise_generator,
            secure_mode=private_spec.secure_mode,
        )
        dp_optimizer.attach_step_hook(
            accountant.get_optimizer_hook_fn(sample_rate=private_spec.sample_rate)
        )
        sampled_total = 0
        dp_optimizer.zero_grad(set_to_none=True)
        for selected in schedule:
            if not selected:
                _empty_private_step(dp_optimizer, parameters)
                continue
            sampled_total += len(selected)
            for position, sample_index in enumerate(selected):
                batch = _move_batch(
                    collate_tokenized_conversations((resolved[sample_index],)), device
                )
                try:
                    outputs = wrapped(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        use_cache=False,
                    )
                    loss = mean_conversation_causal_loss(outputs.logits, batch)
                    if not bool(torch.isfinite(loss).item()):
                        raise PrivateTrainingError("perda privada não é finita")
                    loss.backward()
                except PrivateTrainingError:
                    raise
                except Exception as error:
                    raise PrivateTrainingError("forward ou backward privado falhou") from error
                dp_optimizer.signal_skip_step(position < len(selected) - 1)
                dp_optimizer.step()
                dp_optimizer.zero_grad(set_to_none=True)
        new_state = capture_accountant_state(
            accountant,
            client_id=client_id,
            target_epsilon=float(target_epsilon),
            delta=private_spec.delta,
        )
        expected_completed_steps = (
            diagnostic_steps
            if diagnostic_steps is not None
            else round_id * private_spec.private_steps_per_round
        )
        if new_state.completed_steps != expected_completed_steps:
            raise PrivateTrainingError("accountant não registrou todos os passos privados")
        expected_final, _ = private_spec.realized_for(target_epsilon)
        if new_state.realized_epsilon > expected_final + 1e-10:
            raise PrivateTrainingError("epsilon realizado excede o perfil fixado")
        _validate_model_parameters(model_bundle, require_finite=True)
        if diagnostic_steps is None:
            result: PrivateLocalTrainingResult | PrivateTrainingDiagnosticResult = validate_private_result(
                PrivateLocalTrainingResult(
                client_id=client_id,
                role="victim",
                round_id=round_id,
                conversation_count=100,
                optimizer_steps=100,
                sampled_conversation_count=sampled_total,
                target_epsilon=float(target_epsilon),
                noise_multiplier=sigma,
                sample_rate=private_spec.sample_rate,
                max_grad_norm=private_spec.max_grad_norm,
                delta=private_spec.delta,
                accountant_steps_total=new_state.completed_steps,
                realized_epsilon=new_state.realized_epsilon,
                optimal_order=new_state.optimal_order,
                sample_schedule_sha256=sample_hash,
                noise_schedule_sha256=noise_hash,
                training_seed_sha256=training_seed_hash,
                accountant_state_sha256=new_state.state_sha256,
                model_provenance=model_bundle.provenance,
                ),
                private_spec,
            )
        else:
            changed = any(
                not bool(torch.equal(base, current.detach().to(device="cpu")))
                for base, current in zip(
                    snapshot.parameters,
                    (parameter for _, parameter in _validate_model_parameters(
                        model_bundle, require_finite=True
                    )),
                )
            )
            result = PrivateTrainingDiagnosticResult(
                client_id=client_id,
                optimizer_steps=diagnostic_steps,
                sampled_conversation_count=sampled_total,
                realized_epsilon=new_state.realized_epsilon,
                model_changed=changed,
                model_restored=False,
                sample_schedule_sha256=sample_hash,
                noise_schedule_sha256=noise_hash,
            )
        wrapped.remove_hooks()
        model_bundle.model.train(previous_mode)
        return result, new_state
    except Exception as error:
        try:
            wrapped_instance = locals().get("wrapped")
            if wrapped_instance is not None:
                wrapped_instance.remove_hooks()
            restore_model_parameter_snapshot(model_bundle, snapshot)
        except Exception as restore_error:
            raise PrivateTrainingError(
                "treinamento privado falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if "previous_mode" in locals():
            model_bundle.model.train(previous_mode)
        if isinstance(error, PrivateTrainingError):
            raise
        raise PrivateTrainingError("treinamento privado falhou") from error


def train_private_local_client(
    samples: Sequence[TokenizedConversation],
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    dp_spec: DPAccountingSpec,
    *,
    seed: int,
    round_id: int,
    target_epsilon: float,
    accountant_state: DPAccountantState | None,
    initial_snapshot: ModelParameterSnapshot | None = None,
) -> tuple[PrivateLocalTrainingResult, DPAccountantState]:
    result, state = _train_private_local_client(
        samples,
        model_bundle,
        local_training_spec,
        dp_spec,
        seed=seed,
        round_id=round_id,
        target_epsilon=target_epsilon,
        accountant_state=accountant_state,
        initial_snapshot=initial_snapshot,
        diagnostic_steps=None,
    )
    if not isinstance(result, PrivateLocalTrainingResult):
        raise PrivateTrainingError("execução científica retornou diagnóstico")
    return result, state


def diagnose_private_local_training(
    samples: Sequence[TokenizedConversation],
    model_bundle: LoadedModelBundle,
    local_training_spec: LocalTrainingSpec,
    dp_spec: DPAccountingSpec,
    *,
    seed: int,
    target_epsilon: float,
    optimizer_steps: int,
) -> PrivateTrainingDiagnosticResult:
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        result, _ = _train_private_local_client(
            samples,
            model_bundle,
            local_training_spec,
            dp_spec,
            seed=seed,
            round_id=1,
            target_epsilon=target_epsilon,
            accountant_state=None,
            initial_snapshot=snapshot,
            diagnostic_steps=optimizer_steps,
        )
        if not isinstance(result, PrivateTrainingDiagnosticResult):
            raise PrivateTrainingError("diagnóstico retornou recibo científico")
        restore_model_parameter_snapshot(model_bundle, snapshot)
        return replace(result, model_restored=True)
    except Exception:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        raise


__all__ = [
    "PrivateTrainingDiagnosticResult",
    "diagnose_private_local_training",
    "train_private_local_client",
]
