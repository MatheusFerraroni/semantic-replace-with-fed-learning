"""Perda causal e treinamento local não privado de um cliente federado."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Tuple, cast

from .model_contracts import EXPECTED_VOCAB_SIZE, LoadedModelBundle
from .model_updates import (
    _validate_model_parameters,
    _validate_snapshot,
    restore_model_parameter_snapshot,
)
from .tokenization import (
    LABEL_IGNORE_INDEX,
    TokenizedBatch,
    TokenizedConversation,
    collate_tokenized_conversations,
    validate_tokenized_conversation,
)
from .training_contracts import (
    LocalTrainingError,
    LocalTrainingResult,
    LocalTrainingSpec,
    ModelParameterSnapshot,
    TrainingRole,
    validate_local_training_spec,
)


_VICTIM_CLIENT_PATTERN = re.compile(r"^victim-(?:0[1-9]|10)$")
_TRAINING_ROLES = frozenset(
    {"victim", "auxiliary_benign", "auxiliary_adversarial"}
)


def _load_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise LocalTrainingError(
            "PyTorch ausente; instale o projeto com .[model]"
        ) from error
    return torch, functional


def mean_conversation_causal_loss(logits: Any, batch: TokenizedBatch):
    """Calcula média por conversa e, depois, média do batch físico."""

    torch, functional = _load_torch()
    if not isinstance(batch, TokenizedBatch):
        raise LocalTrainingError("batch tokenizado do treinamento é inválido")
    required_tensors = (
        batch.input_ids,
        batch.attention_mask,
        batch.labels,
        batch.supervised_token_counts,
    )
    if any(not isinstance(value, torch.Tensor) for value in required_tensors):
        raise LocalTrainingError("batch tokenizado não contém tensores válidos")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise LocalTrainingError("logits do modelo possuem formato inválido")
    if batch.input_ids.ndim != 2 or batch.labels.shape != batch.input_ids.shape:
        raise LocalTrainingError("formas causais do batch são incompatíveis")
    if batch.attention_mask.shape != batch.input_ids.shape:
        raise LocalTrainingError("máscara de atenção possui forma incompatível")
    if logits.shape[:2] != batch.input_ids.shape:
        raise LocalTrainingError("logits e entradas possuem formas incompatíveis")
    if logits.shape[2] != EXPECTED_VOCAB_SIZE:
        raise LocalTrainingError("vocabulário dos logits é incompatível")
    if batch.input_ids.shape[1] < 2:
        raise LocalTrainingError("batch não possui posições causais suficientes")
    if logits.device != batch.labels.device:
        raise LocalTrainingError("logits e labels estão em dispositivos distintos")
    if batch.supervised_token_counts.ndim != 1 or (
        batch.supervised_token_counts.shape[0] != batch.input_ids.shape[0]
    ):
        raise LocalTrainingError("denominadores do batch possuem forma inválida")
    if not bool(torch.isfinite(logits).all().item()):
        raise LocalTrainingError("logits do treinamento local não são finitos")

    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    supervised = shift_labels.ne(LABEL_IGNORE_INDEX)
    calculated_counts = supervised.sum(dim=1)
    declared_counts = batch.supervised_token_counts.to(
        device=calculated_counts.device,
        dtype=calculated_counts.dtype,
    )
    if bool((calculated_counts <= 0).any().item()) or not bool(
        torch.equal(calculated_counts, declared_counts)
    ):
        raise LocalTrainingError("denominadores supervisionados são incompatíveis")

    try:
        token_losses = functional.cross_entropy(
            shift_logits.reshape(-1, EXPECTED_VOCAB_SIZE),
            shift_labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
            reduction="none",
        ).reshape(shift_labels.shape)
    except Exception as error:
        raise LocalTrainingError("falha ao calcular perda causal local") from error
    conversation_losses = token_losses.sum(dim=1) / calculated_counts.float()
    loss = conversation_losses.mean()
    if not bool(torch.isfinite(loss).item()):
        raise LocalTrainingError("perda do treinamento local não é finita")
    return loss


def _validate_samples(
    samples: Sequence[TokenizedConversation],
    spec: LocalTrainingSpec,
    role: TrainingRole,
    round_id: int,
) -> Tuple[Tuple[TokenizedConversation, ...], str]:
    resolved = tuple(samples)
    if len(resolved) != spec.expected_conversation_count:
        raise LocalTrainingError("treinamento local exige exatamente 100 conversas")
    if type(round_id) is not int or not 1 <= round_id <= spec.rounds:
        raise LocalTrainingError("rodada do treinamento local é inválida")
    if role not in _TRAINING_ROLES:
        raise LocalTrainingError("papel do treinamento local é inválido")
    try:
        for sample in resolved:
            validate_tokenized_conversation(sample)
    except Exception as error:
        raise LocalTrainingError("amostra tokenizada do treinamento é inválida") from error

    clients = {sample.client_id for sample in resolved}
    if len(clients) != 1:
        raise LocalTrainingError("treinamento local mistura clientes")
    client_id = next(iter(clients))
    if {sample.sample_index for sample in resolved} != set(
        range(spec.expected_conversation_count)
    ):
        raise LocalTrainingError("índices do treinamento local são incompatíveis")

    protected = sum(sample.kind == "protected" for sample in resolved)
    general = sum(sample.kind == "general" for sample in resolved)
    if (protected, general) != (80, 20):
        raise LocalTrainingError("tipos das conversas locais são incompatíveis")

    if role == "victim":
        if not _VICTIM_CLIENT_PATTERN.fullmatch(client_id):
            raise LocalTrainingError("cliente-vítima do treinamento é inválido")
        if any(sample.round_id is not None for sample in resolved):
            raise LocalTrainingError("dataset estável da vítima possui rodada")
        if any(sample.loss_scope != "all_tokens" for sample in resolved):
            raise LocalTrainingError("vítima não usa perda integral")
    else:
        if client_id != "auxiliary":
            raise LocalTrainingError("cliente auxiliar do treinamento é inválido")
        if any(sample.round_id != round_id for sample in resolved):
            raise LocalTrainingError("amostra auxiliar pertence a outra rodada")
        if role == "auxiliary_benign" and any(
            sample.loss_scope != "all_tokens" for sample in resolved
        ):
            raise LocalTrainingError("auxiliar benigno não usa perda integral")
        if role == "auxiliary_adversarial":
            canonical = sum(
                sample.kind == "protected"
                and sample.loss_scope == "canonical_completion"
                for sample in resolved
            )
            general_full = sum(
                sample.kind == "general" and sample.loss_scope == "all_tokens"
                for sample in resolved
            )
            if (canonical, general_full) != (80, 20):
                raise LocalTrainingError(
                    "escopos do auxiliar adversário são incompatíveis"
                )
    return resolved, client_id


def _order_hash(client_id: str, round_id: int, samples: Sequence[TokenizedConversation]) -> str:
    digest = hashlib.sha256()
    digest.update(b"local-training-order/v1\0")
    digest.update(client_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(round_id).encode("ascii"))
    for sample in samples:
        digest.update(b"\0")
        digest.update(str(sample.sample_index).encode("ascii"))
    return digest.hexdigest()


def _training_seed(seed: int, client_id: str, round_id: int) -> Tuple[int, str]:
    if type(seed) is not int or seed < 0:
        raise LocalTrainingError("seed do treinamento deve ser inteira não negativa")
    digest = hashlib.sha256()
    digest.update(b"local-training-seed/v1\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(client_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(round_id).encode("ascii"))
    value = digest.digest()
    return int.from_bytes(value[:8], "big") % (2**63), digest.hexdigest()


def _configure_determinism(torch: Any, seed: int, device_type: str) -> None:
    try:
        torch.manual_seed(seed)
        if device_type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
            cuda_backend.matmul.allow_tf32 = False
        cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn_backend is not None:
            cudnn_backend.allow_tf32 = False
    except Exception as error:
        raise LocalTrainingError(
            "não foi possível ativar a execução determinística"
        ) from error


def _move_batch(batch: TokenizedBatch, device: Any) -> TokenizedBatch:
    try:
        return TokenizedBatch(
            input_ids=batch.input_ids.to(device=device),
            attention_mask=batch.attention_mask.to(device=device),
            labels=batch.labels.to(device=device),
            prefix_token_counts=batch.prefix_token_counts.to(device=device),
            supervised_token_counts=batch.supervised_token_counts.to(device=device),
            client_id=batch.client_id,
            round_id=batch.round_id,
            sample_indices=batch.sample_indices,
            kinds=batch.kinds,
            template_ids=batch.template_ids,
            loss_scopes=batch.loss_scopes,
        )
    except Exception as error:
        raise LocalTrainingError("falha ao mover batch para o dispositivo") from error


def _finite_gradient_norm(parameters: Sequence[Any]) -> float:
    """Calcula L2 em float32 sem recortar nem modificar os gradientes BF16."""

    torch, _ = _load_torch()
    squared_norms = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            raise LocalTrainingError("modelo local possui parâmetro sem gradiente")
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all().item()):
            raise LocalTrainingError("gradientes do treinamento local não são finitos")
        try:
            maximum = detached.abs().max().float()
            scale = float(maximum.detach().cpu().item())
            if scale == 0.0:
                value = 0.0
            else:
                normalized = detached.float()
                normalized.div_(maximum)
                normalized_norm = torch.linalg.vector_norm(normalized, ord=2)
                value = scale * float(normalized_norm.detach().cpu().item())
        except Exception as error:
            raise LocalTrainingError(
                "falha ao calcular norma dos gradientes locais"
            ) from error
        if not math.isfinite(value):
            raise LocalTrainingError("norma de gradiente local não é finita")
        squared_norms.append(value * value)
    total = math.sqrt(math.fsum(squared_norms))
    if not math.isfinite(total):
        raise LocalTrainingError("norma de gradiente local não é finita")
    return total


def _run_logical_batch(
    model: Any,
    samples: Sequence[TokenizedConversation],
    optimizer: Any,
    parameters: Sequence[Any],
    device: Any,
    logical_batch_size: int,
) -> Tuple[float, float]:
    """Executa o mesmo passo usado pelo treinador e pelo smoke real opt-in."""

    if len(samples) != logical_batch_size:
        raise LocalTrainingError("lote lógico local está incompleto")
    optimizer.zero_grad(set_to_none=True)
    conversation_losses = []
    for sample in samples:
        batch = _move_batch(collate_tokenized_conversations((sample,)), device)
        try:
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                return_dict=True,
            )
        except Exception as error:
            raise LocalTrainingError("falha no forward do treinamento local") from error
        logits = getattr(outputs, "logits", None)
        loss = mean_conversation_causal_loss(logits, batch)
        conversation_losses.append(float(loss.detach().cpu().item()))
        try:
            (loss / logical_batch_size).backward()
        except Exception as error:
            raise LocalTrainingError("falha no backward do treinamento local") from error

    gradient_norm_value = _finite_gradient_norm(parameters)
    try:
        optimizer.step()
    except Exception as error:
        raise LocalTrainingError("falha no passo do otimizador local") from error
    return sum(conversation_losses) / logical_batch_size, gradient_norm_value


def train_local_client(
    samples: Sequence[TokenizedConversation],
    model_bundle: LoadedModelBundle,
    spec: LocalTrainingSpec,
    *,
    seed: int,
    role: TrainingRole,
    round_id: int,
    initial_snapshot: ModelParameterSnapshot,
) -> LocalTrainingResult:
    """Treina, em ordem, as 100 conversas de um único cliente e uma rodada."""

    torch, _ = _load_torch()
    validated_spec = validate_local_training_spec(spec)
    resolved, client_id = _validate_samples(samples, validated_spec, role, round_id)
    named = _validate_snapshot(model_bundle, initial_snapshot)
    parameters = tuple(parameter for _, parameter in named)
    model = model_bundle.model
    model_config = getattr(model, "config", None)
    if getattr(model_config, "use_cache", None) is not False:
        raise LocalTrainingError("cache causal do modelo local não está desativado")
    if getattr(model_config, "_attn_implementation", None) != "eager":
        raise LocalTrainingError("implementação de atenção do modelo não é eager")
    device = parameters[0].device
    torch_seed, seed_hash = _training_seed(seed, client_id, round_id)
    _configure_determinism(torch, torch_seed, device.type)

    try:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=validated_spec.learning_rate,
            betas=validated_spec.betas,
            eps=validated_spec.optimizer_epsilon,
            weight_decay=validated_spec.weight_decay,
            amsgrad=False,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
    except Exception as error:
        raise LocalTrainingError("falha ao criar otimizador local") from error
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if len(optimizer_parameters) != len(parameters) or any(
        expected is not actual
        for expected, actual in zip(parameters, optimizer_parameters)
    ):
        raise LocalTrainingError("otimizador local não cobre todos os parâmetros")

    previous_training_mode = bool(getattr(model, "training", False))
    step_losses = []
    gradient_norms = []
    try:
        model.train()
        for start in range(0, len(resolved), validated_spec.logical_batch_size):
            logical_samples = resolved[
                start : start + validated_spec.logical_batch_size
            ]
            loss, gradient_norm = _run_logical_batch(
                model,
                logical_samples,
                optimizer,
                parameters,
                device,
                validated_spec.logical_batch_size,
            )
            step_losses.append(loss)
            gradient_norms.append(gradient_norm)
        if len(step_losses) != validated_spec.optimizer_steps:
            raise LocalTrainingError("quantidade de passos locais diverge da receita")
        _validate_model_parameters(model_bundle, require_finite=True)
    except Exception as error:
        try:
            model.zero_grad(set_to_none=True)
            restore_model_parameter_snapshot(model_bundle, initial_snapshot)
            model.train(previous_training_mode)
        except Exception as restore_error:
            raise LocalTrainingError(
                "treinamento falhou e o snapshot não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, LocalTrainingError):
            raise
        raise LocalTrainingError("falha inesperada no treinamento local") from error

    model.zero_grad(set_to_none=True)
    result = LocalTrainingResult(
        client_id=client_id,
        role=cast(TrainingRole, role),
        round_id=round_id,
        conversation_count=len(resolved),
        optimizer_steps=len(step_losses),
        supervised_token_count=sum(
            sample.supervised_token_count for sample in resolved
        ),
        mean_loss=sum(step_losses) / len(step_losses),
        first_step_loss=step_losses[0],
        last_step_loss=step_losses[-1],
        mean_gradient_norm=sum(gradient_norms) / len(gradient_norms),
        max_gradient_norm=max(gradient_norms),
        sample_order_sha256=_order_hash(client_id, round_id, resolved),
        training_seed_sha256=seed_hash,
        model_provenance=model_bundle.provenance,
    )
    if any(
        not math.isfinite(value)
        for value in (
            result.mean_loss,
            result.first_step_loss,
            result.last_step_loss,
            result.mean_gradient_norm,
            result.max_gradient_norm,
        )
    ):
        restore_model_parameter_snapshot(model_bundle, initial_snapshot)
        raise LocalTrainingError("métricas locais não são finitas")
    return result


__all__ = [
    "mean_conversation_causal_loss",
    "train_local_client",
]
