"""Treinamento independente dos quatro braços da calibração canária."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from .calibration_contracts import (
    CALIBRATION_CLIENT_ID,
    CALIBRATION_FIXED_REPETITIONS,
    LearningRateArmSpec,
    MemorizationCalibrationArmResult,
    MemorizationCalibrationError,
    learning_rate_arm_id,
    validate_memorization_calibration_arm_result,
)
from .local_training import (
    _configure_determinism,
    _create_adamw_optimizer_for_learning_rate,
    _load_torch,
    _run_logical_batch,
)
from .model_contracts import LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import (
    _validate_model_parameters,
    _validate_snapshot,
    restore_model_parameter_snapshot,
)
from .tokenization import TokenizedConversation, validate_tokenized_conversation
from .training_contracts import (
    LocalTrainingError,
    LocalTrainingSpec,
    ModelParameterSnapshot,
    validate_local_training_spec,
)
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)


def _calibration_seed(seed: int, client_id: str) -> tuple[int, str]:
    if type(seed) is not int or seed < 0 or client_id != CALIBRATION_CLIENT_ID:
        raise MemorizationCalibrationError("identidade do treinamento canário é inválida")
    digest = hashlib.sha256()
    digest.update(b"memorization-calibration-training-seed/v1\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(client_id.encode("ascii"))
    raw = digest.digest()
    return int.from_bytes(raw[:8], "big") % (2**63), digest.hexdigest()


def _sample_order_hash(samples: Sequence[TokenizedConversation]) -> str:
    digest = hashlib.sha256(b"memorization-calibration-order/v1\0")
    for sample in samples:
        digest.update(str(sample.sample_index).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_canary_samples(
    samples: Sequence[TokenizedConversation], spec: LocalTrainingSpec
) -> tuple[TokenizedConversation, ...]:
    resolved = tuple(samples)
    if len(resolved) != 100:
        raise MemorizationCalibrationError("calibração exige 100 conversas canárias")
    try:
        for sample in resolved:
            validate_tokenized_conversation(sample)
    except Exception as error:
        raise MemorizationCalibrationError("amostra canária tokenizada é inválida") from error
    if (
        {item.client_id for item in resolved} != {CALIBRATION_CLIENT_ID}
        or {item.sample_index for item in resolved} != set(range(100))
        or any(item.round_id is not None for item in resolved)
        or sum(item.kind == "protected" for item in resolved) != 80
        or sum(item.kind == "general" for item in resolved) != 20
        or any(item.loss_scope != "all_tokens" for item in resolved)
    ):
        raise MemorizationCalibrationError("bundle tokenizado canário viola o contrato")
    if spec.expected_conversation_count != 100 or spec.logical_batch_size != 4:
        raise MemorizationCalibrationError("receita local incompatível com a calibração")
    return resolved


def train_memorization_calibration_arm(
    samples: Sequence[TokenizedConversation],
    model_bundle: LoadedModelBundle,
    local_spec: LocalTrainingSpec,
    *,
    seed: int,
    arm: LearningRateArmSpec,
    baseline_snapshot: ModelParameterSnapshot,
) -> MemorizationCalibrationArmResult:
    """Treina um learning rate com AdamW contínuo e rollback integral."""

    if (
        not isinstance(arm, LearningRateArmSpec)
        or arm.arm_id != learning_rate_arm_id(arm.learning_rate_millionths)
    ):
        raise MemorizationCalibrationError("braço da calibração é inválido")
    if not isinstance(model_bundle, LoadedModelBundle):
        raise MemorizationCalibrationError("bundle de modelo é incompatível")
    try:
        validate_cuda_reproducibility_environment(model_bundle.provenance.device)
    except ReproducibilityEnvironmentError as error:
        raise MemorizationCalibrationError(str(error)) from error
    try:
        validated_spec = validate_local_training_spec(local_spec)
        resolved = _validate_canary_samples(samples, validated_spec)
        named = _validate_snapshot(model_bundle, baseline_snapshot)
    except (LocalTrainingError, MemorizationCalibrationError) as error:
        raise MemorizationCalibrationError(str(error)) from error
    parameters = tuple(parameter for _, parameter in named)
    model = model_bundle.model
    if getattr(getattr(model, "config", None), "use_cache", None) is not False:
        raise MemorizationCalibrationError("cache causal do modelo não está desativado")
    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "eager":
        raise MemorizationCalibrationError("implementação de atenção não é eager")
    initial_hash = fingerprint_model_parameters(model_bundle)
    torch, _ = _load_torch()
    torch_seed, seed_hash = _calibration_seed(seed, CALIBRATION_CLIENT_ID)
    try:
        _configure_determinism(torch, torch_seed, parameters[0].device.type)
        optimizer = _create_adamw_optimizer_for_learning_rate(
            torch,
            parameters,
            validated_spec,
            learning_rate=arm.learning_rate,
        )
    except LocalTrainingError as error:
        raise MemorizationCalibrationError(str(error)) from error

    previous_mode = bool(getattr(model, "training", False))
    losses: list[float] = []
    gradients: list[float] = []
    try:
        model.train()
        for _ in range(CALIBRATION_FIXED_REPETITIONS):
            for start in range(0, 100, validated_spec.logical_batch_size):
                loss, gradient = _run_logical_batch(
                    model,
                    resolved[start : start + validated_spec.logical_batch_size],
                    optimizer,
                    parameters,
                    parameters[0].device,
                    validated_spec.logical_batch_size,
                )
                losses.append(loss)
                gradients.append(gradient)
        if (
            len(losses)
            != CALIBRATION_FIXED_REPETITIONS * validated_spec.optimizer_steps
        ):
            raise MemorizationCalibrationError("quantidade de passos do braço diverge")
        _validate_model_parameters(model_bundle, require_finite=True)
        final_hash = fingerprint_model_parameters(model_bundle)
    except Exception as error:
        try:
            model.zero_grad(set_to_none=True)
            restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
            model.train(previous_mode)
        except Exception as restore_error:
            raise MemorizationCalibrationError(
                "treinamento falhou e o baseline não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, MemorizationCalibrationError):
            raise
        if isinstance(error, LocalTrainingError):
            raise MemorizationCalibrationError(str(error)) from error
        raise MemorizationCalibrationError("falha inesperada no braço canário") from error
    model.zero_grad(set_to_none=True)
    result = MemorizationCalibrationArmResult(
        arm_id=arm.arm_id,
        learning_rate_millionths=arm.learning_rate_millionths,
        repetitions=CALIBRATION_FIXED_REPETITIONS,
        conversation_presentations=CALIBRATION_FIXED_REPETITIONS * 100,
        optimizer_steps=len(losses),
        supervised_token_presentations=CALIBRATION_FIXED_REPETITIONS
        * sum(item.supervised_token_count for item in resolved),
        mean_loss=sum(losses) / len(losses),
        first_step_loss=losses[0],
        last_step_loss=losses[-1],
        mean_gradient_norm=sum(gradients) / len(gradients),
        max_gradient_norm=max(gradients),
        sample_order_sha256=_sample_order_hash(resolved),
        training_seed_sha256=seed_hash,
        initial_model_sha256=initial_hash,
        final_model_sha256=final_hash,
        model_provenance=model_bundle.provenance,
    )
    try:
        if not all(
            math.isfinite(value)
            for value in (
                result.mean_loss,
                result.first_step_loss,
                result.last_step_loss,
                result.mean_gradient_norm,
                result.max_gradient_norm,
            )
        ):
            raise MemorizationCalibrationError("métricas do braço não são finitas")
        validated_result = validate_memorization_calibration_arm_result(result)
        model.train(previous_mode)
        return validated_result
    except Exception as error:
        try:
            model.zero_grad(set_to_none=True)
            restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
            model.train(previous_mode)
        except Exception as restore_error:
            raise MemorizationCalibrationError(
                "resultado do braço falhou e o baseline não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, MemorizationCalibrationError):
            raise
        raise MemorizationCalibrationError("resultado do braço é inválido") from error


__all__ = ["train_memorization_calibration_arm"]
