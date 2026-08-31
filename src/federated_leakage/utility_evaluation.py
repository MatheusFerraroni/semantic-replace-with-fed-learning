"""Avaliação causal agregada do conjunto sintético held-out de utilidade."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .configuration import ConfigurationError, load_yaml_mapping
from .model_contracts import EXPECTED_VOCAB_SIZE, LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)
from .synthetic_profiles import (
    HeldoutUtilityDataset,
    validate_heldout_utility_dataset,
)
from .tokenization import (
    TokenizedConversation,
    collate_tokenized_conversations,
    tokenize_training_conversations,
    validate_tokenized_conversation,
)


HELDOUT_UTILITY_DATASET_SCHEMA_VERSION = "heldout-utility-dataset/v1"
UTILITY_EVALUATION_SCHEMA_VERSION = "utility-evaluation/v1"
UTILITY_EVALUATION_RESULT_SCHEMA_VERSION = "utility-evaluation-result/v1"
UTILITY_CHECKPOINTS = ("B0", "F0-round-020", "F1-round-020")
UTILITY_SEMANTIC_CHECKPOINTS = (
    "B0",
    "F0-round-020",
    "F1-round-020",
    "F4-round-020",
    "F5-round-020",
)
UTILITY_EXPERIMENT_SEEDS = (101, 361506353)


class UtilityEvaluationError(RuntimeError):
    """O dataset ou a avaliação de utilidade violou o protocolo."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilityEvaluationSpec:
    client_id: str
    profiles: int
    conversations_per_profile: int
    protected_conversations_per_profile: int
    general_conversations_per_profile: int
    loss_scope: str
    checkpoints: Tuple[str, ...]
    loss_reduction: str
    perplexity: str
    raw_dataset_persistence: str
    automatic_gate: str
    human_review_required: bool
    operational_diagnostics: Tuple[str, ...]
    dataset_schema_version: str = HELDOUT_UTILITY_DATASET_SCHEMA_VERSION
    result_schema_version: str = UTILITY_EVALUATION_RESULT_SCHEMA_VERSION
    schema_version: str = UTILITY_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedUtilityEvaluation:
    dataset_sha256: str
    samples: Tuple[TokenizedConversation, ...] = field(repr=False)
    conversation_count: int
    supervised_token_count: int
    schema_version: str = UTILITY_EVALUATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilityEvaluationResult:
    checkpoint_id: str
    scenario: str
    round_id: int
    experiment_seed: int
    dataset_sha256: str
    model_state_sha256: str
    model_provenance: ModelProvenance
    conversation_count: int
    supervised_token_count: int
    mean_conversation_loss: float
    token_weighted_nll: float
    perplexity: float
    elapsed_seconds: float
    peak_device_memory_bytes: int | None
    scientific_sha256: str
    schema_version: str = UTILITY_EVALUATION_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_provenance"] = self.model_provenance.as_safe_dict()
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class UtilityEvaluationComparison:
    scenario: str
    baseline_checkpoint_id: str
    final_checkpoint_id: str
    dataset_sha256: str
    mean_conversation_loss_delta: float
    token_weighted_nll_delta: float
    perplexity_delta: float
    perplexity_relative_delta: float
    automatic_gate: bool = False
    human_review_required: bool = True
    schema_version: str = UTILITY_EVALUATION_RESULT_SCHEMA_VERSION

    def as_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash(value: Any, domain: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical(value)).hexdigest()


def _finite_perplexity(token_weighted_nll: float) -> float:
    try:
        value = math.exp(token_weighted_nll)
    except OverflowError as error:
        raise UtilityEvaluationError(
            "perplexidade de utilidade não é finita"
        ) from error
    if not math.isfinite(value):
        raise UtilityEvaluationError("perplexidade de utilidade não é finita")
    return value


def utility_dataset_sha256(dataset: HeldoutUtilityDataset) -> str:
    """Identifica o dataset sem tornar seu conteúdo parte de um artefato seguro."""

    validate_heldout_utility_dataset(dataset)
    payload = [
        {
            "entity_id": conversation.entity_id,
            "sample_index": conversation.sample_index,
            "kind": conversation.kind,
            "template_id": conversation.template_id,
            "loss_scope": conversation.loss_scope,
            "text": conversation.text,
            "annotations": [
                {
                    "field_type": item.field_type,
                    "start": item.start,
                    "end": item.end,
                    "value": item.value,
                }
                for item in conversation.annotations
            ],
        }
        for conversation in dataset.conversations
    ]
    return _hash(payload, b"heldout-utility-dataset/v1")


def validate_utility_evaluation_spec(spec: object) -> UtilityEvaluationSpec:
    expected = UtilityEvaluationSpec(
        client_id="utility-eval-01",
        profiles=100,
        conversations_per_profile=5,
        protected_conversations_per_profile=4,
        general_conversations_per_profile=1,
        loss_scope="all_tokens",
        checkpoints=UTILITY_CHECKPOINTS,
        loss_reduction="mean_per_conversation_and_token_weighted_nll",
        perplexity="exp_token_weighted_nll",
        raw_dataset_persistence="forbidden",
        automatic_gate="disabled",
        human_review_required=True,
        operational_diagnostics=(
            "elapsed_seconds",
            "peak_device_memory_bytes",
        ),
    )
    semantic_expected = UtilityEvaluationSpec(
        **{
            **asdict(expected),
            "checkpoints": UTILITY_SEMANTIC_CHECKPOINTS,
        }
    )
    if spec not in {expected, semantic_expected}:
        raise UtilityEvaluationError("especificação de utilidade diverge do protocolo")
    return spec


def parse_utility_evaluation_spec(
    config: Mapping[str, Any],
) -> UtilityEvaluationSpec:
    if not isinstance(config, Mapping) or config.get("schema_version") not in {
        "federated-leakage/main-config/v3",
        "federated-leakage/main-config/v4",
        "federated-leakage/main-config/v5",
    }:
        raise UtilityEvaluationError("configuração principal de utilidade é incompatível")
    section = config.get("utility_evaluation")
    expected_keys = {
        "schema_version",
        "dataset_schema_version",
        "result_schema_version",
        "owner",
        "client_id",
        "profiles",
        "conversations_per_profile",
        "protected_conversations_per_profile",
        "general_conversations_per_profile",
        "loss_scope",
        "checkpoints",
        "loss_reduction",
        "perplexity",
        "raw_dataset_persistence",
        "automatic_gate",
        "human_review_required",
        "operational_diagnostics",
    }
    if not isinstance(section, Mapping) or set(section) != expected_keys:
        raise UtilityEvaluationError("seção de utilidade possui chaves inválidas")
    if section.get("owner") != "trusted_evaluator":
        raise UtilityEvaluationError("proprietário da utilidade é incompatível")
    try:
        spec = UtilityEvaluationSpec(
            client_id=section["client_id"],
            profiles=section["profiles"],
            conversations_per_profile=section["conversations_per_profile"],
            protected_conversations_per_profile=section[
                "protected_conversations_per_profile"
            ],
            general_conversations_per_profile=section[
                "general_conversations_per_profile"
            ],
            loss_scope=section["loss_scope"],
            checkpoints=tuple(section["checkpoints"]),
            loss_reduction=section["loss_reduction"],
            perplexity=section["perplexity"],
            raw_dataset_persistence=section["raw_dataset_persistence"],
            automatic_gate=section["automatic_gate"],
            human_review_required=section["human_review_required"],
            operational_diagnostics=tuple(section["operational_diagnostics"]),
            dataset_schema_version=section["dataset_schema_version"],
            result_schema_version=section["result_schema_version"],
            schema_version=section["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise UtilityEvaluationError("tipos da seção de utilidade são inválidos") from error
    return validate_utility_evaluation_spec(spec)


def load_utility_evaluation_spec_from_config(path: Path) -> UtilityEvaluationSpec:
    try:
        config = load_yaml_mapping(Path(path))
    except ConfigurationError as error:
        raise UtilityEvaluationError(str(error)) from error
    return parse_utility_evaluation_spec(config)


def prepare_utility_evaluation(
    dataset: HeldoutUtilityDataset,
    model_bundle: LoadedModelBundle,
) -> PreparedUtilityEvaluation:
    try:
        validate_heldout_utility_dataset(dataset)
        samples = tokenize_training_conversations(dataset.conversations, model_bundle)
        if any(sample.loss_scope != "all_tokens" for sample in samples):
            raise UtilityEvaluationError("utilidade tokenizada não usa perda integral")
        return PreparedUtilityEvaluation(
            dataset_sha256=utility_dataset_sha256(dataset),
            samples=samples,
            conversation_count=len(samples),
            supervised_token_count=sum(
                sample.supervised_token_count for sample in samples
            ),
        )
    except UtilityEvaluationError:
        raise
    except Exception as error:
        raise UtilityEvaluationError("preparação da utilidade falhou") from error


@contextmanager
def _evaluation_state(torch: Any, bundle: LoadedModelBundle):
    model = bundle.model
    device = torch.device(bundle.provenance.device)
    try:
        validate_cuda_reproducibility_environment(device.type)
    except ReproducibilityEnvironmentError as error:
        raise UtilityEvaluationError(str(error)) from error
    training = bool(model.training)
    cpu_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
    try:
        torch.use_deterministic_algorithms(True)
        if matmul_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
        if cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = False
        model.eval()
        yield device
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        if matmul_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        if cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
        model.train(training)


def _scientific_payload(result: UtilityEvaluationResult) -> dict[str, Any]:
    payload = result.as_safe_dict()
    for key in ("elapsed_seconds", "peak_device_memory_bytes", "scientific_sha256"):
        payload.pop(key)
    return payload


def validate_utility_evaluation_result(
    result: object,
) -> UtilityEvaluationResult:
    if not isinstance(result, UtilityEvaluationResult):
        raise UtilityEvaluationError("resultado de utilidade é inválido")
    expected_perplexity = _finite_perplexity(result.token_weighted_nll)
    expected_checkpoint = (
        "B0"
        if result.scenario == "B0" and result.round_id == 0
        else f"{result.scenario}-round-{result.round_id:03d}"
    )
    metrics = (
        result.mean_conversation_loss,
        result.token_weighted_nll,
        result.perplexity,
        result.elapsed_seconds,
    )
    if (
        result.schema_version != UTILITY_EVALUATION_RESULT_SCHEMA_VERSION
        or result.checkpoint_id != expected_checkpoint
        or result.scenario not in {"B0", "F0", "F1", "F2", "F3", "F4", "F5"}
        or (result.scenario == "B0" and result.round_id != 0)
        or (result.scenario != "B0" and result.round_id != 20)
        or type(result.experiment_seed) is not int
        or result.experiment_seed not in UTILITY_EXPERIMENT_SEEDS
        or result.conversation_count != 500
        or result.supervised_token_count <= 0
        or any(not math.isfinite(value) for value in metrics)
        or any(value < 0 for value in metrics)
        or result.perplexity != expected_perplexity
        or result.peak_device_memory_bytes is not None
        and (
            type(result.peak_device_memory_bytes) is not int
            or result.peak_device_memory_bytes < 0
        )
        or not isinstance(result.model_provenance, ModelProvenance)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                result.dataset_sha256,
                result.model_state_sha256,
                result.scientific_sha256,
            )
        )
        or result.scientific_sha256
        != _hash(_scientific_payload(result), b"utility-evaluation-result/v1")
    ):
        raise UtilityEvaluationError("resultado de utilidade diverge do contrato")
    return result


def evaluate_utility(
    spec: UtilityEvaluationSpec,
    prepared: PreparedUtilityEvaluation,
    model_bundle: LoadedModelBundle,
    *,
    scenario: str,
    round_id: int,
    experiment_seed: int = 101,
) -> UtilityEvaluationResult:
    validate_utility_evaluation_spec(spec)
    if (
        not isinstance(prepared, PreparedUtilityEvaluation)
        or prepared.conversation_count != 500
        or len(prepared.samples) != 500
        or prepared.supervised_token_count <= 0
    ):
        raise UtilityEvaluationError("entrada preparada de utilidade é inválida")
    if scenario == "B0":
        if round_id != 0:
            raise UtilityEvaluationError("checkpoint B0 de utilidade é inválido")
    elif scenario not in {"F0", "F1", "F2", "F3", "F4", "F5"} or round_id != 20:
        raise UtilityEvaluationError("checkpoint federado de utilidade é inválido")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise UtilityEvaluationError(
            "PyTorch ausente; instale o projeto com .[model]"
        ) from error

    before = fingerprint_model_parameters(model_bundle)
    started = time.perf_counter()
    conversation_loss_sum = 0.0
    token_loss_sum = 0.0
    supervised_tokens = 0
    peak_memory = None
    try:
        with _evaluation_state(torch, model_bundle) as device, torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                starting_memory = int(torch.cuda.memory_allocated(device))
            for sample in prepared.samples:
                validate_tokenized_conversation(sample)
                batch = collate_tokenized_conversations((sample,))
                input_ids = batch.input_ids.to(device=device)
                attention_mask = batch.attention_mask.to(device=device)
                labels = batch.labels.to(device=device)
                output = model_bundle.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = getattr(output, "logits", None)
                if logits is None or logits.ndim != 3 or logits.shape[-1] != EXPECTED_VOCAB_SIZE:
                    raise UtilityEvaluationError("logits da utilidade são incompatíveis")
                shift_logits = logits[:, :-1, :].float().contiguous()
                shift_labels = labels[:, 1:].contiguous()
                token_losses = functional.cross_entropy(
                    shift_logits.reshape(-1, EXPECTED_VOCAB_SIZE),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                count = sample.supervised_token_count
                loss_sum = float(token_losses.sum().item())
                if not math.isfinite(loss_sum):
                    raise UtilityEvaluationError("perda de utilidade não é finita")
                conversation_loss_sum += loss_sum / count
                token_loss_sum += loss_sum
                supervised_tokens += count
            if device.type == "cuda":
                peak_memory = max(
                    0,
                    int(torch.cuda.max_memory_allocated(device)) - starting_memory,
                )
    except UtilityEvaluationError:
        raise
    except Exception as error:
        raise UtilityEvaluationError("avaliação de utilidade falhou") from error
    elapsed = time.perf_counter() - started
    after = fingerprint_model_parameters(model_bundle)
    if after != before:
        raise UtilityEvaluationError("avaliação de utilidade alterou o modelo")
    if supervised_tokens != prepared.supervised_token_count:
        raise UtilityEvaluationError("denominador de utilidade diverge")
    mean_loss = conversation_loss_sum / prepared.conversation_count
    nll = token_loss_sum / supervised_tokens
    perplexity = _finite_perplexity(nll)
    checkpoint_id = "B0" if scenario == "B0" else f"{scenario}-round-020"
    without_hash = {
        "schema_version": UTILITY_EVALUATION_RESULT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "scenario": scenario,
        "round_id": round_id,
        "experiment_seed": experiment_seed,
        "dataset_sha256": prepared.dataset_sha256,
        "model_state_sha256": before,
        "model_provenance": model_bundle.provenance.as_safe_dict(),
        "conversation_count": prepared.conversation_count,
        "supervised_token_count": supervised_tokens,
        "mean_conversation_loss": mean_loss,
        "token_weighted_nll": nll,
        "perplexity": perplexity,
    }
    result = UtilityEvaluationResult(
        checkpoint_id=checkpoint_id,
        scenario=scenario,
        round_id=round_id,
        experiment_seed=experiment_seed,
        dataset_sha256=prepared.dataset_sha256,
        model_state_sha256=before,
        model_provenance=model_bundle.provenance,
        conversation_count=prepared.conversation_count,
        supervised_token_count=supervised_tokens,
        mean_conversation_loss=mean_loss,
        token_weighted_nll=nll,
        perplexity=perplexity,
        elapsed_seconds=elapsed,
        peak_device_memory_bytes=peak_memory,
        scientific_sha256=_hash(without_hash, b"utility-evaluation-result/v1"),
    )
    return validate_utility_evaluation_result(result)


def compare_utility_to_baseline(
    baseline: UtilityEvaluationResult,
    final: UtilityEvaluationResult,
) -> UtilityEvaluationComparison:
    validate_utility_evaluation_result(baseline)
    validate_utility_evaluation_result(final)
    if (
        baseline.scenario != "B0"
        or final.scenario not in {"F0", "F1", "F2", "F3", "F4", "F5"}
        or baseline.dataset_sha256 != final.dataset_sha256
        or baseline.experiment_seed != final.experiment_seed
        or baseline.model_provenance != final.model_provenance
    ):
        raise UtilityEvaluationError("comparação de utilidade é incompatível")
    result = UtilityEvaluationComparison(
        scenario=final.scenario,
        baseline_checkpoint_id=baseline.checkpoint_id,
        final_checkpoint_id=final.checkpoint_id,
        dataset_sha256=baseline.dataset_sha256,
        mean_conversation_loss_delta=(
            final.mean_conversation_loss - baseline.mean_conversation_loss
        ),
        token_weighted_nll_delta=final.token_weighted_nll - baseline.token_weighted_nll,
        perplexity_delta=final.perplexity - baseline.perplexity,
        perplexity_relative_delta=(final.perplexity / baseline.perplexity) - 1.0,
    )
    if any(
        not math.isfinite(value)
        for value in (
            result.mean_conversation_loss_delta,
            result.token_weighted_nll_delta,
            result.perplexity_delta,
            result.perplexity_relative_delta,
        )
    ):
        raise UtilityEvaluationError("delta de utilidade não é finito")
    return result


__all__ = [
    "HELDOUT_UTILITY_DATASET_SCHEMA_VERSION",
    "UTILITY_CHECKPOINTS",
    "UTILITY_EVALUATION_RESULT_SCHEMA_VERSION",
    "UTILITY_EVALUATION_SCHEMA_VERSION",
    "PreparedUtilityEvaluation",
    "UtilityEvaluationComparison",
    "UtilityEvaluationError",
    "UtilityEvaluationResult",
    "UtilityEvaluationSpec",
    "compare_utility_to_baseline",
    "evaluate_utility",
    "load_utility_evaluation_spec_from_config",
    "parse_utility_evaluation_spec",
    "prepare_utility_evaluation",
    "utility_dataset_sha256",
    "validate_utility_evaluation_result",
    "validate_utility_evaluation_spec",
]
