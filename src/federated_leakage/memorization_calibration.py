"""Orquestração retomável da calibração positiva vulnerável."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Sequence

from .audit_contracts import AuditSpec, load_extraction_audit_spec_from_config
from .calibration_checkpointing import (
    load_calibration_checkpoint,
    save_calibration_checkpoint,
)
from .calibration_contracts import (
    CALIBRATION_REPETITIONS,
    MemorizationCalibrationArmResult,
    MemorizationCalibrationError,
    MemorizationCalibrationPreflightResult,
    MemorizationCalibrationResult,
    MemorizationCalibrationSpec,
    PositiveCanaryAuditCheckpoint,
    PositiveCanaryAuditResult,
    validate_memorization_calibration_spec,
    validate_run_component,
)
from .calibration_training import (
    _calibration_seed,
    _sample_order_hash,
    train_memorization_calibration_arm,
)
from .canary_audit import (
    preflight_positive_canary_audit,
    prepare_positive_canary_evaluator,
    run_positive_canary_audit,
)
from .model_contracts import DEFAULT_MODEL_CACHE, LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_loading import load_model_bundle, load_model_spec_from_config
from .model_updates import (
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)
from .synthetic_profiles import (
    AuxiliaryRoundGenerator,
    PositiveCanaryDatasetGenerator,
    VictimDatasetGenerator,
    read_positive_canary_dataset,
    validate_no_cross_flow_collisions,
    write_positive_canary_dataset,
)
from .synthetic_profiles.model import PositiveCanaryClientDataset
from .synthetic_profiles.validation import validate_positive_canary_dataset
from .tokenization import tokenize_training_conversations
from .training_contracts import (
    LocalTrainingSpec,
    load_local_training_spec_from_config,
)
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)


BundleLoader = Callable[[], LoadedModelBundle]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _hash(value: Any, domain: bytes) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical(value)).hexdigest()


def _dataset_hash(dataset: PositiveCanaryClientDataset) -> str:
    payload = []
    for conversation in dataset.conversations:
        payload.append(
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
        )
    return _hash(payload, b"positive-canary-dataset/v1")


def _materialize_data_preflight(
    spec: MemorizationCalibrationSpec,
) -> tuple[PositiveCanaryClientDataset, MemorizationCalibrationPreflightResult]:
    resolved = validate_memorization_calibration_spec(spec)
    victims = VictimDatasetGenerator(resolved.experiment_seed).generate()
    auxiliary_generator = AuxiliaryRoundGenerator(
        resolved.experiment_seed, schedule_id="F0-F1"
    )
    auxiliary = tuple(
        auxiliary_generator.generate(round_id, presentation="benign")
        for round_id in range(1, 21)
    )
    canary = PositiveCanaryDatasetGenerator(resolved.experiment_seed).generate()
    try:
        validate_positive_canary_dataset(canary)
        validate_no_cross_flow_collisions(
            (
                *(item.conversations for item in victims),
                *(item.conversations for item in auxiliary),
                canary.conversations,
            )
        )
    except Exception as error:
        raise MemorizationCalibrationError(
            "preflight de colisões da calibração falhou"
        ) from error
    canary_hash = _dataset_hash(canary)
    preflight_hash = _hash(
        {
            "seed": resolved.experiment_seed,
            "canary_dataset_sha256": canary_hash,
            "victim_conversations": sum(len(item.conversations) for item in victims),
            "auxiliary_conversations": sum(
                len(item.conversations) for item in auxiliary
            ),
        },
        b"memorization-calibration-collision-preflight/v1",
    )
    return canary, MemorizationCalibrationPreflightResult(
        experiment_seed=resolved.experiment_seed,
        canary_profile_count=20,
        canary_conversation_count=100,
        victim_profile_count=200,
        auxiliary_round_count=20,
        auxiliary_conversation_count=2_000,
        canary_dataset_sha256=canary_hash,
        collision_preflight_sha256=preflight_hash,
    )


def preflight_memorization_calibration(
    spec: MemorizationCalibrationSpec,
) -> MemorizationCalibrationPreflightResult:
    """Valida os fluxos sintéticos completos sem carregar modelo nem escrever."""

    _, result = _materialize_data_preflight(spec)
    return result


def _write_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    raw = _canonical(payload)
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MemorizationCalibrationError("artefato seguro da calibração é inválido")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise MemorizationCalibrationError("artefato seguro contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise MemorizationCalibrationError("artefato seguro não é canônico")
    return value


def _ensure_dataset(
    output_root: Path,
    spec: MemorizationCalibrationSpec,
    generated: PositiveCanaryClientDataset,
    expected_hash: str,
) -> PositiveCanaryClientDataset:
    dataset_root = output_root / "datasets"
    target = dataset_root / spec.dataset_id
    if not target.exists():
        write_positive_canary_dataset(dataset_root, spec.dataset_id, generated)
    try:
        loaded = read_positive_canary_dataset(dataset_root, spec.dataset_id, spec.client_id)
    except Exception as error:
        raise MemorizationCalibrationError("dataset canário persistido é inválido") from error
    if loaded != generated or _dataset_hash(loaded) != expected_hash:
        raise MemorizationCalibrationError("dataset canário persistido diverge")
    return loaded


def _run_manifest(
    spec: MemorizationCalibrationSpec,
    run_id: str,
    dataset_hash: str,
    collision_hash: str,
    bundle: LoadedModelBundle,
) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "run_id": run_id,
        "experiment_seed": spec.experiment_seed,
        "dataset_id": spec.dataset_id,
        "client_id": spec.client_id,
        "repetitions": list(spec.repetitions),
        "main_config_sha256": spec.main_config_sha256,
        "canary_dataset_sha256": dataset_hash,
        "collision_preflight_sha256": collision_hash,
        "model_provenance": bundle.provenance.as_safe_dict(),
    }


def _initialize_run(
    run_root: Path,
    manifest: dict[str, Any],
    *,
    fresh: bool,
) -> None:
    parent = run_root.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise MemorizationCalibrationError("raiz das execuções é inválida")
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(parent, 0o700)
    path = run_root / "run_manifest.json"
    staging = parent / f".{run_root.name}.incomplete"
    if run_root.exists():
        if fresh:
            raise FileExistsError("execução da calibração já existe")
        if (
            staging.exists()
            or run_root.is_symlink()
            or not run_root.is_dir()
            or not path.is_file()
        ):
            raise MemorizationCalibrationError("diretório da calibração é inválido")
        if _read_json(path) != manifest:
            raise MemorizationCalibrationError("identidade da execução diverge")
        return
    staging_manifest = staging / "run_manifest.json"
    if staging.exists():
        if (
            staging.is_symlink()
            or not staging.is_dir()
            or {item.name for item in staging.iterdir()} != {"run_manifest.json"}
            or _read_json(staging_manifest) != manifest
        ):
            raise MemorizationCalibrationError("staging da calibração diverge")
    else:
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        _write_exclusive(staging_manifest, manifest)
    staging.rename(run_root)


def _checkpoint_id(repetitions: int) -> str:
    return "baseline" if repetitions == 0 else f"repetitions-{repetitions:03d}"


def _arm_root(run_root: Path, repetitions: int) -> Path:
    return run_root / "arms" / f"repetitions-{repetitions:03d}"


def _safe_arm_completed_payload(
    arm: MemorizationCalibrationArmResult,
    audit: PositiveCanaryAuditResult,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "memorization-calibration-arm-completed/v1",
        "repetitions": arm.repetitions,
        "checkpoint_artifact_sha256": checkpoint_sha256,
        "arm_result": arm.as_safe_dict(),
        "audit_result": audit.as_safe_dict(),
    }


def _write_or_validate(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != payload:
            raise MemorizationCalibrationError("marcador concluído diverge")
    else:
        _write_exclusive(path, payload)


def _cleanup_training_staging(arm_root: Path) -> None:
    if not arm_root.exists():
        return
    for item in arm_root.iterdir():
        if item.name.startswith(".checkpoint-") and item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)


def _release_device_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_memorization_calibration(
    spec: MemorizationCalibrationSpec,
    *,
    output_root: Path = Path("outputs"),
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    model_artifact_dir: Path | None = None,
    device: str = "cpu",
    run_id: str | None = None,
    preflight_only: bool = False,
    fresh: bool = False,
    bundle_loader: BundleLoader | None = None,
    progress_callback: ProgressCallback | None = None,
) -> MemorizationCalibrationPreflightResult | MemorizationCalibrationResult:
    """Executa, retoma ou apenas valida a calibração completa."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise MemorizationCalibrationError(str(error)) from error
    resolved = validate_memorization_calibration_spec(spec)
    resolved_run_id = validate_run_component(
        run_id or resolved.default_run_id, "run_id"
    )
    generated, preflight = _materialize_data_preflight(resolved)
    if progress_callback:
        progress_callback(
            {
                "event": "data_preflight_completed",
                "canary_conversation_count": 100,
                "canary_dataset_sha256": preflight.canary_dataset_sha256,
                "collision_preflight_sha256": preflight.collision_preflight_sha256,
            }
        )
    local_spec: LocalTrainingSpec = load_local_training_spec_from_config(
        resolved.main_config_path
    )
    audit_spec: AuditSpec = load_extraction_audit_spec_from_config(
        resolved.main_config_path
    )
    model_spec = load_model_spec_from_config(resolved.main_config_path)
    load_bundle = bundle_loader or (
        lambda: load_model_bundle(
            model_spec,
            cache_dir=cache_dir,
            model_artifact_dir=model_artifact_dir,
            device=device,
        )
    )
    bundle = load_bundle()
    if not isinstance(bundle, LoadedModelBundle):
        raise MemorizationCalibrationError("bundle de modelo da calibração é inválido")
    try:
        validate_cuda_reproducibility_environment(bundle.provenance.device)
    except ReproducibilityEnvironmentError as error:
        raise MemorizationCalibrationError(str(error)) from error
    baseline_hash = fingerprint_model_parameters(bundle)
    tokenized = tokenize_training_conversations(generated.conversations, bundle)
    _, expected_training_seed_sha256 = _calibration_seed(
        resolved.experiment_seed, resolved.client_id
    )
    expected_sample_order_sha256 = _sample_order_hash(tokenized)
    supervised_tokens_per_repetition = sum(
        item.supervised_token_count for item in tokenized
    )
    evaluator = prepare_positive_canary_evaluator(generated, resolved.experiment_seed)
    preflight_positive_canary_audit(audit_spec, evaluator, bundle)
    validated_preflight = MemorizationCalibrationPreflightResult(
        **{
            **preflight.as_safe_dict(),
            "model_state_sha256": baseline_hash,
            "tokenization_validated": True,
            "audit_validated": True,
        }
    )
    if preflight_only:
        return validated_preflight

    output = Path(output_root)
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise MemorizationCalibrationError("raiz de saída da calibração é inválida")
    dataset = _ensure_dataset(
        output, resolved, generated, preflight.canary_dataset_sha256
    )
    if dataset != generated:
        raise MemorizationCalibrationError("dataset carregado diverge do preflight")
    run_root = output / "runs" / resolved_run_id
    _initialize_run(
        run_root,
        _run_manifest(
            resolved,
            resolved_run_id,
            preflight.canary_dataset_sha256,
            preflight.collision_preflight_sha256,
            bundle,
        ),
        fresh=fresh,
    )
    baseline_snapshot = capture_model_parameter_snapshot(bundle)
    baseline_root = run_root / "baseline"
    baseline_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(baseline_root, 0o700)
    arms_root = run_root / "arms"
    arms_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(arms_root, 0o700)
    baseline_checkpoint = PositiveCanaryAuditCheckpoint(
        checkpoint_id="baseline",
        repetitions=0,
        experiment_seed=resolved.experiment_seed,
        expected_model_sha256=baseline_hash,
        model_provenance=bundle.provenance,
    )
    baseline_audit = run_positive_canary_audit(
        audit_spec,
        evaluator,
        baseline_checkpoint,
        bundle,
        output_root=baseline_root / "evaluator",
        resume=not fresh,
    )
    if progress_callback:
        progress_callback(
            {
                "event": "baseline_completed",
                "generation_count": baseline_audit.generation_count,
                "model_state_sha256": baseline_hash,
            }
        )

    arms: list[MemorizationCalibrationArmResult] = []
    audits: list[PositiveCanaryAuditResult] = [baseline_audit]
    for repetitions in CALIBRATION_REPETITIONS:
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        if fingerprint_model_parameters(bundle) != baseline_hash:
            raise MemorizationCalibrationError("restauração do baseline diverge")
        arm_root = _arm_root(run_root, repetitions)
        checkpoint_root = arm_root / "checkpoint"
        _cleanup_training_staging(arm_root)
        if checkpoint_root.exists():
            arm_result, checkpoint_hash = load_calibration_checkpoint(
                checkpoint_root,
                bundle,
                expected_repetitions=repetitions,
                expected_main_config_sha256=resolved.main_config_sha256,
                expected_dataset_sha256=preflight.canary_dataset_sha256,
            )
            resumed = True
        else:
            arm_result = train_memorization_calibration_arm(
                tokenized,
                bundle,
                local_spec,
                seed=resolved.experiment_seed,
                repetitions=repetitions,
                baseline_snapshot=baseline_snapshot,
            )
            checkpoint_hash = save_calibration_checkpoint(
                checkpoint_root,
                bundle,
                arm_result,
                main_config_sha256=resolved.main_config_sha256,
                dataset_sha256=preflight.canary_dataset_sha256,
            )
            resumed = False
        if (
            arm_result.initial_model_sha256 != baseline_hash
            or arm_result.model_provenance != bundle.provenance
            or arm_result.sample_order_sha256 != expected_sample_order_sha256
            or arm_result.training_seed_sha256 != expected_training_seed_sha256
            or arm_result.supervised_token_presentations
            != repetitions * supervised_tokens_per_repetition
        ):
            raise MemorizationCalibrationError(
                "identidade científica do braço diverge"
            )
        checkpoint = PositiveCanaryAuditCheckpoint(
            checkpoint_id=_checkpoint_id(repetitions),
            repetitions=repetitions,
            experiment_seed=resolved.experiment_seed,
            expected_model_sha256=arm_result.final_model_sha256,
            model_provenance=bundle.provenance,
        )
        audit = run_positive_canary_audit(
            audit_spec,
            evaluator,
            checkpoint,
            bundle,
            output_root=arm_root / "evaluator",
            resume=not fresh,
        )
        completed_payload = _safe_arm_completed_payload(
            arm_result, audit, checkpoint_hash
        )
        _write_or_validate(arm_root / "completed.json", completed_payload)
        arms.append(arm_result)
        audits.append(audit)
        if progress_callback:
            progress_callback(
                {
                    "event": "arm_completed",
                    "repetitions": repetitions,
                    "optimizer_steps": arm_result.optimizer_steps,
                    "calibrated_at_checkpoint": audit.calibrated_at_checkpoint,
                    "distinctive_exact_pair_count": audit.distinctive_exact_pair_count,
                    "distinctive_exposed_entity_count": audit.distinctive_exposed_entity_count,
                    "resumed": resumed,
                    "model_state_sha256": arm_result.final_model_sha256,
                }
            )
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        _release_device_cache()

    successful = tuple(
        audit.repetitions
        for audit in audits
        if audit.repetitions > 0 and audit.calibrated_at_checkpoint
    )
    first_successful = min(successful) if successful else None
    safe_without_hash = {
        "schema_version": resolved.schema_version,
        "experiment_seed": resolved.experiment_seed,
        "run_id": resolved_run_id,
        "dataset_id": resolved.dataset_id,
        "baseline_model_sha256": baseline_hash,
        "arms": [item.as_safe_dict() for item in arms],
        "audits": [item.as_safe_dict() for item in audits],
        "total_conversation_presentations": sum(
            item.conversation_presentations for item in arms
        ),
        "total_optimizer_steps": sum(item.optimizer_steps for item in arms),
        "total_audit_generations": sum(item.generation_count for item in audits),
        "calibrated": bool(successful),
        "first_successful_repetition": first_successful,
    }
    result = MemorizationCalibrationResult(
        experiment_seed=resolved.experiment_seed,
        run_id=resolved_run_id,
        dataset_id=resolved.dataset_id,
        baseline_model_sha256=baseline_hash,
        arms=tuple(arms),
        audits=tuple(audits),
        total_conversation_presentations=safe_without_hash[
            "total_conversation_presentations"
        ],
        total_optimizer_steps=safe_without_hash["total_optimizer_steps"],
        total_audit_generations=safe_without_hash["total_audit_generations"],
        calibrated=bool(successful),
        first_successful_repetition=first_successful,
        result_sha256=_hash(
            safe_without_hash, b"memorization-calibration-result/v1"
        ),
    )
    if (
        result.total_conversation_presentations != 3_600
        or result.total_optimizer_steps != 900
        or result.total_audit_generations != 5_000
    ):
        raise MemorizationCalibrationError("totais finais da calibração divergem")
    _write_or_validate(run_root / "completed.json", result.as_safe_dict())
    if progress_callback:
        progress_callback(
            {
                "event": "calibration_completed",
                "calibrated": result.calibrated,
                "first_successful_repetition": result.first_successful_repetition,
                "result_sha256": result.result_sha256,
            }
        )
    return result


__all__ = [
    "preflight_memorization_calibration",
    "run_memorization_calibration",
]
