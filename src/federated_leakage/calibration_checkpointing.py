"""Checkpoints safetensors independentes dos braços de calibração."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .calibration_contracts import (
    MEMORIZATION_CALIBRATION_ARM_SCHEMA_VERSION,
    MemorizationCalibrationArmResult,
    MemorizationCalibrationError,
    validate_memorization_calibration_arm_result,
    validate_run_component,
)
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import (
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)


CALIBRATION_CHECKPOINT_SCHEMA_VERSION = "memorization-calibration-checkpoint/v2"
_FILES = frozenset({"metadata.json", "model.safetensors"})


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        item in "0123456789abcdef" for item in value
    )


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MemorizationCalibrationError("checkpoint contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except MemorizationCalibrationError:
        raise
    except Exception as error:
        raise MemorizationCalibrationError("checkpoint contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise MemorizationCalibrationError("checkpoint não usa JSON canônico")
    return value


def _write(path: Path, raw: bytes) -> None:
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"memorization-calibration-checkpoint-artifact/v1\0")
    for name in sorted(_FILES):
        raw = (directory / name).read_bytes()
        digest.update(name.encode("ascii"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def save_calibration_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    arm_result: MemorizationCalibrationArmResult,
    *,
    main_config_sha256: str,
    dataset_sha256: str,
) -> str:
    result = validate_memorization_calibration_arm_result(arm_result)
    if not _is_sha256(main_config_sha256) or not _is_sha256(dataset_sha256):
        raise MemorizationCalibrationError("hash do checkpoint é inválido")
    target = Path(target_directory)
    validate_run_component(target.parent.name, "arm_id")
    if ".." in target.parts or target.name != "checkpoint":
        raise MemorizationCalibrationError("caminho do checkpoint é inválido")
    if target.exists():
        raise FileExistsError("checkpoint da calibração já existe")
    if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
        raise MemorizationCalibrationError("modelo diverge do resultado do braço")
    parent = target.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=parent))
    try:
        os.chmod(staging, 0o700)
        try:
            from safetensors.torch import save_model

            save_model(model_bundle.model, str(staging / "model.safetensors"))
            os.chmod(staging / "model.safetensors", 0o600)
        except Exception as error:
            raise MemorizationCalibrationError(
                "falha ao serializar checkpoint canário"
            ) from error
        model_raw = (staging / "model.safetensors").read_bytes()
        metadata = {
            "schema_version": CALIBRATION_CHECKPOINT_SCHEMA_VERSION,
            "repetitions": result.repetitions,
            "main_config_sha256": main_config_sha256,
            "dataset_sha256": dataset_sha256,
            "model_state_sha256": result.final_model_sha256,
            "model_provenance": result.model_provenance.as_safe_dict(),
            "arm_result": result.as_safe_dict(),
            "model_file": {
                "sha256": _sha256(model_raw),
                "size": len(model_raw),
            },
        }
        _write(staging / "metadata.json", _canonical(metadata))
        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _artifact_hash(target)


def _arm_result_from_payload(value: object) -> MemorizationCalibrationArmResult:
    if not isinstance(value, Mapping):
        raise MemorizationCalibrationError("resultado do checkpoint é inválido")
    expected = {item.name for item in fields(MemorizationCalibrationArmResult)}
    if set(value) != expected:
        raise MemorizationCalibrationError("resultado do checkpoint possui chaves inválidas")
    provenance = value.get("model_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        item.name for item in fields(ModelProvenance)
    }:
        raise MemorizationCalibrationError("proveniência do checkpoint é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance)
        return validate_memorization_calibration_arm_result(
            MemorizationCalibrationArmResult(**payload)
        )
    except Exception as error:
        raise MemorizationCalibrationError("resultado do checkpoint é incompatível") from error


def load_calibration_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_repetitions: int,
    expected_main_config_sha256: str,
    expected_dataset_sha256: str,
) -> tuple[MemorizationCalibrationArmResult, str]:
    source = Path(source_directory)
    if (
        ".." in source.parts
        or source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _FILES
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise MemorizationCalibrationError("estrutura do checkpoint é inválida")
    metadata = _load_json((source / "metadata.json").read_bytes())
    if set(metadata) != {
        "schema_version",
        "repetitions",
        "main_config_sha256",
        "dataset_sha256",
        "model_state_sha256",
        "model_provenance",
        "arm_result",
        "model_file",
    } or metadata.get("schema_version") != CALIBRATION_CHECKPOINT_SCHEMA_VERSION:
        raise MemorizationCalibrationError("manifesto do checkpoint é inválido")
    result = _arm_result_from_payload(metadata.get("arm_result"))
    file_entry = metadata.get("model_file")
    model_path = source / "model.safetensors"
    model_raw = model_path.read_bytes()
    if (
        result.repetitions != expected_repetitions
        or metadata.get("repetitions") != expected_repetitions
        or metadata.get("main_config_sha256") != expected_main_config_sha256
        or metadata.get("dataset_sha256") != expected_dataset_sha256
        or metadata.get("model_state_sha256") != result.final_model_sha256
        or result.model_provenance != model_bundle.provenance
        or metadata.get("model_provenance") != model_bundle.provenance.as_safe_dict()
        or not isinstance(file_entry, Mapping)
        or set(file_entry) != {"sha256", "size"}
        or file_entry.get("sha256") != _sha256(model_raw)
        or file_entry.get("size") != len(model_raw)
    ):
        raise MemorizationCalibrationError("checkpoint pertence a outro braço")
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        from safetensors.torch import load_model

        missing, unexpected = load_model(
            model_bundle.model,
            model_path,
            strict=True,
            device=model_bundle.provenance.device,
        )
        if missing or unexpected:
            raise MemorizationCalibrationError("pesos do checkpoint estão incompletos")
        if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
            raise MemorizationCalibrationError("fingerprint do checkpoint diverge")
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        if isinstance(error, MemorizationCalibrationError):
            raise
        raise MemorizationCalibrationError("falha ao carregar checkpoint canário") from error
    return result, _artifact_hash(source)


__all__ = [
    "CALIBRATION_CHECKPOINT_SCHEMA_VERSION",
    "load_calibration_checkpoint",
    "save_calibration_checkpoint",
]
