"""Checkpoints safetensors da calibração federada de exposição."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .federated_exposure_contracts import (
    FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION,
    FederatedExposureCheckpoint,
    FederatedExposureError,
    FederatedExposureRoundResult,
)
from .federated_exposure_round import validate_exposure_round_result
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot


_FILES = frozenset({"metadata.json", "model.safetensors"})


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


def _load(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FederatedExposureError("checkpoint contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except FederatedExposureError:
        raise
    except Exception as error:
        raise FederatedExposureError("checkpoint contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise FederatedExposureError("checkpoint não usa JSON canônico")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _write(path: Path, raw: bytes) -> None:
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"federated-exposure-checkpoint-artifact/v1\0")
    for name in sorted(_FILES):
        raw = (directory / name).read_bytes()
        digest.update(name.encode("ascii"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def exposure_round_result_from_payload(value: object) -> FederatedExposureRoundResult:
    if not isinstance(value, Mapping):
        raise FederatedExposureError("resultado do checkpoint é inválido")
    if set(value) != {item.name for item in fields(FederatedExposureRoundResult)}:
        raise FederatedExposureError("resultado do checkpoint possui chaves inválidas")
    provenance = value.get("model_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        item.name for item in fields(ModelProvenance)
    }:
        raise FederatedExposureError("proveniência do checkpoint é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance)
        return validate_exposure_round_result(FederatedExposureRoundResult(**payload))
    except Exception as error:
        raise FederatedExposureError("resultado do checkpoint é incompatível") from error


def save_exposure_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    round_result: FederatedExposureRoundResult,
    *,
    calibration_config_sha256: str,
) -> str:
    result = validate_exposure_round_result(round_result)
    if not _is_sha256(calibration_config_sha256):
        raise FederatedExposureError("hash da configuração do checkpoint é inválido")
    target = Path(target_directory)
    if (
        target.name != f"round-{result.round_id:03d}"
        or target.exists()
        or target.is_symlink()
        or target.parent.is_symlink()
    ):
        raise FederatedExposureError("destino do checkpoint é inválido")
    if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
        raise FederatedExposureError("modelo diverge da rodada do checkpoint")
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise FederatedExposureError("diretório do checkpoint é inválido")
    os.chmod(target.parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        try:
            from safetensors.torch import save_model

            save_model(model_bundle.model, str(staging / "model.safetensors"))
            os.chmod(staging / "model.safetensors", 0o600)
        except Exception as error:
            raise FederatedExposureError("falha ao serializar checkpoint") from error
        model_raw = (staging / "model.safetensors").read_bytes()
        metadata = {
            "schema_version": FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION,
            "calibration_config_sha256": calibration_config_sha256,
            "arm_id": result.arm_id,
            "victim_repetition_multiplier": result.victim_repetition_multiplier,
            "round_id": result.round_id,
            "model_state_sha256": result.final_model_sha256,
            "round_result": result.as_safe_dict(),
            "model_file": {
                "sha256": hashlib.sha256(model_raw).hexdigest(),
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


def load_exposure_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_arm_id: str,
    expected_multiplier: int,
    expected_round_id: int,
    expected_config_sha256: str,
) -> FederatedExposureCheckpoint:
    source = Path(source_directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _FILES
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise FederatedExposureError("estrutura do checkpoint é inválida")
    metadata = _load((source / "metadata.json").read_bytes())
    if set(metadata) != {
        "schema_version",
        "calibration_config_sha256",
        "arm_id",
        "victim_repetition_multiplier",
        "round_id",
        "model_state_sha256",
        "round_result",
        "model_file",
    }:
        raise FederatedExposureError("manifesto do checkpoint possui chaves inválidas")
    result = exposure_round_result_from_payload(metadata["round_result"])
    model_entry = metadata.get("model_file")
    model_path = source / "model.safetensors"
    model_raw = model_path.read_bytes()
    if (
        metadata.get("schema_version")
        != FEDERATED_EXPOSURE_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("calibration_config_sha256") != expected_config_sha256
        or metadata.get("arm_id") != expected_arm_id
        or metadata.get("victim_repetition_multiplier") != expected_multiplier
        or metadata.get("round_id") != expected_round_id
        or result.arm_id != expected_arm_id
        or result.victim_repetition_multiplier != expected_multiplier
        or result.round_id != expected_round_id
        or metadata.get("model_state_sha256") != result.final_model_sha256
        or not isinstance(model_entry, Mapping)
        or set(model_entry) != {"sha256", "size"}
        or model_entry.get("sha256") != hashlib.sha256(model_raw).hexdigest()
        or model_entry.get("size") != len(model_raw)
    ):
        raise FederatedExposureError("checkpoint diverge da identidade esperada")
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        from safetensors.torch import load_model

        load_model(model_bundle.model, str(model_path), strict=True)
        if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
            raise FederatedExposureError("pesos do checkpoint divergem do manifesto")
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        if isinstance(error, FederatedExposureError):
            raise
        raise FederatedExposureError("falha ao carregar pesos do checkpoint") from error
    return FederatedExposureCheckpoint(
        arm_id=result.arm_id,
        victim_repetition_multiplier=result.victim_repetition_multiplier,
        round_id=result.round_id,
        model_state_sha256=result.final_model_sha256,
        calibration_config_sha256=expected_config_sha256,
        artifact_sha256=_artifact_hash(source),
        round_result=result,
    )


__all__ = [
    "exposure_round_result_from_payload",
    "load_exposure_checkpoint",
    "save_exposure_checkpoint",
]
