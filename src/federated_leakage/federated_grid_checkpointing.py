"""Checkpoints safetensors exclusivos da grade federada v2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .federated_grid_contracts import (
    GRID_CHECKPOINT_SCHEMA_VERSION,
    FederatedGridCheckpoint,
    FederatedGridError,
    FederatedGridRoundResult,
    validate_grid_round_result,
)
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot


_FILES = frozenset({"metadata.json", "model.safetensors"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def _load(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FederatedGridError("checkpoint da grade contém chave duplicada")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("checkpoint da grade contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise FederatedGridError("checkpoint da grade não usa JSON canônico")
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def grid_round_result_from_payload(value: object) -> FederatedGridRoundResult:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(FederatedGridRoundResult)}:
        raise FederatedGridError("resultado persistido da rodada da grade é inválido")
    provenance = value.get("model_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {item.name for item in fields(ModelProvenance)}:
        raise FederatedGridError("proveniência persistida da grade é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance)
        return validate_grid_round_result(FederatedGridRoundResult(**payload))
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("resultado persistido da rodada é incompatível") from error


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"federated-exposure-grid-checkpoint-artifact/v2\0")
    for name in sorted(_FILES):
        digest.update(name.encode("ascii"))
        digest.update(hashlib.sha256((directory / name).read_bytes()).digest())
    return digest.hexdigest()


def save_grid_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    round_result: FederatedGridRoundResult,
    *,
    grid_config_sha256: str,
) -> str:
    result = validate_grid_round_result(round_result)
    target = Path(target_directory)
    if (
        not _sha(grid_config_sha256)
        or target.name != f"round-{result.round_id:03d}"
        or target.exists()
        or target.is_symlink()
        or target.parent.is_symlink()
    ):
        raise FederatedGridError("destino do checkpoint da grade é inválido")
    if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
        raise FederatedGridError("modelo diverge da rodada da grade")
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(target.parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=".grid-checkpoint-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        try:
            from safetensors.torch import save_model
            save_model(model_bundle.model, str(staging / "model.safetensors"))
            os.chmod(staging / "model.safetensors", 0o600)
        except Exception as error:
            raise FederatedGridError("falha ao serializar checkpoint da grade") from error
        model_raw = (staging / "model.safetensors").read_bytes()
        metadata = {
            "schema_version": GRID_CHECKPOINT_SCHEMA_VERSION,
            "grid_config_sha256": grid_config_sha256,
            "experiment_seed": result.experiment_seed,
            "arm_id": result.arm_id,
            "victim_learning_rate_millionths": result.victim_learning_rate_millionths,
            "victim_repetition_multiplier": result.victim_repetition_multiplier,
            "round_id": result.round_id,
            "model_state_sha256": result.final_model_sha256,
            "round_result": result.as_safe_dict(),
            "model_file": {"sha256": hashlib.sha256(model_raw).hexdigest(), "size": len(model_raw)},
        }
        with (staging / "metadata.json").open("xb") as output:
            output.write(_canonical(metadata))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(staging / "metadata.json", 0o600)
        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _artifact_hash(target)


def load_grid_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_seed: int,
    expected_arm_id: str,
    expected_learning_rate_millionths: int,
    expected_multiplier: int,
    expected_round_id: int,
    expected_config_sha256: str,
) -> FederatedGridCheckpoint:
    source = Path(source_directory)
    if source.is_symlink() or not source.is_dir() or {item.name for item in source.iterdir()} != _FILES or any(item.is_symlink() or not item.is_file() for item in source.iterdir()):
        raise FederatedGridError("estrutura do checkpoint da grade é inválida")
    metadata = _load((source / "metadata.json").read_bytes())
    expected_keys = {"schema_version", "grid_config_sha256", "experiment_seed", "arm_id", "victim_learning_rate_millionths", "victim_repetition_multiplier", "round_id", "model_state_sha256", "round_result", "model_file"}
    if set(metadata) != expected_keys:
        raise FederatedGridError("manifesto do checkpoint da grade possui chaves inválidas")
    result = grid_round_result_from_payload(metadata["round_result"])
    model_entry = metadata.get("model_file")
    model_path = source / "model.safetensors"
    model_raw = model_path.read_bytes()
    if (
        metadata.get("schema_version") != GRID_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("grid_config_sha256") != expected_config_sha256
        or metadata.get("experiment_seed") != expected_seed
        or metadata.get("arm_id") != expected_arm_id
        or metadata.get("victim_learning_rate_millionths") != expected_learning_rate_millionths
        or metadata.get("victim_repetition_multiplier") != expected_multiplier
        or metadata.get("round_id") != expected_round_id
        or result.experiment_seed != expected_seed
        or result.arm_id != expected_arm_id
        or result.round_id != expected_round_id
        or metadata.get("model_state_sha256") != result.final_model_sha256
        or not isinstance(model_entry, Mapping)
        or set(model_entry) != {"sha256", "size"}
        or model_entry.get("sha256") != hashlib.sha256(model_raw).hexdigest()
        or model_entry.get("size") != len(model_raw)
    ):
        raise FederatedGridError("checkpoint da grade diverge da identidade esperada")
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        from safetensors.torch import load_model
        load_model(model_bundle.model, str(model_path), strict=True)
        if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
            raise FederatedGridError("pesos do checkpoint da grade divergem")
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        if isinstance(error, FederatedGridError):
            raise
        raise FederatedGridError("falha ao carregar checkpoint da grade") from error
    return FederatedGridCheckpoint(
        experiment_seed=result.experiment_seed,
        arm_id=result.arm_id,
        victim_learning_rate_millionths=result.victim_learning_rate_millionths,
        victim_repetition_multiplier=result.victim_repetition_multiplier,
        round_id=result.round_id,
        model_state_sha256=result.final_model_sha256,
        grid_config_sha256=expected_config_sha256,
        artifact_sha256=_artifact_hash(source),
        round_result=result,
    )


__all__ = ["grid_round_result_from_payload", "load_grid_checkpoint", "save_grid_checkpoint"]
