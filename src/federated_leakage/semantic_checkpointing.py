"""Checkpoints safetensors do piloto de substituição semântica."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_contracts import LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot
from .semantic_pilot_contracts import (
    SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
    SemanticFederatedRoundResult,
    SemanticPilotError,
    validate_semantic_round_result,
)
from .semantic_pilot_storage import semantic_round_result_from_payload


_FILES = frozenset({"metadata.json", "model.safetensors"})


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedSemanticCheckpoint:
    round_result: SemanticFederatedRoundResult
    config_sha256: str
    artifact_sha256: str
    schema_version: str = SEMANTIC_CHECKPOINT_SCHEMA_VERSION


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
                raise SemanticPilotError("checkpoint contém chave duplicada")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except SemanticPilotError:
        raise
    except Exception as error:
        raise SemanticPilotError("checkpoint contém JSON inválido") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        raise SemanticPilotError("checkpoint não usa JSON canônico")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"semantic-substitution-checkpoint-artifact/v1\0")
    for name in sorted(_FILES):
        digest.update(name.encode("ascii"))
        digest.update(hashlib.sha256((directory / name).read_bytes()).digest())
    return digest.hexdigest()


def save_semantic_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    round_result: SemanticFederatedRoundResult,
    *,
    config_sha256: str,
) -> str:
    result = validate_semantic_round_result(round_result)
    target = Path(target_directory)
    if (
        not _is_sha256(config_sha256)
        or target.name != f"round-{result.round_id:03d}"
        or target.exists()
        or target.is_symlink()
        or target.parent.is_symlink()
    ):
        raise SemanticPilotError("destino do checkpoint é inválido")
    if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
        raise SemanticPilotError("modelo diverge da rodada")
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(target.parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=".semantic-checkpoint-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        try:
            from safetensors.torch import save_model
            save_model(model_bundle.model, str(staging / "model.safetensors"))
            os.chmod(staging / "model.safetensors", 0o600)
        except Exception as error:
            raise SemanticPilotError("falha ao serializar checkpoint") from error
        model_raw = (staging / "model.safetensors").read_bytes()
        metadata = {
            "schema_version": SEMANTIC_CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "experiment_seed": result.experiment_seed,
            "scenario": result.scenario,
            "round_id": result.round_id,
            "model_state_sha256": result.final_model_sha256,
            "round_result": result.as_safe_dict(),
            "model_file": {
                "sha256": hashlib.sha256(model_raw).hexdigest(),
                "size": len(model_raw),
            },
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


def load_semantic_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_seed: int,
    expected_scenario: str,
    expected_round_id: int,
    expected_config_sha256: str,
) -> LoadedSemanticCheckpoint:
    source = Path(source_directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _FILES
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise SemanticPilotError("estrutura do checkpoint é inválida")
    metadata = _load((source / "metadata.json").read_bytes())
    expected_keys = {
        "schema_version", "config_sha256", "experiment_seed", "scenario",
        "round_id", "model_state_sha256", "round_result", "model_file",
    }
    if set(metadata) != expected_keys:
        raise SemanticPilotError("manifesto do checkpoint possui chaves inválidas")
    result = semantic_round_result_from_payload(metadata["round_result"])
    model_entry = metadata.get("model_file")
    model_path = source / "model.safetensors"
    model_raw = model_path.read_bytes()
    if (
        metadata.get("schema_version") != SEMANTIC_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("config_sha256") != expected_config_sha256
        or metadata.get("experiment_seed") != expected_seed
        or metadata.get("scenario") != expected_scenario
        or metadata.get("round_id") != expected_round_id
        or result.experiment_seed != expected_seed
        or result.scenario != expected_scenario
        or result.round_id != expected_round_id
        or metadata.get("model_state_sha256") != result.final_model_sha256
        or not isinstance(model_entry, Mapping)
        or set(model_entry) != {"sha256", "size"}
        or model_entry.get("sha256") != hashlib.sha256(model_raw).hexdigest()
        or model_entry.get("size") != len(model_raw)
    ):
        raise SemanticPilotError("checkpoint diverge da identidade esperada")
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        from safetensors.torch import load_model
        load_model(model_bundle.model, str(model_path), strict=True)
        if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
            raise SemanticPilotError("pesos do checkpoint divergem")
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        if isinstance(error, SemanticPilotError):
            raise
        raise SemanticPilotError("falha ao carregar checkpoint") from error
    return LoadedSemanticCheckpoint(
        round_result=result,
        config_sha256=expected_config_sha256,
        artifact_sha256=_artifact_hash(source),
    )


__all__ = [
    "LoadedSemanticCheckpoint",
    "load_semantic_checkpoint",
    "save_semantic_checkpoint",
]
