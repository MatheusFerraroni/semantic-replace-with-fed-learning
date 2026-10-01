"""Checkpoints safetensors com accountants do piloto refinado."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

from .dp_accounting import validate_accountant_state
from .dp_contracts import DPAccountantState
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot
from .private_federated_round import (
    PrivateFederatedRoundResult,
    private_round_result_from_payload,
    validate_private_federated_round_result,
)
from .refined_pilot_contracts import REFINED_CHECKPOINT_SCHEMA_VERSION, RefinedPilotError
from .semantic_pilot_contracts import (
    SemanticFederatedRoundResult,
    validate_semantic_round_result,
)
from .semantic_pilot_storage import canonical_json_bytes, semantic_round_result_from_payload


RoundResult = SemanticFederatedRoundResult | PrivateFederatedRoundResult
_FILES = frozenset({"metadata.json", "model.safetensors"})


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedRefinedCheckpoint:
    round_result: RoundResult
    accountant_states: Tuple[DPAccountantState, ...]
    config_sha256: str
    artifact_sha256: str
    schema_version: str = REFINED_CHECKPOINT_SCHEMA_VERSION


def _scenario_identity(result: RoundResult) -> str:
    if isinstance(result, PrivateFederatedRoundResult):
        return f"{result.scenario}-epsilon-{int(result.target_epsilon)}"
    if isinstance(result, SemanticFederatedRoundResult):
        return result.scenario
    raise RefinedPilotError("resultado de checkpoint possui tipo inválido")


def _read_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RefinedPilotError("checkpoint refinado contém chave duplicada")
            value[key] = item
        return value
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except RefinedPilotError:
        raise
    except Exception as error:
        raise RefinedPilotError("checkpoint refinado contém JSON inválido") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise RefinedPilotError("checkpoint refinado não usa JSON canônico")
    return value


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256(b"refined-defense-checkpoint-artifact/v1\0")
    for name in sorted(_FILES):
        digest.update(name.encode("ascii"))
        digest.update(hashlib.sha256((directory / name).read_bytes()).digest())
    return digest.hexdigest()


def _round_from_payload(kind: str, value: object) -> RoundResult:
    if kind == "non_private":
        return semantic_round_result_from_payload(value)
    if kind == "private":
        return private_round_result_from_payload(value)
    raise RefinedPilotError("tipo de rodada do checkpoint é inválido")


def _accountant_from_payload(value: object) -> DPAccountantState:
    if not isinstance(value, dict) or set(value) != {
        field.name for field in fields(DPAccountantState)
    }:
        raise RefinedPilotError("accountant persistido é inválido")
    try:
        payload = dict(value)
        payload["history"] = tuple(tuple(item) for item in payload["history"])
        state = DPAccountantState(**payload)
    except Exception as error:
        raise RefinedPilotError("accountant persistido é incompatível") from error
    try:
        return validate_accountant_state(state)
    except Exception as error:
        raise RefinedPilotError("accountant persistido diverge") from error


def save_refined_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    round_result: RoundResult,
    accountant_states: Tuple[DPAccountantState, ...],
    *,
    config_sha256: str,
    scenario_id: str,
) -> str:
    target = Path(target_directory)
    private = isinstance(round_result, PrivateFederatedRoundResult)
    if private:
        validate_private_federated_round_result(round_result)
        if len(accountant_states) != 10:
            raise RefinedPilotError("checkpoint privado exige dez accountants")
    elif isinstance(round_result, SemanticFederatedRoundResult):
        validate_semantic_round_result(round_result)
        if accountant_states:
            raise RefinedPilotError("checkpoint não privado recebeu accountants")
    else:
        raise RefinedPilotError("resultado de checkpoint possui tipo inválido")
    if (
        len(config_sha256) != 64
        or scenario_id != _scenario_identity(round_result)
        or target.name != f"round-{round_result.round_id:03d}"
        or target.exists()
        or target.is_symlink()
        or target.parent.is_symlink()
        or fingerprint_model_parameters(model_bundle) != round_result.final_model_sha256
    ):
        raise RefinedPilotError("destino ou modelo do checkpoint é inválido")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=".refined-checkpoint-", dir=target.parent))
    try:
        os.chmod(staging, 0o700)
        from safetensors.torch import save_model
        save_model(model_bundle.model, str(staging / "model.safetensors"))
        os.chmod(staging / "model.safetensors", 0o600)
        model_bytes = (staging / "model.safetensors").read_bytes()
        payload = {
            "schema_version": REFINED_CHECKPOINT_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "scenario_id": scenario_id,
            "experiment_seed": round_result.experiment_seed,
            "round_id": round_result.round_id,
            "model_state_sha256": round_result.final_model_sha256,
            "round_kind": "private" if private else "non_private",
            "round_result": round_result.as_safe_dict(),
            "accountant_states": [value.as_safe_dict() for value in accountant_states],
            "model_file": {
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "size": len(model_bytes),
            },
        }
        with (staging / "metadata.json").open("xb") as output:
            output.write(canonical_json_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(staging / "metadata.json", 0o600)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _artifact_hash(target)


def load_refined_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_seed: int,
    expected_scenario_id: str,
    expected_round_id: int,
    expected_config_sha256: str,
) -> LoadedRefinedCheckpoint:
    source = Path(source_directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _FILES
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise RefinedPilotError("estrutura do checkpoint refinado é inválida")
    metadata = _read_json(source / "metadata.json")
    expected_keys = {
        "schema_version", "config_sha256", "scenario_id", "experiment_seed",
        "round_id", "model_state_sha256", "round_kind", "round_result",
        "accountant_states", "model_file",
    }
    if set(metadata) != expected_keys:
        raise RefinedPilotError("checkpoint refinado possui chaves inválidas")
    result = _round_from_payload(str(metadata.get("round_kind")), metadata.get("round_result"))
    if isinstance(result, SemanticFederatedRoundResult):
        validate_semantic_round_result(result)
    accountant_raw = metadata.get("accountant_states")
    if not isinstance(accountant_raw, list):
        raise RefinedPilotError("accountants do checkpoint são inválidos")
    accountants = tuple(_accountant_from_payload(value) for value in accountant_raw)
    model_path = source / "model.safetensors"
    model_bytes = model_path.read_bytes()
    model_file = metadata.get("model_file")
    if (
        metadata.get("schema_version") != REFINED_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("config_sha256") != expected_config_sha256
        or metadata.get("scenario_id") != expected_scenario_id
        or expected_scenario_id != _scenario_identity(result)
        or metadata.get("experiment_seed") != expected_seed
        or metadata.get("round_id") != expected_round_id
        or result.experiment_seed != expected_seed
        or result.round_id != expected_round_id
        or metadata.get("model_state_sha256") != result.final_model_sha256
        or not isinstance(model_file, Mapping)
        or set(model_file) != {"sha256", "size"}
        or model_file.get("sha256") != hashlib.sha256(model_bytes).hexdigest()
        or model_file.get("size") != len(model_bytes)
        or (isinstance(result, PrivateFederatedRoundResult) and len(accountants) != 10)
        or (isinstance(result, SemanticFederatedRoundResult) and accountants)
    ):
        raise RefinedPilotError("checkpoint refinado diverge da identidade esperada")
    snapshot = capture_model_parameter_snapshot(model_bundle)
    try:
        from safetensors.torch import load_model
        load_model(model_bundle.model, str(model_path), strict=True)
        if fingerprint_model_parameters(model_bundle) != result.final_model_sha256:
            raise RefinedPilotError("pesos do checkpoint refinado divergem")
    except Exception as error:
        restore_model_parameter_snapshot(model_bundle, snapshot)
        if isinstance(error, RefinedPilotError):
            raise
        raise RefinedPilotError("falha ao carregar checkpoint refinado") from error
    return LoadedRefinedCheckpoint(
        round_result=result,
        accountant_states=accountants,
        config_sha256=expected_config_sha256,
        artifact_sha256=_artifact_hash(source),
    )


__all__ = [
    "LoadedRefinedCheckpoint",
    "load_refined_checkpoint",
    "save_refined_checkpoint",
]
