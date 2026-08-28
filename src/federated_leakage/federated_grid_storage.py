"""Persistência segura, exclusiva e retomável da grade federada v2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .federated_grid_checkpointing import grid_round_result_from_payload
from .federated_grid_contracts import (
    FederatedGridError,
    FederatedGridRoundResult,
    FederatedGridSpec,
    GridArmSpec,
    validate_federated_grid_spec,
)
from .model_contracts import ModelProvenance
from .synthetic_profiles.storage import validate_storage_component


RUN_MANIFEST_SCHEMA_VERSION = "federated-memorization-grid-run-manifest/v2"
ARM_STATE_SCHEMA_VERSION = "federated-exposure-grid-arm-state/v2"


@dataclass(frozen=True, slots=True)
class FederatedGridPaths:
    output_root: Path
    run_root: Path
    dataset_root: Path

    def arm_root(self, arm_id: str) -> Path:
        return self.run_root / "arms" / validate_storage_component(arm_id, "arm_id")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def safe_payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_safe_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise FederatedGridError("artefato seguro da grade está ausente")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FederatedGridError("artefato da grade contém chave duplicada")
            result[key] = value
        return result
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("artefato seguro da grade é inválido") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise FederatedGridError("artefato da grade não usa JSON canônico")
    return value


def write_idempotent(path: Path, payload: Any) -> None:
    raw = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise FederatedGridError("artefato existente diverge da grade")
        os.chmod(path, 0o600)
        return
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def write_atomic(path: Path, payload: Any) -> None:
    raw = canonical_json_bytes(payload)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_grid_run(
    output_root: Path,
    run_id: str,
    seed: int,
    spec: FederatedGridSpec,
    model_provenance: ModelProvenance,
    *,
    grid_config_sha256: str,
    baseline_model_sha256: str,
    fresh: bool,
) -> FederatedGridPaths:
    resolved = validate_federated_grid_spec(spec)
    if run_id != resolved.run_id_for_seed(seed) or not isinstance(model_provenance, ModelProvenance):
        raise FederatedGridError("identidade da execução da grade é inválida")
    root = Path(output_root)
    if ".." in root.parts or (root.exists() and (root.is_symlink() or not root.is_dir())):
        raise FederatedGridError("raiz de saída da grade é inválida")
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    datasets_root = root / "datasets"
    for directory in (runs_root, datasets_root):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    run_root = runs_root / validate_storage_component(run_id, "run_id")
    if fresh and run_root.exists():
        raise FileExistsError("execução oficial da grade já existe")
    if not run_root.exists():
        run_root.mkdir(mode=0o700)
    elif run_root.is_symlink() or not run_root.is_dir():
        raise FederatedGridError("diretório da execução da grade é inválido")
    for name in ("baseline", "arms"):
        child = run_root / name
        child.mkdir(mode=0o700, exist_ok=True)
        if child.is_symlink() or not child.is_dir():
            raise FederatedGridError("estrutura da execução da grade é inválida")
        os.chmod(child, 0o700)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_seed": seed,
        "spec": resolved.as_safe_dict(),
        "grid_config_sha256": grid_config_sha256,
        "baseline_model_sha256": baseline_model_sha256,
        "model_provenance": model_provenance.as_safe_dict(),
    }
    write_idempotent(run_root / "run_manifest.json", manifest)
    return FederatedGridPaths(root, run_root, datasets_root)


def initialize_grid_arm(paths: FederatedGridPaths, arm: GridArmSpec) -> Path:
    root = paths.arm_root(arm.arm_id)
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise FederatedGridError("diretório do braço da grade é inválido")
    for name in ("rounds", "checkpoints"):
        child = root / name
        child.mkdir(mode=0o700, exist_ok=True)
        if child.is_symlink() or not child.is_dir():
            raise FederatedGridError("estrutura do braço da grade é inválida")
    return root


def load_grid_arm_state(arm_root: Path, seed: int, arm: GridArmSpec) -> dict[str, Any]:
    path = arm_root / "state.json"
    if not path.exists():
        return {
            "schema_version": ARM_STATE_SCHEMA_VERSION,
            "experiment_seed": seed,
            "arm_id": arm.arm_id,
            "victim_learning_rate_millionths": arm.victim_learning_rate_millionths,
            "victim_repetition_multiplier": arm.victim_repetition_multiplier,
            "completed_round": 0,
            "current_model_sha256": None,
            "checkpoint_artifact_sha256": None,
        }
    state = read_safe_json(path)
    expected = {"schema_version", "experiment_seed", "arm_id", "victim_learning_rate_millionths", "victim_repetition_multiplier", "completed_round", "current_model_sha256", "checkpoint_artifact_sha256"}
    if (
        set(state) != expected
        or state.get("schema_version") != ARM_STATE_SCHEMA_VERSION
        or state.get("experiment_seed") != seed
        or state.get("arm_id") != arm.arm_id
        or state.get("victim_learning_rate_millionths") != arm.victim_learning_rate_millionths
        or state.get("victim_repetition_multiplier") != arm.victim_repetition_multiplier
        or type(state.get("completed_round")) is not int
        or not 0 <= state["completed_round"] <= 20
    ):
        raise FederatedGridError("estado persistido do braço da grade diverge")
    if state["completed_round"] == 0:
        if state["current_model_sha256"] is not None or state["checkpoint_artifact_sha256"] is not None:
            raise FederatedGridError("estado inicial persistido da grade é inválido")
    elif any(
        not isinstance(state[key], str)
        or len(state[key]) != 64
        or any(character not in "0123456789abcdef" for character in state[key])
        for key in ("current_model_sha256", "checkpoint_artifact_sha256")
    ):
        raise FederatedGridError("hashes persistidos do braço da grade são inválidos")
    return state


def read_grid_round_results(arm_root: Path, completed_round: int) -> tuple[FederatedGridRoundResult, ...]:
    results = []
    previous = None
    for round_id in range(1, completed_round + 1):
        result = grid_round_result_from_payload(read_safe_json(arm_root / "rounds" / f"round-{round_id:03d}.json"))
        if result.round_id != round_id or (previous is not None and result.initial_model_sha256 != previous):
            raise FederatedGridError("continuidade persistida da grade diverge")
        previous = result.final_model_sha256
        results.append(result)
    return tuple(results)


def commit_grid_round(arm_root: Path, result: FederatedGridRoundResult, checkpoint_artifact_sha256: str) -> None:
    write_idempotent(arm_root / "rounds" / f"round-{result.round_id:03d}.json", result.as_safe_dict())
    write_atomic(
        arm_root / "state.json",
        {
            "schema_version": ARM_STATE_SCHEMA_VERSION,
            "experiment_seed": result.experiment_seed,
            "arm_id": result.arm_id,
            "victim_learning_rate_millionths": result.victim_learning_rate_millionths,
            "victim_repetition_multiplier": result.victim_repetition_multiplier,
            "completed_round": result.round_id,
            "current_model_sha256": result.final_model_sha256,
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        },
    )
    checkpoints = arm_root / "checkpoints"
    for candidate in checkpoints.iterdir():
        if candidate.name != f"round-{result.round_id:03d}":
            if candidate.is_symlink() or not candidate.is_dir():
                raise FederatedGridError("resíduo de checkpoint da grade é inválido")
            shutil.rmtree(candidate)


def validate_grid_checkpoint_residue(arm_root: Path, completed_round: int) -> None:
    expected = None if completed_round == 0 else f"round-{completed_round:03d}"
    for candidate in tuple((arm_root / "checkpoints").iterdir()):
        if candidate.name != expected:
            if candidate.is_symlink() or not candidate.is_dir():
                raise FederatedGridError("resíduo incompleto da grade é inválido")
            shutil.rmtree(candidate)


def aggregate_grid_round_results_sha256(results: Sequence[FederatedGridRoundResult]) -> str:
    if len(results) != 20:
        raise FederatedGridError("resultado agregado da grade exige vinte rodadas")
    return hashlib.sha256(
        b"federated-exposure-grid-round-results/v2\0" + b"".join(
            canonical_json_bytes(value.as_safe_dict()) for value in results
        )
    ).hexdigest()


__all__ = ["FederatedGridPaths", "aggregate_grid_round_results_sha256", "commit_grid_round", "initialize_grid_arm", "initialize_grid_run", "load_grid_arm_state", "read_grid_round_results", "read_safe_json", "safe_payload_sha256", "validate_grid_checkpoint_residue", "write_atomic", "write_idempotent"]
