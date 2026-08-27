"""Persistência segura e retomável da calibração federada de exposição."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .federated_exposure_checkpointing import exposure_round_result_from_payload
from .federated_exposure_contracts import (
    FederatedExposureError,
    FederatedExposureRoundResult,
    FederatedMemorizationCalibrationSpec,
    validate_federated_exposure_spec,
)
from .model_contracts import ModelProvenance
from .synthetic_profiles.storage import validate_storage_component


RUN_MANIFEST_SCHEMA_VERSION = "federated-exposure-run-manifest/v1"
ARM_STATE_SCHEMA_VERSION = "federated-exposure-arm-state/v1"


@dataclass(frozen=True, slots=True)
class FederatedExposurePaths:
    output_root: Path
    run_root: Path
    dataset_root: Path

    def arm_root(self, arm_id: str) -> Path:
        return self.run_root / "arms" / validate_storage_component(arm_id, "arm_id")


def canonical_json_bytes(value: Any) -> bytes:
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


def safe_payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FederatedExposureError("artefato contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except FederatedExposureError:
        raise
    except Exception as error:
        raise FederatedExposureError("artefato contém JSON inválido") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise FederatedExposureError("artefato não usa JSON canônico")
    return value


def read_safe_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise FederatedExposureError("artefato seguro está ausente")
    try:
        return _load(target.read_bytes())
    except OSError as error:
        raise FederatedExposureError("artefato seguro é inacessível") from error


def _write_exclusive(path: Path, payload: Any) -> None:
    raw = canonical_json_bytes(payload)
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, 0o600)


def write_idempotent(path: Path, payload: Any) -> None:
    raw = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise FederatedExposureError("artefato existente diverge da execução")
        os.chmod(path, 0o600)
        return
    _write_exclusive(path, payload)


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


def initialize_exposure_run(
    output_root: Path,
    run_id: str,
    spec: FederatedMemorizationCalibrationSpec,
    model_provenance: ModelProvenance,
    *,
    calibration_config_sha256: str,
    baseline_model_sha256: str,
    fresh: bool,
) -> FederatedExposurePaths:
    validate_federated_exposure_spec(spec)
    if not isinstance(model_provenance, ModelProvenance):
        raise FederatedExposureError("proveniência da execução é inválida")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (calibration_config_sha256, baseline_model_sha256)
    ):
        raise FederatedExposureError("identidade criptográfica da execução é inválida")
    resolved_run_id = validate_storage_component(run_id, "run_id")
    root = Path(output_root)
    if ".." in root.parts:
        raise FederatedExposureError("raiz de saída contém travessia de caminho")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise FederatedExposureError("raiz de saída é inválida")
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    datasets_root = root / "datasets"
    for directory in (runs_root, datasets_root):
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise FederatedExposureError("raiz de artefatos é inválida")
        os.chmod(directory, 0o700)
    run_root = runs_root / resolved_run_id
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "spec": spec.as_safe_dict(),
        "calibration_config_sha256": calibration_config_sha256,
        "baseline_model_sha256": baseline_model_sha256,
        "model_provenance": model_provenance.as_safe_dict(),
    }
    if fresh and run_root.exists():
        raise FileExistsError("execução federada de exposição já existe")
    if not run_root.exists():
        run_root.mkdir(parents=True, mode=0o700)
        os.chmod(run_root, 0o700)
        for name in ("baseline", "arms"):
            (run_root / name).mkdir(mode=0o700)
    elif run_root.is_symlink() or not run_root.is_dir():
        raise FederatedExposureError("diretório da execução é inválido")
    for name in ("baseline", "arms"):
        child = run_root / name
        child.mkdir(mode=0o700, exist_ok=True)
        if child.is_symlink() or not child.is_dir():
            raise FederatedExposureError("estrutura da execução é inválida")
        os.chmod(child, 0o700)
    write_idempotent(run_root / "run_manifest.json", manifest)
    return FederatedExposurePaths(
        output_root=root,
        run_root=run_root,
        dataset_root=datasets_root,
    )


def initialize_arm(paths: FederatedExposurePaths, arm_id: str) -> Path:
    root = paths.arm_root(arm_id)
    if not root.exists():
        root.mkdir(mode=0o700)
    elif root.is_symlink() or not root.is_dir():
        raise FederatedExposureError("diretório do braço é inválido")
    for name in ("rounds", "checkpoints"):
        child = root / name
        child.mkdir(mode=0o700, exist_ok=True)
        if child.is_symlink() or not child.is_dir():
            raise FederatedExposureError("estrutura do braço é inválida")
    return root


def load_arm_state(arm_root: Path, arm_id: str, multiplier: int) -> dict[str, Any]:
    state_path = arm_root / "state.json"
    if not state_path.exists():
        return {
            "schema_version": ARM_STATE_SCHEMA_VERSION,
            "arm_id": arm_id,
            "victim_repetition_multiplier": multiplier,
            "completed_round": 0,
            "current_model_sha256": None,
            "checkpoint_artifact_sha256": None,
        }
    state = read_safe_json(state_path)
    if (
        set(state)
        != {
            "schema_version",
            "arm_id",
            "victim_repetition_multiplier",
            "completed_round",
            "current_model_sha256",
            "checkpoint_artifact_sha256",
        }
        or state.get("schema_version") != ARM_STATE_SCHEMA_VERSION
        or state.get("arm_id") != arm_id
        or state.get("victim_repetition_multiplier") != multiplier
        or type(state.get("completed_round")) is not int
        or not 0 <= state["completed_round"] <= 20
    ):
        raise FederatedExposureError("estado do braço diverge")
    if state["completed_round"] == 0:
        if (
            state["current_model_sha256"] is not None
            or state["checkpoint_artifact_sha256"] is not None
        ):
            raise FederatedExposureError("estado inicial do braço é inválido")
    elif any(
        not isinstance(state[key], str)
        or len(state[key]) != 64
        or any(character not in "0123456789abcdef" for character in state[key])
        for key in ("current_model_sha256", "checkpoint_artifact_sha256")
    ):
        raise FederatedExposureError("hashes do estado do braço são inválidos")
    return state


def read_round_results(
    arm_root: Path, completed_round: int
) -> tuple[FederatedExposureRoundResult, ...]:
    results = []
    previous = None
    for round_id in range(1, completed_round + 1):
        result = exposure_round_result_from_payload(
            read_safe_json(arm_root / "rounds" / f"round-{round_id:03d}.json")
        )
        if result.round_id != round_id or (
            previous is not None and result.initial_model_sha256 != previous
        ):
            raise FederatedExposureError("continuidade persistida do braço diverge")
        previous = result.final_model_sha256
        results.append(result)
    return tuple(results)


def commit_exposure_round(
    arm_root: Path,
    result: FederatedExposureRoundResult,
    checkpoint_artifact_sha256: str,
) -> None:
    write_idempotent(
        arm_root / "rounds" / f"round-{result.round_id:03d}.json",
        result.as_safe_dict(),
    )
    state = {
        "schema_version": ARM_STATE_SCHEMA_VERSION,
        "arm_id": result.arm_id,
        "victim_repetition_multiplier": result.victim_repetition_multiplier,
        "completed_round": result.round_id,
        "current_model_sha256": result.final_model_sha256,
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
    }
    write_atomic(arm_root / "state.json", state)
    checkpoints = arm_root / "checkpoints"
    for candidate in checkpoints.iterdir():
        if candidate.name != f"round-{result.round_id:03d}":
            if candidate.is_symlink() or not candidate.is_dir():
                raise FederatedExposureError("resíduo de checkpoint é inválido")
            shutil.rmtree(candidate)


def validate_checkpoint_residue(arm_root: Path, completed_round: int) -> None:
    checkpoints = arm_root / "checkpoints"
    expected = None if completed_round == 0 else f"round-{completed_round:03d}"
    for candidate in tuple(checkpoints.iterdir()):
        if candidate.name != expected:
            if candidate.is_symlink() or not candidate.is_dir():
                raise FederatedExposureError("resíduo incompleto é inválido")
            shutil.rmtree(candidate)


def aggregate_round_results_sha256(
    results: Sequence[FederatedExposureRoundResult],
) -> str:
    return safe_payload_sha256([item.as_safe_dict() for item in results])


__all__ = [
    "FederatedExposurePaths",
    "aggregate_round_results_sha256",
    "canonical_json_bytes",
    "commit_exposure_round",
    "initialize_arm",
    "initialize_exposure_run",
    "load_arm_state",
    "read_round_results",
    "read_safe_json",
    "safe_payload_sha256",
    "validate_checkpoint_residue",
    "write_atomic",
    "write_idempotent",
]
