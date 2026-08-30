"""Persistência segura e retomável do piloto de substituição semântica."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .model_contracts import ModelProvenance
from .semantic_pilot_contracts import (
    SEMANTIC_PILOT_SCHEMA_VERSION,
    SEMANTIC_TRAJECTORY_SCHEMA_VERSION,
    SemanticFederatedRoundResult,
    SemanticPilotError,
    SemanticPilotSpec,
    validate_semantic_pilot_spec,
)
from .synthetic_profiles.storage import validate_storage_component


SEMANTIC_RUN_MANIFEST_SCHEMA_VERSION = "semantic-substitution-run-manifest/v1"
SEMANTIC_TRAJECTORY_STATE_SCHEMA_VERSION = "semantic-substitution-trajectory-state/v1"


@dataclass(frozen=True, slots=True)
class SemanticPilotPaths:
    output_root: Path
    run_root: Path

    def trajectory_root(self, scenario: str) -> Path:
        if scenario not in {"F0", "F1", "F4", "F5"}:
            raise SemanticPilotError("cenário da trajetória é inválido")
        return self.run_root / "trajectories" / scenario


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


def read_safe_json(path: Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise SemanticPilotError("artefato seguro está ausente")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SemanticPilotError("artefato contém chave duplicada")
            result[key] = value
        return result
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except SemanticPilotError:
        raise
    except Exception as error:
        raise SemanticPilotError("artefato seguro é inválido") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise SemanticPilotError("artefato seguro não é canônico")
    return value


def write_idempotent(path: Path, payload: Any) -> None:
    raw = canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise SemanticPilotError("artefato existente diverge da execução")
        os.chmod(path, 0o600)
        return
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(path, 0o600)
    except OSError as error:
        raise SemanticPilotError("falha ao publicar artefato seguro") from error


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
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise SemanticPilotError("falha ao atualizar estado da execução") from error


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise SemanticPilotError("diretório da execução é inválido")
    os.chmod(path, 0o700)


def initialize_semantic_pilot_run(
    output_root: Path,
    run_id: str,
    seed: int,
    spec: SemanticPilotSpec,
    model_provenance: ModelProvenance,
    *,
    config_sha256: str,
    baseline_model_sha256: str,
    fresh: bool,
) -> SemanticPilotPaths:
    resolved = validate_semantic_pilot_spec(spec)
    if (
        run_id != resolved.run_id_for_seed(seed)
        or config_sha256 != hashlib.sha256(
            (resolved.main_config_path.parent / "semantic-substitution-pilot-v1.yaml").read_bytes()
        ).hexdigest()
        or not isinstance(model_provenance, ModelProvenance)
    ):
        raise SemanticPilotError("identidade da execução é inválida")
    root = Path(output_root)
    if ".." in root.parts or (root.exists() and (root.is_symlink() or not root.is_dir())):
        raise SemanticPilotError("raiz de saída é inválida")
    _ensure_directory(root)
    _ensure_directory(root / "runs")
    run_root = root / "runs" / validate_storage_component(run_id, "run_id")
    if fresh and run_root.exists():
        raise FileExistsError("execução oficial já existe")
    _ensure_directory(run_root)
    _ensure_directory(run_root / "baseline")
    _ensure_directory(run_root / "trajectories")
    _ensure_directory(run_root / "paired")
    for scenario in resolved.scenario_order:
        trajectory = run_root / "trajectories" / scenario
        for directory in (
            trajectory,
            trajectory / "rounds",
            trajectory / "checkpoints",
            trajectory / "checkpoints" / "permanent",
            trajectory / "checkpoints" / "resume",
        ):
            _ensure_directory(directory)
    manifest = {
        "schema_version": SEMANTIC_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_seed": seed,
        "semantic_config_sha256": config_sha256,
        "main_config_sha256": resolved.main_config_sha256,
        "baseline_model_sha256": baseline_model_sha256,
        "model_provenance": model_provenance.as_safe_dict(),
        "spec": {
            "schema_version": resolved.schema_version,
            "scenario_order": list(resolved.scenario_order),
            "rounds": resolved.rounds,
            "victim_learning_rate_millionths": resolved.victim_learning_rate_millionths,
            "victim_repetition_multiplier": resolved.victim_repetition_multiplier,
            "auxiliary_learning_rate_millionths": resolved.auxiliary_learning_rate_millionths,
            "grid_combined_result_sha256": resolved.grid_combined_result_sha256,
        },
    }
    write_idempotent(run_root / "run_manifest.json", manifest)
    return SemanticPilotPaths(root, run_root)


def initialize_trajectory(paths: SemanticPilotPaths, scenario: str) -> Path:
    root = paths.trajectory_root(scenario)
    if root.is_symlink() or not root.is_dir():
        raise SemanticPilotError("trajetória está ausente")
    return root


def load_trajectory_state(
    trajectory_root: Path,
    *,
    seed: int,
    scenario: str,
    baseline_model_sha256: str,
) -> dict[str, Any]:
    path = trajectory_root / "state.json"
    if not path.exists():
        return {
            "schema_version": SEMANTIC_TRAJECTORY_STATE_SCHEMA_VERSION,
            "experiment_seed": seed,
            "scenario": scenario,
            "completed_round": 0,
            "baseline_model_sha256": baseline_model_sha256,
            "current_model_sha256": baseline_model_sha256,
            "checkpoint_artifact_sha256": None,
        }
    state = read_safe_json(path)
    expected = {
        "schema_version", "experiment_seed", "scenario", "completed_round",
        "baseline_model_sha256", "current_model_sha256", "checkpoint_artifact_sha256",
    }
    if (
        set(state) != expected
        or state.get("schema_version") != SEMANTIC_TRAJECTORY_STATE_SCHEMA_VERSION
        or state.get("experiment_seed") != seed
        or state.get("scenario") != scenario
        or state.get("baseline_model_sha256") != baseline_model_sha256
        or type(state.get("completed_round")) is not int
        or not 0 <= state["completed_round"] <= 20
    ):
        raise SemanticPilotError("estado da trajetória diverge")
    return state


def semantic_round_result_from_payload(value: object) -> SemanticFederatedRoundResult:
    from dataclasses import fields
    from .semantic_pilot_contracts import validate_semantic_round_result
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(SemanticFederatedRoundResult)}:
        raise SemanticPilotError("resultado persistido da rodada é inválido")
    provenance = value.get("model_provenance")
    if not isinstance(provenance, dict):
        raise SemanticPilotError("proveniência persistida é inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance)
        return validate_semantic_round_result(SemanticFederatedRoundResult(**payload))
    except Exception as error:
        raise SemanticPilotError("resultado persistido é incompatível") from error


def read_round_results(
    trajectory_root: Path,
    completed_round: int,
) -> Tuple[SemanticFederatedRoundResult, ...]:
    results = []
    previous = None
    for round_id in range(1, completed_round + 1):
        result = semantic_round_result_from_payload(
            read_safe_json(trajectory_root / "rounds" / f"round-{round_id:03d}.json")
        )
        if result.round_id != round_id or (
            previous is not None and result.initial_model_sha256 != previous
        ):
            raise SemanticPilotError("continuidade persistida diverge")
        previous = result.final_model_sha256
        results.append(result)
    return tuple(results)


def checkpoint_directory(
    trajectory_root: Path,
    round_id: int,
    retained_rounds: Sequence[int],
) -> Path:
    category = "permanent" if round_id in set(retained_rounds) else "resume"
    return trajectory_root / "checkpoints" / category / f"round-{round_id:03d}"


def commit_round(
    trajectory_root: Path,
    result: SemanticFederatedRoundResult,
    checkpoint_artifact_sha256: str,
) -> None:
    write_idempotent(
        trajectory_root / "rounds" / f"round-{result.round_id:03d}.json",
        result.as_safe_dict(),
    )
    write_atomic(
        trajectory_root / "state.json",
        {
            "schema_version": SEMANTIC_TRAJECTORY_STATE_SCHEMA_VERSION,
            "experiment_seed": result.experiment_seed,
            "scenario": result.scenario,
            "completed_round": result.round_id,
            "baseline_model_sha256": read_safe_json(
                trajectory_root.parent.parent / "run_manifest.json"
            )["baseline_model_sha256"],
            "current_model_sha256": result.final_model_sha256,
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        },
    )
    resume_root = trajectory_root / "checkpoints" / "resume"
    for candidate in tuple(resume_root.iterdir()):
        if candidate.name != f"round-{result.round_id:03d}":
            if candidate.is_symlink() or not candidate.is_dir():
                raise SemanticPilotError("resíduo de checkpoint é inválido")
            shutil.rmtree(candidate)


def aggregate_round_results_sha256(
    results: Sequence[SemanticFederatedRoundResult],
) -> str:
    if len(results) != 20:
        raise SemanticPilotError("trajetória exige vinte resultados de rodada")
    return hashlib.sha256(
        b"semantic-substitution-round-results/v1\0"
        + b"".join(canonical_json_bytes(value.as_safe_dict()) for value in results)
    ).hexdigest()


__all__ = [
    "SemanticPilotPaths", "aggregate_round_results_sha256", "canonical_json_bytes",
    "checkpoint_directory", "commit_round", "initialize_semantic_pilot_run",
    "initialize_trajectory", "load_trajectory_state", "read_round_results",
    "read_safe_json", "safe_payload_sha256", "semantic_round_result_from_payload",
    "write_atomic", "write_idempotent",
]
