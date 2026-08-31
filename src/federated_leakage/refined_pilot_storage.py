"""Journal seguro e retomável do piloto refinado."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .refined_pilot_contracts import (
    REFINED_JOURNAL_SCHEMA_VERSION,
    REFINED_PILOT_SCHEMA_VERSION,
    RefinedPilotError,
    RefinedPilotSpec,
)
from .semantic_pilot_storage import read_safe_json, write_atomic, write_idempotent
from .synthetic_profiles.storage import validate_storage_component


@dataclass(frozen=True, slots=True)
class RefinedPilotPaths:
    output_root: Path
    run_root: Path

    def trajectory_root(self, scenario_id: str) -> Path:
        validate_storage_component(scenario_id, "scenario_id")
        return self.run_root / "trajectories" / scenario_id


def _directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise RefinedPilotError("diretório refinado é inválido")
    os.chmod(path, 0o700)


def initialize_refined_run(
    output_root: Path,
    run_id: str,
    seed: int,
    spec: RefinedPilotSpec,
    *,
    config_sha256: str,
    baseline_model_sha256: str,
    model_provenance: Mapping[str, Any],
    fresh: bool,
) -> RefinedPilotPaths:
    validate_storage_component(run_id, "run_id")
    root = Path(output_root) / "runs" / run_id
    if fresh and root.exists():
        raise FileExistsError("start exige que a execução refinada ainda não exista")
    _directory(root)
    for name in ("baseline", "trajectories", "gates"):
        _directory(root / name)
    manifest = {
        "schema_version": REFINED_PILOT_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_seed": seed,
        "config_sha256": config_sha256,
        "main_config_sha256": spec.main_config_sha256,
        "baseline_model_sha256": baseline_model_sha256,
        "model_provenance": dict(model_provenance),
        "scenario_order": list(spec.scenario_order),
    }
    write_idempotent(root / "run_manifest.json", manifest)
    return RefinedPilotPaths(output_root=Path(output_root), run_root=root)


def initialize_refined_trajectory(
    paths: RefinedPilotPaths,
    scenario_id: str,
    *,
    seed: int,
    baseline_model_sha256: str,
) -> Path:
    root = paths.trajectory_root(scenario_id)
    for path in (root, root / "rounds", root / "checkpoints", root / "checkpoints" / "permanent", root / "checkpoints" / "resume", root / "evaluator"):
        _directory(path)
    state_path = root / "state.json"
    if not state_path.exists():
        write_idempotent(
            state_path,
            {
            "schema_version": REFINED_JOURNAL_SCHEMA_VERSION,
            "seed": seed,
            "scenario_id": scenario_id,
            "baseline_model_sha256": baseline_model_sha256,
            "completed_round": 0,
            "current_model_sha256": baseline_model_sha256,
            "checkpoint_artifact_sha256": None,
            },
        )
    return root


def load_refined_trajectory_state(
    trajectory_root: Path,
    *,
    seed: int,
    scenario_id: str,
    baseline_model_sha256: str,
) -> dict[str, Any]:
    state = read_safe_json(Path(trajectory_root) / "state.json")
    expected = {
        "schema_version", "seed", "scenario_id", "baseline_model_sha256",
        "completed_round", "current_model_sha256", "checkpoint_artifact_sha256",
    }
    if (
        set(state) != expected
        or state.get("schema_version") != REFINED_JOURNAL_SCHEMA_VERSION
        or state.get("seed") != seed
        or state.get("scenario_id") != scenario_id
        or state.get("baseline_model_sha256") != baseline_model_sha256
        or type(state.get("completed_round")) is not int
        or not 0 <= state["completed_round"] <= 20
    ):
        raise RefinedPilotError("journal da trajetória refinada diverge")
    return state


def refined_checkpoint_directory(
    trajectory_root: Path,
    round_id: int,
    retained_rounds: Sequence[int],
) -> Path:
    category = "permanent" if round_id in set(retained_rounds) else "resume"
    return Path(trajectory_root) / "checkpoints" / category / f"round-{round_id:03d}"


def commit_refined_round(
    trajectory_root: Path,
    *,
    seed: int,
    scenario_id: str,
    baseline_model_sha256: str,
    round_id: int,
    final_model_sha256: str,
    checkpoint_artifact_sha256: str,
    round_payload: Mapping[str, Any],
    retained_rounds: Sequence[int],
) -> None:
    root = Path(trajectory_root)
    state = load_refined_trajectory_state(
        root,
        seed=seed,
        scenario_id=scenario_id,
        baseline_model_sha256=baseline_model_sha256,
    )
    if state["completed_round"] != round_id - 1:
        raise RefinedPilotError("commit refinado não é a próxima rodada")
    write_idempotent(root / "rounds" / f"round-{round_id:03d}.json", dict(round_payload))
    write_atomic(
        root / "state.json",
        {
            "schema_version": REFINED_JOURNAL_SCHEMA_VERSION,
            "seed": seed,
            "scenario_id": scenario_id,
            "baseline_model_sha256": baseline_model_sha256,
            "completed_round": round_id,
            "current_model_sha256": final_model_sha256,
            "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        },
    )
    if round_id not in set(retained_rounds):
        resume_root = root / "checkpoints" / "resume"
        current = f"round-{round_id:03d}"
        for item in tuple(resume_root.iterdir()):
            if item.name != current and item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)


def aggregate_refined_round_hash(trajectory_root: Path, completed_round: int) -> str:
    digest = hashlib.sha256(b"refined-defense-round-results/v1\0")
    for round_id in range(1, completed_round + 1):
        raw = (Path(trajectory_root) / "rounds" / f"round-{round_id:03d}.json").read_bytes()
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


__all__ = [
    "RefinedPilotPaths",
    "aggregate_refined_round_hash",
    "commit_refined_round",
    "initialize_refined_run",
    "initialize_refined_trajectory",
    "load_refined_trajectory_state",
    "refined_checkpoint_directory",
]
