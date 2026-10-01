"""Estado canônico, atômico e retomável do piloto não privado."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .aggregation_contracts import FedAvgRoundResult
from .audit_contracts import (
    ExtractionAuditResult,
    FieldAuditMetric,
    validate_extraction_audit_result,
)
from .execution_contracts import (
    FEDERATED_TRAJECTORY_SCHEMA_VERSION,
    PILOT_EXECUTION_SCHEMA_VERSION,
    FederatedTrajectoryState,
    PilotExecutionError,
    PilotExecutionResult,
    PilotExecutionSpec,
    PilotRunIdentity,
    validate_pilot_execution_spec,
)
from .federated_round import validate_federated_round_result
from .model_contracts import ModelProvenance
from .synthetic_profiles.storage import validate_storage_component
from .utility_evaluation import UtilityEvaluationResult, validate_utility_evaluation_result


RUN_MANIFEST_SCHEMA_VERSION = "pilot-run-manifest/v3"
ROUND_COMMIT_SCHEMA_VERSION = "federated-round-commit/v3"
PAIRED_ROUND_SCHEMA_VERSION = "paired-federated-round/v3"


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotRunPaths:
    output_root: Path
    run_root: Path
    dataset_root: Path
    identity: PilotRunIdentity

    def trajectory_root(self, scenario: str) -> Path:
        if scenario not in {"F0", "F1"}:
            raise PilotExecutionError("cenário de trajetória é inválido")
        return self.run_root / "trajectories" / f"{scenario}-k01"


def _canonical_json_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _load_json(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PilotExecutionError("artefato contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except PilotExecutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotExecutionError("artefato da execução é inválido") from error
    if not isinstance(value, dict):
        raise PilotExecutionError("artefato da execução deve ser objeto")
    return value


def _read_canonical_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PilotExecutionError("artefato da execução está ausente")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PilotExecutionError("artefato da execução é inacessível") from error
    payload = _load_json(raw)
    if _canonical_json_bytes(payload) != raw:
        raise PilotExecutionError("artefato da execução não é canônico")
    return payload


def _write_exclusive(path: Path, payload: Any) -> None:
    raw = _canonical_json_bytes(payload)
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(path, 0o600)
    except FileExistsError:
        raise
    except OSError as error:
        raise PilotExecutionError("falha ao publicar artefato da execução") from error


def _write_idempotent(path: Path, payload: Any) -> None:
    raw = _canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise PilotExecutionError("artefato existente diverge da execução")
        os.chmod(path, 0o600)
        return
    _write_exclusive(path, payload)


def _write_idempotent_atomic(path: Path, payload: Any) -> None:
    raw = _canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise PilotExecutionError("artefato existente diverge da execução")
        os.chmod(path, 0o600)
        return
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise PilotExecutionError("artefato existente diverge da execução")
            os.chmod(path, 0o600)
    except PilotExecutionError:
        raise
    except OSError as error:
        raise PilotExecutionError("falha ao publicar artefato da execução") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_atomic(path: Path, payload: Any) -> None:
    raw = _canonical_json_bytes(payload)
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise PilotExecutionError("diretório da execução é inválido")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=parent)
    temporary = Path(temporary_name)
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
        if isinstance(error, PilotExecutionError):
            raise
        raise PilotExecutionError("falha ao atualizar estado da execução") from error


def _ensure_private_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise PilotExecutionError("diretório da execução é inválido")
    os.chmod(path, 0o700)


def _provenance_from_payload(value: object) -> ModelProvenance:
    if not isinstance(value, Mapping):
        raise PilotExecutionError("proveniência persistida é inválida")
    expected = frozenset(field.name for field in fields(ModelProvenance))
    if frozenset(value) != expected:
        raise PilotExecutionError("proveniência persistida possui chaves inválidas")
    try:
        return ModelProvenance(**value)
    except (TypeError, ValueError) as error:
        raise PilotExecutionError("proveniência persistida possui tipos inválidos") from error


def round_result_from_safe_payload(value: object) -> FedAvgRoundResult:
    if not isinstance(value, Mapping):
        raise PilotExecutionError("resultado persistido da rodada é inválido")
    expected = frozenset(field.name for field in fields(FedAvgRoundResult))
    if frozenset(value) != expected:
        raise PilotExecutionError("resultado da rodada possui chaves inválidas")
    try:
        payload = dict(value)
        payload["model_provenance"] = _provenance_from_payload(
            payload["model_provenance"]
        )
        result = FedAvgRoundResult(**payload)
    except (TypeError, ValueError) as error:
        raise PilotExecutionError("resultado da rodada possui tipos inválidos") from error
    try:
        return validate_federated_round_result(result)
    except Exception as error:
        raise PilotExecutionError("resultado da rodada diverge do contrato") from error


_AUDIT_DERIVED_KEYS = frozenset(
    {
        "targeted_exact_pair_recall",
        "targeted_partial_pair_recall",
        "targeted_complete_generation_rate",
        "targeted_ordered_complete_generation_rate",
        "targeted_any_field_profile_exposure_rate",
        "targeted_misassociation_rate",
        "targeted_unseen_synthetic_value_rate",
    }
)


def audit_result_from_safe_payload(value: object) -> ExtractionAuditResult:
    if not isinstance(value, Mapping):
        raise PilotExecutionError("resultado persistido da auditoria é inválido")
    base_keys = frozenset(field.name for field in fields(ExtractionAuditResult))
    if frozenset(value) != base_keys | _AUDIT_DERIVED_KEYS:
        raise PilotExecutionError("resultado da auditoria possui chaves inválidas")
    payload = {key: value[key] for key in base_keys}
    metrics = payload.get("field_metrics")
    if not isinstance(metrics, list):
        raise PilotExecutionError("métricas persistidas da auditoria são inválidas")
    metric_base_keys = frozenset(field.name for field in fields(FieldAuditMetric))
    metric_keys = metric_base_keys | frozenset({"exact_recall", "partial_recall"})
    if any(
        not isinstance(metric, Mapping) or frozenset(metric) != metric_keys
        for metric in metrics
    ):
        raise PilotExecutionError("métrica persistida da auditoria é inválida")
    try:
        payload["field_metrics"] = tuple(
            FieldAuditMetric(**{key: metric[key] for key in metric_base_keys})
            for metric in metrics
        )
        payload["model_provenance"] = _provenance_from_payload(
            payload["model_provenance"]
        )
        result = ExtractionAuditResult(**payload)
        validate_extraction_audit_result(result)
    except (TypeError, ValueError) as error:
        raise PilotExecutionError("resultado persistido da auditoria é inválido") from error
    except Exception as error:
        raise PilotExecutionError("resultado persistido da auditoria diverge") from error
    if result.as_safe_dict() != dict(value):
        raise PilotExecutionError("taxas persistidas da auditoria divergem")
    return result


def utility_result_from_safe_payload(value: object) -> UtilityEvaluationResult:
    if not isinstance(value, Mapping):
        raise PilotExecutionError("resultado persistido de utilidade é inválido")
    expected = frozenset(field.name for field in fields(UtilityEvaluationResult))
    if frozenset(value) != expected:
        raise PilotExecutionError("resultado de utilidade possui chaves inválidas")
    try:
        payload = dict(value)
        payload["model_provenance"] = _provenance_from_payload(
            payload["model_provenance"]
        )
        result = UtilityEvaluationResult(**payload)
        validate_utility_evaluation_result(result)
    except Exception as error:
        raise PilotExecutionError("resultado persistido de utilidade diverge") from error
    if result.as_safe_dict() != dict(value):
        raise PilotExecutionError("resultado persistido de utilidade não é canônico")
    return result


def _run_manifest_payload(
    identity: PilotRunIdentity,
    spec: PilotExecutionSpec,
    provenance: ModelProvenance,
    baseline_model_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "execution_schema_version": spec.schema_version,
        "trajectory_schema_version": spec.trajectory_schema_version,
        "checkpoint_schema_version": spec.checkpoint_schema_version,
        "identity": identity.as_safe_dict(),
        "baseline_model_sha256": baseline_model_sha256,
        "model_provenance": provenance.as_safe_dict(),
        "rounds": spec.rounds,
        "scenario_order": list(spec.scenario_order),
        "target_counts": list(spec.target_counts),
        "reference_target_count": spec.reference_target_count,
        "sensitivity_target_counts": list(spec.sensitivity_target_counts),
        "retained_rounds": list(spec.retained_rounds),
        "expected_generation_count": spec.expected_generation_count,
        "learning_rate_millionths": identity.learning_rate_millionths,
        "utility_evaluation": {
            "schema_version": spec.utility_evaluation.schema_version,
            "dataset_schema_version": spec.utility_evaluation.dataset_schema_version,
            "result_schema_version": spec.utility_evaluation.result_schema_version,
            "checkpoints": list(spec.utility_evaluation.checkpoints),
            "automatic_gate": False,
            "human_review_required": True,
        },
        "paths": {
            "dataset": f"datasets/{identity.dataset_id}",
            "baseline": "baseline",
            "f0": "trajectories/F0-k01",
            "f1": "trajectories/F1-k01",
            "paired": "paired",
        },
    }


def initialize_pilot_run(
    *,
    output_root: Path,
    identity: PilotRunIdentity,
    spec: PilotExecutionSpec,
    model_provenance: ModelProvenance,
    baseline_model_sha256: str,
    fresh: bool,
) -> PilotRunPaths:
    validate_pilot_execution_spec(spec)
    try:
        validate_storage_component(identity.run_id, "run_id")
        validate_storage_component(identity.dataset_id, "dataset_id")
    except Exception as error:
        raise PilotExecutionError("identidade de armazenamento é inválida") from error
    root = Path(output_root)
    if ".." in root.parts:
        raise PilotExecutionError("raiz de saídas contém travessia de caminho")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise PilotExecutionError("raiz de saídas é inválida")
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs"
    datasets_root = root / "datasets"
    for directory in (runs_root, datasets_root):
        if not directory.exists():
            directory.mkdir(mode=0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise PilotExecutionError("raiz de artefatos é inválida")
        os.chmod(directory, 0o700)
    run_root = runs_root / identity.run_id
    expected_manifest = _run_manifest_payload(
        identity,
        spec,
        model_provenance,
        baseline_model_sha256,
    )
    if run_root.exists():
        if fresh:
            raise FileExistsError("execução do piloto já existe")
        if run_root.is_symlink() or not run_root.is_dir():
            raise PilotExecutionError("diretório da execução é inválido")
        manifest = _read_canonical_json(run_root / "run_manifest.json")
        if manifest != expected_manifest:
            raise PilotExecutionError("execução existente possui identidade divergente")
    else:
        run_root.mkdir(mode=0o700)
        _write_exclusive(run_root / "run_manifest.json", expected_manifest)
    os.chmod(run_root, 0o700)
    for relative in ("baseline", "trajectories", "paired"):
        _ensure_private_directory(run_root / relative)
    for scenario in ("F0", "F1"):
        trajectory = run_root / "trajectories" / f"{scenario}-k01"
        if not trajectory.exists():
            trajectory.mkdir(mode=0o700)
        if trajectory.is_symlink() or not trajectory.is_dir():
            raise PilotExecutionError("diretório de trajetória é inválido")
        os.chmod(trajectory, 0o700)
        for child in ("rounds", "checkpoints"):
            _ensure_private_directory(trajectory / child)
    return PilotRunPaths(
        output_root=root,
        run_root=run_root,
        dataset_root=datasets_root,
        identity=identity,
    )


def mark_baseline_completed(
    paths: PilotRunPaths,
    baseline_model_sha256: str,
    audits: Sequence[ExtractionAuditResult],
    utility_result: UtilityEvaluationResult,
) -> str:
    resolved = tuple(sorted(audits, key=lambda result: result.target_count))
    if tuple(result.target_count for result in resolved) != (1, 5, 20, 200):
        raise PilotExecutionError("auditorias B0 estão incompletas")
    audit_payload = [
        {
            "target_count": result.target_count,
            "result_sha256": safe_payload_sha256(result.as_safe_dict()),
            "generation_count": result.generation_count,
        }
        for result in resolved
    ]
    try:
        utility = validate_utility_evaluation_result(utility_result)
    except Exception as error:
        raise PilotExecutionError("utilidade B0 é inválida") from error
    if utility.scenario != "B0" or utility.model_state_sha256 != baseline_model_sha256:
        raise PilotExecutionError("utilidade B0 diverge do baseline")
    utility_sha256 = safe_payload_sha256(utility.as_safe_dict())
    baseline_audit_sha256 = safe_payload_sha256(
        {
            "schema_version": "pilot-baseline-audits/v3",
            "audits": audit_payload,
        }
    )
    payload = {
        "schema_version": "pilot-baseline/v3",
        "baseline_model_sha256": baseline_model_sha256,
        "baseline_audit_sha256": baseline_audit_sha256,
        "audits": audit_payload,
        "utility_result_sha256": utility_sha256,
    }
    write_utility_result(paths, utility)
    _write_idempotent(paths.run_root / "baseline" / "completed.json", payload)
    return baseline_audit_sha256


def persisted_audit_summary_path(
    paths: PilotRunPaths,
    *,
    scenario: str,
    round_id: int,
    target_count: int,
) -> Path:
    budget = f"targets-{target_count:03d}"
    if scenario == "B0":
        if round_id != 0:
            raise PilotExecutionError("rodada B0 é inválida")
        audit_id = f"B0-{budget}-round-000"
        run_root = paths.run_root / "baseline"
    elif scenario in {"F0", "F1"}:
        audit_id = f"{scenario}-k01-{budget}-round-{round_id:03d}"
        run_root = paths.trajectory_root(scenario)
    else:
        raise PilotExecutionError("cenário da auditoria é inválido")
    return run_root / "evaluator" / "summaries" / f"{audit_id}.json"


def read_persisted_audit_result(
    paths: PilotRunPaths,
    *,
    scenario: str,
    round_id: int,
    target_count: int,
) -> ExtractionAuditResult:
    return audit_result_from_safe_payload(
        _read_canonical_json(
            persisted_audit_summary_path(
                paths,
                scenario=scenario,
                round_id=round_id,
                target_count=target_count,
            )
        )
    )


def read_trajectory_state(
    paths: PilotRunPaths,
    scenario: str,
    *,
    baseline_model_sha256: str,
) -> FederatedTrajectoryState:
    root = paths.trajectory_root(scenario)
    state_path = root / "state.json"
    if not state_path.exists():
        return FederatedTrajectoryState(
            scenario=scenario,
            completed_round=0,
            baseline_model_sha256=baseline_model_sha256,
            current_model_sha256=baseline_model_sha256,
            checkpoint_artifact_sha256=None,
        )
    payload = _read_canonical_json(state_path)
    expected = frozenset(field.name for field in fields(FederatedTrajectoryState))
    if frozenset(payload) != expected:
        raise PilotExecutionError("estado da trajetória possui chaves inválidas")
    try:
        state = FederatedTrajectoryState(**payload)
    except (TypeError, ValueError) as error:
        raise PilotExecutionError("estado da trajetória possui tipos inválidos") from error
    if (
        state.schema_version != FEDERATED_TRAJECTORY_SCHEMA_VERSION
        or state.scenario != scenario
        or type(state.completed_round) is not int
        or not 0 <= state.completed_round <= 20
        or state.baseline_model_sha256 != baseline_model_sha256
        or len(state.current_model_sha256) != 64
        or (
            state.completed_round == 0
            and state.checkpoint_artifact_sha256 is not None
        )
        or (
            state.completed_round > 0
            and not isinstance(state.checkpoint_artifact_sha256, str)
        )
    ):
        raise PilotExecutionError("estado da trajetória diverge do contrato")
    return state


def checkpoint_id_for_round(round_id: int, retained_rounds: Sequence[int]) -> str:
    if type(round_id) is not int or not 1 <= round_id <= 20:
        raise PilotExecutionError("rodada do checkpoint é inválida")
    return (
        f"round-{round_id:03d}"
        if round_id in tuple(retained_rounds)
        else f"resume-round-{round_id:03d}"
    )


def checkpoint_path_for_round(
    paths: PilotRunPaths,
    scenario: str,
    round_id: int,
    retained_rounds: Sequence[int],
) -> Path:
    return (
        paths.trajectory_root(scenario)
        / "checkpoints"
        / checkpoint_id_for_round(round_id, retained_rounds)
    )


def _round_commit_payload(
    round_result: FedAvgRoundResult,
    audits: Sequence[ExtractionAuditResult],
    auxiliary_manifest: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_artifact_sha256: str,
    utility_result: UtilityEvaluationResult | None,
) -> dict[str, Any]:
    return {
        "schema_version": ROUND_COMMIT_SCHEMA_VERSION,
        "round_result": round_result.as_safe_dict(),
        "audits": [
            result.as_safe_dict()
            for result in sorted(audits, key=lambda item: item.target_count)
        ],
        "auxiliary_manifest": dict(auxiliary_manifest),
        "checkpoint_id": checkpoint_id,
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        "utility_result": (
            utility_result.as_safe_dict() if utility_result is not None else None
        ),
    }


def _rebuild_training_metrics(trajectory_root: Path, completed_round: int) -> None:
    raw = b""
    auxiliary_raw = b""
    for round_id in range(1, completed_round + 1):
        payload = _read_canonical_json(
            trajectory_root / "rounds" / f"round-{round_id:03d}.json"
        )
        raw += _canonical_json_bytes(payload["round_result"])
        auxiliary_raw += _canonical_json_bytes(payload["auxiliary_manifest"])
    path = trajectory_root / "training_metrics.jsonl"
    handle, name = tempfile.mkstemp(prefix=".training-metrics-", dir=trajectory_root)
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
        raise PilotExecutionError("falha ao reconstruir métricas") from error
    auxiliary_path = trajectory_root / "round_auxiliary_manifest.jsonl"
    handle, name = tempfile.mkstemp(prefix=".auxiliary-manifest-", dir=trajectory_root)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(auxiliary_raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, auxiliary_path)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise PilotExecutionError("falha ao reconstruir manifestos auxiliares") from error


def commit_trajectory_round(
    paths: PilotRunPaths,
    round_result: FedAvgRoundResult,
    audits: Sequence[ExtractionAuditResult],
    *,
    auxiliary_manifest: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_artifact_sha256: str,
    baseline_model_sha256: str,
    utility_result: UtilityEvaluationResult | None = None,
) -> FederatedTrajectoryState:
    result = validate_federated_round_result(round_result)
    if result.scenario not in {"F0", "F1"}:
        raise PilotExecutionError("cenário da rodada é inválido")
    resolved_audits = tuple(audits)
    expected_targets = (1, 5, 20, 200) if result.round_id == 20 else (20,)
    if tuple(sorted(audit.target_count for audit in resolved_audits)) != expected_targets:
        raise PilotExecutionError("auditorias da rodada estão incompletas")
    if result.round_id == 20:
        if utility_result is None:
            raise PilotExecutionError("utilidade da rodada final está ausente")
        try:
            utility = validate_utility_evaluation_result(utility_result)
        except Exception as error:
            raise PilotExecutionError("utilidade da rodada final é inválida") from error
        if (
            utility.scenario != result.scenario
            or utility.round_id != result.round_id
            or utility.model_state_sha256 != result.final_model_sha256
        ):
            raise PilotExecutionError("utilidade diverge da rodada final")
        write_utility_result(paths, utility)
    elif utility_result is not None:
        raise PilotExecutionError("utilidade só pode integrar a rodada final")
    current = read_trajectory_state(
        paths,
        result.scenario,
        baseline_model_sha256=baseline_model_sha256,
    )
    if current.completed_round not in {result.round_id - 1, result.round_id}:
        raise PilotExecutionError("commit da rodada está fora de ordem")
    payload = _round_commit_payload(
        result,
        resolved_audits,
        auxiliary_manifest,
        checkpoint_id,
        checkpoint_artifact_sha256,
        utility_result,
    )
    round_path = (
        paths.trajectory_root(result.scenario)
        / "rounds"
        / f"round-{result.round_id:03d}.json"
    )
    _write_idempotent(round_path, payload)
    state = FederatedTrajectoryState(
        scenario=result.scenario,
        completed_round=result.round_id,
        baseline_model_sha256=baseline_model_sha256,
        current_model_sha256=result.final_model_sha256,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
    )
    _rebuild_training_metrics(
        paths.trajectory_root(result.scenario),
        result.round_id,
    )
    _write_atomic(paths.trajectory_root(result.scenario) / "state.json", {
        field.name: getattr(state, field.name) for field in fields(FederatedTrajectoryState)
    })
    return state


def read_committed_round(
    paths: PilotRunPaths,
    scenario: str,
    round_id: int,
) -> tuple[
    FedAvgRoundResult,
    Tuple[ExtractionAuditResult, ...],
    str,
    str,
    UtilityEvaluationResult | None,
]:
    payload = _read_canonical_json(
        paths.trajectory_root(scenario)
        / "rounds"
        / f"round-{round_id:03d}.json"
    )
    expected_keys = frozenset(
        {
            "schema_version",
            "round_result",
            "audits",
            "auxiliary_manifest",
            "checkpoint_id",
            "checkpoint_artifact_sha256",
            "utility_result",
        }
    )
    if (
        frozenset(payload) != expected_keys
        or payload.get("schema_version") != ROUND_COMMIT_SCHEMA_VERSION
    ):
        raise PilotExecutionError("commit persistido da rodada é inválido")
    audits_raw = payload.get("audits")
    if not isinstance(audits_raw, list):
        raise PilotExecutionError("auditorias persistidas da rodada são inválidas")
    result = round_result_from_safe_payload(payload.get("round_result"))
    audits = tuple(audit_result_from_safe_payload(item) for item in audits_raw)
    checkpoint_id = payload.get("checkpoint_id")
    artifact_sha = payload.get("checkpoint_artifact_sha256")
    raw_utility = payload.get("utility_result")
    utility = (
        utility_result_from_safe_payload(raw_utility)
        if raw_utility is not None
        else None
    )
    if (
        result.scenario != scenario
        or result.round_id != round_id
        or not isinstance(checkpoint_id, str)
        or not isinstance(artifact_sha, str)
        or len(artifact_sha) != 64
        or (round_id == 20) != (utility is not None)
    ):
        raise PilotExecutionError("identidade persistida da rodada diverge")
    return result, audits, checkpoint_id, artifact_sha, utility


def commit_paired_round(
    paths: PilotRunPaths,
    benign: FedAvgRoundResult,
    adversarial: FedAvgRoundResult,
    audit_pairs: Sequence[tuple[ExtractionAuditResult, ExtractionAuditResult]],
    utility_pair: tuple[
        UtilityEvaluationResult,
        UtilityEvaluationResult,
    ] | None = None,
) -> None:
    if benign.round_id == 20:
        if utility_pair is None:
            raise PilotExecutionError("par de utilidade da rodada final está ausente")
        first, second = utility_pair
        try:
            validate_utility_evaluation_result(first)
            validate_utility_evaluation_result(second)
        except Exception as error:
            raise PilotExecutionError("par de utilidade é inválido") from error
        if (
            first.scenario != "F0"
            or second.scenario != "F1"
            or first.dataset_sha256 != second.dataset_sha256
            or first.model_state_sha256 != benign.final_model_sha256
            or second.model_state_sha256 != adversarial.final_model_sha256
        ):
            raise PilotExecutionError("par de utilidade diverge da rodada final")
    elif utility_pair is not None:
        raise PilotExecutionError("par de utilidade só pode existir na rodada final")
    payload = {
        "schema_version": PAIRED_ROUND_SCHEMA_VERSION,
        "round_id": benign.round_id,
        "experiment_seed": benign.experiment_seed,
        "auxiliary_weight_units": benign.auxiliary_weight_units,
        "benign_result_sha256": safe_payload_sha256(benign.as_safe_dict()),
        "adversarial_result_sha256": safe_payload_sha256(adversarial.as_safe_dict()),
        "audit_pairs": [
            {
                "target_count": first.target_count,
                "benign_result_sha256": safe_payload_sha256(first.as_safe_dict()),
                "adversarial_result_sha256": safe_payload_sha256(second.as_safe_dict()),
            }
            for first, second in sorted(audit_pairs, key=lambda pair: pair[0].target_count)
        ],
        "utility_pair": (
            {
                "benign_result_sha256": safe_payload_sha256(
                    utility_pair[0].as_safe_dict()
                ),
                "adversarial_result_sha256": safe_payload_sha256(
                    utility_pair[1].as_safe_dict()
                ),
            }
            if utility_pair is not None
            else None
        ),
    }
    _write_idempotent(
        paths.run_root / "paired" / f"round-{benign.round_id:03d}.json",
        payload,
    )


def utility_result_path(paths: PilotRunPaths, scenario: str) -> Path:
    if scenario == "B0":
        return paths.run_root / "baseline" / "evaluator" / "utility" / "summary.json"
    if scenario in {"F0", "F1"}:
        return paths.trajectory_root(scenario) / "evaluator" / "utility" / "round-020.json"
    raise PilotExecutionError("cenário da utilidade é inválido")


def write_utility_result(
    paths: PilotRunPaths,
    result: UtilityEvaluationResult,
) -> None:
    try:
        resolved = validate_utility_evaluation_result(result)
    except Exception as error:
        raise PilotExecutionError("resultado de utilidade é inválido") from error
    path = utility_result_path(paths, resolved.scenario)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PilotExecutionError("diretório da utilidade é inválido")
    os.chmod(path.parent, 0o700)
    _write_idempotent_atomic(path, resolved.as_safe_dict())


def read_utility_result(
    paths: PilotRunPaths,
    scenario: str,
) -> UtilityEvaluationResult:
    return utility_result_from_safe_payload(
        _read_canonical_json(utility_result_path(paths, scenario))
    )


def remove_obsolete_resume_checkpoints(
    paths: PilotRunPaths,
    scenario: str,
    *,
    keep_round: int,
) -> None:
    root = paths.trajectory_root(scenario) / "checkpoints"
    keep_name = f"resume-round-{keep_round:03d}"
    for candidate in root.iterdir():
        if (
            candidate.name.startswith("resume-round-")
            and candidate.name != keep_name
        ):
            if candidate.is_symlink() or not candidate.is_dir():
                raise PilotExecutionError("checkpoint móvel é inválido")
            shutil.rmtree(candidate)


def write_pilot_completed(
    paths: PilotRunPaths,
    result: PilotExecutionResult,
) -> None:
    if not result.completed:
        raise PilotExecutionError("resultado incompleto não pode ser publicado")
    _write_exclusive(paths.run_root / "completed.json", result.as_safe_dict())


def read_pilot_completed(paths: PilotRunPaths) -> dict[str, Any] | None:
    path = paths.run_root / "completed.json"
    return _read_canonical_json(path) if path.exists() else None


__all__ = [
    "PilotRunPaths",
    "audit_result_from_safe_payload",
    "checkpoint_id_for_round",
    "checkpoint_path_for_round",
    "commit_paired_round",
    "commit_trajectory_round",
    "initialize_pilot_run",
    "mark_baseline_completed",
    "read_committed_round",
    "read_pilot_completed",
    "read_persisted_audit_result",
    "read_trajectory_state",
    "remove_obsolete_resume_checkpoints",
    "round_result_from_safe_payload",
    "safe_payload_sha256",
    "read_utility_result",
    "utility_result_from_safe_payload",
    "utility_result_path",
    "write_utility_result",
    "write_pilot_completed",
]
