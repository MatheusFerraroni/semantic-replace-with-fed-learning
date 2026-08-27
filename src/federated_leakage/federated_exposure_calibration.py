"""Orquestração retomável da calibração federada de exposição."""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .aggregation_contracts import load_fedavg_spec_from_config
from .audit_contracts import AuditCheckpoint, load_extraction_audit_spec_from_config
from .execution_contracts import PILOT_BASELINE_MODEL_SHA256
from .federated_exposure_checkpointing import (
    load_exposure_checkpoint,
    save_exposure_checkpoint,
)
from .federated_exposure_contracts import (
    DEFAULT_RUN_ID,
    ExposureArmSpec,
    FederatedExposureArmResult,
    FederatedExposureAuditResult,
    FederatedExposureError,
    FederatedExposurePreflightResult,
    FederatedMemorizationCalibrationResult,
    FederatedMemorizationCalibrationSpec,
    calibration_result_from_payload,
    exposure_arm_result_from_payload,
    result_sha256,
    validate_exposure_audit_result,
    validate_federated_exposure_spec,
)
from .federated_exposure_round import run_federated_exposure_round
from .federated_exposure_storage import (
    FederatedExposurePaths,
    aggregate_round_results_sha256,
    commit_exposure_round,
    initialize_arm,
    initialize_exposure_run,
    load_arm_state,
    read_round_results,
    read_safe_json,
    safe_payload_sha256,
    validate_checkpoint_residue,
    write_idempotent,
)
from .federated_round import (
    PreparedVictimTrainingInputs,
    prepare_auxiliary_training_input,
    prepare_victim_training_inputs,
)
from .model_contracts import DEFAULT_MODEL_CACHE, LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_loading import load_model_bundle, load_model_spec_from_config
from .model_updates import (
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)
from .reproducibility import (
    ReproducibilityEnvironmentError,
    validate_cuda_reproducibility_environment,
)
from .synthetic_profiles import (
    AuxiliaryRound,
    AuxiliaryRoundGenerator,
    HeldoutUtilityDatasetGenerator,
    PositiveCanaryDatasetGenerator,
    VictimClientDataset,
    VictimDatasetGenerator,
    build_round_manifest,
    build_victim_dataset_manifest,
    read_auxiliary_round,
    read_victim_client_dataset,
    validate_conversation_preflight,
    validate_no_cross_flow_collisions,
    write_auxiliary_round,
    write_victim_datasets,
)
from .training_contracts import load_local_training_spec_from_config
from .trusted_evaluator import (
    preflight_extraction_audit,
    prepare_trusted_evaluator,
    read_completed_distinctive_exposure_counts,
    run_extraction_audit,
)
from .utility_evaluation import (
    PreparedUtilityEvaluation,
    UtilityEvaluationResult,
    compare_utility_to_baseline,
    evaluate_utility,
    load_utility_evaluation_spec_from_config,
    prepare_utility_evaluation,
    utility_dataset_sha256,
)


BundleLoader = Callable[[], LoadedModelBundle]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _canonical_hash(payload: Any, domain: bytes) -> str:
    import json

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


def _calibration_config_sha256(path: Path) -> str:
    target = Path(path)
    try:
        if target.is_symlink() or not target.is_file():
            raise FederatedExposureError("configuração da calibração está ausente")
        return hashlib.sha256(target.read_bytes()).hexdigest()
    except OSError as error:
        raise FederatedExposureError("configuração da calibração é inacessível") from error


def _materialize_data_preflight(
    spec: FederatedMemorizationCalibrationSpec,
    victims: Sequence[VictimClientDataset],
) -> tuple[FederatedExposurePreflightResult, tuple[AuxiliaryRound, ...]]:
    resolved = validate_federated_exposure_spec(spec)
    victim_datasets = tuple(victims)
    generator = AuxiliaryRoundGenerator(
        resolved.experiment_seed,
        schedule_id=resolved.schedule_id,
    )
    benign_rounds = tuple(
        generator.generate(round_id, presentation="benign")
        for round_id in range(1, resolved.rounds + 1)
    )
    canary = PositiveCanaryDatasetGenerator(resolved.experiment_seed).generate()
    utility = HeldoutUtilityDatasetGenerator(resolved.experiment_seed).generate()
    try:
        validate_conversation_preflight(victim_datasets, benign_rounds)
        validate_no_cross_flow_collisions(
            (
                *(dataset.conversations for dataset in victim_datasets),
                *(item.conversations for item in benign_rounds),
                canary.conversations,
                utility.conversations,
            )
        )
        victim_manifest = build_victim_dataset_manifest(victim_datasets)
        manifests = tuple(build_round_manifest(item) for item in benign_rounds)
        utility_hash = utility_dataset_sha256(utility)
    except Exception as error:
        raise FederatedExposureError("preflight dos dados sintéticos falhou") from error
    schedule_hash = _canonical_hash(
        [manifest["schedule_sha256"] for manifest in manifests],
        b"pilot-auxiliary-schedule/v1",
    )
    if (
        victim_manifest["dataset_sha256"]
        != resolved.expected_victim_dataset_sha256
        or schedule_hash != resolved.expected_benign_schedule_sha256
        or utility_hash != resolved.expected_utility_dataset_sha256
    ):
        raise FederatedExposureError("hashes do preflight divergem da configuração")
    result = FederatedExposurePreflightResult(
        experiment_seed=resolved.experiment_seed,
        victim_client_count=len(victim_datasets),
        victim_conversation_count=sum(
            len(dataset.conversations) for dataset in victim_datasets
        ),
        auxiliary_round_count=len(benign_rounds),
        auxiliary_conversation_count=sum(
            len(item.conversations) for item in benign_rounds
        ),
        utility_profile_count=100,
        utility_conversation_count=len(utility.conversations),
        victim_dataset_sha256=victim_manifest["dataset_sha256"],
        benign_schedule_sha256=schedule_hash,
        utility_dataset_sha256=utility_hash,
    )
    return result, benign_rounds


def preflight_federated_memorization_calibration(
    spec: FederatedMemorizationCalibrationSpec,
    *,
    victim_datasets: Sequence[VictimClientDataset] | None = None,
) -> FederatedExposurePreflightResult:
    resolved = validate_federated_exposure_spec(spec)
    victims = (
        tuple(victim_datasets)
        if victim_datasets is not None
        else VictimDatasetGenerator(resolved.experiment_seed).generate()
    )
    result, _ = _materialize_data_preflight(resolved, victims)
    return result


def _validate_reference_pilot(
    output_root: Path,
    spec: FederatedMemorizationCalibrationSpec,
) -> Mapping[str, Any]:
    root = Path(output_root)
    if ".." in root.parts or root.is_symlink() or not root.is_dir():
        raise FederatedExposureError("raiz histórica do piloto é inválida")
    runs_root = root / "runs"
    run_root = runs_root / spec.reference_pilot_run_id
    trajectory_root = run_root / "trajectories" / "F0-k01"
    for directory in (runs_root, run_root, run_root / "trajectories", trajectory_root):
        if directory.is_symlink() or not directory.is_dir():
            raise FederatedExposureError("estrutura histórica do piloto é inválida")
    completed = read_safe_json(run_root / "completed.json")
    if set(completed) != {
        "schema_version",
        "identity",
        "baseline_model_sha256",
        "baseline_audit_sha256",
        "baseline_audit_count",
        "baseline_utility",
        "trajectories",
        "utility_comparisons",
        "total_federated_rounds",
        "total_conversation_count",
        "total_optimizer_steps",
        "total_audit_generations",
        "paired_results_sha256",
        "completed",
    }:
        raise FederatedExposureError("marcador histórico do piloto é inválido")
    identity = completed.get("identity")
    trajectories = completed.get("trajectories")
    if (
        not isinstance(identity, Mapping)
        or set(identity)
        != {
            "run_id",
            "dataset_id",
            "experiment_seed",
            "auxiliary_weight_units",
            "schedule_id",
            "config_sha256",
            "calibration_result_sha256",
            "calibration_manifest_sha256",
            "learning_rate_millionths",
            "schema_version",
        }
        or not isinstance(trajectories, list)
        or len(trajectories) != 2
    ):
        raise FederatedExposureError("referência histórica do piloto é inválida")
    f0 = next(
        (
            item
            for item in trajectories
            if isinstance(item, Mapping) and item.get("scenario") == "F0"
        ),
        None,
    )
    utility = f0.get("utility_result") if isinstance(f0, Mapping) else None
    if (
        completed.get("schema_version") != "pilot-execution/v3"
        or identity.get("schema_version") != "pilot-execution/v3"
        or identity.get("run_id") != spec.reference_pilot_run_id
        or identity.get("experiment_seed") != spec.experiment_seed
        or identity.get("auxiliary_weight_units") != 1
        or identity.get("schedule_id") != spec.schedule_id
        or identity.get("config_sha256") != spec.main_config_sha256
        or identity.get("learning_rate_millionths") != 30
        or completed.get("baseline_model_sha256")
        != PILOT_BASELINE_MODEL_SHA256
        or completed.get("baseline_audit_count") != 4
        or completed.get("completed") is not True
        or completed.get("paired_results_sha256")
        != spec.reference_paired_result_sha256
        or completed.get("total_federated_rounds") != 40
        or completed.get("total_conversation_count") != 44_000
        or completed.get("total_optimizer_steps") != 11_000
        or completed.get("total_audit_generations") != 12_992
        or not isinstance(f0, Mapping)
        or set(f0)
        != {
            "schema_version",
            "scenario",
            "experiment_seed",
            "auxiliary_weight_units",
            "completed_rounds",
            "conversation_count",
            "optimizer_steps",
            "baseline_model_sha256",
            "baseline_audit_sha256",
            "final_model_sha256",
            "round_result_count",
            "audit_result_count",
            "utility_result",
            "result_sha256",
        }
        or f0.get("schema_version") != "federated-trajectory/v3"
        or f0.get("scenario") != "F0"
        or f0.get("experiment_seed") != spec.experiment_seed
        or f0.get("auxiliary_weight_units") != 1
        or f0.get("completed_rounds") != 20
        or f0.get("conversation_count") != 22_000
        or f0.get("optimizer_steps") != 5_500
        or f0.get("baseline_model_sha256") != PILOT_BASELINE_MODEL_SHA256
        or f0.get("round_result_count") != 20
        or f0.get("audit_result_count") != 23
        or f0.get("result_sha256") != spec.reference_f0_trajectory_sha256
        or f0.get("final_model_sha256") != spec.reference_f0_final_model_sha256
        or not isinstance(utility, Mapping)
        or utility.get("scenario") != "F0"
        or utility.get("round_id") != 20
        or utility.get("conversation_count") != 500
        or utility.get("dataset_sha256")
        != spec.expected_utility_dataset_sha256
        or utility.get("model_state_sha256")
        != spec.reference_f0_final_model_sha256
        or utility.get("scientific_sha256") != spec.reference_f0_utility_sha256
    ):
        raise FederatedExposureError("referência histórica do piloto diverge")
    summary = read_safe_json(
        trajectory_root
        / "evaluator"
        / "summaries"
        / "F0-k01-targets-200-round-020.json"
    )
    if (
        summary.get("scenario") != "F0"
        or summary.get("round_id") != 20
        or summary.get("target_count") != 200
        or summary.get("model_state_sha256")
        != spec.reference_f0_final_model_sha256
    ):
        raise FederatedExposureError("auditoria histórica F0 é incompatível")
    return summary


def _ensure_victim_artifacts(
    paths: FederatedExposurePaths,
    spec: FederatedMemorizationCalibrationSpec,
    victims: Sequence[VictimClientDataset],
    *,
    allow_create: bool = True,
) -> tuple[VictimClientDataset, ...]:
    target = paths.dataset_root / spec.dataset_id
    if not target.exists():
        if not allow_create:
            raise FederatedExposureError(
                "datasets de uma execução concluída estão ausentes"
            )
        try:
            write_victim_datasets(paths.dataset_root, spec.dataset_id, victims)
        except Exception as error:
            raise FederatedExposureError("falha ao publicar datasets das vítimas") from error
    try:
        loaded = tuple(
            read_victim_client_dataset(
                paths.dataset_root,
                spec.dataset_id,
                f"victim-{index:02d}",
            )
            for index in range(1, 11)
        )
    except Exception as error:
        raise FederatedExposureError(
            "datasets persistidos das vítimas são inválidos"
        ) from error
    if loaded != tuple(victims):
        raise FederatedExposureError("datasets persistidos divergem da seed")
    return loaded


def _restore_confirmed_arm_state(
    *,
    arm_root: Path,
    arm: ExposureArmSpec,
    model_bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    config_sha256: str,
) -> None:
    state = load_arm_state(
        arm_root,
        arm.arm_id,
        arm.victim_repetition_multiplier,
    )
    completed_round = state["completed_round"]
    if completed_round == 0:
        restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
        return
    checkpoint = load_exposure_checkpoint(
        arm_root / "checkpoints" / f"round-{completed_round:03d}",
        model_bundle,
        expected_arm_id=arm.arm_id,
        expected_multiplier=arm.victim_repetition_multiplier,
        expected_round_id=completed_round,
        expected_config_sha256=config_sha256,
    )
    if (
        checkpoint.artifact_sha256 != state["checkpoint_artifact_sha256"]
        or checkpoint.model_state_sha256 != state["current_model_sha256"]
    ):
        raise FederatedExposureError("rollback do braço diverge do estado confirmado")


def _ensure_auxiliary_artifact(
    paths: FederatedExposurePaths,
    spec: FederatedMemorizationCalibrationSpec,
    round_data: AuxiliaryRound,
) -> AuxiliaryRound:
    target = (
        paths.dataset_root
        / spec.dataset_id
        / "clients"
        / "auxiliary"
        / spec.schedule_id
        / "benign"
        / f"round-{round_data.round_id:03d}"
    )
    if not target.exists():
        try:
            write_auxiliary_round(
                paths.dataset_root,
                spec.dataset_id,
                spec.schedule_id,
                round_data,
            )
        except Exception as error:
            raise FederatedExposureError("falha ao publicar rodada auxiliar") from error
    try:
        loaded = read_auxiliary_round(
            paths.dataset_root,
            spec.dataset_id,
            spec.schedule_id,
            "benign",
            round_data.round_id,
        )
    except Exception as error:
        raise FederatedExposureError("rodada auxiliar persistida é inválida") from error
    if loaded != round_data:
        raise FederatedExposureError("rodada auxiliar persistida diverge da agenda")
    return loaded


def _audit_identity(
    paths: FederatedExposurePaths,
    arm: ExposureArmSpec | None,
) -> tuple[Path, str]:
    if arm is None:
        return paths.run_root, "baseline"
    return paths.run_root / "arms", arm.arm_id


def _run_or_reuse_audit(
    *,
    audit_spec: Any,
    context: Any,
    model_bundle: LoadedModelBundle,
    model_sha256: str,
    paths: FederatedExposurePaths,
    arm: ExposureArmSpec | None,
) -> FederatedExposureAuditResult:
    output_root, run_id = _audit_identity(paths, arm)
    checkpoint = AuditCheckpoint(
        scenario="B0" if arm is None else "F0",
        experiment_seed=101,
        round_id=0 if arm is None else 20,
        auxiliary_weight_units=None if arm is None else 1,
        expected_model_sha256=model_sha256,
        model_provenance=model_bundle.provenance,
    )
    try:
        extraction = run_extraction_audit(
            audit_spec,
            context,
            checkpoint,
            model_bundle,
            output_root=output_root,
            run_id=run_id,
            resume=True,
        )
        pair_count, entity_count = read_completed_distinctive_exposure_counts(
            audit_spec,
            context,
            checkpoint,
            model_bundle,
            output_root=output_root,
            run_id=run_id,
        )
    except Exception as error:
        raise FederatedExposureError("auditoria federada de exposição falhou") from error
    result = FederatedExposureAuditResult(
        arm_id=None if arm is None else arm.arm_id,
        victim_repetition_multiplier=(
            0 if arm is None else arm.victim_repetition_multiplier
        ),
        extraction_result_sha256=safe_payload_sha256(extraction.as_safe_dict()),
        target_count=extraction.target_count,
        distinctive_exact_pair_count=pair_count,
        distinctive_exact_pair_denominator=1_000,
        distinctive_exposed_entity_count=entity_count,
        distinctive_entity_denominator=200,
        calibrated_at_checkpoint=pair_count >= 10 and entity_count >= 5,
        model_state_sha256=model_sha256,
    )
    validate_exposure_audit_result(result)
    relative = (
        paths.run_root / "baseline" / "exposure_audit.json"
        if arm is None
        else paths.arm_root(arm.arm_id) / "exposure_audit.json"
    )
    write_idempotent(relative, result.as_safe_dict())
    return result


def _run_or_reuse_utility(
    *,
    utility_spec: Any,
    prepared: PreparedUtilityEvaluation,
    model_bundle: LoadedModelBundle,
    model_sha256: str,
    paths: FederatedExposurePaths,
    arm: ExposureArmSpec | None,
) -> UtilityEvaluationResult:
    expected_scenario = "B0" if arm is None else "F0"
    expected_round = 0 if arm is None else 20
    path = (
        paths.run_root / "baseline" / "utility.json"
        if arm is None
        else paths.arm_root(arm.arm_id) / "utility.json"
    )
    if path.exists():
        from .execution_storage import utility_result_from_safe_payload

        try:
            result = utility_result_from_safe_payload(read_safe_json(path))
        except Exception as error:
            raise FederatedExposureError("utilidade persistida é inválida") from error
        if (
            result.dataset_sha256 != prepared.dataset_sha256
            or result.model_state_sha256 != model_sha256
            or result.model_provenance != model_bundle.provenance
            or result.scenario != expected_scenario
            or result.round_id != expected_round
            or result.experiment_seed != 101
        ):
            raise FederatedExposureError("utilidade persistida diverge do checkpoint")
        return result
    try:
        result = evaluate_utility(
            utility_spec,
            prepared,
            model_bundle,
            scenario=expected_scenario,
            round_id=expected_round,
            experiment_seed=101,
        )
    except Exception as error:
        raise FederatedExposureError("avaliação federada de utilidade falhou") from error
    write_idempotent(path, result.as_safe_dict())
    return result


def _load_completed_arm(
    paths: FederatedExposurePaths,
    arm: ExposureArmSpec,
    model_bundle: LoadedModelBundle,
    config_sha256: str,
) -> FederatedExposureArmResult | None:
    root = paths.arm_root(arm.arm_id)
    completed_path = root / "completed.json"
    if not completed_path.exists():
        return None
    result = exposure_arm_result_from_payload(read_safe_json(completed_path))
    checkpoint = load_exposure_checkpoint(
        root / "checkpoints" / "round-020",
        model_bundle,
        expected_arm_id=arm.arm_id,
        expected_multiplier=arm.victim_repetition_multiplier,
        expected_round_id=20,
        expected_config_sha256=config_sha256,
    )
    if (
        result.checkpoint_artifact_sha256 != checkpoint.artifact_sha256
        or fingerprint_model_parameters(model_bundle) != result.final_model_sha256
    ):
        raise FederatedExposureError("braço concluído diverge do checkpoint final")
    return result


def _revalidate_completed_run(
    *,
    paths: FederatedExposurePaths,
    spec: FederatedMemorizationCalibrationSpec,
    result: FederatedMemorizationCalibrationResult,
    model_bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    baseline_model_sha256: str,
    audit_spec: Any,
    evaluator_context: Any,
    utility_spec: Any,
    prepared_utility: PreparedUtilityEvaluation,
    config_sha256: str,
    reference_audit_summary: Mapping[str, Any],
) -> FederatedMemorizationCalibrationResult:
    restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
    baseline_audit = _run_or_reuse_audit(
        audit_spec=audit_spec,
        context=evaluator_context,
        model_bundle=model_bundle,
        model_sha256=baseline_model_sha256,
        paths=paths,
        arm=None,
    )
    baseline_utility = _run_or_reuse_utility(
        utility_spec=utility_spec,
        prepared=prepared_utility,
        model_bundle=model_bundle,
        model_sha256=baseline_model_sha256,
        paths=paths,
        arm=None,
    )
    if (
        baseline_audit != result.baseline_audit
        or baseline_utility != result.baseline_utility
    ):
        raise FederatedExposureError("baseline concluído diverge do marcador final")
    loaded_arms = []
    for arm, expected in zip(spec.arms, result.arms):
        restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
        loaded = _load_completed_arm(paths, arm, model_bundle, config_sha256)
        if loaded is None:
            raise FederatedExposureError("braço confirmado está ausente")
        audit = _run_or_reuse_audit(
            audit_spec=audit_spec,
            context=evaluator_context,
            model_bundle=model_bundle,
            model_sha256=loaded.final_model_sha256,
            paths=paths,
            arm=arm,
        )
        utility = _run_or_reuse_utility(
            utility_spec=utility_spec,
            prepared=prepared_utility,
            model_bundle=model_bundle,
            model_sha256=loaded.final_model_sha256,
            paths=paths,
            arm=arm,
        )
        if loaded != expected or audit != loaded.audit or utility != loaded.utility:
            raise FederatedExposureError("braço confirmado diverge do marcador final")
        if arm.victim_repetition_multiplier == 1:
            summary = read_safe_json(
                paths.arm_root(arm.arm_id)
                / "evaluator"
                / "summaries"
                / "F0-k01-targets-200-round-020.json"
            )
            if summary != dict(reference_audit_summary):
                raise FederatedExposureError("regressão histórica do braço 1x diverge")
        loaded_arms.append(loaded)
    comparisons = tuple(
        compare_utility_to_baseline(baseline_utility, item.utility)
        for item in loaded_arms
    )
    if comparisons != result.utility_comparisons:
        raise FederatedExposureError("comparações de utilidade concluídas divergem")
    return result


def _run_arm(
    *,
    spec: FederatedMemorizationCalibrationSpec,
    arm: ExposureArmSpec,
    paths: FederatedExposurePaths,
    model_bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    baseline_model_sha256: str,
    victim_inputs: PreparedVictimTrainingInputs,
    local_spec: Any,
    fedavg_spec: Any,
    audit_spec: Any,
    evaluator_context: Any,
    utility_spec: Any,
    prepared_utility: PreparedUtilityEvaluation,
    config_sha256: str,
    reference_audit_summary: Mapping[str, Any],
    progress_callback: ProgressCallback | None,
) -> FederatedExposureArmResult:
    restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
    initialize_arm(paths, arm.arm_id)
    completed = _load_completed_arm(paths, arm, model_bundle, config_sha256)
    if completed is not None:
        _run_or_reuse_audit(
            audit_spec=audit_spec,
            context=evaluator_context,
            model_bundle=model_bundle,
            model_sha256=completed.final_model_sha256,
            paths=paths,
            arm=arm,
        )
        _run_or_reuse_utility(
            utility_spec=utility_spec,
            prepared=prepared_utility,
            model_bundle=model_bundle,
            model_sha256=completed.final_model_sha256,
            paths=paths,
            arm=arm,
        )
        return completed

    arm_root = paths.arm_root(arm.arm_id)
    state = load_arm_state(
        arm_root,
        arm.arm_id,
        arm.victim_repetition_multiplier,
    )
    validate_checkpoint_residue(arm_root, state["completed_round"])
    rounds = list(read_round_results(arm_root, state["completed_round"]))
    if state["completed_round"]:
        checkpoint = load_exposure_checkpoint(
            arm_root
            / "checkpoints"
            / f"round-{state['completed_round']:03d}",
            model_bundle,
            expected_arm_id=arm.arm_id,
            expected_multiplier=arm.victim_repetition_multiplier,
            expected_round_id=state["completed_round"],
            expected_config_sha256=config_sha256,
        )
        if (
            checkpoint.round_result != rounds[-1]
            or checkpoint.artifact_sha256
            != state["checkpoint_artifact_sha256"]
            or checkpoint.model_state_sha256 != state["current_model_sha256"]
        ):
            raise FederatedExposureError("estado retomado do braço diverge")
    elif fingerprint_model_parameters(model_bundle) != baseline_model_sha256:
        raise FederatedExposureError("braço não iniciou no baseline")

    generator = AuxiliaryRoundGenerator(
        spec.experiment_seed,
        schedule_id=spec.schedule_id,
    )
    for round_id in range(state["completed_round"] + 1, spec.rounds + 1):
        round_data = _ensure_auxiliary_artifact(
            paths,
            spec,
            generator.generate(round_id, presentation="benign"),
        )
        try:
            auxiliary_input = prepare_auxiliary_training_input(
                round_data,
                model_bundle,
            )
            result = run_federated_exposure_round(
                victim_inputs,
                auxiliary_input,
                model_bundle,
                local_spec,
                fedavg_spec,
                seed=spec.experiment_seed,
                round_id=round_id,
                arm=arm,
            )
            expected_initial = (
                baseline_model_sha256
                if not rounds
                else rounds[-1].final_model_sha256
            )
            if (
                result.initial_model_sha256 != expected_initial
                or result.victim_dataset_sha256
                != spec.expected_victim_dataset_sha256
            ):
                raise FederatedExposureError("continuidade da rodada diverge")
            checkpoint_path = (
                arm_root / "checkpoints" / f"round-{round_id:03d}"
            )
            artifact_hash = save_exposure_checkpoint(
                checkpoint_path,
                model_bundle,
                result,
                calibration_config_sha256=config_sha256,
            )
            commit_exposure_round(arm_root, result, artifact_hash)
        except Exception as error:
            try:
                _restore_confirmed_arm_state(
                    arm_root=arm_root,
                    arm=arm,
                    model_bundle=model_bundle,
                    baseline_snapshot=baseline_snapshot,
                    config_sha256=config_sha256,
                )
            except Exception as rollback_error:
                raise FederatedExposureError(
                    f"rollback do braço falhou na rodada {round_id}"
                ) from rollback_error
            if isinstance(error, FederatedExposureError):
                raise
            raise FederatedExposureError(
                f"execução do braço falhou na rodada {round_id}"
            ) from error
        rounds.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "round_completed",
                    "arm_id": arm.arm_id,
                    "round_id": round_id,
                    "victim_repetition_multiplier": (
                        arm.victim_repetition_multiplier
                    ),
                    "final_model_sha256": result.final_model_sha256,
                    "mean_client_loss": result.mean_client_loss,
                }
            )

    final_model_sha256 = fingerprint_model_parameters(model_bundle)
    audit = _run_or_reuse_audit(
        audit_spec=audit_spec,
        context=evaluator_context,
        model_bundle=model_bundle,
        model_sha256=final_model_sha256,
        paths=paths,
        arm=arm,
    )
    utility = _run_or_reuse_utility(
        utility_spec=utility_spec,
        prepared=prepared_utility,
        model_bundle=model_bundle,
        model_sha256=final_model_sha256,
        paths=paths,
        arm=arm,
    )
    state = load_arm_state(
        arm_root,
        arm.arm_id,
        arm.victim_repetition_multiplier,
    )
    result = FederatedExposureArmResult(
        arm_id=arm.arm_id,
        victim_repetition_multiplier=arm.victim_repetition_multiplier,
        completed_rounds=len(rounds),
        conversation_presentations=sum(
            item.conversation_presentations for item in rounds
        ),
        optimizer_steps=sum(item.optimizer_steps for item in rounds),
        baseline_model_sha256=baseline_model_sha256,
        final_model_sha256=final_model_sha256,
        round_result_sha256=aggregate_round_results_sha256(rounds),
        audit=audit,
        utility=utility,
        checkpoint_artifact_sha256=state["checkpoint_artifact_sha256"],
    )
    exposure_arm_result_from_payload(result.as_safe_dict())
    if arm.victim_repetition_multiplier == 1:
        extraction_summary = read_safe_json(
            arm_root
            / "evaluator"
            / "summaries"
            / "F0-k01-targets-200-round-020.json"
        )
        if (
            result.final_model_sha256 != spec.reference_f0_final_model_sha256
            or result.utility.scientific_sha256
            != spec.reference_f0_utility_sha256
            or extraction_summary != dict(reference_audit_summary)
        ):
            raise FederatedExposureError("regressão científica do braço 1x falhou")
    write_idempotent(arm_root / "completed.json", result.as_safe_dict())
    if progress_callback is not None:
        progress_callback(
            {
                "event": "arm_completed",
                "arm_id": arm.arm_id,
                "victim_repetition_multiplier": arm.victim_repetition_multiplier,
                "distinctive_exact_pair_count": (
                    audit.distinctive_exact_pair_count
                ),
                "distinctive_exposed_entity_count": (
                    audit.distinctive_exposed_entity_count
                ),
                "calibrated_at_checkpoint": audit.calibrated_at_checkpoint,
                "final_model_sha256": final_model_sha256,
            }
        )
    return result


def _default_bundle_loader(
    *,
    config_path: Path,
    cache_dir: Path,
    model_artifact_dir: Path | None,
    device: str,
) -> BundleLoader:
    model_spec = load_model_spec_from_config(config_path)

    def load() -> LoadedModelBundle:
        return load_model_bundle(
            model_spec,
            cache_dir=cache_dir,
            model_artifact_dir=model_artifact_dir,
            device=device,
        )

    return load


def run_federated_memorization_calibration(
    spec: FederatedMemorizationCalibrationSpec,
    *,
    config_path: Path,
    output_root: Path = Path("outputs"),
    run_id: str | None = None,
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    model_artifact_dir: Path | None = None,
    device: str,
    preflight_only: bool = False,
    fresh: bool = False,
    model_loader: BundleLoader | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FederatedMemorizationCalibrationResult | FederatedExposurePreflightResult:
    """Executa B0 e três trajetórias F0 independentes e retomáveis."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise FederatedExposureError(str(error)) from error
    resolved = validate_federated_exposure_spec(spec)
    effective_run_id = run_id or resolved.default_run_id
    if effective_run_id != DEFAULT_RUN_ID:
        raise FederatedExposureError("run_id diverge da execução oficial")
    calibration_config_sha256 = _calibration_config_sha256(config_path)
    reference_audit = _validate_reference_pilot(output_root, resolved)
    try:
        victims = VictimDatasetGenerator(resolved.experiment_seed).generate()
        data_preflight, _ = _materialize_data_preflight(resolved, victims)
        audit_spec = load_extraction_audit_spec_from_config(
            resolved.main_config_path
        )
        local_spec = load_local_training_spec_from_config(
            resolved.main_config_path
        )
        fedavg_spec = load_fedavg_spec_from_config(resolved.main_config_path)
        utility_spec = load_utility_evaluation_spec_from_config(
            resolved.main_config_path
        )
        utility_dataset = HeldoutUtilityDatasetGenerator(
            resolved.experiment_seed
        ).generate()
        if (
            local_spec.learning_rate != 3e-5
            or utility_dataset_sha256(utility_dataset)
            != resolved.expected_utility_dataset_sha256
        ):
            raise FederatedExposureError("receita científica da calibração diverge")
    except FederatedExposureError:
        raise
    except Exception as error:
        raise FederatedExposureError("preflight da configuração falhou") from error
    if progress_callback is not None:
        progress_callback(
            {
                "event": "data_preflight_completed",
                "victim_conversation_count": data_preflight.victim_conversation_count,
                "auxiliary_conversation_count": (
                    data_preflight.auxiliary_conversation_count
                ),
                "benign_schedule_sha256": data_preflight.benign_schedule_sha256,
                "utility_dataset_sha256": data_preflight.utility_dataset_sha256,
            }
        )
    loader = model_loader or _default_bundle_loader(
        config_path=resolved.main_config_path,
        cache_dir=cache_dir,
        model_artifact_dir=model_artifact_dir,
        device=device,
    )
    try:
        bundle = loader()
        baseline_model_sha256 = fingerprint_model_parameters(bundle)
        if baseline_model_sha256 != PILOT_BASELINE_MODEL_SHA256:
            raise FederatedExposureError("modelo inicial diverge do Tucano pinado")
        victim_inputs = prepare_victim_training_inputs(victims, bundle)
        prepared_utility = prepare_utility_evaluation(utility_dataset, bundle)
        evaluator_context = prepare_trusted_evaluator(
            victims,
            resolved.experiment_seed,
            target_count=resolved.audit_target_count,
        )
        preflight_extraction_audit(audit_spec, evaluator_context, bundle)
    except FederatedExposureError:
        raise
    except Exception as error:
        raise FederatedExposureError("preflight do modelo e tokenizador falhou") from error
    if preflight_only:
        return FederatedExposurePreflightResult(
            **{
                **data_preflight.as_safe_dict(),
                "model_state_sha256": baseline_model_sha256,
                "tokenization_validated": True,
                "audit_validated": True,
            }
        )

    paths = initialize_exposure_run(
        output_root,
        effective_run_id,
        resolved,
        bundle.provenance,
        calibration_config_sha256=calibration_config_sha256,
        baseline_model_sha256=baseline_model_sha256,
        fresh=fresh,
    )
    completed_path = paths.run_root / "completed.json"
    if completed_path.exists():
        persisted_victims = _ensure_victim_artifacts(
            paths,
            resolved,
            victims,
            allow_create=False,
        )
    else:
        persisted_victims = _ensure_victim_artifacts(paths, resolved, victims)
    if persisted_victims != victims:
        raise FederatedExposureError("datasets retomados divergem do preflight")
    baseline_snapshot = capture_model_parameter_snapshot(bundle)
    if completed_path.exists():
        if fresh:
            raise FileExistsError("execução federada de exposição já existe")
        return _revalidate_completed_run(
            paths=paths,
            spec=resolved,
            result=calibration_result_from_payload(
                read_safe_json(completed_path)
            ),
            model_bundle=bundle,
            baseline_snapshot=baseline_snapshot,
            baseline_model_sha256=baseline_model_sha256,
            audit_spec=audit_spec,
            evaluator_context=evaluator_context,
            utility_spec=utility_spec,
            prepared_utility=prepared_utility,
            config_sha256=calibration_config_sha256,
            reference_audit_summary=reference_audit,
        )

    baseline_audit = _run_or_reuse_audit(
        audit_spec=audit_spec,
        context=evaluator_context,
        model_bundle=bundle,
        model_sha256=baseline_model_sha256,
        paths=paths,
        arm=None,
    )
    baseline_utility = _run_or_reuse_utility(
        utility_spec=utility_spec,
        prepared=prepared_utility,
        model_bundle=bundle,
        model_sha256=baseline_model_sha256,
        paths=paths,
        arm=None,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "baseline_completed",
                "generation_count": 1_801,
                "distinctive_exact_pair_count": (
                    baseline_audit.distinctive_exact_pair_count
                ),
                "model_state_sha256": baseline_model_sha256,
            }
        )

    arms = []
    for arm in resolved.arms:
        result = _run_arm(
            spec=resolved,
            arm=arm,
            paths=paths,
            model_bundle=bundle,
            baseline_snapshot=baseline_snapshot,
            baseline_model_sha256=baseline_model_sha256,
            victim_inputs=victim_inputs,
            local_spec=local_spec,
            fedavg_spec=fedavg_spec,
            audit_spec=audit_spec,
            evaluator_context=evaluator_context,
            utility_spec=utility_spec,
            prepared_utility=prepared_utility,
            config_sha256=calibration_config_sha256,
            reference_audit_summary=reference_audit,
            progress_callback=progress_callback,
        )
        arms.append(result)
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        gc.collect()

    successful = tuple(
        item.victim_repetition_multiplier
        for item in arms
        if item.audit.calibrated_at_checkpoint
    )
    comparisons = tuple(
        compare_utility_to_baseline(baseline_utility, item.utility)
        for item in arms
    )
    unsigned = {
        "schema_version": "federated-memorization-calibration/v1",
        "run_id": effective_run_id,
        "experiment_seed": resolved.experiment_seed,
        "baseline_model_sha256": baseline_model_sha256,
        "baseline_gate_passed": baseline_audit.calibrated_at_checkpoint,
        "calibrated": bool(successful)
        and not baseline_audit.calibrated_at_checkpoint,
        "first_successful_multiplier": successful[0] if successful else None,
        "baseline_audit": baseline_audit.as_safe_dict(),
        "baseline_utility": baseline_utility.as_safe_dict(),
        "arms": [item.as_safe_dict() for item in arms],
        "utility_comparisons": [item.as_safe_dict() for item in comparisons],
        "total_federated_rounds": sum(item.completed_rounds for item in arms),
        "total_conversation_presentations": sum(
            item.conversation_presentations for item in arms
        ),
        "total_optimizer_steps": sum(item.optimizer_steps for item in arms),
        "total_audit_generations": 1_801 * (1 + len(arms)),
        "total_utility_conversations": baseline_utility.conversation_count
        + sum(item.utility.conversation_count for item in arms),
    }
    result = calibration_result_from_payload(
        {**unsigned, "result_sha256": result_sha256(unsigned)}
    )
    write_idempotent(completed_path, result.as_safe_dict())
    if progress_callback is not None:
        progress_callback(
            {
                "event": "calibration_completed",
                "baseline_gate_passed": result.baseline_gate_passed,
                "calibrated": result.calibrated,
                "first_successful_multiplier": result.first_successful_multiplier,
                "result_sha256": result.result_sha256,
            }
        )
    return result


__all__ = [
    "preflight_federated_memorization_calibration",
    "run_federated_memorization_calibration",
]
