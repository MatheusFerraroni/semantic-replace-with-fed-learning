"""Orquestração retomável da grade federada de intensidade v2."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .aggregation_contracts import load_fedavg_spec_from_config
from .audit_contracts import AuditCheckpoint, load_extraction_audit_spec_from_config
from .execution_contracts import PILOT_BASELINE_MODEL_SHA256
from .federated_exposure_contracts import calibration_result_from_payload
from .federated_grid_checkpointing import load_grid_checkpoint, save_grid_checkpoint
from .federated_grid_contracts import (
    DISTINCTIVE_FIELD_TYPES,
    EXPERIMENT_SEEDS,
    FederatedGridArmResult,
    FederatedGridAuditResult,
    FederatedGridError,
    FederatedGridPreflightResult,
    FederatedGridSeedResult,
    FederatedGridSpec,
    GridArmSpec,
    grid_arm_result_from_payload,
    grid_seed_result_from_payload,
    safe_result_sha256,
    validate_federated_grid_spec,
    validate_grid_audit_result,
)
from .federated_grid_round import run_federated_grid_round
from .federated_grid_storage import (
    FederatedGridPaths,
    aggregate_grid_round_results_sha256,
    commit_grid_round,
    initialize_grid_arm,
    initialize_grid_run,
    load_grid_arm_state,
    read_grid_round_results,
    read_safe_json,
    safe_payload_sha256,
    validate_grid_checkpoint_residue,
    write_idempotent,
)
from .federated_round import prepare_auxiliary_training_input, prepare_victim_training_inputs
from .model_contracts import DEFAULT_MODEL_CACHE, LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_loading import load_model_bundle, load_model_spec_from_config
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot
from .reproducibility import ReproducibilityEnvironmentError, validate_cuda_reproducibility_environment
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
    read_completed_distinctive_exposure_breakdown,
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


def _canonical_hash(value: Any, domain: bytes) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


def _config_sha256(path: Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise FederatedGridError("configuração da grade está ausente")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _seed_material(
    spec: FederatedGridSpec,
    seed: int,
) -> tuple[tuple[VictimClientDataset, ...], tuple[AuxiliaryRound, ...], Any, Any, str, str, str]:
    victims = VictimDatasetGenerator(seed).generate()
    generator = AuxiliaryRoundGenerator(seed, schedule_id=spec.schedule_id)
    auxiliary = tuple(generator.generate(round_id, presentation="benign") for round_id in range(1, 21))
    canary = PositiveCanaryDatasetGenerator(seed).generate()
    utility = HeldoutUtilityDatasetGenerator(seed).generate()
    validate_conversation_preflight(victims, auxiliary)
    victim_hash = build_victim_dataset_manifest(victims)["dataset_sha256"]
    manifests = tuple(build_round_manifest(value) for value in auxiliary)
    schedule_hash = _canonical_hash(
        [value["schedule_sha256"] for value in manifests],
        b"pilot-auxiliary-schedule/v1",
    )
    utility_hash = utility_dataset_sha256(utility)
    expected = spec.hashes_for_seed(seed)
    if (
        victim_hash != expected.victim_dataset_sha256
        or schedule_hash != expected.benign_schedule_sha256
        or utility_hash != expected.utility_dataset_sha256
    ):
        raise FederatedGridError("hashes dos dados da seed divergem da grade")
    return victims, auxiliary, canary, utility, victim_hash, schedule_hash, utility_hash


def preflight_federated_memorization_grid(
    spec: FederatedGridSpec,
    *,
    selected_seed: int,
) -> tuple[FederatedGridPreflightResult, tuple[VictimClientDataset, ...], Any]:
    """Reconstrói as duas seeds e valida colisões globais antes de selecionar uma."""

    resolved = validate_federated_grid_spec(spec)
    if selected_seed not in EXPERIMENT_SEEDS:
        raise FederatedGridError("seed selecionada não pertence à grade")
    materials = {seed: _seed_material(resolved, seed) for seed in EXPERIMENT_SEEDS}
    conversations = []
    manifest_identity = []
    for seed in EXPERIMENT_SEEDS:
        victims, auxiliary, canary, utility, victim_hash, schedule_hash, utility_hash = materials[seed]
        conversations.extend(dataset.conversations for dataset in victims)
        conversations.extend(value.conversations for value in auxiliary)
        conversations.extend((canary.conversations, utility.conversations))
        manifest_identity.append((seed, victim_hash, schedule_hash, utility_hash))
    try:
        validate_no_cross_flow_collisions(tuple(conversations))
    except Exception as error:
        raise FederatedGridError("preflight cruzado das duas seeds falhou") from error
    selected = materials[selected_seed]
    collision_hash = _canonical_hash(manifest_identity, b"federated-grid-cross-seed-preflight/v2")
    result = FederatedGridPreflightResult(
        selected_seed=selected_seed,
        validated_seeds=EXPERIMENT_SEEDS,
        victim_conversation_count=1_000,
        auxiliary_conversation_count=2_000,
        utility_conversation_count=500,
        cross_seed_collision_preflight_sha256=collision_hash,
        selected_victim_dataset_sha256=selected[4],
        selected_benign_schedule_sha256=selected[5],
        selected_utility_dataset_sha256=selected[6],
    )
    return result, selected[0], selected[3]


def _validate_reference_v1(output_root: Path, spec: FederatedGridSpec) -> tuple[Any, Mapping[str, Any]]:
    run_root = Path(output_root) / "runs" / spec.reference_v1_run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise FederatedGridError("referência científica v1 está ausente")
    manifest = read_safe_json(run_root / "run_manifest.json")
    if manifest.get("calibration_config_sha256") != spec.reference_v1_config_sha256:
        raise FederatedGridError("configuração da referência v1 diverge")
    try:
        result = calibration_result_from_payload(read_safe_json(run_root / "completed.json"))
    except Exception as error:
        raise FederatedGridError("marcador concluído da referência v1 é inválido") from error
    arm = next((value for value in result.arms if value.victim_repetition_multiplier == 4), None)
    if (
        result.result_sha256 != spec.reference_v1_result_sha256
        or arm is None
        or arm.final_model_sha256 != spec.reference_v1_final_model_sha256
        or arm.audit.distinctive_exact_pair_count != spec.reference_v1_distinctive_exact_pairs
        or arm.audit.distinctive_exposed_entity_count != spec.reference_v1_distinctive_entities
    ):
        raise FederatedGridError("resultado científico da referência v1 diverge")
    summary = read_safe_json(
        run_root / "arms" / "victim-repetitions-004" / "evaluator" / "summaries" / "F0-k01-targets-200-round-020.json"
    )
    return arm, summary


def _ensure_victims(paths: FederatedGridPaths, spec: FederatedGridSpec, seed: int, victims: Sequence[VictimClientDataset], *, allow_create: bool = True) -> tuple[VictimClientDataset, ...]:
    dataset_id = spec.dataset_id_for_seed(seed)
    target = paths.dataset_root / dataset_id
    if not target.exists():
        if not allow_create:
            raise FederatedGridError("datasets de uma grade concluída estão ausentes")
        try:
            write_victim_datasets(paths.dataset_root, dataset_id, victims)
        except Exception as error:
            raise FederatedGridError("falha ao publicar vítimas da grade") from error
    try:
        loaded = tuple(read_victim_client_dataset(paths.dataset_root, dataset_id, f"victim-{index:02d}") for index in range(1, 11))
    except Exception as error:
        raise FederatedGridError("vítimas persistidas da grade são inválidas") from error
    if loaded != tuple(victims):
        raise FederatedGridError("vítimas persistidas divergem da seed")
    return loaded


def _ensure_auxiliary(paths: FederatedGridPaths, spec: FederatedGridSpec, seed: int, value: AuxiliaryRound) -> AuxiliaryRound:
    dataset_id = spec.dataset_id_for_seed(seed)
    target = paths.dataset_root / dataset_id / "clients" / "auxiliary" / spec.schedule_id / "benign" / f"round-{value.round_id:03d}"
    if not target.exists():
        try:
            write_auxiliary_round(paths.dataset_root, dataset_id, spec.schedule_id, value)
        except Exception as error:
            raise FederatedGridError("falha ao publicar auxiliar da grade") from error
    try:
        loaded = read_auxiliary_round(paths.dataset_root, dataset_id, spec.schedule_id, "benign", value.round_id)
    except Exception as error:
        raise FederatedGridError("auxiliar persistido da grade é inválido") from error
    if loaded != value:
        raise FederatedGridError("auxiliar persistido diverge da agenda")
    return loaded


def _run_or_reuse_audit(
    *,
    spec: FederatedGridSpec,
    seed: int,
    audit_spec: Any,
    context: Any,
    model_bundle: LoadedModelBundle,
    model_sha256: str,
    paths: FederatedGridPaths,
    arm: GridArmSpec | None,
) -> FederatedGridAuditResult:
    output_root, run_id = ((paths.run_root, "baseline") if arm is None else (paths.run_root / "arms", arm.arm_id))
    checkpoint = AuditCheckpoint(
        scenario="B0" if arm is None else "F0",
        experiment_seed=seed,
        round_id=0 if arm is None else 20,
        auxiliary_weight_units=None if arm is None else 1,
        expected_model_sha256=model_sha256,
        model_provenance=model_bundle.provenance,
    )
    try:
        extraction = run_extraction_audit(audit_spec, context, checkpoint, model_bundle, output_root=output_root, run_id=run_id, resume=True)
        pair_count, entity_count, breakdown = read_completed_distinctive_exposure_breakdown(
            audit_spec, context, checkpoint, model_bundle, output_root=output_root, run_id=run_id
        )
    except Exception as error:
        raise FederatedGridError("auditoria da grade falhou") from error
    field_count = sum(value > 0 for _, value in breakdown)
    gate = (
        pair_count >= spec.distinctive_exact_pair_threshold
        and entity_count >= spec.distinctive_entity_threshold
        and field_count >= spec.distinctive_field_type_threshold
    )
    result = FederatedGridAuditResult(
        experiment_seed=seed,
        arm_id=None if arm is None else arm.arm_id,
        victim_learning_rate_millionths=0 if arm is None else arm.victim_learning_rate_millionths,
        victim_repetition_multiplier=0 if arm is None else arm.victim_repetition_multiplier,
        extraction_result_sha256=safe_payload_sha256(extraction.as_safe_dict()),
        target_count=extraction.target_count,
        distinctive_exact_pair_count=pair_count,
        distinctive_exposed_entity_count=entity_count,
        distinctive_exact_pairs_by_type=breakdown,
        distinctive_field_type_count=field_count,
        gate_passed=gate,
        model_state_sha256=model_sha256,
    )
    validate_grid_audit_result(result, spec)
    path = paths.run_root / "baseline" / "exposure_audit.json" if arm is None else paths.arm_root(arm.arm_id) / "exposure_audit.json"
    write_idempotent(path, result.as_safe_dict())
    return result


def _run_or_reuse_utility(
    *,
    seed: int,
    utility_spec: Any,
    prepared: PreparedUtilityEvaluation,
    model_bundle: LoadedModelBundle,
    model_sha256: str,
    paths: FederatedGridPaths,
    arm: GridArmSpec | None,
) -> UtilityEvaluationResult:
    scenario = "B0" if arm is None else "F0"
    round_id = 0 if arm is None else 20
    path = paths.run_root / "baseline" / "utility.json" if arm is None else paths.arm_root(arm.arm_id) / "utility.json"
    if path.exists():
        from .execution_storage import utility_result_from_safe_payload
        try:
            result = utility_result_from_safe_payload(read_safe_json(path))
        except Exception as error:
            raise FederatedGridError("utilidade persistida da grade é inválida") from error
        if result.dataset_sha256 != prepared.dataset_sha256 or result.model_state_sha256 != model_sha256 or result.experiment_seed != seed or result.scenario != scenario or result.round_id != round_id:
            raise FederatedGridError("utilidade persistida diverge da grade")
        return result
    try:
        result = evaluate_utility(utility_spec, prepared, model_bundle, scenario=scenario, round_id=round_id, experiment_seed=seed)
    except Exception as error:
        raise FederatedGridError("avaliação de utilidade da grade falhou") from error
    write_idempotent(path, result.as_safe_dict())
    return result


def _restore_confirmed(
    *,
    arm_root: Path,
    seed: int,
    arm: GridArmSpec,
    model_bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    config_sha256: str,
) -> None:
    state = load_grid_arm_state(arm_root, seed, arm)
    if state["completed_round"] == 0:
        restore_model_parameter_snapshot(model_bundle, baseline_snapshot)
        return
    checkpoint = load_grid_checkpoint(
        arm_root / "checkpoints" / f"round-{state['completed_round']:03d}",
        model_bundle,
        expected_seed=seed,
        expected_arm_id=arm.arm_id,
        expected_learning_rate_millionths=arm.victim_learning_rate_millionths,
        expected_multiplier=arm.victim_repetition_multiplier,
        expected_round_id=state["completed_round"],
        expected_config_sha256=config_sha256,
    )
    if (
        checkpoint.artifact_sha256 != state["checkpoint_artifact_sha256"]
        or checkpoint.model_state_sha256 != state["current_model_sha256"]
    ):
        raise FederatedGridError("checkpoint confirmado diverge do estado da grade")


def _load_completed_arm(paths: FederatedGridPaths, spec: FederatedGridSpec, seed: int, arm: GridArmSpec, bundle: LoadedModelBundle, config_sha256: str) -> FederatedGridArmResult | None:
    path = paths.arm_root(arm.arm_id) / "completed.json"
    if not path.exists():
        return None
    result = grid_arm_result_from_payload(read_safe_json(path), spec)
    checkpoint = load_grid_checkpoint(
        paths.arm_root(arm.arm_id) / "checkpoints" / "round-020",
        bundle,
        expected_seed=seed,
        expected_arm_id=arm.arm_id,
        expected_learning_rate_millionths=arm.victim_learning_rate_millionths,
        expected_multiplier=arm.victim_repetition_multiplier,
        expected_round_id=20,
        expected_config_sha256=config_sha256,
    )
    if result.experiment_seed != seed or result.checkpoint_artifact_sha256 != checkpoint.artifact_sha256 or checkpoint.round_result.final_model_sha256 != result.final_model_sha256 or fingerprint_model_parameters(bundle) != result.final_model_sha256:
        raise FederatedGridError("braço concluído diverge do checkpoint final")
    return result


def _run_arm(
    *,
    spec: FederatedGridSpec,
    seed: int,
    arm: GridArmSpec,
    paths: FederatedGridPaths,
    bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    baseline_sha256: str,
    victim_inputs: Any,
    local_spec: Any,
    fedavg_spec: Any,
    audit_spec: Any,
    evaluator_context: Any,
    utility_spec: Any,
    prepared_utility: PreparedUtilityEvaluation,
    config_sha256: str,
    reference_arm: Any,
    reference_summary: Mapping[str, Any],
    progress_callback: ProgressCallback | None,
) -> FederatedGridArmResult:
    restore_model_parameter_snapshot(bundle, baseline_snapshot)
    arm_root = initialize_grid_arm(paths, arm)
    completed = _load_completed_arm(paths, spec, seed, arm, bundle, config_sha256)
    if completed is not None:
        audit = _run_or_reuse_audit(spec=spec, seed=seed, audit_spec=audit_spec, context=evaluator_context, model_bundle=bundle, model_sha256=completed.final_model_sha256, paths=paths, arm=arm)
        utility = _run_or_reuse_utility(seed=seed, utility_spec=utility_spec, prepared=prepared_utility, model_bundle=bundle, model_sha256=completed.final_model_sha256, paths=paths, arm=arm)
        if audit != completed.audit or utility != completed.utility:
            raise FederatedGridError("artefatos do braço concluído divergem")
        return completed
    state = load_grid_arm_state(arm_root, seed, arm)
    validate_grid_checkpoint_residue(arm_root, state["completed_round"])
    rounds = list(read_grid_round_results(arm_root, state["completed_round"]))
    if state["completed_round"]:
        _restore_confirmed(arm_root=arm_root, seed=seed, arm=arm, model_bundle=bundle, baseline_snapshot=baseline_snapshot, config_sha256=config_sha256)
        if rounds[-1].final_model_sha256 != state["current_model_sha256"]:
            raise FederatedGridError("estado retomado do braço diverge")
    elif fingerprint_model_parameters(bundle) != baseline_sha256:
        raise FederatedGridError("braço da grade não iniciou no baseline")

    generator = AuxiliaryRoundGenerator(seed, schedule_id=spec.schedule_id)
    for round_id in range(state["completed_round"] + 1, 21):
        round_data = _ensure_auxiliary(paths, spec, seed, generator.generate(round_id, presentation="benign"))
        try:
            auxiliary_input = prepare_auxiliary_training_input(round_data, bundle)
            result = run_federated_grid_round(victim_inputs, auxiliary_input, bundle, local_spec, fedavg_spec, seed=seed, round_id=round_id, arm=arm)
            expected_initial = baseline_sha256 if not rounds else rounds[-1].final_model_sha256
            if result.initial_model_sha256 != expected_initial or result.victim_dataset_sha256 != spec.hashes_for_seed(seed).victim_dataset_sha256:
                raise FederatedGridError("continuidade científica da rodada diverge")
            checkpoint_path = arm_root / "checkpoints" / f"round-{round_id:03d}"
            artifact = save_grid_checkpoint(checkpoint_path, bundle, result, grid_config_sha256=config_sha256)
            commit_grid_round(arm_root, result, artifact)
        except Exception as error:
            try:
                _restore_confirmed(arm_root=arm_root, seed=seed, arm=arm, model_bundle=bundle, baseline_snapshot=baseline_snapshot, config_sha256=config_sha256)
            except Exception as rollback_error:
                raise FederatedGridError(f"rollback da grade falhou na rodada {round_id}") from rollback_error
            if isinstance(error, FederatedGridError):
                raise
            raise FederatedGridError(f"braço da grade falhou na rodada {round_id}") from error
        rounds.append(result)
        if progress_callback:
            progress_callback({"event": "round_completed", "seed": seed, "arm_id": arm.arm_id, "round_id": round_id, "victim_learning_rate_millionths": arm.victim_learning_rate_millionths, "victim_repetition_multiplier": arm.victim_repetition_multiplier, "final_model_sha256": result.final_model_sha256, "mean_client_loss": result.mean_client_loss})

    final_sha = fingerprint_model_parameters(bundle)
    audit = _run_or_reuse_audit(spec=spec, seed=seed, audit_spec=audit_spec, context=evaluator_context, model_bundle=bundle, model_sha256=final_sha, paths=paths, arm=arm)
    utility = _run_or_reuse_utility(seed=seed, utility_spec=utility_spec, prepared=prepared_utility, model_bundle=bundle, model_sha256=final_sha, paths=paths, arm=arm)
    state = load_grid_arm_state(arm_root, seed, arm)
    result = FederatedGridArmResult(
        experiment_seed=seed,
        arm_id=arm.arm_id,
        victim_learning_rate_millionths=arm.victim_learning_rate_millionths,
        auxiliary_learning_rate_millionths=30,
        victim_repetition_multiplier=arm.victim_repetition_multiplier,
        completed_rounds=len(rounds),
        conversation_presentations=sum(value.conversation_presentations for value in rounds),
        optimizer_steps=sum(value.optimizer_steps for value in rounds),
        baseline_model_sha256=baseline_sha256,
        final_model_sha256=final_sha,
        round_result_sha256=aggregate_grid_round_results_sha256(rounds),
        audit=audit,
        utility=utility,
        checkpoint_artifact_sha256=state["checkpoint_artifact_sha256"],
    )
    grid_arm_result_from_payload(result.as_safe_dict(), spec)
    if seed == 101 and arm.victim_learning_rate_millionths == 30 and arm.victim_repetition_multiplier == 4:
        summary = read_safe_json(arm_root / "evaluator" / "summaries" / "F0-k01-targets-200-round-020.json")
        if (
            result.final_model_sha256 != reference_arm.final_model_sha256
            or result.audit.distinctive_exact_pair_count != reference_arm.audit.distinctive_exact_pair_count
            or result.audit.distinctive_exposed_entity_count != reference_arm.audit.distinctive_exposed_entity_count
            or result.utility.scientific_sha256 != reference_arm.utility.scientific_sha256
            or summary != dict(reference_summary)
        ):
            raise FederatedGridError("regressão científica 4x/3e-5 contra v1 falhou")
    write_idempotent(arm_root / "completed.json", result.as_safe_dict())
    if progress_callback:
        progress_callback({"event": "arm_completed", "seed": seed, "arm_id": arm.arm_id, "gate_passed": audit.gate_passed, "distinctive_exact_pair_count": audit.distinctive_exact_pair_count, "distinctive_exposed_entity_count": audit.distinctive_exposed_entity_count, "distinctive_field_type_count": audit.distinctive_field_type_count, "final_model_sha256": final_sha})
    return result


def _default_loader(config_path: Path, cache_dir: Path, artifact: Path | None, device: str) -> BundleLoader:
    model_spec = load_model_spec_from_config(config_path)
    def load() -> LoadedModelBundle:
        return load_model_bundle(model_spec, cache_dir=cache_dir, model_artifact_dir=artifact, device=device)
    return load


def run_federated_memorization_grid(
    spec: FederatedGridSpec,
    *,
    seed: int,
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
) -> FederatedGridSeedResult | FederatedGridPreflightResult:
    """Executa B0 e seis braços F0 independentes para uma das duas seeds."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise FederatedGridError(str(error)) from error
    resolved = validate_federated_grid_spec(spec)
    effective_run_id = run_id or resolved.run_id_for_seed(seed)
    if effective_run_id != resolved.run_id_for_seed(seed):
        raise FederatedGridError("run_id diverge da seed oficial da grade")
    config_sha = _config_sha256(config_path)
    reference_arm, reference_summary = _validate_reference_v1(output_root, resolved)
    data_preflight, victims, utility_dataset = preflight_federated_memorization_grid(resolved, selected_seed=seed)
    try:
        audit_spec = load_extraction_audit_spec_from_config(resolved.main_config_path)
        local_spec = load_local_training_spec_from_config(resolved.main_config_path)
        fedavg_spec = load_fedavg_spec_from_config(resolved.main_config_path)
        utility_spec = load_utility_evaluation_spec_from_config(resolved.main_config_path)
        if local_spec.learning_rate != 3e-5:
            raise FederatedGridError("taxa oficial do auxiliar diverge da grade")
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("preflight das receitas da grade falhou") from error
    if progress_callback:
        progress_callback({"event": "data_preflight_completed", "selected_seed": seed, "validated_seeds": list(EXPERIMENT_SEEDS), "victim_conversation_count": 1_000, "auxiliary_conversation_count": 2_000, "utility_conversation_count": 500, "cross_seed_collision_preflight_sha256": data_preflight.cross_seed_collision_preflight_sha256})
    loader = model_loader or _default_loader(resolved.main_config_path, cache_dir, model_artifact_dir, device)
    try:
        bundle = loader()
        baseline_sha = fingerprint_model_parameters(bundle)
        if baseline_sha != PILOT_BASELINE_MODEL_SHA256:
            raise FederatedGridError("modelo inicial diverge do Tucano pinado")
        victim_inputs = prepare_victim_training_inputs(victims, bundle)
        prepared_utility = prepare_utility_evaluation(utility_dataset, bundle)
        context = prepare_trusted_evaluator(victims, seed, target_count=200)
        preflight_extraction_audit(audit_spec, context, bundle)
    except FederatedGridError:
        raise
    except Exception as error:
        raise FederatedGridError("preflight do modelo da grade falhou") from error
    if preflight_only:
        return FederatedGridPreflightResult(**{**data_preflight.as_safe_dict(), "model_state_sha256": baseline_sha})

    paths = initialize_grid_run(output_root, effective_run_id, seed, resolved, bundle.provenance, grid_config_sha256=config_sha, baseline_model_sha256=baseline_sha, fresh=fresh)
    completed_path = paths.run_root / "completed.json"
    persisted = _ensure_victims(paths, resolved, seed, victims, allow_create=not completed_path.exists())
    if persisted != victims:
        raise FederatedGridError("datasets retomados divergem do preflight")
    baseline_snapshot = capture_model_parameter_snapshot(bundle)
    if completed_path.exists():
        if fresh:
            raise FileExistsError("execução oficial da grade já existe")
        result = grid_seed_result_from_payload(read_safe_json(completed_path), resolved)
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        baseline_audit = _run_or_reuse_audit(spec=resolved, seed=seed, audit_spec=audit_spec, context=context, model_bundle=bundle, model_sha256=baseline_sha, paths=paths, arm=None)
        baseline_utility = _run_or_reuse_utility(seed=seed, utility_spec=utility_spec, prepared=prepared_utility, model_bundle=bundle, model_sha256=baseline_sha, paths=paths, arm=None)
        if baseline_audit != result.baseline_audit or baseline_utility != result.baseline_utility:
            raise FederatedGridError("baseline concluído da grade diverge")
        for arm_spec, arm_result in zip(resolved.arms, result.arms):
            restore_model_parameter_snapshot(bundle, baseline_snapshot)
            loaded = _load_completed_arm(paths, resolved, seed, arm_spec, bundle, config_sha)
            if loaded != arm_result:
                raise FederatedGridError("resultado concluído da grade diverge")
            audit = _run_or_reuse_audit(spec=resolved, seed=seed, audit_spec=audit_spec, context=context, model_bundle=bundle, model_sha256=loaded.final_model_sha256, paths=paths, arm=arm_spec)
            utility = _run_or_reuse_utility(seed=seed, utility_spec=utility_spec, prepared=prepared_utility, model_bundle=bundle, model_sha256=loaded.final_model_sha256, paths=paths, arm=arm_spec)
            if audit != loaded.audit or utility != loaded.utility:
                raise FederatedGridError("artefatos concluídos da grade divergem")
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        return result

    baseline_audit = _run_or_reuse_audit(spec=resolved, seed=seed, audit_spec=audit_spec, context=context, model_bundle=bundle, model_sha256=baseline_sha, paths=paths, arm=None)
    baseline_utility = _run_or_reuse_utility(seed=seed, utility_spec=utility_spec, prepared=prepared_utility, model_bundle=bundle, model_sha256=baseline_sha, paths=paths, arm=None)
    if progress_callback:
        progress_callback({"event": "baseline_completed", "seed": seed, "generation_count": 1_801, "gate_passed": baseline_audit.gate_passed, "model_state_sha256": baseline_sha})

    arms = []
    for arm in resolved.arms:
        result = _run_arm(
            spec=resolved, seed=seed, arm=arm, paths=paths, bundle=bundle,
            baseline_snapshot=baseline_snapshot, baseline_sha256=baseline_sha,
            victim_inputs=victim_inputs, local_spec=local_spec, fedavg_spec=fedavg_spec,
            audit_spec=audit_spec, evaluator_context=context, utility_spec=utility_spec,
            prepared_utility=prepared_utility, config_sha256=config_sha,
            reference_arm=reference_arm, reference_summary=reference_summary,
            progress_callback=progress_callback,
        )
        arms.append(result)
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        gc.collect()
    successful = tuple(value.arm_id for value in arms if value.audit.gate_passed)
    comparisons = tuple(compare_utility_to_baseline(baseline_utility, value.utility) for value in arms)
    unsigned = {
        "schema_version": "federated-memorization-grid/v2",
        "run_id": effective_run_id,
        "experiment_seed": seed,
        "baseline_model_sha256": baseline_sha,
        "baseline_gate_passed": baseline_audit.gate_passed,
        "any_arm_passed": not baseline_audit.gate_passed and bool(successful),
        "first_successful_arm": successful[0] if successful and not baseline_audit.gate_passed else None,
        "baseline_audit": baseline_audit.as_safe_dict(),
        "baseline_utility": baseline_utility.as_safe_dict(),
        "arms": [value.as_safe_dict() for value in arms],
        "utility_comparisons": [value.as_safe_dict() for value in comparisons],
        "total_federated_rounds": sum(value.completed_rounds for value in arms),
        "total_conversation_presentations": sum(value.conversation_presentations for value in arms),
        "total_optimizer_steps": sum(value.optimizer_steps for value in arms),
        "total_audit_generations": 1_801 * (1 + len(arms)),
        "total_utility_conversations": baseline_utility.conversation_count + sum(value.utility.conversation_count for value in arms),
    }
    result = grid_seed_result_from_payload({**unsigned, "result_sha256": safe_result_sha256(unsigned, b"federated-memorization-grid-result/v2")}, resolved)
    write_idempotent(completed_path, result.as_safe_dict())
    if progress_callback:
        progress_callback({"event": "grid_seed_completed", "seed": seed, "any_arm_passed": result.any_arm_passed, "first_successful_arm": result.first_successful_arm, "result_sha256": result.result_sha256})
    return result


__all__ = ["preflight_federated_memorization_grid", "run_federated_memorization_grid"]
