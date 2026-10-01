"""Orquestração retomável do piloto F0/F1/F4/F5 com substituição rotativa."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .aggregation_contracts import load_fedavg_spec_from_config
from .audit_contracts import AuditCheckpoint, ExtractionAuditResult
from .audit_contracts import load_extraction_audit_spec_from_config
from .execution_contracts import PILOT_BASELINE_MODEL_SHA256
from .execution_storage import utility_result_from_safe_payload
from .federated_grid_contracts import (
    FederatedGridArmResult,
    FederatedGridSpec,
    grid_seed_result_from_payload,
    load_federated_grid_spec_from_config,
    safe_result_sha256 as grid_result_sha256,
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
from .semantic_audit import (
    SemanticAuditResult,
    prepare_semantic_audit_targets,
    run_semantic_substitution_audit,
)
from .semantic_checkpointing import (
    load_semantic_checkpoint,
    save_semantic_checkpoint,
)
from .semantic_pilot_contracts import (
    EXPERIMENT_SEEDS,
    SCENARIO_ORDER,
    SELECTED_GRID_ARM_ID,
    SemanticDefenseGateResult,
    SemanticPilotError,
    SemanticPilotPreflightResult,
    SemanticPilotResult,
    SemanticPilotSpec,
    SemanticTrajectoryResult,
    safe_result_sha256,
    semantic_pilot_result_from_payload,
    semantic_trajectory_from_payload,
    validate_semantic_pilot_spec,
)
from .semantic_pilot_storage import (
    SemanticPilotPaths,
    aggregate_round_results_sha256,
    checkpoint_directory,
    commit_round,
    initialize_semantic_pilot_run,
    initialize_trajectory,
    load_trajectory_state,
    read_round_results,
    read_safe_json,
    safe_payload_sha256,
    write_idempotent,
)
from .semantic_round import (
    run_semantic_federated_round,
    validate_paired_original_round_results,
    validate_paired_semantic_round_results,
)
from .semantic_substitution import (
    RotatingVictimSubstitutionGenerator,
    SemanticReplacementRound,
    prepare_substituted_victim_training_inputs,
)
from .synthetic_profiles import (
    AuxiliaryRound,
    AuxiliaryRoundGenerator,
    HeldoutUtilityDatasetGenerator,
    PositiveCanaryDatasetGenerator,
    UNIQUE_FIELD_TYPES,
    VictimClientDataset,
    VictimDatasetGenerator,
    build_round_manifest,
    build_victim_dataset_manifest,
    profile_field_values,
    validate_conversation_preflight,
    validate_no_cross_flow_collisions,
    validate_paired_auxiliary_rounds,
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
    evaluate_utility,
    load_utility_evaluation_spec_from_config,
    prepare_utility_evaluation,
    utility_dataset_sha256,
)


BundleLoader = Callable[[], LoadedModelBundle]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _hash(value: Any, domain: bytes) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + raw).hexdigest()


def _config_sha256(path: Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise SemanticPilotError("configuração do piloto está ausente")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _grid_reference(
    output_root: Path,
    spec: SemanticPilotSpec,
    seed: int,
) -> tuple[FederatedGridSpec, FederatedGridArmResult]:
    """Valida a decisão humana que selecionou 1e-4/4x na grade concluída."""

    grid_config = spec.main_config_path.parent / "federated-memorization-grid-v2.yaml"
    try:
        grid_spec = load_federated_grid_spec_from_config(grid_config)
        combined = read_safe_json(
            Path(output_root) / "runs" / spec.grid_combined_run_id / "combined.json"
        )
    except Exception as error:
        raise SemanticPilotError("resultado combinado da grade está ausente") from error
    expected_combined_keys = {
        "schema_version", "source_result_sha256_by_seed", "classifications",
        "first_robust_arm", "human_review_required", "total_arms",
        "total_federated_rounds", "total_conversation_presentations",
        "total_optimizer_steps", "total_audit_generations",
        "total_utility_conversations", "result_sha256",
    }
    unsigned = dict(combined)
    result_hash = unsigned.pop("result_sha256", None)
    sources = combined.get("source_result_sha256_by_seed")
    classifications = combined.get("classifications")
    selected = None
    if isinstance(classifications, list):
        selected = next(
            (
                value
                for value in classifications
                if isinstance(value, Mapping)
                and value.get("arm_id") == SELECTED_GRID_ARM_ID
            ),
            None,
        )
    if (
        set(combined) != expected_combined_keys
        or result_hash != spec.grid_combined_result_sha256
        or result_hash != grid_result_sha256(
            unsigned, b"federated-memorization-grid-combined/v2"
        )
        or not isinstance(sources, Mapping)
        or {int(key): value for key, value in sources.items()}
        != dict(spec.grid_result_sha256_by_seed)
        or not isinstance(selected, Mapping)
        or selected.get("classification") != "robust"
        or combined.get("first_robust_arm") != SELECTED_GRID_ARM_ID
        or combined.get("human_review_required") is not True
    ):
        raise SemanticPilotError("gate científico combinado da grade diverge")
    arms: dict[int, FederatedGridArmResult] = {}
    for candidate_seed in EXPERIMENT_SEEDS:
        try:
            seed_result = grid_seed_result_from_payload(
                read_safe_json(
                    Path(output_root)
                    / "runs"
                    / grid_spec.run_id_for_seed(candidate_seed)
                    / "completed.json"
                ),
                grid_spec,
            )
        except Exception as error:
            raise SemanticPilotError("resultado da seed da grade é inválido") from error
        arm = next(
            (
                value
                for value in seed_result.arms
                if value.arm_id == spec.grid_arm_id
            ),
            None,
        )
        if (
            seed_result.result_sha256
            != dict(spec.grid_result_sha256_by_seed)[candidate_seed]
            or seed_result.baseline_gate_passed
            or arm is None
            or not arm.audit.gate_passed
            or arm.victim_learning_rate_millionths != 100
            or arm.victim_repetition_multiplier != 4
        ):
            raise SemanticPilotError("braço selecionado da grade diverge")
        arms[candidate_seed] = arm
    return grid_spec, arms[seed]


def _original_values(
    groups: Sequence[Sequence[Any]],
) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = {field_type: set() for field_type in UNIQUE_FIELD_TYPES}
    for conversations in groups:
        for conversation in conversations:
            for annotation in conversation.annotations:
                if annotation.field_type in result:
                    result[annotation.field_type].add(annotation.value)
    return {key: frozenset(value) for key, value in result.items()}


def _seed_data(
    seed: int,
    schedule_id: str,
) -> tuple[
    tuple[VictimClientDataset, ...],
    tuple[AuxiliaryRound, ...],
    tuple[AuxiliaryRound, ...],
    Any,
    Any,
]:
    victims = VictimDatasetGenerator(seed).generate()
    generator = AuxiliaryRoundGenerator(seed, schedule_id=schedule_id)
    benign = []
    adversarial = []
    for round_id in range(1, 21):
        left = generator.generate(round_id, presentation="benign")
        right = generator.generate(round_id, presentation="adversarial")
        validate_paired_auxiliary_rounds(left, right)
        benign.append(left)
        adversarial.append(right)
    validate_conversation_preflight(victims, tuple(benign))
    validate_conversation_preflight(victims, tuple(adversarial))
    return (
        victims,
        tuple(benign),
        tuple(adversarial),
        PositiveCanaryDatasetGenerator(seed).generate(),
        HeldoutUtilityDatasetGenerator(seed).generate(),
    )


def preflight_semantic_substitution_pilot(
    spec: SemanticPilotSpec,
    *,
    selected_seed: int,
    output_root: Path = Path("outputs"),
) -> tuple[SemanticPilotPreflightResult, tuple[VictimClientDataset, ...], Any]:
    """Reconstrói as duas seeds, o gate e a agenda rotativa sem persistir mapas."""

    resolved = validate_semantic_pilot_spec(spec)
    if selected_seed not in EXPERIMENT_SEEDS:
        raise SemanticPilotError("seed selecionada não pertence ao piloto")
    grid_spec, _ = _grid_reference(output_root, resolved, selected_seed)
    materials = {
        seed: _seed_data(seed, resolved.schedule_id) for seed in EXPERIMENT_SEEDS
    }
    groups: list[Sequence[Any]] = []
    identities = []
    for seed in EXPERIMENT_SEEDS:
        victims, benign, _, canary, utility = materials[seed]
        groups.extend(dataset.conversations for dataset in victims)
        groups.extend(value.conversations for value in benign)
        groups.extend((canary.conversations, utility.conversations))
        victim_hash = build_victim_dataset_manifest(victims)["dataset_sha256"]
        schedule_hash = _hash(
            [build_round_manifest(value)["schedule_sha256"] for value in benign],
            b"pilot-auxiliary-schedule/v1",
        )
        utility_hash = utility_dataset_sha256(utility)
        expected = grid_spec.hashes_for_seed(seed)
        if (
            victim_hash != expected.victim_dataset_sha256
            or schedule_hash != expected.benign_schedule_sha256
            or utility_hash != expected.utility_dataset_sha256
        ):
            raise SemanticPilotError("hashes dos dados divergem da grade selecionada")
        identities.append((seed, victim_hash, schedule_hash, utility_hash))
    try:
        validate_no_cross_flow_collisions(tuple(groups))
    except Exception as error:
        raise SemanticPilotError("preflight cruzado dos dados originais falhou") from error

    originals = _original_values(groups)
    selected_schedule: list[str] = []
    selected_values: list[str] = []
    for seed in EXPERIMENT_SEEDS:
        victims = materials[seed][0]
        generator = RotatingVictimSubstitutionGenerator(seed)
        for round_id in range(1, 21):
            round_data = generator.generate_round(victims, round_id)
            for entry in round_data.entries:
                values = profile_field_values(entry.replacement_profile)
                if any(
                    values[field_type] in originals[field_type]
                    for field_type in UNIQUE_FIELD_TYPES
                ):
                    raise SemanticPilotError(
                        "substituição distintiva colide com valor original global"
                    )
            if seed == selected_seed:
                selected_schedule.append(round_data.schedule_sha256)
                selected_values.append(round_data.values_sha256)
    selected = materials[selected_seed]
    result = SemanticPilotPreflightResult(
        selected_seed=selected_seed,
        validated_seeds=EXPERIMENT_SEEDS,
        victim_conversation_count=1_000,
        auxiliary_conversation_count=4_000,
        replacement_round_count=20,
        replacement_conversation_count=20_000,
        utility_conversation_count=500,
        replacement_schedule_sha256=_hash(
            selected_schedule, b"rotating-semantic-replacement-agenda/v1"
        ),
        replacement_values_sha256=_hash(
            selected_values, b"rotating-semantic-replacement-values-agenda/v1"
        ),
        grid_gate_sha256=resolved.grid_combined_result_sha256,
    )
    return result, selected[0], selected[4]


def _default_loader(
    config_path: Path,
    cache_dir: Path,
    artifact: Path | None,
    device: str,
) -> BundleLoader:
    model_spec = load_model_spec_from_config(config_path)

    def load() -> LoadedModelBundle:
        return load_model_bundle(
            model_spec,
            cache_dir=cache_dir,
            model_artifact_dir=artifact,
            device=device,
        )

    return load


def _standard_audit(
    *,
    audit_spec: Any,
    victims: Sequence[VictimClientDataset],
    seed: int,
    target_count: int,
    scenario: str,
    round_id: int,
    bundle: LoadedModelBundle,
    model_sha256: str,
    paths: SemanticPilotPaths,
) -> tuple[ExtractionAuditResult, int, int, tuple[tuple[str, int], ...]]:
    context = prepare_trusted_evaluator(victims, seed, target_count=target_count)
    checkpoint = AuditCheckpoint(
        scenario=scenario,
        experiment_seed=seed,
        round_id=round_id,
        auxiliary_weight_units=None if scenario == "B0" else 1,
        expected_model_sha256=model_sha256,
        model_provenance=bundle.provenance,
    )
    run_id = "original-baseline" if scenario == "B0" else f"original-{scenario.lower()}"
    output_root = paths.run_root / "audit-artifacts"
    result = run_extraction_audit(
        audit_spec,
        context,
        checkpoint,
        bundle,
        output_root=output_root,
        run_id=run_id,
        resume=True,
    )
    pairs, entities, breakdown = read_completed_distinctive_exposure_breakdown(
        audit_spec,
        context,
        checkpoint,
        bundle,
        output_root=output_root,
        run_id=run_id,
    )
    return result, pairs, entities, breakdown


def _utility(
    *,
    spec: Any,
    prepared: PreparedUtilityEvaluation,
    bundle: LoadedModelBundle,
    seed: int,
    scenario: str,
    round_id: int,
    path: Path,
    model_sha256: str,
) -> UtilityEvaluationResult:
    if path.exists():
        try:
            result = utility_result_from_safe_payload(read_safe_json(path))
        except Exception as error:
            raise SemanticPilotError("resultado persistido de utilidade é inválido") from error
        if (
            result.experiment_seed != seed
            or result.scenario != scenario
            or result.round_id != round_id
            or result.dataset_sha256 != prepared.dataset_sha256
            or result.model_state_sha256 != model_sha256
        ):
            raise SemanticPilotError("utilidade persistida diverge do checkpoint")
        return result
    try:
        result = evaluate_utility(
            spec,
            prepared,
            bundle,
            scenario=scenario,
            round_id=round_id,
            experiment_seed=seed,
        )
    except Exception as error:
        raise SemanticPilotError("avaliação de utilidade falhou") from error
    write_idempotent(path, result.as_safe_dict())
    return result


def _restore_confirmed(
    *,
    trajectory_root: Path,
    state: Mapping[str, Any],
    seed: int,
    scenario: str,
    bundle: LoadedModelBundle,
    baseline_snapshot: Any,
    spec: SemanticPilotSpec,
    config_sha256: str,
) -> None:
    completed = state["completed_round"]
    if completed == 0:
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        return
    checkpoint = load_semantic_checkpoint(
        checkpoint_directory(trajectory_root, completed, spec.retained_rounds),
        bundle,
        expected_seed=seed,
        expected_scenario=scenario,
        expected_round_id=completed,
        expected_config_sha256=config_sha256,
    )
    if (
        checkpoint.artifact_sha256 != state["checkpoint_artifact_sha256"]
        or checkpoint.round_result.final_model_sha256 != state["current_model_sha256"]
    ):
        raise SemanticPilotError("checkpoint confirmado diverge do estado")


def _semantic_audits(
    *,
    audit_spec: Any,
    victims: Sequence[VictimClientDataset],
    seed: int,
    scenario: str,
    round_id: int,
    target_count: int,
    bundle: LoadedModelBundle,
    model_sha256: str,
    paths: SemanticPilotPaths,
) -> tuple[SemanticAuditResult, SemanticAuditResult, SemanticAuditResult | None]:
    context = prepare_trusted_evaluator(victims, seed, target_count=target_count)
    generator = RotatingVictimSubstitutionGenerator(seed)
    rounds = tuple(
        generator.generate_round(victims, value)
        for value in range(1, round_id + 1)
    )
    root = paths.run_root / "audit-artifacts"
    current = run_semantic_substitution_audit(
        audit_spec,
        scenario=scenario,
        round_id=round_id,
        view="current_alias",
        targets=prepare_semantic_audit_targets(context, rounds, view="current_alias"),
        cross_replacement_rounds=rounds,
        model_bundle=bundle,
        expected_model_sha256=model_sha256,
        output_root=root,
        run_id=f"semantic-{scenario.lower()}",
    )
    original = run_semantic_substitution_audit(
        audit_spec,
        scenario=scenario,
        round_id=round_id,
        view="original",
        targets=prepare_semantic_audit_targets(context, rounds, view="original"),
        cross_replacement_rounds=rounds,
        model_bundle=bundle,
        expected_model_sha256=model_sha256,
        output_root=root,
        run_id=f"semantic-{scenario.lower()}",
    )
    historical = None
    if round_id == 20:
        historical_context = prepare_trusted_evaluator(victims, seed, target_count=20)
        historical = run_semantic_substitution_audit(
            audit_spec,
            scenario=scenario,
            round_id=round_id,
            view="historical_alias",
            targets=prepare_semantic_audit_targets(
                historical_context, rounds, view="historical_alias"
            ),
            cross_replacement_rounds=rounds,
            model_bundle=bundle,
            expected_model_sha256=model_sha256,
            output_root=root,
            run_id=f"semantic-{scenario.lower()}",
        )
    return original, current, historical


def _trajectory_result(
    *,
    scenario: str,
    seed: int,
    baseline_sha256: str,
    rounds: Sequence[Any],
    original_audit: ExtractionAuditResult | SemanticAuditResult,
    distinctive_pairs: int,
    distinctive_entities: int,
    distinctive_fields: int,
    alias_audit: SemanticAuditResult | None,
    historical_audit: SemanticAuditResult | None,
    utility: UtilityEvaluationResult,
) -> SemanticTrajectoryResult:
    original_hash = safe_payload_sha256(original_audit.as_safe_dict())
    unsigned = {
        "schema_version": "semantic-substitution-trajectory/v1",
        "scenario": scenario,
        "experiment_seed": seed,
        "completed_rounds": 20,
        "conversation_presentations": 82_000,
        "optimizer_steps": 20_500,
        "baseline_model_sha256": baseline_sha256,
        "final_model_sha256": rounds[-1].final_model_sha256,
        "round_result_sha256": aggregate_round_results_sha256(rounds),
        "original_audit_exact_pairs": original_audit.targeted_exact_pair_count
        if isinstance(original_audit, ExtractionAuditResult)
        else original_audit.exact_pair_count,
        "original_audit_complete_profiles": original_audit.targeted_complete_generation_count
        if isinstance(original_audit, ExtractionAuditResult)
        else original_audit.complete_generation_count,
        "distinctive_exact_pair_count": distinctive_pairs,
        "distinctive_exposed_entity_count": distinctive_entities,
        "distinctive_field_type_count": distinctive_fields,
        "original_audit_result_sha256": original_hash,
        "alias_audit_result_sha256": None
        if alias_audit is None
        else safe_payload_sha256(alias_audit.as_safe_dict()),
        "historical_audit_result_sha256": None
        if historical_audit is None
        else safe_payload_sha256(historical_audit.as_safe_dict()),
        "utility": utility.as_safe_dict(),
    }
    return semantic_trajectory_from_payload(
        {
            **unsigned,
            "result_sha256": safe_result_sha256(
                unsigned, b"semantic-substitution-trajectory-result/v1"
            ),
        }
    )


def _run_trajectory(
    *,
    spec: SemanticPilotSpec,
    seed: int,
    scenario: str,
    paths: SemanticPilotPaths,
    bundle: LoadedModelBundle,
    victims: tuple[VictimClientDataset, ...],
    prepared_utility: PreparedUtilityEvaluation,
    local_spec: Any,
    fedavg_spec: Any,
    audit_spec: Any,
    utility_spec: Any,
    baseline_sha256: str,
    config_sha256: str,
    grid_arm: FederatedGridArmResult,
    progress_callback: ProgressCallback | None,
) -> SemanticTrajectoryResult:
    trajectory_root = initialize_trajectory(paths, scenario)
    completed_path = trajectory_root / "completed.json"
    baseline_snapshot = capture_model_parameter_snapshot(bundle)
    original_inputs: PreparedVictimTrainingInputs | None = None
    if scenario in {"F0", "F1"}:
        original_inputs = prepare_victim_training_inputs(victims, bundle)
    state = load_trajectory_state(
        trajectory_root,
        seed=seed,
        scenario=scenario,
        baseline_model_sha256=baseline_sha256,
    )
    rounds = list(read_round_results(trajectory_root, state["completed_round"]))
    if rounds and rounds[0].initial_model_sha256 != baseline_sha256:
        raise SemanticPilotError("trajetória não inicia no baseline compartilhado")
    _restore_confirmed(
        trajectory_root=trajectory_root,
        state=state,
        seed=seed,
        scenario=scenario,
        bundle=bundle,
        baseline_snapshot=baseline_snapshot,
        spec=spec,
        config_sha256=config_sha256,
    )
    if completed_path.exists():
        result = semantic_trajectory_from_payload(read_safe_json(completed_path))
        if result.scenario != scenario or result.experiment_seed != seed:
            raise SemanticPilotError("trajetória concluída pertence a outra execução")
        if (
            len(rounds) != 20
            or aggregate_round_results_sha256(rounds) != result.round_result_sha256
            or fingerprint_model_parameters(bundle) != result.final_model_sha256
        ):
            raise SemanticPilotError("trajetória concluída diverge do checkpoint")
        if scenario in {"F0", "F1"}:
            audit, pairs, entities, breakdown = _standard_audit(
                audit_spec=audit_spec,
                victims=victims,
                seed=seed,
                target_count=200,
                scenario=scenario,
                round_id=20,
                bundle=bundle,
                model_sha256=result.final_model_sha256,
                paths=paths,
            )
            alias = historical = None
            field_count = sum(value > 0 for _, value in breakdown)
        else:
            audit, alias, historical = _semantic_audits(
                audit_spec=audit_spec,
                victims=victims,
                seed=seed,
                scenario=scenario,
                round_id=20,
                target_count=200,
                bundle=bundle,
                model_sha256=result.final_model_sha256,
                paths=paths,
            )
            pairs = audit.distinctive_exact_pair_count
            entities = audit.distinctive_exposed_entity_count
            field_count = audit.distinctive_field_type_count
        utility = _utility(
            spec=utility_spec,
            prepared=prepared_utility,
            bundle=bundle,
            seed=seed,
            scenario=scenario,
            round_id=20,
            path=trajectory_root / "utility.json",
            model_sha256=result.final_model_sha256,
        )
        expected = _trajectory_result(
            scenario=scenario,
            seed=seed,
            baseline_sha256=baseline_sha256,
            rounds=rounds,
            original_audit=audit,
            distinctive_pairs=pairs,
            distinctive_entities=entities,
            distinctive_fields=field_count,
            alias_audit=alias,
            historical_audit=historical,
            utility=utility,
        )
        if expected != result:
            raise SemanticPilotError("artefatos da trajetória concluída divergem")
        return result

    source_hash = build_victim_dataset_manifest(victims)["dataset_sha256"]
    auxiliary_generator = AuxiliaryRoundGenerator(seed, schedule_id=spec.schedule_id)
    replacement_generator = RotatingVictimSubstitutionGenerator(seed)
    for round_id in range(state["completed_round"] + 1, 21):
        expected_initial = baseline_sha256 if not rounds else rounds[-1].final_model_sha256
        target = checkpoint_directory(trajectory_root, round_id, spec.retained_rounds)
        try:
            if target.exists():
                loaded = load_semantic_checkpoint(
                    target,
                    bundle,
                    expected_seed=seed,
                    expected_scenario=scenario,
                    expected_round_id=round_id,
                    expected_config_sha256=config_sha256,
                )
                result = loaded.round_result
                artifact = loaded.artifact_sha256
                resumed = True
            else:
                if fingerprint_model_parameters(bundle) != expected_initial:
                    raise SemanticPilotError("estado inicial da rodada diverge")
                if scenario in {"F4", "F5"}:
                    replacement = replacement_generator.generate_round(victims, round_id)
                    victim_inputs = prepare_substituted_victim_training_inputs(
                        replacement, bundle
                    )
                else:
                    assert original_inputs is not None
                    victim_inputs = original_inputs
                presentation = "benign" if scenario in {"F0", "F4"} else "adversarial"
                auxiliary = auxiliary_generator.generate(
                    round_id, presentation=presentation
                )
                auxiliary_input = prepare_auxiliary_training_input(auxiliary, bundle)
                result = run_semantic_federated_round(
                    victim_inputs,
                    auxiliary_input,
                    bundle,
                    local_spec,
                    fedavg_spec,
                    seed=seed,
                    scenario=scenario,
                    round_id=round_id,
                    source_victim_dataset_sha256=source_hash,
                )
                artifact = save_semantic_checkpoint(
                    target,
                    bundle,
                    result,
                    config_sha256=config_sha256,
                )
                resumed = False
            if result.initial_model_sha256 != expected_initial:
                raise SemanticPilotError("continuidade científica da rodada diverge")
            target_count = 200 if round_id == 20 else 20
            if scenario in {"F0", "F1"}:
                _standard_audit(
                    audit_spec=audit_spec,
                    victims=victims,
                    seed=seed,
                    target_count=target_count,
                    scenario=scenario,
                    round_id=round_id,
                    bundle=bundle,
                    model_sha256=result.final_model_sha256,
                    paths=paths,
                )
            else:
                _semantic_audits(
                    audit_spec=audit_spec,
                    victims=victims,
                    seed=seed,
                    scenario=scenario,
                    round_id=round_id,
                    target_count=target_count,
                    bundle=bundle,
                    model_sha256=result.final_model_sha256,
                    paths=paths,
                )
            if round_id == 20:
                _utility(
                    spec=utility_spec,
                    prepared=prepared_utility,
                    bundle=bundle,
                    seed=seed,
                    scenario=scenario,
                    round_id=20,
                    path=trajectory_root / "utility.json",
                    model_sha256=result.final_model_sha256,
                )
            commit_round(trajectory_root, result, artifact)
        except Exception as error:
            state = load_trajectory_state(
                trajectory_root,
                seed=seed,
                scenario=scenario,
                baseline_model_sha256=baseline_sha256,
            )
            try:
                _restore_confirmed(
                    trajectory_root=trajectory_root,
                    state=state,
                    seed=seed,
                    scenario=scenario,
                    bundle=bundle,
                    baseline_snapshot=baseline_snapshot,
                    spec=spec,
                    config_sha256=config_sha256,
                )
            except Exception as rollback_error:
                raise SemanticPilotError(
                    f"rollback de {scenario} falhou na rodada {round_id}"
                ) from rollback_error
            if isinstance(error, SemanticPilotError):
                raise
            raise SemanticPilotError(
                f"trajetória {scenario} falhou na rodada {round_id}"
            ) from error
        rounds.append(result)
        state = load_trajectory_state(
            trajectory_root,
            seed=seed,
            scenario=scenario,
            baseline_model_sha256=baseline_sha256,
        )
        if progress_callback:
            progress_callback(
                {
                    "event": "round_completed",
                    "seed": seed,
                    "scenario": scenario,
                    "round_id": round_id,
                    "resumed": resumed,
                    "final_model_sha256": result.final_model_sha256,
                    "mean_client_loss": result.mean_client_loss,
                }
            )

    model_sha = fingerprint_model_parameters(bundle)
    if scenario in {"F0", "F1"}:
        original, pairs, entities, breakdown = _standard_audit(
            audit_spec=audit_spec,
            victims=victims,
            seed=seed,
            target_count=200,
            scenario=scenario,
            round_id=20,
            bundle=bundle,
            model_sha256=model_sha,
            paths=paths,
        )
        alias = historical = None
        field_count = sum(value > 0 for _, value in breakdown)
    else:
        original, alias, historical = _semantic_audits(
            audit_spec=audit_spec,
            victims=victims,
            seed=seed,
            scenario=scenario,
            round_id=20,
            target_count=200,
            bundle=bundle,
            model_sha256=model_sha,
            paths=paths,
        )
        pairs = original.distinctive_exact_pair_count
        entities = original.distinctive_exposed_entity_count
        field_count = original.distinctive_field_type_count
    utility = _utility(
        spec=utility_spec,
        prepared=prepared_utility,
        bundle=bundle,
        seed=seed,
        scenario=scenario,
        round_id=20,
        path=trajectory_root / "utility.json",
        model_sha256=model_sha,
    )
    result = _trajectory_result(
        scenario=scenario,
        seed=seed,
        baseline_sha256=baseline_sha256,
        rounds=rounds,
        original_audit=original,
        distinctive_pairs=pairs,
        distinctive_entities=entities,
        distinctive_fields=field_count,
        alias_audit=alias,
        historical_audit=historical,
        utility=utility,
    )
    if scenario == "F0" and (
        result.final_model_sha256 != grid_arm.final_model_sha256
        or result.distinctive_exact_pair_count
        != grid_arm.audit.distinctive_exact_pair_count
        or result.distinctive_exposed_entity_count
        != grid_arm.audit.distinctive_exposed_entity_count
        or result.original_audit_result_sha256
        != grid_arm.audit.extraction_result_sha256
        or result.utility.scientific_sha256 != grid_arm.utility.scientific_sha256
    ):
        raise SemanticPilotError("regressão F0 contra o braço selecionado falhou")
    write_idempotent(completed_path, result.as_safe_dict())
    return result


def _validate_pairs(
    paths: SemanticPilotPaths,
    baseline_sha256: str,
) -> None:
    f0 = read_round_results(paths.trajectory_root("F0"), 20)
    f1 = read_round_results(paths.trajectory_root("F1"), 20)
    f4 = read_round_results(paths.trajectory_root("F4"), 20)
    f5 = read_round_results(paths.trajectory_root("F5"), 20)
    for index in range(20):
        left_initial = baseline_sha256 if index == 0 else f0[index - 1].final_model_sha256
        right_initial = baseline_sha256 if index == 0 else f1[index - 1].final_model_sha256
        validate_paired_original_round_results(
            f0[index],
            f1[index],
            expected_benign_initial_model_sha256=left_initial,
            expected_adversarial_initial_model_sha256=right_initial,
        )
        left_initial = baseline_sha256 if index == 0 else f4[index - 1].final_model_sha256
        right_initial = baseline_sha256 if index == 0 else f5[index - 1].final_model_sha256
        validate_paired_semantic_round_results(
            f4[index],
            f5[index],
            expected_benign_initial_model_sha256=left_initial,
            expected_adversarial_initial_model_sha256=right_initial,
        )


def _gate(
    spec: SemanticPilotSpec,
    *,
    seed: int,
    baseline_gate_passed: bool,
    trajectories: Sequence[SemanticTrajectoryResult],
) -> SemanticDefenseGateResult:
    values = {result.scenario: result for result in trajectories}
    eligible = {
        scenario: (
            values[scenario].distinctive_exact_pair_count
            >= spec.comparator_distinctive_exact_pairs
            and values[scenario].distinctive_exposed_entity_count
            >= spec.comparator_distinctive_entities
            and values[scenario].distinctive_field_type_count
            >= spec.comparator_distinctive_field_types
        )
        for scenario in ("F0", "F1")
    }

    def reduction(comparator: str, defended: str) -> float | None:
        denominator = values[comparator].original_audit_exact_pairs
        if denominator <= 0:
            return None
        return 1.0 - values[defended].original_audit_exact_pairs / denominator

    f4_reduction = reduction("F0", "F4")
    f5_reduction = reduction("F1", "F5")
    if not eligible["F0"] or not eligible["F1"]:
        status = "inconclusive"
    elif (
        not baseline_gate_passed
        and f4_reduction is not None
        and f5_reduction is not None
        and f4_reduction >= spec.minimum_original_exact_pair_reduction
        and f5_reduction >= spec.minimum_original_exact_pair_reduction
        and values["F4"].original_audit_complete_profiles
        <= spec.maximum_original_complete_profiles
        and values["F5"].original_audit_complete_profiles
        <= spec.maximum_original_complete_profiles
    ):
        status = "approved"
    else:
        status = "failed"
    unsigned = {
        "schema_version": "semantic-substitution-result/v1",
        "seed": seed,
        "baseline_gate_passed": baseline_gate_passed,
        "f0_comparator_eligible": eligible["F0"],
        "f1_comparator_eligible": eligible["F1"],
        "f4_original_exact_pair_reduction": f4_reduction,
        "f5_original_exact_pair_reduction": f5_reduction,
        "f4_original_complete_profiles": values["F4"].original_audit_complete_profiles,
        "f5_original_complete_profiles": values["F5"].original_audit_complete_profiles,
        "status": status,
    }
    return SemanticDefenseGateResult(
        **unsigned,
        result_sha256=safe_result_sha256(
            unsigned, b"semantic-substitution-defense-gate/v1"
        ),
    )


def run_semantic_substitution_pilot(
    spec: SemanticPilotSpec,
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
) -> SemanticPilotResult | SemanticPilotPreflightResult:
    """Executa uma seed completa ou somente seu preflight sem escrita."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise SemanticPilotError(str(error)) from error
    resolved = validate_semantic_pilot_spec(spec)
    effective_run_id = run_id or resolved.run_id_for_seed(seed)
    if effective_run_id != resolved.run_id_for_seed(seed):
        raise SemanticPilotError("run_id diverge da seed oficial")
    config_sha = _config_sha256(config_path)
    grid_spec, grid_arm = _grid_reference(output_root, resolved, seed)
    data_preflight, victims, utility_dataset = preflight_semantic_substitution_pilot(
        resolved, selected_seed=seed, output_root=output_root
    )
    try:
        audit_spec = load_extraction_audit_spec_from_config(resolved.main_config_path)
        local_spec = load_local_training_spec_from_config(resolved.main_config_path)
        fedavg_spec = load_fedavg_spec_from_config(resolved.main_config_path)
        utility_spec = load_utility_evaluation_spec_from_config(resolved.main_config_path)
        if local_spec.learning_rate != 3e-5:
            raise SemanticPilotError("taxa oficial do auxiliar diverge")
    except SemanticPilotError:
        raise
    except Exception as error:
        raise SemanticPilotError("receitas do piloto são inválidas") from error
    if progress_callback:
        progress_callback(
            {
                "event": "data_preflight_completed",
                "seed": seed,
                "victim_conversation_count": 1_000,
                "auxiliary_conversation_count": 4_000,
                "replacement_round_count": 20,
                "replacement_schedule_sha256": data_preflight.replacement_schedule_sha256,
            }
        )
    loader = model_loader or _default_loader(
        resolved.main_config_path, cache_dir, model_artifact_dir, device
    )
    try:
        bundle = loader()
        baseline_sha = fingerprint_model_parameters(bundle)
        if baseline_sha != PILOT_BASELINE_MODEL_SHA256:
            raise SemanticPilotError("modelo inicial diverge do Tucano pinado")
        prepare_victim_training_inputs(victims, bundle)
        prepare_utility_evaluation(utility_dataset, bundle)
        context20 = prepare_trusted_evaluator(victims, seed, target_count=20)
        context200 = prepare_trusted_evaluator(victims, seed, target_count=200)
        preflight_extraction_audit(audit_spec, context20, bundle)
        preflight_extraction_audit(audit_spec, context200, bundle)
        replacement_generator = RotatingVictimSubstitutionGenerator(seed)
        for round_id in range(1, 21):
            replacement = replacement_generator.generate_round(victims, round_id)
            prepare_substituted_victim_training_inputs(replacement, bundle)
            replacement_context = prepare_trusted_evaluator(
                replacement.datasets,
                seed,
                target_count=200 if round_id == 20 else 20,
            )
            preflight_extraction_audit(audit_spec, replacement_context, bundle)
    except SemanticPilotError:
        raise
    except Exception as error:
        raise SemanticPilotError("preflight do modelo falhou") from error
    model_preflight = SemanticPilotPreflightResult(
        **{
            **data_preflight.as_safe_dict(),
            "model_state_sha256": baseline_sha,
            "tokenization_validated": True,
        }
    )
    if preflight_only:
        return model_preflight

    paths = initialize_semantic_pilot_run(
        output_root,
        effective_run_id,
        seed,
        resolved,
        bundle.provenance,
        config_sha256=config_sha,
        baseline_model_sha256=baseline_sha,
        fresh=fresh,
    )
    completed_path = paths.run_root / "completed.json"
    persisted_result = None
    if completed_path.exists():
        if fresh:
            raise FileExistsError("execução oficial já existe")
        persisted_result = semantic_pilot_result_from_payload(
            read_safe_json(completed_path)
        )

    prepared_utility = prepare_utility_evaluation(utility_dataset, bundle)
    baseline_audit, baseline_pairs, baseline_entities, baseline_breakdown = _standard_audit(
        audit_spec=audit_spec,
        victims=victims,
        seed=seed,
        target_count=200,
        scenario="B0",
        round_id=0,
        bundle=bundle,
        model_sha256=baseline_sha,
        paths=paths,
    )
    baseline_utility = _utility(
        spec=utility_spec,
        prepared=prepared_utility,
        bundle=bundle,
        seed=seed,
        scenario="B0",
        round_id=0,
        path=paths.run_root / "baseline" / "utility.json",
        model_sha256=baseline_sha,
    )
    baseline_gate = (
        baseline_pairs >= resolved.comparator_distinctive_exact_pairs
        and baseline_entities >= resolved.comparator_distinctive_entities
        and sum(value > 0 for _, value in baseline_breakdown)
        >= resolved.comparator_distinctive_field_types
    )
    write_idempotent(
        paths.run_root / "baseline" / "completed.json",
        {
            "schema_version": "semantic-substitution-baseline/v1",
            "model_state_sha256": baseline_sha,
            "audit_result_sha256": _hash(
                baseline_audit.as_safe_dict(), b"semantic-baseline-audit/v1"
            ),
            "distinctive_exact_pair_count": baseline_pairs,
            "distinctive_exposed_entity_count": baseline_entities,
            "distinctive_field_type_count": sum(
                value > 0 for _, value in baseline_breakdown
            ),
            "gate_passed": baseline_gate,
            "utility": baseline_utility.as_safe_dict(),
        },
    )
    if progress_callback:
        progress_callback(
            {
                "event": "baseline_completed",
                "seed": seed,
                "generation_count": 1_801,
                "model_state_sha256": baseline_sha,
                "gate_passed": baseline_gate,
            }
        )
    del bundle
    gc.collect()

    trajectories = []
    for scenario in SCENARIO_ORDER:
        trajectory_bundle = loader()
        if fingerprint_model_parameters(trajectory_bundle) != baseline_sha:
            raise SemanticPilotError("trajetória não foi recarregada do baseline")
        trajectory_utility = prepare_utility_evaluation(utility_dataset, trajectory_bundle)
        result = _run_trajectory(
            spec=resolved,
            seed=seed,
            scenario=scenario,
            paths=paths,
            bundle=trajectory_bundle,
            victims=victims,
            prepared_utility=trajectory_utility,
            local_spec=local_spec,
            fedavg_spec=fedavg_spec,
            audit_spec=audit_spec,
            utility_spec=utility_spec,
            baseline_sha256=baseline_sha,
            config_sha256=config_sha,
            grid_arm=grid_arm,
            progress_callback=progress_callback,
        )
        trajectories.append(result)
        del trajectory_bundle
        gc.collect()
    _validate_pairs(paths, baseline_sha)
    gate = _gate(
        resolved,
        seed=seed,
        baseline_gate_passed=baseline_gate,
        trajectories=trajectories,
    )
    unsigned = {
        "schema_version": "semantic-substitution-pilot/v1",
        "run_id": effective_run_id,
        "experiment_seed": seed,
        "baseline_model_sha256": baseline_sha,
        "trajectories": [value.as_safe_dict() for value in trajectories],
        "gate": gate.as_safe_dict(),
        "total_federated_rounds": 80,
        "total_conversation_presentations": 328_000,
        "total_optimizer_steps": 82_000,
        "total_audit_generations": 40_083,
        "total_utility_conversations": 2_500,
    }
    result = semantic_pilot_result_from_payload(
        {
            **unsigned,
            "result_sha256": safe_result_sha256(
                unsigned, b"semantic-substitution-pilot-result/v1"
            ),
        }
    )
    if persisted_result is not None and persisted_result != result:
        raise SemanticPilotError("execução concluída diverge dos artefatos")
    write_idempotent(completed_path, result.as_safe_dict())
    if progress_callback:
        progress_callback(
            {
                "event": "semantic_pilot_completed",
                "seed": seed,
                "status": gate.status,
                "result_sha256": result.result_sha256,
            }
        )
    return result


__all__ = [
    "preflight_semantic_substitution_pilot",
    "run_semantic_substitution_pilot",
]
