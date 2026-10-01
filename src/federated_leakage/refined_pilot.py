"""Orquestrador retomável F0-F5 iniciado no Fórum/Tec refinado."""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Tuple

from .aggregation_contracts import load_fedavg_spec_from_config
from .audit_contracts import load_extraction_audit_spec_from_config
from .dp_accounting import validate_accounting_profile
from .dp_contracts import DPAccountantState, load_dp_accounting_spec_from_config
from .federated_round import (
    prepare_auxiliary_training_input,
    prepare_victim_training_inputs,
)
from .model_contracts import DEFAULT_MODEL_CACHE, LoadedModelBundle
from .model_fingerprint import fingerprint_model_parameters
from .model_loading import load_model_bundle, load_model_spec_from_config
from .model_updates import capture_model_parameter_snapshot, restore_model_parameter_snapshot
from .private_federated_round import (
    PrivateFederatedRoundResult,
    private_round_result_from_payload,
    run_private_federated_round,
    validate_paired_private_round_results,
)
from .refined_checkpointing import load_refined_checkpoint, save_refined_checkpoint
from .refined_pilot_contracts import (
    EXPERIMENT_SEEDS,
    REFINED_PILOT_SCHEMA_VERSION,
    RefinedDefenseResult,
    RefinedGatePendingResult,
    RefinedPilotError,
    RefinedPilotResult,
    RefinedPilotSpec,
    RefinedPreflightResult,
    RefinedTrajectoryResult,
    classify_reduction,
    default_run_id,
    refined_trajectory_result_from_payload,
    safe_result_sha256,
    validate_refined_pilot_spec,
)
from .refined_pilot_storage import (
    RefinedPilotPaths,
    aggregate_refined_round_hash,
    commit_refined_round,
    initialize_refined_run,
    initialize_refined_trajectory,
    load_refined_trajectory_state,
    refined_checkpoint_directory,
)
from .reproducibility import ReproducibilityEnvironmentError, validate_cuda_reproducibility_environment
from .semantic_pilot import (
    _original_values,
    _seed_data,
    _semantic_audits,
    _standard_audit,
    _utility,
)
from .semantic_pilot_storage import (
    read_safe_json,
    semantic_round_result_from_payload,
    write_idempotent,
)
from .semantic_round import (
    run_semantic_federated_round,
    validate_paired_original_round_results,
    validate_paired_semantic_round_results,
)
from .semantic_substitution import (
    RotatingVictimSubstitutionGenerator,
    prepare_substituted_victim_training_inputs,
)
from .synthetic_profiles import (
    AuxiliaryRoundGenerator,
    UNIQUE_FIELD_TYPES,
    build_victim_dataset_manifest,
    profile_field_values,
    validate_no_cross_flow_collisions,
)
from .training_contracts import load_local_training_spec_from_config
from .trusted_evaluator import preflight_extraction_audit, prepare_trusted_evaluator
from .utility_evaluation import (
    load_utility_evaluation_spec_from_config,
    prepare_utility_evaluation,
    validate_utility_evaluation_result,
)


BundleLoader = Callable[[], LoadedModelBundle]
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _file_sha256(path: Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RefinedPilotError("configuração do piloto refinado está ausente")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _default_loader(
    main_config: Path,
    cache_dir: Path,
    model_artifact_dir: Path | None,
    device: str,
) -> BundleLoader:
    model_spec = load_model_spec_from_config(main_config)
    if model_artifact_dir is None:
        raise RefinedPilotError("diretório do artefato refinado é obrigatório")
    artifact = Path(model_artifact_dir)
    if not artifact.is_absolute():
        raise RefinedPilotError("diretório do artefato refinado deve ser absoluto")

    def load() -> LoadedModelBundle:
        return load_model_bundle(
            model_spec,
            cache_dir=cache_dir,
            model_artifact_dir=artifact,
            device=device,
        )

    return load


def _preflight_data(spec: RefinedPilotSpec, selected_seed: int):
    materials = {
        seed: _seed_data(seed, spec.schedule_id) for seed in EXPERIMENT_SEEDS
    }
    groups = []
    for seed in EXPERIMENT_SEEDS:
        victims, benign, adversarial, canary, utility = materials[seed]
        groups.extend(dataset.conversations for dataset in victims)
        groups.extend(value.conversations for value in benign)
        groups.extend(value.conversations for value in adversarial)
        groups.extend((canary.conversations, utility.conversations))
    try:
        validate_no_cross_flow_collisions(tuple(groups))
    except Exception as error:
        raise RefinedPilotError("preflight cruzado dos dados originais falhou") from error
    originals = _original_values(groups)
    for seed in EXPERIMENT_SEEDS:
        victims = materials[seed][0]
        generator = RotatingVictimSubstitutionGenerator(seed)
        for round_id in range(1, 21):
            replacement = generator.generate_round(victims, round_id)
            for entry in replacement.entries:
                values = profile_field_values(entry.replacement_profile)
                if any(
                    values[field_type] in originals[field_type]
                    for field_type in UNIQUE_FIELD_TYPES
                ):
                    raise RefinedPilotError(
                        "substituição distintiva colide com original validado"
                    )
    return materials[selected_seed]


def preflight_refined_defense_pilot(
    spec: RefinedPilotSpec,
    *,
    seed: int,
    loader: BundleLoader,
) -> tuple[RefinedPreflightResult, Any, Any, LoadedModelBundle]:
    resolved = validate_refined_pilot_spec(spec)
    if seed not in EXPERIMENT_SEEDS:
        raise RefinedPilotError("seed selecionada é inválida")
    victims, benign, adversarial, _, utility = _preflight_data(resolved, seed)
    dp_spec = load_dp_accounting_spec_from_config(resolved.main_config_path)
    validate_accounting_profile(dp_spec)
    try:
        bundle = loader()
        prepared_victims = prepare_victim_training_inputs(victims, bundle)
        prepared_utility = prepare_utility_evaluation(utility, bundle)
        audit_spec = load_extraction_audit_spec_from_config(resolved.main_config_path)
        for target_count in (20, 200):
            preflight_extraction_audit(
                audit_spec,
                prepare_trusted_evaluator(victims, seed, target_count=target_count),
                bundle,
            )
        replacement_generator = RotatingVictimSubstitutionGenerator(seed)
        for round_id in range(1, 21):
            replacement = replacement_generator.generate_round(victims, round_id)
            prepare_substituted_victim_training_inputs(replacement, bundle)
        if len(prepared_victims.client_samples) != 10 or prepared_utility.conversation_count != 500:
            raise RefinedPilotError("tokenização do preflight possui contagem inválida")
        baseline = fingerprint_model_parameters(bundle)
    except RefinedPilotError:
        raise
    except Exception as error:
        raise RefinedPilotError("preflight do modelo refinado falhou") from error
    return (
        RefinedPreflightResult(
            seed=seed,
            validated_seeds=EXPERIMENT_SEEDS,
            baseline_model_sha256=baseline,
            victim_conversation_count=1_000,
            auxiliary_conversation_count=len(benign) * 100 + len(adversarial) * 100,
            replacement_round_count=20,
            utility_conversation_count=500,
            accounting_profile_validated=True,
            artifact_validated=True,
            tokenization_validated=True,
        ),
        victims,
        utility,
        bundle,
    )


def _trajectory_from_payload(value: object) -> RefinedTrajectoryResult:
    return refined_trajectory_result_from_payload(value)


def _validate_persisted_trajectory(
    result: RefinedTrajectoryResult,
    *,
    seed: int,
    scenario_id: str,
    baseline_sha256: str,
) -> RefinedTrajectoryResult:
    scenario = scenario_id.split("-", 1)[0]
    private = scenario in {"F2", "F3"}
    expected_epsilon = (
        3.0 if scenario_id.endswith("-3") else 8.0 if scenario_id.endswith("-8") else None
    )
    try:
        utility = validate_utility_evaluation_result(result.utility)
    except Exception as error:
        raise RefinedPilotError("utilidade persistida da trajetória diverge") from error
    if (
        result.seed != seed
        or result.scenario_id != scenario_id
        or result.baseline_model_sha256 != baseline_sha256
        or result.optimizer_steps != 20_500
        or result.non_private_conversation_presentations != (0 if private else 82_000)
        or private != (result.private_sampled_conversation_count is not None)
        or private != (result.max_realized_epsilon is not None)
        or result.target_epsilon != expected_epsilon
        or utility.experiment_seed != seed
        or utility.scenario != scenario
        or utility.round_id != 20
        or utility.model_state_sha256 != result.final_model_sha256
    ):
        raise RefinedPilotError("trajetória refinada persistida possui identidade divergente")
    return result


def _checkpoint_restore(
    trajectory_root: Path,
    state: Mapping[str, Any],
    *,
    spec: RefinedPilotSpec,
    seed: int,
    scenario_id: str,
    config_sha256: str,
    bundle: LoadedModelBundle,
    baseline_snapshot: Any,
) -> Tuple[DPAccountantState, ...]:
    completed = int(state["completed_round"])
    if completed == 0:
        restore_model_parameter_snapshot(bundle, baseline_snapshot)
        return ()
    loaded = load_refined_checkpoint(
        refined_checkpoint_directory(trajectory_root, completed, spec.retained_rounds),
        bundle,
        expected_seed=seed,
        expected_scenario_id=scenario_id,
        expected_round_id=completed,
        expected_config_sha256=config_sha256,
    )
    if (
        loaded.artifact_sha256 != state["checkpoint_artifact_sha256"]
        or loaded.round_result.final_model_sha256 != state["current_model_sha256"]
    ):
        raise RefinedPilotError("checkpoint refinado confirmado diverge do journal")
    return loaded.accountant_states


def _endpoint_metrics(
    *,
    audit_spec: Any,
    victims: Any,
    seed: int,
    scenario_id: str,
    bundle: LoadedModelBundle,
    model_sha256: str,
    paths: RefinedPilotPaths,
):
    audit_paths = RefinedPilotPaths(paths.output_root, paths.trajectory_root(scenario_id))
    scenario = scenario_id.split("-", 1)[0]
    if scenario in {"F4", "F5"}:
        original, alias, historical = _semantic_audits(
            audit_spec=audit_spec,
            victims=victims,
            seed=seed,
            scenario=scenario,
            round_id=20,
            target_count=200,
            bundle=bundle,
            model_sha256=model_sha256,
            paths=audit_paths,
        )
        return (
            original,
            original.distinctive_exact_pair_count,
            original.distinctive_exposed_entity_count,
            original.distinctive_field_type_count,
            original.exact_pair_count,
            original.complete_generation_count,
        )
    audit, pairs, entities, breakdown = _standard_audit(
        audit_spec=audit_spec,
        victims=victims,
        seed=seed,
        target_count=200,
        scenario=scenario,
        round_id=20,
        bundle=bundle,
        model_sha256=model_sha256,
        paths=audit_paths,
    )
    return (
        audit,
        pairs,
        entities,
        sum(value > 0 for _, value in breakdown),
        audit.targeted_exact_pair_count,
        audit.targeted_complete_generation_count,
    )


def _run_trajectory(
    *,
    spec: RefinedPilotSpec,
    seed: int,
    scenario_id: str,
    paths: RefinedPilotPaths,
    bundle: LoadedModelBundle,
    victims: Any,
    utility_dataset: Any,
    config_sha256: str,
    local_spec: Any,
    dp_spec: Any,
    fedavg_spec: Any,
    audit_spec: Any,
    utility_spec: Any,
    baseline_sha256: str,
    progress_callback: ProgressCallback | None,
) -> RefinedTrajectoryResult:
    root = initialize_refined_trajectory(
        paths, scenario_id, seed=seed, baseline_model_sha256=baseline_sha256
    )
    completed_path = root / "completed.json"
    baseline_snapshot = capture_model_parameter_snapshot(bundle)
    if completed_path.exists():
        persisted = _validate_persisted_trajectory(
            _trajectory_from_payload(read_safe_json(completed_path)),
            seed=seed,
            scenario_id=scenario_id,
            baseline_sha256=baseline_sha256,
        )
        state = load_refined_trajectory_state(
            root,
            seed=seed,
            scenario_id=scenario_id,
            baseline_model_sha256=baseline_sha256,
        )
        _checkpoint_restore(
            root,
            state,
            spec=spec,
            seed=seed,
            scenario_id=scenario_id,
            config_sha256=config_sha256,
            bundle=bundle,
            baseline_snapshot=baseline_snapshot,
        )
        if fingerprint_model_parameters(bundle) != persisted.final_model_sha256:
            raise RefinedPilotError("trajetória concluída diverge do checkpoint")
        return persisted

    original_inputs = prepare_victim_training_inputs(victims, bundle)
    prepared_utility = prepare_utility_evaluation(utility_dataset, bundle)
    source_hash = build_victim_dataset_manifest(victims)["dataset_sha256"]
    state = load_refined_trajectory_state(
        root,
        seed=seed,
        scenario_id=scenario_id,
        baseline_model_sha256=baseline_sha256,
    )
    accountant_states = _checkpoint_restore(
        root,
        state,
        spec=spec,
        seed=seed,
        scenario_id=scenario_id,
        config_sha256=config_sha256,
        bundle=bundle,
        baseline_snapshot=baseline_snapshot,
    )
    scenario = scenario_id.split("-", 1)[0]
    private = scenario in {"F2", "F3"}
    target_epsilon = (
        3.0 if scenario_id.endswith("-3") else 8.0 if scenario_id.endswith("-8") else None
    )
    if private and state["completed_round"] == 0:
        accountant_states = tuple(None for _ in range(10))
    auxiliary_generator = AuxiliaryRoundGenerator(seed, schedule_id=spec.schedule_id)
    replacement_generator = RotatingVictimSubstitutionGenerator(seed)
    round_results = []
    sampled_total = 0
    for round_id in range(1, int(state["completed_round"]) + 1):
        payload = read_safe_json(root / "rounds" / f"round-{round_id:03d}.json")
        value = (
            private_round_result_from_payload(payload)
            if private
            else semantic_round_result_from_payload(payload)
        )
        round_results.append(value)
        if private:
            sampled_total += value.sampled_conversation_count

    for round_id in range(int(state["completed_round"]) + 1, 21):
        expected_initial = baseline_sha256 if not round_results else round_results[-1].final_model_sha256
        target = refined_checkpoint_directory(root, round_id, spec.retained_rounds)
        try:
            if target.exists():
                loaded = load_refined_checkpoint(
                    target,
                    bundle,
                    expected_seed=seed,
                    expected_scenario_id=scenario_id,
                    expected_round_id=round_id,
                    expected_config_sha256=config_sha256,
                )
                result = loaded.round_result
                next_accountants = loaded.accountant_states
                artifact_hash = loaded.artifact_sha256
                resumed = True
            else:
                if fingerprint_model_parameters(bundle) != expected_initial:
                    raise RefinedPilotError("estado inicial refinado diverge")
                presentation = "benign" if scenario in {"F0", "F2", "F4"} else "adversarial"
                auxiliary = auxiliary_generator.generate(round_id, presentation=presentation)
                auxiliary_input = prepare_auxiliary_training_input(auxiliary, bundle)
                if private:
                    result, next_accountants = run_private_federated_round(
                        original_inputs,
                        auxiliary_input,
                        bundle,
                        local_spec,
                        dp_spec,
                        fedavg_spec,
                        seed=seed,
                        scenario=scenario,
                        round_id=round_id,
                        target_epsilon=float(target_epsilon),
                        accountant_states=tuple(accountant_states),
                        source_victim_dataset_sha256=source_hash,
                    )
                else:
                    victim_inputs = original_inputs
                    if scenario in {"F4", "F5"}:
                        replacement = replacement_generator.generate_round(victims, round_id)
                        victim_inputs = prepare_substituted_victim_training_inputs(
                            replacement, bundle
                        )
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
                    next_accountants = ()
                artifact_hash = save_refined_checkpoint(
                    target,
                    bundle,
                    result,
                    tuple(next_accountants),
                    config_sha256=config_sha256,
                    scenario_id=scenario_id,
                )
                resumed = False
            if result.initial_model_sha256 != expected_initial:
                raise RefinedPilotError("continuidade refinada da rodada diverge")
            audit_paths = RefinedPilotPaths(paths.output_root, root)
            target_count = 200 if round_id == 20 else 20
            if scenario in {"F4", "F5"}:
                _semantic_audits(
                    audit_spec=audit_spec,
                    victims=victims,
                    seed=seed,
                    scenario=scenario,
                    round_id=round_id,
                    target_count=target_count,
                    bundle=bundle,
                    model_sha256=result.final_model_sha256,
                    paths=audit_paths,
                )
            else:
                _standard_audit(
                    audit_spec=audit_spec,
                    victims=victims,
                    seed=seed,
                    target_count=target_count,
                    scenario=scenario,
                    round_id=round_id,
                    bundle=bundle,
                    model_sha256=result.final_model_sha256,
                    paths=audit_paths,
                )
            if round_id == 20:
                _utility(
                    spec=utility_spec,
                    prepared=prepared_utility,
                    bundle=bundle,
                    seed=seed,
                    scenario=scenario,
                    round_id=20,
                    path=root / "utility.json",
                    model_sha256=result.final_model_sha256,
                )
            commit_refined_round(
                root,
                seed=seed,
                scenario_id=scenario_id,
                baseline_model_sha256=baseline_sha256,
                round_id=round_id,
                final_model_sha256=result.final_model_sha256,
                checkpoint_artifact_sha256=artifact_hash,
                round_payload=result.as_safe_dict(),
                retained_rounds=spec.retained_rounds,
            )
            accountant_states = tuple(next_accountants)
        except Exception as error:
            current_state = load_refined_trajectory_state(
                root,
                seed=seed,
                scenario_id=scenario_id,
                baseline_model_sha256=baseline_sha256,
            )
            try:
                accountant_states = _checkpoint_restore(
                    root,
                    current_state,
                    spec=spec,
                    seed=seed,
                    scenario_id=scenario_id,
                    config_sha256=config_sha256,
                    bundle=bundle,
                    baseline_snapshot=baseline_snapshot,
                )
            except Exception as rollback_error:
                raise RefinedPilotError(
                    f"rollback de {scenario_id} falhou na rodada {round_id}"
                ) from rollback_error
            if isinstance(error, RefinedPilotError):
                raise
            raise RefinedPilotError(
                f"trajetória {scenario_id} falhou na rodada {round_id}"
            ) from error
        round_results.append(result)
        if private:
            sampled_total += result.sampled_conversation_count
        if progress_callback:
            payload = {
                "event": "round_completed",
                "seed": seed,
                "scenario_id": scenario_id,
                "round_id": round_id,
                "resumed": resumed,
                "final_model_sha256": result.final_model_sha256,
            }
            if private:
                payload["max_realized_epsilon"] = result.max_realized_epsilon
                payload["sampled_conversation_count"] = result.sampled_conversation_count
            progress_callback(payload)

    model_hash = fingerprint_model_parameters(bundle)
    audit, distinctive_pairs, entities, field_count, exact_pairs, complete = _endpoint_metrics(
        audit_spec=audit_spec,
        victims=victims,
        seed=seed,
        scenario_id=scenario_id,
        bundle=bundle,
        model_sha256=model_hash,
        paths=paths,
    )
    utility = _utility(
        spec=utility_spec,
        prepared=prepared_utility,
        bundle=bundle,
        seed=seed,
        scenario=scenario,
        round_id=20,
        path=root / "utility.json",
        model_sha256=model_hash,
    )
    audit_hash = safe_result_sha256(
        audit.as_safe_dict(), b"refined-defense-endpoint-audit/v1"
    )
    unsigned = {
        "schema_version": "refined-defense-trajectory/v1",
        "scenario_id": scenario_id,
        "seed": seed,
        "completed_rounds": 20,
        "optimizer_steps": 20_500,
        "non_private_conversation_presentations": 0 if private else 82_000,
        "private_sampled_conversation_count": sampled_total if private else None,
        "target_epsilon": target_epsilon,
        "max_realized_epsilon": (
            round_results[-1].max_realized_epsilon if private else None
        ),
        "baseline_model_sha256": baseline_sha256,
        "final_model_sha256": model_hash,
        "original_exact_pair_count": exact_pairs,
        "original_complete_profile_count": complete,
        "distinctive_exact_pair_count": distinctive_pairs,
        "distinctive_exposed_entity_count": entities,
        "distinctive_field_type_count": field_count,
        "audit_result_sha256": audit_hash,
        "utility": utility.as_safe_dict(),
    }
    result = RefinedTrajectoryResult(
        **{**unsigned, "utility": utility},
        result_sha256=safe_result_sha256(
            unsigned, b"refined-defense-trajectory-result/v1"
        ),
    )
    if len(aggregate_refined_round_hash(root, 20)) != 64:
        raise RefinedPilotError("hash das rodadas refinadas é inválido")
    write_idempotent(completed_path, result.as_safe_dict())
    return result


def _vulnerability_gate(
    spec: RefinedPilotSpec,
    *,
    seed: int,
    baseline_gate_passed: bool,
    baseline_model_sha256: str,
    config_sha256: str,
    f0: RefinedTrajectoryResult,
    f1: RefinedTrajectoryResult,
) -> dict[str, Any]:
    thresholds = spec.vulnerability_gate
    eligible = {
        result.scenario_id: (
            result.distinctive_exact_pair_count >= thresholds[0]
            and result.distinctive_exposed_entity_count >= thresholds[1]
            and result.distinctive_field_type_count >= thresholds[2]
        )
        for result in (f0, f1)
    }
    payload = {
        "schema_version": "refined-vulnerability-gate/v1",
        "seed": seed,
        "config_sha256": config_sha256,
        "main_config_sha256": spec.main_config_sha256,
        "baseline_model_sha256": baseline_model_sha256,
        "baseline_gate_passed": baseline_gate_passed,
        "f0_eligible": eligible["F0"],
        "f1_eligible": eligible["F1"],
        "f0_result_sha256": f0.result_sha256,
        "f1_result_sha256": f1.result_sha256,
        "passed": not baseline_gate_passed and all(eligible.values()),
    }
    return {
        **payload,
        "result_sha256": safe_result_sha256(
            payload, b"refined-vulnerability-gate/v1"
        ),
    }


def _validate_vulnerability_gate(
    value: object,
    *,
    expected_seed: int,
    expected_config_sha256: str,
    expected_main_config_sha256: str,
    expected_baseline_model_sha256: str,
) -> Mapping[str, Any]:
    expected_keys = {
        "schema_version", "seed", "config_sha256", "main_config_sha256",
        "baseline_model_sha256", "baseline_gate_passed", "f0_eligible",
        "f1_eligible", "f0_result_sha256", "f1_result_sha256", "passed",
        "result_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RefinedPilotError("gate vulnerável persistido possui estrutura inválida")
    unsigned = dict(value)
    digest = unsigned.pop("result_sha256")
    hashes = (
        value.get("config_sha256"), value.get("main_config_sha256"),
        value.get("baseline_model_sha256"), value.get("f0_result_sha256"),
        value.get("f1_result_sha256"), digest,
    )
    if (
        value.get("schema_version") != "refined-vulnerability-gate/v1"
        or value.get("seed") != expected_seed
        or value.get("config_sha256") != expected_config_sha256
        or value.get("main_config_sha256") != expected_main_config_sha256
        or value.get("baseline_model_sha256") != expected_baseline_model_sha256
        or any(type(value.get(key)) is not bool for key in (
            "baseline_gate_passed", "f0_eligible", "f1_eligible", "passed"
        ))
        or value.get("passed")
        != (
            not value.get("baseline_gate_passed")
            and value.get("f0_eligible")
            and value.get("f1_eligible")
        )
        or any(
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in hashes
        )
        or digest != safe_result_sha256(unsigned, b"refined-vulnerability-gate/v1")
    ):
        raise RefinedPilotError("gate vulnerável persistido diverge da execução")
    return value


def _peer_gate(
    paths: RefinedPilotPaths,
    seed: int,
    *,
    config_sha256: str,
    main_config_sha256: str,
    baseline_model_sha256: str,
) -> Mapping[str, Any] | None:
    peer = next(value for value in EXPERIMENT_SEEDS if value != seed)
    path = paths.output_root / "runs" / default_run_id(peer) / "gates" / "vulnerability.json"
    if not path.exists():
        return None
    return _validate_vulnerability_gate(
        read_safe_json(path),
        expected_seed=peer,
        expected_config_sha256=config_sha256,
        expected_main_config_sha256=main_config_sha256,
        expected_baseline_model_sha256=baseline_model_sha256,
    )


def _defense_result(
    spec: RefinedPilotSpec,
    *,
    seed: int,
    baseline_gate_passed: bool,
    trajectories: Sequence[RefinedTrajectoryResult],
) -> RefinedDefenseResult:
    values = {value.scenario_id: value for value in trajectories}
    vulnerability = all(
        values[name].distinctive_exact_pair_count >= spec.vulnerability_gate[0]
        and values[name].distinctive_exposed_entity_count >= spec.vulnerability_gate[1]
        and values[name].distinctive_field_type_count >= spec.vulnerability_gate[2]
        for name in ("F0", "F1")
    ) and not baseline_gate_passed
    epsilon_statuses = []
    for epsilon in (3.0, 8.0):
        suffix = str(int(epsilon))
        left_status, left_reduction = classify_reduction(
            values["F0"].original_exact_pair_count,
            values[f"F2-epsilon-{suffix}"].original_exact_pair_count,
            values[f"F2-epsilon-{suffix}"].original_complete_profile_count,
            spec.minimum_reduction,
        )
        right_status, right_reduction = classify_reduction(
            values["F1"].original_exact_pair_count,
            values[f"F3-epsilon-{suffix}"].original_exact_pair_count,
            values[f"F3-epsilon-{suffix}"].original_complete_profile_count,
            spec.minimum_reduction,
        )
        status = "approved" if left_status == right_status == "approved" else (
            "inconclusive" if "inconclusive" in {left_status, right_status} else "insufficient"
        )
        epsilon_statuses.append((epsilon, status, left_reduction, right_reduction))
    f4_status, f4_reduction = classify_reduction(
        values["F0"].original_exact_pair_count,
        values["F4"].original_exact_pair_count,
        values["F4"].original_complete_profile_count,
        spec.minimum_reduction,
    )
    f5_status, f5_reduction = classify_reduction(
        values["F1"].original_exact_pair_count,
        values["F5"].original_exact_pair_count,
        values["F5"].original_complete_profile_count,
        spec.minimum_reduction,
    )
    substitution = "approved" if f4_status == f5_status == "approved" else (
        "inconclusive" if "inconclusive" in {f4_status, f5_status} else "insufficient"
    )
    overall = (
        "approved"
        if vulnerability
        and substitution == "approved"
        and all(item[1] == "approved" for item in epsilon_statuses)
        else "inconclusive" if not vulnerability else "insufficient"
    )
    unsigned = {
        "schema_version": "refined-defense-result/v1",
        "seed": seed,
        "baseline_gate_passed": baseline_gate_passed,
        "vulnerability_gate_passed": vulnerability,
        "epsilon_statuses": epsilon_statuses,
        "substitution_status": substitution,
        "f4_reduction": f4_reduction,
        "f5_reduction": f5_reduction,
        "status": overall,
    }
    return RefinedDefenseResult(
        **{**unsigned, "epsilon_statuses": tuple(epsilon_statuses)},
        result_sha256=safe_result_sha256(unsigned, b"refined-defense-result/v1"),
    )


def run_refined_defense_pilot(
    spec: RefinedPilotSpec,
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
) -> RefinedPilotResult | RefinedPreflightResult | RefinedGatePendingResult:
    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise RefinedPilotError(str(error)) from error
    resolved = validate_refined_pilot_spec(spec)
    effective_run_id = run_id or resolved.run_id_for_seed(seed)
    if effective_run_id != resolved.run_id_for_seed(seed):
        raise RefinedPilotError("run_id refinado diverge da seed")
    config_sha = _file_sha256(config_path)
    loader = model_loader or _default_loader(
        resolved.main_config_path, cache_dir, model_artifact_dir, device
    )
    preflight, victims, utility_dataset, bundle = preflight_refined_defense_pilot(
        resolved, seed=seed, loader=loader
    )
    if progress_callback:
        progress_callback(
            {
                "event": "preflight_completed",
                "seed": seed,
                "baseline_model_sha256": preflight.baseline_model_sha256,
                "accounting_profile_validated": True,
            }
        )
    if preflight_only:
        return preflight
    baseline_sha = str(preflight.baseline_model_sha256)
    paths = initialize_refined_run(
        output_root,
        effective_run_id,
        seed,
        resolved,
        config_sha256=config_sha,
        baseline_model_sha256=baseline_sha,
        model_provenance=bundle.provenance.as_safe_dict(),
        fresh=fresh,
    )
    audit_spec = load_extraction_audit_spec_from_config(resolved.main_config_path)
    local_spec = load_local_training_spec_from_config(resolved.main_config_path)
    dp_spec = load_dp_accounting_spec_from_config(resolved.main_config_path)
    fedavg_spec = load_fedavg_spec_from_config(resolved.main_config_path)
    utility_spec = load_utility_evaluation_spec_from_config(resolved.main_config_path)
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
        baseline_pairs >= resolved.vulnerability_gate[0]
        and baseline_entities >= resolved.vulnerability_gate[1]
        and sum(value > 0 for _, value in baseline_breakdown)
        >= resolved.vulnerability_gate[2]
    )
    write_idempotent(
        paths.run_root / "baseline" / "completed.json",
        {
            "schema_version": "refined-defense-baseline/v1",
            "model_state_sha256": baseline_sha,
            "audit_result_sha256": safe_result_sha256(
                baseline_audit.as_safe_dict(), b"refined-defense-baseline-audit/v1"
            ),
            "distinctive_exact_pair_count": baseline_pairs,
            "distinctive_exposed_entity_count": baseline_entities,
            "distinctive_field_type_count": sum(value > 0 for _, value in baseline_breakdown),
            "gate_passed": baseline_gate,
            "utility": baseline_utility.as_safe_dict(),
        },
    )
    del bundle
    gc.collect()

    trajectories: list[RefinedTrajectoryResult] = []
    for scenario_id in ("F0", "F1"):
        current = loader()
        if fingerprint_model_parameters(current) != baseline_sha:
            raise RefinedPilotError("trajetória não iniciou no baseline refinado")
        trajectories.append(
            _run_trajectory(
                spec=resolved,
                seed=seed,
                scenario_id=scenario_id,
                paths=paths,
                bundle=current,
                victims=victims,
                utility_dataset=utility_dataset,
                config_sha256=config_sha,
                local_spec=local_spec,
                dp_spec=dp_spec,
                fedavg_spec=fedavg_spec,
                audit_spec=audit_spec,
                utility_spec=utility_spec,
                baseline_sha256=baseline_sha,
                progress_callback=progress_callback,
            )
        )
        del current
        gc.collect()
    gate = _vulnerability_gate(
        resolved,
        seed=seed,
        baseline_gate_passed=baseline_gate,
        baseline_model_sha256=baseline_sha,
        config_sha256=config_sha,
        f0=trajectories[0],
        f1=trajectories[1],
    )
    write_idempotent(paths.run_root / "gates" / "vulnerability.json", gate)
    peer_gate = _peer_gate(
        paths,
        seed,
        config_sha256=config_sha,
        main_config_sha256=resolved.main_config_sha256,
        baseline_model_sha256=baseline_sha,
    )
    if not gate["passed"]:
        inconclusive = RefinedGatePendingResult(
            run_id=effective_run_id,
            seed=seed,
            phase="inconclusive-vulnerability",
            own_vulnerability_gate_passed=False,
            peer_gate_available=peer_gate is not None,
        )
        write_idempotent(paths.run_root / "inconclusive.json", inconclusive.as_safe_dict())
        return inconclusive
    if peer_gate is None:
        return RefinedGatePendingResult(
            run_id=effective_run_id,
            seed=seed,
            phase="awaiting-peer-vulnerability-gate",
            own_vulnerability_gate_passed=True,
            peer_gate_available=False,
        )
    if peer_gate.get("passed") is not True:
        inconclusive = RefinedGatePendingResult(
            run_id=effective_run_id,
            seed=seed,
            phase="inconclusive-peer-vulnerability",
            own_vulnerability_gate_passed=True,
            peer_gate_available=True,
        )
        write_idempotent(paths.run_root / "inconclusive.json", inconclusive.as_safe_dict())
        return inconclusive

    for scenario_id in resolved.scenario_order[2:]:
        current = loader()
        if fingerprint_model_parameters(current) != baseline_sha:
            raise RefinedPilotError("defesa não iniciou no baseline refinado")
        trajectories.append(
            _run_trajectory(
                spec=resolved,
                seed=seed,
                scenario_id=scenario_id,
                paths=paths,
                bundle=current,
                victims=victims,
                utility_dataset=utility_dataset,
                config_sha256=config_sha,
                local_spec=local_spec,
                dp_spec=dp_spec,
                fedavg_spec=fedavg_spec,
                audit_spec=audit_spec,
                utility_spec=utility_spec,
                baseline_sha256=baseline_sha,
                progress_callback=progress_callback,
            )
        )
        del current
        gc.collect()
    _validate_paired_results(paths, baseline_sha, resolved)
    defense = _defense_result(
        resolved,
        seed=seed,
        baseline_gate_passed=baseline_gate,
        trajectories=trajectories,
    )
    private_selected = sum(
        value.private_sampled_conversation_count or 0 for value in trajectories
    )
    unsigned = {
        "schema_version": REFINED_PILOT_SCHEMA_VERSION,
        "run_id": effective_run_id,
        "seed": seed,
        "baseline_model_sha256": baseline_sha,
        "trajectories": [value.as_safe_dict() for value in trajectories],
        "defense": defense.as_safe_dict(),
        "total_federated_rounds": 160,
        "total_optimizer_steps": 164_000,
        "non_private_conversation_presentations": 328_000,
        "private_sampled_conversation_count": private_selected,
        "total_audit_generations": 61_043,
        "total_utility_conversations": 4_500,
    }
    result = RefinedPilotResult(
        run_id=effective_run_id,
        seed=seed,
        baseline_model_sha256=baseline_sha,
        trajectories=tuple(trajectories),
        defense=defense,
        total_federated_rounds=160,
        total_optimizer_steps=164_000,
        non_private_conversation_presentations=328_000,
        private_sampled_conversation_count=private_selected,
        total_audit_generations=61_043,
        total_utility_conversations=4_500,
        result_sha256=safe_result_sha256(unsigned, b"refined-defense-pilot-result/v1"),
    )
    write_idempotent(paths.run_root / "completed.json", result.as_safe_dict())
    if progress_callback:
        progress_callback(
            {
                "event": "refined_defense_pilot_completed",
                "seed": seed,
                "status": defense.status,
                "result_sha256": result.result_sha256,
            }
        )
    return result


def _validate_paired_results(
    paths: RefinedPilotPaths,
    baseline_sha: str,
    spec: RefinedPilotSpec,
) -> None:
    def nonprivate(name: str):
        return tuple(
            semantic_round_result_from_payload(
                read_safe_json(paths.trajectory_root(name) / "rounds" / f"round-{index:03d}.json")
            )
            for index in range(1, 21)
        )
    f0, f1, f4, f5 = (nonprivate(name) for name in ("F0", "F1", "F4", "F5"))
    for index in range(20):
        validate_paired_original_round_results(
            f0[index], f1[index],
            expected_benign_initial_model_sha256=baseline_sha if index == 0 else f0[index - 1].final_model_sha256,
            expected_adversarial_initial_model_sha256=baseline_sha if index == 0 else f1[index - 1].final_model_sha256,
        )
        validate_paired_semantic_round_results(
            f4[index], f5[index],
            expected_benign_initial_model_sha256=baseline_sha if index == 0 else f4[index - 1].final_model_sha256,
            expected_adversarial_initial_model_sha256=baseline_sha if index == 0 else f5[index - 1].final_model_sha256,
        )
    for epsilon in (3, 8):
        left_name = f"F2-epsilon-{epsilon}"
        right_name = f"F3-epsilon-{epsilon}"
        left = tuple(
            private_round_result_from_payload(
                read_safe_json(paths.trajectory_root(left_name) / "rounds" / f"round-{index:03d}.json")
            ) for index in range(1, 21)
        )
        right = tuple(
            private_round_result_from_payload(
                read_safe_json(paths.trajectory_root(right_name) / "rounds" / f"round-{index:03d}.json")
            ) for index in range(1, 21)
        )
        for index in range(20):
            validate_paired_private_round_results(
                left[index], right[index],
                expected_benign_initial_model_sha256=baseline_sha if index == 0 else left[index - 1].final_model_sha256,
                expected_adversarial_initial_model_sha256=baseline_sha if index == 0 else right[index - 1].final_model_sha256,
            )


__all__ = [
    "preflight_refined_defense_pilot",
    "run_refined_defense_pilot",
]
