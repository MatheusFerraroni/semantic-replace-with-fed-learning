"""Orquestração retomável do piloto pareado B0/F0/F1."""

from __future__ import annotations

import gc
import hashlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Tuple

from .aggregation_contracts import (
    FedAvgError,
    FedAvgRoundResult,
    FedAvgSpec,
    load_fedavg_spec_from_config,
)
from .audit_contracts import (
    AuditCheckpoint,
    AuditSpec,
    ExtractionAuditError,
    ExtractionAuditResult,
    TrustedEvaluatorContext,
    load_extraction_audit_spec_from_config,
)
from .audit_storage import read_completed_audit_artifacts
from .checkpointing import (
    build_federated_checkpoint_metadata,
    load_federated_checkpoint,
    save_federated_checkpoint,
)
from .calibration_gate import load_completed_calibration_gate
from .execution_contracts import (
    PILOT_EXPECTED_GENERATION_COUNT,
    FederatedTrajectoryResult,
    PilotExecutionError,
    PilotExecutionResult,
    PilotExecutionSpec,
    PilotPreflightResult,
    PilotRunIdentity,
    TrajectoryScenario,
    validate_pilot_execution_spec,
)
from .execution_storage import (
    PilotRunPaths,
    checkpoint_id_for_round,
    checkpoint_path_for_round,
    commit_paired_round,
    commit_trajectory_round,
    initialize_pilot_run,
    mark_baseline_completed,
    read_committed_round,
    read_persisted_audit_result,
    read_pilot_completed,
    read_trajectory_state,
    remove_obsolete_resume_checkpoints,
    round_result_from_safe_payload,
    safe_payload_sha256,
    write_pilot_completed,
)
from .federated_round import (
    PreparedVictimTrainingInputs,
    prepare_auxiliary_training_input,
    prepare_victim_training_inputs,
    run_non_private_federated_round,
    validate_federated_round_result,
    validate_paired_federated_trajectory_round_results,
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
    VictimClientDataset,
    VictimDatasetGenerator,
    build_round_manifest,
    build_victim_dataset_manifest,
    read_auxiliary_round,
    read_victim_client_dataset,
    validate_conversation_preflight,
    validate_paired_auxiliary_rounds,
    write_auxiliary_round,
    write_victim_datasets,
)
from .training_contracts import LocalTrainingSpec, load_local_training_spec_from_config
from .trusted_evaluator import (
    preflight_extraction_audit,
    prepare_trusted_evaluator,
    run_extraction_audit,
    score_extraction_audit,
    validate_paired_extraction_audit_results,
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


def _materialize_data_preflight(
    spec: PilotExecutionSpec,
    victims: Sequence[VictimClientDataset],
) -> PilotPreflightResult:
    resolved = validate_pilot_execution_spec(spec)
    victim_datasets = tuple(victims)
    generator = AuxiliaryRoundGenerator(
        resolved.experiment_seed,
        schedule_id=resolved.schedule_id,
    )
    benign_rounds = tuple(
        generator.generate(round_id, presentation="benign")
        for round_id in range(1, resolved.rounds + 1)
    )
    adversarial_rounds = tuple(
        generator.generate(round_id, presentation="adversarial")
        for round_id in range(1, resolved.rounds + 1)
    )
    try:
        for benign, adversarial in zip(benign_rounds, adversarial_rounds):
            validate_paired_auxiliary_rounds(benign, adversarial)
        validate_conversation_preflight(victim_datasets, benign_rounds)
        validate_conversation_preflight(victim_datasets, adversarial_rounds)
        victim_manifest = build_victim_dataset_manifest(victim_datasets)
        benign_manifests = tuple(build_round_manifest(item) for item in benign_rounds)
        adversarial_manifests = tuple(
            build_round_manifest(item) for item in adversarial_rounds
        )
    except Exception as error:
        raise PilotExecutionError("preflight dos dados sintéticos falhou") from error
    benign_schedule_sha256 = _canonical_hash(
        [manifest["schedule_sha256"] for manifest in benign_manifests],
        b"pilot-auxiliary-schedule/v1",
    )
    adversarial_schedule_sha256 = _canonical_hash(
        [manifest["schedule_sha256"] for manifest in adversarial_manifests],
        b"pilot-auxiliary-schedule/v1",
    )
    paired_schedule_sha256 = _canonical_hash(
        [
            {
                "round": benign["round"],
                "schedule_sha256": benign["schedule_sha256"],
                "values_sha256": benign["values_sha256"],
            }
            for benign in benign_manifests
        ],
        b"pilot-paired-schedule/v1",
    )
    if benign_schedule_sha256 != adversarial_schedule_sha256:
        raise PilotExecutionError("preflight pareado possui agendas divergentes")
    return PilotPreflightResult(
        experiment_seed=resolved.experiment_seed,
        auxiliary_weight_units=resolved.auxiliary_weight_units,
        victim_client_count=len(victim_datasets),
        victim_conversation_count=sum(
            len(dataset.conversations) for dataset in victim_datasets
        ),
        auxiliary_round_count=len(benign_rounds) + len(adversarial_rounds),
        auxiliary_conversation_count=sum(
            len(round_data.conversations)
            for round_data in (*benign_rounds, *adversarial_rounds)
        ),
        victim_dataset_sha256=victim_manifest["dataset_sha256"],
        benign_schedule_sha256=benign_schedule_sha256,
        adversarial_schedule_sha256=adversarial_schedule_sha256,
        paired_schedule_sha256=paired_schedule_sha256,
    )


def preflight_pilot_execution(
    spec: PilotExecutionSpec,
    *,
    victim_datasets: Sequence[VictimClientDataset] | None = None,
) -> PilotPreflightResult:
    """Valida a agenda sintética completa sem carregar modelo nem escrever saídas."""

    resolved = validate_pilot_execution_spec(spec)
    victims = tuple(victim_datasets) if victim_datasets is not None else (
        VictimDatasetGenerator(resolved.experiment_seed).generate()
    )
    return _materialize_data_preflight(resolved, victims)


def _ensure_victim_artifacts(
    paths: PilotRunPaths,
    victims: Sequence[VictimClientDataset],
) -> Tuple[VictimClientDataset, ...]:
    target = paths.dataset_root / paths.identity.dataset_id
    if not target.exists():
        try:
            write_victim_datasets(
                paths.dataset_root,
                paths.identity.dataset_id,
                victims,
            )
        except Exception as error:
            raise PilotExecutionError("falha ao publicar datasets das vítimas") from error
    loaded = []
    try:
        for index in range(1, 11):
            loaded.append(
                read_victim_client_dataset(
                    paths.dataset_root,
                    paths.identity.dataset_id,
                    f"victim-{index:02d}",
                )
            )
    except Exception as error:
        raise PilotExecutionError("datasets persistidos das vítimas são inválidos") from error
    if tuple(loaded) != tuple(victims):
        raise PilotExecutionError("datasets persistidos divergem da seed do piloto")
    return tuple(loaded)


def _ensure_auxiliary_artifact(
    paths: PilotRunPaths,
    round_data: AuxiliaryRound,
) -> AuxiliaryRound:
    artifact_directory = (
        paths.dataset_root
        / paths.identity.dataset_id
        / "clients"
        / "auxiliary"
        / paths.identity.schedule_id
        / round_data.presentation
        / f"round-{round_data.round_id:03d}"
    )
    if not artifact_directory.exists():
        try:
            write_auxiliary_round(
                paths.dataset_root,
                paths.identity.dataset_id,
                paths.identity.schedule_id,
                round_data,
            )
        except Exception as error:
            raise PilotExecutionError("falha ao publicar rodada auxiliar") from error
    try:
        loaded = read_auxiliary_round(
            paths.dataset_root,
            paths.identity.dataset_id,
            paths.identity.schedule_id,
            round_data.presentation,
            round_data.round_id,
        )
    except Exception as error:
        raise PilotExecutionError("rodada auxiliar persistida é inválida") from error
    if loaded != round_data:
        raise PilotExecutionError("rodada auxiliar persistida diverge da agenda")
    return loaded


def _audit_output_identity(
    paths: PilotRunPaths,
    scenario: str,
) -> tuple[Path, str]:
    if scenario == "B0":
        return paths.run_root, "baseline"
    return paths.run_root / "trajectories", f"{scenario}-k01"


def _run_audits(
    *,
    spec: AuditSpec,
    contexts: Mapping[int, TrustedEvaluatorContext],
    target_counts: Sequence[int],
    scenario: str,
    round_id: int,
    model_bundle: LoadedModelBundle,
    model_state_sha256: str,
    paths: PilotRunPaths,
) -> Tuple[ExtractionAuditResult, ...]:
    output_root, run_id = _audit_output_identity(paths, scenario)
    checkpoint = AuditCheckpoint(
        scenario=scenario,
        experiment_seed=paths.identity.experiment_seed,
        round_id=round_id,
        auxiliary_weight_units=(
            None if scenario == "B0" else paths.identity.auxiliary_weight_units
        ),
        expected_model_sha256=model_state_sha256,
        model_provenance=model_bundle.provenance,
    )
    results = []
    for target_count in sorted(target_counts):
        context = contexts.get(target_count)
        if context is None:
            raise PilotExecutionError("contexto de auditoria está ausente")
        try:
            result = run_extraction_audit(
                spec,
                context,
                checkpoint,
                model_bundle,
                output_root=output_root,
                run_id=run_id,
                resume=True,
            )
        except Exception as error:
            raise PilotExecutionError("auditoria do checkpoint falhou") from error
        results.append(result)
    return tuple(results)


def _revalidate_completed_audit(
    paths: PilotRunPaths,
    audit_spec: AuditSpec,
    contexts: Mapping[int, TrustedEvaluatorContext],
    result: ExtractionAuditResult,
) -> ExtractionAuditResult:
    context = contexts.get(result.target_count)
    if context is None:
        raise PilotExecutionError("contexto da auditoria retomada está ausente")
    output_root, run_id = _audit_output_identity(paths, result.scenario)
    checkpoint = AuditCheckpoint(
        scenario=result.scenario,
        experiment_seed=result.experiment_seed,
        round_id=result.round_id,
        auxiliary_weight_units=result.auxiliary_weight_units,
        expected_model_sha256=result.model_state_sha256,
        model_provenance=result.model_provenance,
    )
    try:
        completed = read_completed_audit_artifacts(
            output_root=output_root,
            run_id=run_id,
            spec=audit_spec,
            context=context,
            checkpoint=checkpoint,
            generation_schedule_sha256=result.generation_schedule_sha256,
        )
        if completed is None:
            raise PilotExecutionError("auditoria confirmada está ausente")
        records, stored_summary = completed
        rescored = score_extraction_audit(
            audit_spec,
            context,
            checkpoint,
            records,
            generation_schedule_sha256=result.generation_schedule_sha256,
        )
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("auditoria confirmada é inválida") from error
    if rescored != result or stored_summary != result.as_safe_dict():
        raise PilotExecutionError("auditoria retomada diverge do commit da rodada")
    return result


def _load_committed_trajectory(
    paths: PilotRunPaths,
    scenario: str,
    completed_round: int,
    audit_spec: AuditSpec,
    contexts: Mapping[int, TrustedEvaluatorContext],
    baseline_model_sha256: str,
) -> tuple[list[FedAvgRoundResult], list[ExtractionAuditResult]]:
    rounds: list[FedAvgRoundResult] = []
    audits: list[ExtractionAuditResult] = []
    for round_id in range(1, completed_round + 1):
        result, round_audits, _, _ = read_committed_round(
            paths,
            scenario,
            round_id,
        )
        expected_initial = (
            baseline_model_sha256
            if not rounds
            else rounds[-1].final_model_sha256
        )
        _validate_trajectory_round_continuity(
            result,
            scenario=scenario,
            round_id=round_id,
            expected_initial_model_sha256=expected_initial,
        )
        rounds.append(result)
        audits.extend(
            _revalidate_completed_audit(paths, audit_spec, contexts, audit)
            for audit in round_audits
        )
    return rounds, audits


def _validate_trajectory_round_continuity(
    result: FedAvgRoundResult,
    *,
    scenario: str,
    round_id: int,
    expected_initial_model_sha256: str,
) -> None:
    try:
        resolved = validate_federated_round_result(result)
    except Exception as error:
        raise PilotExecutionError(
            f"resultado {scenario} da rodada {round_id} é incompatível"
        ) from error
    if resolved.scenario != scenario or resolved.round_id != round_id:
        raise PilotExecutionError(
            f"identidade {scenario} da rodada {round_id} diverge"
        )
    if resolved.initial_model_sha256 != expected_initial_model_sha256:
        raise PilotExecutionError(
            f"continuidade {scenario} da rodada {round_id} diverge"
        )


def _validate_trajectory_sequence(
    scenario: str,
    baseline_model_sha256: str,
    rounds: Sequence[FedAvgRoundResult],
) -> None:
    expected_initial = baseline_model_sha256
    for expected_round, result in enumerate(rounds, start=1):
        _validate_trajectory_round_continuity(
            result,
            scenario=scenario,
            round_id=expected_round,
            expected_initial_model_sha256=expected_initial,
        )
        expected_initial = result.final_model_sha256


def _trajectory_result(
    scenario: TrajectoryScenario,
    baseline_model_sha256: str,
    baseline_audit_sha256: str,
    rounds: Sequence[FedAvgRoundResult],
    audits: Sequence[ExtractionAuditResult],
) -> FederatedTrajectoryResult:
    round_tuple = tuple(rounds)
    audit_tuple = tuple(audits)
    if len(round_tuple) != 20 or not round_tuple:
        raise PilotExecutionError("trajetória não possui vinte rodadas")
    _validate_trajectory_sequence(scenario, baseline_model_sha256, round_tuple)
    payload = {
        "scenario": scenario,
        "baseline_audit_sha256": baseline_audit_sha256,
        "rounds": [safe_payload_sha256(item.as_safe_dict()) for item in round_tuple],
        "audits": [safe_payload_sha256(item.as_safe_dict()) for item in audit_tuple],
    }
    return FederatedTrajectoryResult(
        scenario=scenario,
        experiment_seed=round_tuple[0].experiment_seed,
        auxiliary_weight_units=round_tuple[0].auxiliary_weight_units,
        completed_rounds=len(round_tuple),
        conversation_count=sum(item.conversation_count for item in round_tuple),
        optimizer_steps=sum(item.optimizer_steps for item in round_tuple),
        baseline_model_sha256=baseline_model_sha256,
        baseline_audit_sha256=baseline_audit_sha256,
        final_model_sha256=round_tuple[-1].final_model_sha256,
        round_results=round_tuple,
        audit_results=audit_tuple,
        result_sha256=_canonical_hash(payload, b"federated-trajectory-result/v2"),
    )


def validate_paired_federated_trajectory_results(
    benign: FederatedTrajectoryResult,
    adversarial: FederatedTrajectoryResult,
) -> tuple[FederatedTrajectoryResult, FederatedTrajectoryResult]:
    """Valida integralmente duas trajetórias F0/F1 já concluídas."""

    if (
        not isinstance(benign, FederatedTrajectoryResult)
        or not isinstance(adversarial, FederatedTrajectoryResult)
        or benign.scenario != "F0"
        or adversarial.scenario != "F1"
        or benign.schema_version != adversarial.schema_version
        or benign.experiment_seed != adversarial.experiment_seed
        or benign.auxiliary_weight_units != adversarial.auxiliary_weight_units
        or benign.completed_rounds != 20
        or adversarial.completed_rounds != 20
        or benign.baseline_model_sha256 != adversarial.baseline_model_sha256
        or benign.baseline_audit_sha256 != adversarial.baseline_audit_sha256
        or len(benign.round_results) != 20
        or len(adversarial.round_results) != 20
    ):
        raise PilotExecutionError("trajetórias pareadas são incompatíveis")
    expected_benign_initial = benign.baseline_model_sha256
    expected_adversarial_initial = adversarial.baseline_model_sha256
    for expected_round, (first, second) in enumerate(
        zip(benign.round_results, adversarial.round_results),
        start=1,
    ):
        if first.round_id != expected_round or second.round_id != expected_round:
            raise PilotExecutionError("trajetórias pareadas estão fora de ordem")
        try:
            validate_paired_federated_trajectory_round_results(
                first,
                second,
                expected_benign_initial_model_sha256=expected_benign_initial,
                expected_adversarial_initial_model_sha256=(
                    expected_adversarial_initial
                ),
            )
        except FedAvgError as error:
            raise PilotExecutionError(str(error)) from error
        except Exception as error:
            raise PilotExecutionError(
                f"pareamento das trajetórias diverge na rodada {expected_round}"
            ) from error
        expected_benign_initial = first.final_model_sha256
        expected_adversarial_initial = second.final_model_sha256
    for trajectory in (benign, adversarial):
        expected = _trajectory_result(
            trajectory.scenario,
            trajectory.baseline_model_sha256,
            trajectory.baseline_audit_sha256,
            trajectory.round_results,
            trajectory.audit_results,
        )
        if expected != trajectory:
            raise PilotExecutionError("hash ou totais da trajetória divergem")
    benign_audits = {
        (audit.round_id, audit.target_count): audit
        for audit in benign.audit_results
    }
    adversarial_audits = {
        (audit.round_id, audit.target_count): audit
        for audit in adversarial.audit_results
    }
    if (
        len(benign_audits) != len(benign.audit_results)
        or len(adversarial_audits) != len(adversarial.audit_results)
        or frozenset(benign_audits) != frozenset(adversarial_audits)
    ):
        raise PilotExecutionError("agendas de auditoria das trajetórias divergem")
    for key in sorted(benign_audits):
        try:
            validate_paired_extraction_audit_results(
                benign_audits[key],
                adversarial_audits[key],
            )
        except Exception as error:
            raise PilotExecutionError("auditoria das trajetórias pareadas diverge") from error
    return benign, adversarial


def _validate_and_commit_pair(
    paths: PilotRunPaths,
    benign: FedAvgRoundResult,
    benign_audits: Sequence[ExtractionAuditResult],
    adversarial: FedAvgRoundResult,
    adversarial_audits: Sequence[ExtractionAuditResult],
    *,
    expected_benign_initial_model_sha256: str,
    expected_adversarial_initial_model_sha256: str,
    commit: bool = True,
) -> None:
    try:
        validate_paired_federated_trajectory_round_results(
            benign,
            adversarial,
            expected_benign_initial_model_sha256=(
                expected_benign_initial_model_sha256
            ),
            expected_adversarial_initial_model_sha256=(
                expected_adversarial_initial_model_sha256
            ),
        )
        benign_by_count = {result.target_count: result for result in benign_audits}
        adversarial_by_count = {
            result.target_count: result for result in adversarial_audits
        }
        if frozenset(benign_by_count) != frozenset(adversarial_by_count):
            raise PilotExecutionError("budgets pareados da auditoria divergem")
        pairs = tuple(
            (benign_by_count[count], adversarial_by_count[count])
            for count in sorted(benign_by_count)
        )
        for first, second in pairs:
            validate_paired_extraction_audit_results(first, second)
        if commit:
            commit_paired_round(paths, benign, adversarial, pairs)
    except PilotExecutionError:
        raise
    except (FedAvgError, ExtractionAuditError) as error:
        raise PilotExecutionError(str(error)) from error
    except Exception as error:
        raise PilotExecutionError(
            f"validação pareada da rodada {benign.round_id} falhou"
        ) from error


def _restore_committed_checkpoint(
    *,
    paths: PilotRunPaths,
    scenario: str,
    state_round: int,
    model_bundle: LoadedModelBundle,
    spec: PilotExecutionSpec,
    victim_dataset_sha256: str,
    baseline_model_sha256: str,
    baseline_audit_sha256: str,
) -> None:
    if state_round == 0:
        if fingerprint_model_parameters(model_bundle) != baseline_model_sha256:
            raise PilotExecutionError("modelo inicial da trajetória diverge do baseline")
        return
    _, _, checkpoint_id, expected_artifact_sha = read_committed_round(
        paths,
        scenario,
        state_round,
    )
    checkpoint = load_federated_checkpoint(
        paths.trajectory_root(scenario) / "checkpoints" / checkpoint_id,
        model_bundle,
        expected_scenario=scenario,
        expected_round_id=state_round,
        expected_config_sha256=spec.config_sha256,
        expected_victim_dataset_sha256=victim_dataset_sha256,
        expected_baseline_model_sha256=baseline_model_sha256,
        expected_baseline_audit_sha256=baseline_audit_sha256,
    )
    if checkpoint.artifact_sha256 != expected_artifact_sha:
        raise PilotExecutionError("hash do checkpoint retomado diverge do estado")


def run_non_private_trajectory(
    spec: PilotExecutionSpec,
    paths: PilotRunPaths,
    model_bundle: LoadedModelBundle,
    victims: Sequence[VictimClientDataset],
    victim_inputs: PreparedVictimTrainingInputs,
    audit_spec: AuditSpec,
    evaluator_contexts: Mapping[int, TrustedEvaluatorContext],
    local_training_spec: LocalTrainingSpec,
    fedavg_spec: FedAvgSpec,
    *,
    scenario: TrajectoryScenario,
    baseline_model_sha256: str,
    baseline_audit_sha256: str,
    progress_callback: ProgressCallback | None = None,
) -> FederatedTrajectoryResult:
    """Executa ou retoma uma trajetória F0/F1 sem reter deltas locais."""

    resolved = validate_pilot_execution_spec(spec)
    if scenario not in {"F0", "F1"}:
        raise PilotExecutionError("trajetória deve ser F0 ou F1")
    victim_manifest = build_victim_dataset_manifest(tuple(victims))
    state = read_trajectory_state(
        paths,
        scenario,
        baseline_model_sha256=baseline_model_sha256,
    )
    _restore_committed_checkpoint(
        paths=paths,
        scenario=scenario,
        state_round=state.completed_round,
        model_bundle=model_bundle,
        spec=resolved,
        victim_dataset_sha256=victim_manifest["dataset_sha256"],
        baseline_model_sha256=baseline_model_sha256,
        baseline_audit_sha256=baseline_audit_sha256,
    )
    round_results, audit_results = _load_committed_trajectory(
        paths,
        scenario,
        state.completed_round,
        audit_spec,
        evaluator_contexts,
        baseline_model_sha256,
    )
    presentation = "benign" if scenario == "F0" else "adversarial"
    generator = AuxiliaryRoundGenerator(
        resolved.experiment_seed,
        schedule_id=resolved.schedule_id,
    )
    for round_id in range(state.completed_round + 1, resolved.rounds + 1):
        expected_initial_model_sha256 = (
            baseline_model_sha256
            if not round_results
            else round_results[-1].final_model_sha256
        )
        generated_round = generator.generate(round_id, presentation=presentation)
        round_data = _ensure_auxiliary_artifact(paths, generated_round)
        auxiliary_manifest = build_round_manifest(round_data)
        try:
            auxiliary_input = prepare_auxiliary_training_input(round_data, model_bundle)
        except Exception as error:
            raise PilotExecutionError("tokenização da rodada auxiliar falhou") from error
        checkpoint_id = checkpoint_id_for_round(round_id, resolved.retained_rounds)
        checkpoint_path = checkpoint_path_for_round(
            paths,
            scenario,
            round_id,
            resolved.retained_rounds,
        )

        if checkpoint_path.exists():
            recovery_snapshot = capture_model_parameter_snapshot(model_bundle)
            recovered_committed = False
            try:
                recovered = load_federated_checkpoint(
                    checkpoint_path,
                    model_bundle,
                    expected_scenario=scenario,
                    expected_round_id=round_id,
                    expected_config_sha256=resolved.config_sha256,
                    expected_victim_dataset_sha256=victim_manifest["dataset_sha256"],
                    expected_baseline_model_sha256=baseline_model_sha256,
                    expected_baseline_audit_sha256=baseline_audit_sha256,
                )
                recovered_round = round_result_from_safe_payload(
                    recovered.round_result_payload
                )
                _validate_trajectory_round_continuity(
                    recovered_round,
                    scenario=scenario,
                    round_id=round_id,
                    expected_initial_model_sha256=expected_initial_model_sha256,
                )
                targets = (1, 5, 20, 200) if round_id == 20 else (20,)
                recovered_audits = tuple(
                    _revalidate_completed_audit(
                        paths,
                        audit_spec,
                        evaluator_contexts,
                        read_persisted_audit_result(
                            paths,
                            scenario=scenario,
                            round_id=round_id,
                            target_count=count,
                        ),
                    )
                    for count in targets
                )
                marker_hashes = {
                    marker.target_count: marker.result_sha256
                    for marker in recovered.metadata.audit_markers
                }
                if any(
                    marker_hashes.get(audit.target_count)
                    != safe_payload_sha256(audit.as_safe_dict())
                    for audit in recovered_audits
                ):
                    raise PilotExecutionError("auditoria recuperada diverge do checkpoint")
                if scenario == "F1":
                    benign, benign_audits, _, _ = read_committed_round(
                        paths, "F0", round_id
                    )
                    expected_benign_initial_model_sha256 = (
                        baseline_model_sha256
                        if round_id == 1
                        else read_committed_round(paths, "F0", round_id - 1)[
                            0
                        ].final_model_sha256
                    )
                    _validate_and_commit_pair(
                        paths,
                        benign,
                        benign_audits,
                        recovered_round,
                        recovered_audits,
                        expected_benign_initial_model_sha256=(
                            expected_benign_initial_model_sha256
                        ),
                        expected_adversarial_initial_model_sha256=(
                            expected_initial_model_sha256
                        ),
                        commit=False,
                    )
                state = commit_trajectory_round(
                    paths,
                    recovered_round,
                    recovered_audits,
                    auxiliary_manifest=auxiliary_manifest,
                    checkpoint_id=checkpoint_id,
                    checkpoint_artifact_sha256=recovered.artifact_sha256,
                    baseline_model_sha256=baseline_model_sha256,
                )
                recovered_committed = True
                round_results.append(recovered_round)
                audit_results.extend(recovered_audits)
                remove_obsolete_resume_checkpoints(
                    paths,
                    scenario,
                    keep_round=round_id,
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "round_completed",
                            "scenario": scenario,
                            "round_id": round_id,
                            "final_model_sha256": recovered_round.final_model_sha256,
                            "resumed": True,
                        }
                    )
                continue
            except Exception as error:
                if recovered_committed:
                    raise PilotExecutionError(
                        "falha após recuperar rodada já confirmada"
                    ) from error
                if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
                    raise PilotExecutionError("checkpoint candidato é inválido") from error
                shutil.rmtree(checkpoint_path)
                try:
                    restore_model_parameter_snapshot(model_bundle, recovery_snapshot)
                except Exception as restore_error:
                    raise PilotExecutionError(
                        "checkpoint candidato falhou e o modelo não foi restaurado"
                    ) from restore_error

        round_snapshot = None
        round_committed = False
        try:
            round_snapshot = capture_model_parameter_snapshot(model_bundle)
            result = run_non_private_federated_round(
                victim_inputs,
                auxiliary_input,
                model_bundle,
                local_training_spec,
                fedavg_spec,
                seed=resolved.experiment_seed,
                scenario=scenario,
                round_id=round_id,
                auxiliary_weight_units=resolved.auxiliary_weight_units,
                initial_snapshot=round_snapshot,
            )
            _validate_trajectory_round_continuity(
                result,
                scenario=scenario,
                round_id=round_id,
                expected_initial_model_sha256=expected_initial_model_sha256,
            )
            target_counts = (1, 5, 20, 200) if round_id == 20 else (20,)
            audits = _run_audits(
                spec=audit_spec,
                contexts=evaluator_contexts,
                target_counts=target_counts,
                scenario=scenario,
                round_id=round_id,
                model_bundle=model_bundle,
                model_state_sha256=result.final_model_sha256,
                paths=paths,
            )
            if scenario == "F1":
                benign, benign_audits, _, _ = read_committed_round(
                    paths, "F0", round_id
                )
                expected_benign_initial_model_sha256 = (
                    baseline_model_sha256
                    if round_id == 1
                    else read_committed_round(paths, "F0", round_id - 1)[
                        0
                    ].final_model_sha256
                )
                _validate_and_commit_pair(
                    paths,
                    benign,
                    benign_audits,
                    result,
                    audits,
                    expected_benign_initial_model_sha256=(
                        expected_benign_initial_model_sha256
                    ),
                    expected_adversarial_initial_model_sha256=(
                        expected_initial_model_sha256
                    ),
                    commit=False,
                )
            checkpoint_metadata = build_federated_checkpoint_metadata(
                round_result=result,
                audits=audits,
                config_sha256=resolved.config_sha256,
                baseline_model_sha256=baseline_model_sha256,
                baseline_audit_sha256=baseline_audit_sha256,
                canonical_template_sha256=victim_manifest[
                    "canonical_template_sha256"
                ],
            )
            checkpoint = save_federated_checkpoint(
                checkpoint_path,
                model_bundle,
                checkpoint_metadata,
                result,
            )
            state = commit_trajectory_round(
                paths,
                result,
                audits,
                auxiliary_manifest=auxiliary_manifest,
                checkpoint_id=checkpoint_id,
                checkpoint_artifact_sha256=checkpoint.artifact_sha256,
                baseline_model_sha256=baseline_model_sha256,
            )
            round_committed = True
            if scenario == "F1":
                _validate_and_commit_pair(
                    paths,
                    benign,
                    benign_audits,
                    result,
                    audits,
                    expected_benign_initial_model_sha256=(
                        expected_benign_initial_model_sha256
                    ),
                    expected_adversarial_initial_model_sha256=(
                        expected_initial_model_sha256
                    ),
                )
            remove_obsolete_resume_checkpoints(
                paths,
                scenario,
                keep_round=round_id,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "round_completed",
                        "scenario": scenario,
                        "round_id": round_id,
                        "final_model_sha256": result.final_model_sha256,
                        "mean_client_loss": result.mean_client_loss,
                        "resumed": False,
                    }
                )
        except Exception as error:
            if round_committed:
                raise PilotExecutionError(
                    "falha após confirmar rodada federada"
                ) from error
            try:
                if round_snapshot is None:
                    raise PilotExecutionError("snapshot da rodada está ausente")
                restore_model_parameter_snapshot(model_bundle, round_snapshot)
            except Exception as restore_error:
                raise PilotExecutionError(
                    "rodada falhou e o modelo não pôde ser restaurado"
                ) from restore_error
            if isinstance(error, PilotExecutionError):
                raise
            raise PilotExecutionError("execução da rodada federada falhou") from error
        round_results.append(result)
        audit_results.extend(audits)
    remove_obsolete_resume_checkpoints(
        paths,
        scenario,
        keep_round=resolved.rounds,
    )
    return _trajectory_result(
        scenario,
        baseline_model_sha256,
        baseline_audit_sha256,
        round_results,
        audit_results,
    )


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


def run_paired_pilot(
    spec: PilotExecutionSpec,
    identity: PilotRunIdentity,
    *,
    config_path: Path,
    output_root: Path = Path("outputs"),
    cache_dir: Path = DEFAULT_MODEL_CACHE,
    model_artifact_dir: Path | None = None,
    device: str,
    preflight_only: bool = False,
    fresh: bool = False,
    model_loader: BundleLoader | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PilotExecutionResult | PilotPreflightResult:
    """Executa B0 uma vez e as trajetórias F0/F1 sequencialmente."""

    try:
        validate_cuda_reproducibility_environment(device)
    except ReproducibilityEnvironmentError as error:
        raise PilotExecutionError(str(error)) from error
    resolved = validate_pilot_execution_spec(spec)
    calibration_gate = load_completed_calibration_gate(output_root, resolved)
    if identity.experiment_seed != resolved.experiment_seed or (
        identity.config_sha256 != resolved.config_sha256
    ) or identity.calibration_result_sha256 != calibration_gate.result_sha256 or (
        identity.calibration_manifest_sha256 != calibration_gate.manifest_sha256
    ):
        raise PilotExecutionError("identidade da execução diverge da configuração")
    try:
        victims = VictimDatasetGenerator(resolved.experiment_seed).generate()
        data_preflight = _materialize_data_preflight(resolved, victims)
        audit_spec = load_extraction_audit_spec_from_config(config_path)
        local_spec = load_local_training_spec_from_config(config_path)
        fedavg_spec = load_fedavg_spec_from_config(config_path)
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("preflight da configuração falhou") from error
    if progress_callback is not None:
        progress_callback(
            {
                "event": "data_preflight_completed",
                "victim_conversation_count": data_preflight.victim_conversation_count,
                "auxiliary_conversation_count": data_preflight.auxiliary_conversation_count,
                "paired_schedule_sha256": data_preflight.paired_schedule_sha256,
            }
        )
    loader = model_loader or _default_bundle_loader(
        config_path=config_path,
        cache_dir=cache_dir,
        model_artifact_dir=model_artifact_dir,
        device=device,
    )
    try:
        first_bundle = loader()
        baseline_model_sha256 = fingerprint_model_parameters(first_bundle)
        if (
            baseline_model_sha256 != calibration_gate.baseline_model_sha256
            or first_bundle.provenance != calibration_gate.model_provenance
        ):
            raise PilotExecutionError("modelo diverge do gate da calibração")
        victim_inputs = prepare_victim_training_inputs(victims, first_bundle)
        contexts = {
            count: prepare_trusted_evaluator(
                victims,
                resolved.experiment_seed,
                target_count=count,
            )
            for count in resolved.target_counts
        }
        for context in contexts.values():
            preflight_extraction_audit(audit_spec, context, first_bundle)
    except Exception as error:
        raise PilotExecutionError("preflight do modelo e tokenizador falhou") from error
    if preflight_only:
        return PilotPreflightResult(
            **{
                **data_preflight.as_safe_dict(),
                "model_state_sha256": baseline_model_sha256,
                "tokenization_validated": True,
                "calibration_result_sha256": calibration_gate.result_sha256,
                "calibration_manifest_sha256": calibration_gate.manifest_sha256,
            }
        )

    paths = initialize_pilot_run(
        output_root=output_root,
        identity=identity,
        spec=resolved,
        model_provenance=first_bundle.provenance,
        baseline_model_sha256=baseline_model_sha256,
        fresh=fresh,
    )
    persisted_victims = _ensure_victim_artifacts(paths, victims)
    if persisted_victims != victims:
        raise PilotExecutionError("datasets carregados divergem do preflight")
    baseline_audits = _run_audits(
        spec=audit_spec,
        contexts=contexts,
        target_counts=resolved.target_counts,
        scenario="B0",
        round_id=0,
        model_bundle=first_bundle,
        model_state_sha256=baseline_model_sha256,
        paths=paths,
    )
    baseline_audit_sha256 = mark_baseline_completed(
        paths,
        baseline_model_sha256,
        baseline_audits,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "baseline_completed",
                "audit_generation_count": sum(
                    result.generation_count for result in baseline_audits
                ),
                "model_state_sha256": baseline_model_sha256,
            }
        )
    f0 = run_non_private_trajectory(
        resolved,
        paths,
        first_bundle,
        persisted_victims,
        victim_inputs,
        audit_spec,
        contexts,
        local_spec,
        fedavg_spec,
        scenario="F0",
        baseline_model_sha256=baseline_model_sha256,
        baseline_audit_sha256=baseline_audit_sha256,
        progress_callback=progress_callback,
    )
    del first_bundle
    gc.collect()
    try:
        second_bundle = loader()
        second_baseline = fingerprint_model_parameters(second_bundle)
    except Exception as error:
        raise PilotExecutionError("falha ao recarregar baseline para F1") from error
    if second_baseline != baseline_model_sha256:
        raise PilotExecutionError("F1 não iniciou do mesmo baseline de F0")
    f1 = run_non_private_trajectory(
        resolved,
        paths,
        second_bundle,
        persisted_victims,
        victim_inputs,
        audit_spec,
        contexts,
        local_spec,
        fedavg_spec,
        scenario="F1",
        baseline_model_sha256=baseline_model_sha256,
        baseline_audit_sha256=baseline_audit_sha256,
        progress_callback=progress_callback,
    )
    validate_paired_federated_trajectory_results(f0, f1)
    expected_benign_initial_model_sha256 = baseline_model_sha256
    expected_adversarial_initial_model_sha256 = baseline_model_sha256
    for round_id in range(1, resolved.rounds + 1):
        benign, benign_audits, _, _ = read_committed_round(paths, "F0", round_id)
        adversarial, adversarial_audits, _, _ = read_committed_round(
            paths, "F1", round_id
        )
        _validate_and_commit_pair(
            paths,
            benign,
            benign_audits,
            adversarial,
            adversarial_audits,
            expected_benign_initial_model_sha256=(
                expected_benign_initial_model_sha256
            ),
            expected_adversarial_initial_model_sha256=(
                expected_adversarial_initial_model_sha256
            ),
        )
        expected_benign_initial_model_sha256 = benign.final_model_sha256
        expected_adversarial_initial_model_sha256 = adversarial.final_model_sha256
    paired_sha = _canonical_hash(
        {
            "f0": f0.result_sha256,
            "f1": f1.result_sha256,
            "baseline": baseline_model_sha256,
            "baseline_audit": baseline_audit_sha256,
        },
        b"paired-pilot-result/v2",
    )
    result = PilotExecutionResult(
        identity=identity,
        baseline_model_sha256=baseline_model_sha256,
        baseline_audit_sha256=baseline_audit_sha256,
        baseline_audits=baseline_audits,
        trajectories=(f0, f1),
        total_federated_rounds=f0.completed_rounds + f1.completed_rounds,
        total_conversation_count=f0.conversation_count + f1.conversation_count,
        total_optimizer_steps=f0.optimizer_steps + f1.optimizer_steps,
        total_audit_generations=(
            sum(item.generation_count for item in baseline_audits)
            + sum(item.generation_count for item in f0.audit_results)
            + sum(item.generation_count for item in f1.audit_results)
        ),
        paired_results_sha256=paired_sha,
        completed=True,
    )
    if (
        result.total_federated_rounds != 40
        or result.total_conversation_count != 44_000
        or result.total_optimizer_steps != 11_000
        or result.total_audit_generations != PILOT_EXPECTED_GENERATION_COUNT
    ):
        raise PilotExecutionError("totais finais do piloto divergem do protocolo")
    completed = read_pilot_completed(paths)
    if completed is None:
        write_pilot_completed(paths, result)
    elif completed != result.as_safe_dict():
        raise PilotExecutionError("marcador final existente diverge do piloto")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "pilot_completed",
                "total_federated_rounds": result.total_federated_rounds,
                "total_audit_generations": result.total_audit_generations,
                "paired_results_sha256": result.paired_results_sha256,
            }
        )
    return result


__all__ = [
    "preflight_pilot_execution",
    "run_non_private_trajectory",
    "run_paired_pilot",
    "validate_paired_federated_trajectory_results",
]
