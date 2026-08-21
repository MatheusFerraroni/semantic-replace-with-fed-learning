"""Journal privado e retomável da auditoria de extração."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Tuple

from .audit_contracts import (
    EXTRACTION_AUDIT_RECORD_SCHEMA_VERSION,
    AuditCheckpoint,
    AuditGenerationRecord,
    AuditSpec,
    ExtractionAuditError,
    ExtractionAuditResult,
    TrustedEvaluatorContext,
    validate_extraction_audit_spec,
    validate_extraction_audit_result,
)
from .synthetic_profiles.storage import validate_storage_component


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


def _load_json(raw: bytes) -> dict[str, Any]:
    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExtractionAuditError("artefato contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except ExtractionAuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExtractionAuditError("artefato de auditoria é inválido") from error
    if not isinstance(value, dict):
        raise ExtractionAuditError("artefato de auditoria deve ser objeto")
    return value


def _require_keys(value: Mapping[str, Any], expected: frozenset[str]) -> None:
    if frozenset(value) != expected:
        raise ExtractionAuditError("artefato de auditoria possui chaves inválidas")


def _mkdir_private(path: Path, *, exist_ok: bool = False) -> None:
    path.mkdir(mode=0o700, parents=False, exist_ok=exist_ok)
    if path.is_symlink() or not path.is_dir():
        raise ExtractionAuditError("diretório privado da auditoria é inválido")
    os.chmod(path, 0o700)


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(path, 0o600)
    except FileExistsError:
        raise
    except OSError as error:
        raise ExtractionAuditError("falha ao escrever artefato privado") from error


def _safe_component(value: str, label: str) -> str:
    try:
        return validate_storage_component(value, label)
    except Exception as error:
        raise ExtractionAuditError(f"{label} inválido") from error


def _audit_id(checkpoint: AuditCheckpoint) -> str:
    if checkpoint.scenario == "B0":
        return "B0-round-000"
    return (
        f"{checkpoint.scenario}-k{checkpoint.auxiliary_weight_units:02d}-"
        f"round-{checkpoint.round_id:03d}"
    )


def _registry_payload(context: TrustedEvaluatorContext) -> dict[str, Any]:
    return {
        "schema_version": "trusted-evaluator-registry/v1",
        "experiment_seed": context.experiment_seed,
        "record_count": len(context.registry),
        "registry_sha256": context.registry_sha256,
        "records": [
            {
                "client_id": record.client_id,
                "entity_id": record.entity_id,
                "fields": dict(record.field_values),
            }
            for record in context.registry
        ],
    }


def _target_payload(context: TrustedEvaluatorContext) -> dict[str, Any]:
    return {
        "schema_version": "trusted-evaluator-target-manifest/v1",
        "experiment_seed": context.experiment_seed,
        "target_count": len(context.targets),
        "target_schedule_sha256": context.target_schedule_sha256,
        "targets": [
            {
                "client_id": record.client_id,
                "entity_id": record.entity_id,
                "person_name": record.value("PERSON_NAME"),
            }
            for record in context.targets
        ],
    }


def _spec_payload(spec: AuditSpec) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "evaluator_schema_version": spec.evaluator_schema_version,
        "record_schema_version": spec.record_schema_version,
        "result_schema_version": spec.result_schema_version,
        "prompt_catalog_version": spec.prompt_catalog_version,
        "target_profiles": spec.target_profiles,
        "targets_per_client": spec.targets_per_client,
        "generation_seeds_per_target": spec.generation_seeds_per_target,
        "primary_max_new_tokens": spec.primary_max_new_tokens,
        "field_generations_per_pair": spec.field_generations_per_pair,
        "field_max_new_tokens": spec.field_max_new_tokens,
        "untargeted_generations": spec.untargeted_generations,
        "untargeted_max_new_tokens": spec.untargeted_max_new_tokens,
        "partial_match_threshold": spec.partial_match_threshold,
        "exact_match_normalization": spec.exact_match_normalization,
        "generation": {
            "do_sample": spec.generation.do_sample,
            "num_beams": spec.generation.num_beams,
            "temperature": spec.generation.temperature,
            "top_p": spec.generation.top_p,
            "top_k": spec.generation.top_k,
            "repetition_penalty": spec.generation.repetition_penalty,
            "use_cache": spec.generation.use_cache,
        },
    }


def _metadata_payload(
    run_id: str,
    audit_id: str,
    spec: AuditSpec,
    context: TrustedEvaluatorContext,
    checkpoint: AuditCheckpoint,
    generation_schedule_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "extraction-audit-journal/v1",
        "run_id": run_id,
        "audit_id": audit_id,
        "scenario": checkpoint.scenario,
        "experiment_seed": checkpoint.experiment_seed,
        "round_id": checkpoint.round_id,
        "auxiliary_weight_units": checkpoint.auxiliary_weight_units,
        "expected_model_sha256": checkpoint.expected_model_sha256,
        "model_provenance": checkpoint.model_provenance.as_safe_dict(),
        "registry_sha256": context.registry_sha256,
        "target_schedule_sha256": context.target_schedule_sha256,
        "prompt_catalog_sha256": context.prompt_catalog_sha256,
        "generation_schedule_sha256": generation_schedule_sha256,
        "expected_generation_count": spec.expected_generation_count,
        "spec": _spec_payload(spec),
    }


def _record_payload(record: AuditGenerationRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "query_index": record.query_index,
        "mode": record.mode,
        "target_index": record.target_index,
        "target_entity_id": record.target_entity_id,
        "field_type": record.field_type,
        "replicate_index": record.replicate_index,
        "generation_seed": record.generation_seed,
        "max_new_tokens": record.max_new_tokens,
        "finish_reason": record.finish_reason,
        "prompt": record.prompt,
        "generated_text": record.generated_text,
    }


_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "query_index",
        "mode",
        "target_index",
        "target_entity_id",
        "field_type",
        "replicate_index",
        "generation_seed",
        "max_new_tokens",
        "finish_reason",
        "prompt",
        "generated_text",
    }
)


def _record_from_payload(value: Mapping[str, Any]) -> AuditGenerationRecord:
    _require_keys(value, _RECORD_KEYS)
    try:
        record = AuditGenerationRecord(**value)
    except (TypeError, ValueError) as error:
        raise ExtractionAuditError("registro privado possui tipos inválidos") from error
    if (
        record.schema_version != EXTRACTION_AUDIT_RECORD_SCHEMA_VERSION
        or type(record.query_index) is not int
        or record.query_index < 0
        or record.mode not in {"primary", "field_specific", "untargeted"}
        or type(record.replicate_index) is not int
        or record.replicate_index < 0
        or type(record.generation_seed) is not int
        or record.generation_seed < 0
        or type(record.max_new_tokens) is not int
        or record.max_new_tokens <= 0
        or record.finish_reason not in {"eos", "max_tokens"}
        or not isinstance(record.prompt, str)
        or not isinstance(record.generated_text, str)
    ):
        raise ExtractionAuditError("registro privado viola o contrato")
    return record


def _load_records(path: Path) -> Tuple[AuditGenerationRecord, ...]:
    if not path.exists():
        return ()
    if path.is_symlink() or not path.is_file():
        raise ExtractionAuditError("journal privado é inválido")
    os.chmod(path, 0o600)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ExtractionAuditError("journal privado é inacessível") from error
    if b"\r" in raw:
        raise ExtractionAuditError("journal privado não usa LF canônico")
    if raw and not raw.endswith(b"\n"):
        last_lf = raw.rfind(b"\n")
        recovered = raw[: last_lf + 1] if last_lf >= 0 else b""
        try:
            with path.open("r+b") as output:
                output.truncate(len(recovered))
                output.flush()
                os.fsync(output.fileno())
            raw = recovered
        except OSError as error:
            raise ExtractionAuditError("journal parcial não pôde ser recuperado") from error
    records = []
    for expected_index, line in enumerate(raw.splitlines()):
        if not line:
            raise ExtractionAuditError("journal privado contém linha vazia")
        payload = _load_json(line)
        if _canonical_json_bytes(payload) != line + b"\n":
            raise ExtractionAuditError("journal privado não é canônico")
        record = _record_from_payload(payload)
        if record.query_index != expected_index:
            raise ExtractionAuditError("journal privado possui lacuna ou duplicação")
        records.append(record)
    return tuple(records)


class AuditJournal:
    """Journal mutável restrito à execução do avaliador."""

    def __init__(
        self,
        *,
        incomplete_directory: Path,
        final_directory: Path,
        summary_path: Path,
        records: Tuple[AuditGenerationRecord, ...],
        expected_count: int,
    ) -> None:
        self._incomplete_directory = incomplete_directory
        self._final_directory = final_directory
        self._summary_path = summary_path
        self._records = list(records)
        self._expected_count = expected_count

    @property
    def records(self) -> Tuple[AuditGenerationRecord, ...]:
        return tuple(self._records)

    def append(self, record: AuditGenerationRecord) -> None:
        if not isinstance(record, AuditGenerationRecord):
            raise ExtractionAuditError("journal recebeu registro inválido")
        if record.query_index != len(self._records) or len(self._records) >= self._expected_count:
            raise ExtractionAuditError("journal recebeu registro fora de ordem")
        raw = _canonical_json_bytes(_record_payload(record))
        path = self._incomplete_directory / "extraction_results.jsonl"
        try:
            with path.open("ab") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(path, 0o600)
        except OSError as error:
            raise ExtractionAuditError("falha ao anexar resultado privado") from error
        self._records.append(record)

    def finalize(self, result: ExtractionAuditResult) -> None:
        validate_extraction_audit_result(result)
        if (
            not isinstance(result, ExtractionAuditResult)
            or len(self._records) != self._expected_count
            or result.generation_count != self._expected_count
        ):
            raise ExtractionAuditError("auditoria incompleta não pode ser finalizada")
        if self._final_directory.exists() or self._summary_path.exists():
            raise FileExistsError("auditoria final já existe")
        summary_staging = self._summary_path.with_name(
            f".{self._summary_path.name}.incomplete"
        )
        summary_raw = _canonical_json_bytes(result.as_safe_dict())
        if summary_staging.exists():
            if (
                summary_staging.is_symlink()
                or not summary_staging.is_file()
                or summary_staging.read_bytes() != summary_raw
            ):
                raise ExtractionAuditError("resumo parcial diverge da auditoria")
            os.chmod(summary_staging, 0o600)
        else:
            _write_exclusive(summary_staging, summary_raw)
        try:
            self._incomplete_directory.rename(self._final_directory)
            summary_staging.rename(self._summary_path)
        except OSError as error:
            raise ExtractionAuditError("falha ao publicar auditoria final") from error


def _ensure_private_manifest(path: Path, payload: dict[str, Any]) -> None:
    expected = _canonical_json_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ExtractionAuditError("manifesto privado é inválido")
        try:
            current = path.read_bytes()
        except OSError as error:
            raise ExtractionAuditError("manifesto privado é inacessível") from error
        if current != expected:
            raise ExtractionAuditError("manifesto privado diverge da execução")
        os.chmod(path, 0o600)
    else:
        _write_exclusive(path, expected)


def prepare_audit_journal(
    *,
    output_root: Path,
    run_id: str,
    spec: AuditSpec,
    context: TrustedEvaluatorContext,
    checkpoint: AuditCheckpoint,
    generation_schedule_sha256: str,
    resume: bool,
) -> AuditJournal:
    """Cria ou retoma somente o journal correspondente ao checkpoint exato."""

    validate_extraction_audit_spec(spec)
    resolved_run_id = _safe_component(run_id, "run_id")
    audit_id = _safe_component(_audit_id(checkpoint), "audit_id")
    root = Path(output_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ExtractionAuditError("raiz de saídas da auditoria é inválida")
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    run_root = root / resolved_run_id
    if not run_root.exists():
        _mkdir_private(run_root)
    elif run_root.is_symlink() or not run_root.is_dir():
        raise ExtractionAuditError("diretório da execução é inválido")
    else:
        os.chmod(run_root, 0o700)
    evaluator_root = run_root / "evaluator"
    if not evaluator_root.exists():
        _mkdir_private(evaluator_root)
    elif evaluator_root.is_symlink() or not evaluator_root.is_dir():
        raise ExtractionAuditError("diretório do avaliador é inválido")
    else:
        os.chmod(evaluator_root, 0o700)
    private_root = evaluator_root / "private"
    if not private_root.exists():
        _mkdir_private(private_root)
    elif private_root.is_symlink() or not private_root.is_dir():
        raise ExtractionAuditError("diretório privado do avaliador é inválido")
    else:
        os.chmod(private_root, 0o700)
    audit_root = private_root / "audits"
    if not audit_root.exists():
        _mkdir_private(audit_root)
    elif audit_root.is_symlink() or not audit_root.is_dir():
        raise ExtractionAuditError("diretório de auditorias é inválido")
    else:
        os.chmod(audit_root, 0o700)
    summaries_root = evaluator_root / "summaries"
    if not summaries_root.exists():
        _mkdir_private(summaries_root)
    elif summaries_root.is_symlink() or not summaries_root.is_dir():
        raise ExtractionAuditError("diretório de resumos é inválido")
    else:
        os.chmod(summaries_root, 0o700)

    _ensure_private_manifest(
        private_root / "protected_value_registry_evaluator_only.json",
        _registry_payload(context),
    )
    _ensure_private_manifest(
        private_root / "audit_victim_name_manifest_evaluator_only.json",
        _target_payload(context),
    )

    final_directory = audit_root / audit_id
    incomplete_directory = audit_root / f"{audit_id}.incomplete"
    summary_path = summaries_root / f"{audit_id}.json"
    summary_staging = summary_path.with_name(f".{summary_path.name}.incomplete")
    if final_directory.exists() and not summary_path.exists():
        if (
            final_directory.is_symlink()
            or not final_directory.is_dir()
            or not summary_staging.is_file()
            or summary_staging.is_symlink()
        ):
            raise ExtractionAuditError("publicação final da auditoria é inconsistente")
        try:
            summary_staging.rename(summary_path)
        except OSError as error:
            raise ExtractionAuditError("resumo final não pôde ser recuperado") from error
    if final_directory.exists() or summary_path.exists():
        raise FileExistsError("auditoria já foi concluída")
    expected_metadata = _metadata_payload(
        resolved_run_id,
        audit_id,
        spec,
        context,
        checkpoint,
        generation_schedule_sha256,
    )
    if incomplete_directory.exists():
        if not resume:
            raise FileExistsError("auditoria parcial já existe")
        if incomplete_directory.is_symlink() or not incomplete_directory.is_dir():
            raise ExtractionAuditError("journal parcial é inválido")
        metadata_path = incomplete_directory / "metadata.json"
        try:
            metadata_raw = metadata_path.read_bytes()
        except OSError as error:
            raise ExtractionAuditError("metadados do journal estão ausentes") from error
        metadata = _load_json(metadata_raw)
        if metadata != expected_metadata or _canonical_json_bytes(metadata) != metadata_raw:
            raise ExtractionAuditError("metadados do journal divergem")
        os.chmod(metadata_path, 0o600)
    else:
        _mkdir_private(incomplete_directory)
        _write_exclusive(
            incomplete_directory / "metadata.json",
            _canonical_json_bytes(expected_metadata),
        )
    records = _load_records(incomplete_directory / "extraction_results.jsonl")
    if len(records) > spec.expected_generation_count:
        raise ExtractionAuditError("journal excede a contagem esperada")
    return AuditJournal(
        incomplete_directory=incomplete_directory,
        final_directory=final_directory,
        summary_path=summary_path,
        records=records,
        expected_count=spec.expected_generation_count,
    )


__all__ = ["AuditJournal", "prepare_audit_journal"]
