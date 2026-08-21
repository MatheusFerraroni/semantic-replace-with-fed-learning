"""Checkpoints safetensors estritos e atômicos das trajetórias federadas."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aggregation_contracts import FedAvgRoundResult
from .audit_contracts import ExtractionAuditResult, validate_extraction_audit_result
from .execution_contracts import (
    FEDERATED_CHECKPOINT_SCHEMA_VERSION,
    CheckpointAuditMarker,
    FederatedCheckpointMetadata,
    LoadedFederatedCheckpoint,
    PilotExecutionError,
)
from .model_contracts import LoadedModelBundle, ModelProvenance
from .model_fingerprint import fingerprint_model_parameters
from .model_updates import (
    capture_model_parameter_snapshot,
    restore_model_parameter_snapshot,
)
from .synthetic_profiles.storage import validate_storage_component


_CHECKPOINT_FILES = frozenset(
    {"metadata.json", "model.safetensors", "rng_state.json", "round_result.json"}
)
_SHA256_PATTERN = frozenset("0123456789abcdef")


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_PATTERN
    )


def _validate_checkpoint_model(model_bundle: LoadedModelBundle) -> None:
    try:
        import torch

        parameters = tuple(model_bundle.model.parameters())
        expected_device = model_bundle.provenance.device.split(":", 1)[0]
        if (
            not parameters
            or model_bundle.provenance.weight_dtype != "bfloat16"
            or sum(parameter.numel() for parameter in parameters)
            != model_bundle.provenance.parameter_count
            or any(parameter.dtype != torch.bfloat16 for parameter in parameters)
            or any(parameter.device.type != expected_device for parameter in parameters)
            or any(
                not torch.isfinite(parameter.detach()).all().item()
                for parameter in parameters
            )
        ):
            raise PilotExecutionError("modelo do checkpoint é incompatível")
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("modelo do checkpoint não pôde ser validado") from error


def _load_json(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PilotExecutionError("checkpoint contém chave duplicada")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except PilotExecutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotExecutionError("checkpoint contém JSON inválido") from error
    if not isinstance(value, dict):
        raise PilotExecutionError("checkpoint deve conter objeto JSON")
    return value


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
        raise PilotExecutionError("falha ao escrever checkpoint") from error


def _tensor_bytes(tensor: Any) -> bytes:
    try:
        return bytes(tensor.detach().to(device="cpu").tolist())
    except Exception as error:
        raise PilotExecutionError("estado RNG é inválido") from error


def _capture_rng_state(model_bundle: LoadedModelBundle) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise PilotExecutionError("PyTorch é obrigatório para checkpoints") from error
    try:
        cpu_state = _tensor_bytes(torch.get_rng_state())
        device_type = model_bundle.provenance.device.split(":", 1)[0]
        device_state: bytes | None = None
        if device_type == "cuda":
            device_state = _tensor_bytes(
                torch.cuda.get_rng_state(model_bundle.provenance.device)
            )
        elif device_type == "mps":
            device_state = _tensor_bytes(torch.mps.get_rng_state())
        elif device_type != "cpu":
            raise PilotExecutionError("dispositivo do RNG é incompatível")
    except PilotExecutionError:
        raise
    except Exception as error:
        raise PilotExecutionError("falha ao capturar RNG do checkpoint") from error
    return {
        "schema_version": "federated-checkpoint-rng/v1",
        "cpu_state_base64": base64.b64encode(cpu_state).decode("ascii"),
        "device_type": device_type,
        "device_state_base64": (
            base64.b64encode(device_state).decode("ascii")
            if device_state is not None
            else None
        ),
    }


def _decode_rng_state(payload: Mapping[str, Any]) -> tuple[bytes, str, bytes | None]:
    expected = frozenset(
        {"schema_version", "cpu_state_base64", "device_type", "device_state_base64"}
    )
    if frozenset(payload) != expected or payload.get("schema_version") != (
        "federated-checkpoint-rng/v1"
    ):
        raise PilotExecutionError("estado RNG do checkpoint é incompatível")
    device_type = payload.get("device_type")
    encoded_cpu = payload.get("cpu_state_base64")
    encoded_device = payload.get("device_state_base64")
    if (
        device_type not in {"cpu", "cuda", "mps"}
        or not isinstance(encoded_cpu, str)
        or (encoded_device is not None and not isinstance(encoded_device, str))
    ):
        raise PilotExecutionError("estado RNG do checkpoint possui tipos inválidos")
    try:
        cpu_state = base64.b64decode(encoded_cpu, validate=True)
        device_state = (
            base64.b64decode(encoded_device, validate=True)
            if encoded_device is not None
            else None
        )
    except (ValueError, TypeError) as error:
        raise PilotExecutionError("estado RNG do checkpoint é inválido") from error
    if not cpu_state or (device_type == "cpu") != (device_state is None):
        raise PilotExecutionError("estado RNG do checkpoint está incompleto")
    return cpu_state, device_type, device_state


def _restore_rng_state(payload: Mapping[str, Any], model_bundle: LoadedModelBundle) -> None:
    cpu_state, device_type, device_state = _decode_rng_state(payload)
    expected_device = model_bundle.provenance.device.split(":", 1)[0]
    if device_type != expected_device:
        raise PilotExecutionError("estado RNG pertence a outro dispositivo")
    try:
        import torch

        torch.set_rng_state(torch.tensor(list(cpu_state), dtype=torch.uint8))
        if device_type == "cuda" and device_state is not None:
            torch.cuda.set_rng_state(
                torch.tensor(list(device_state), dtype=torch.uint8),
                model_bundle.provenance.device,
            )
        elif device_type == "mps" and device_state is not None:
            torch.mps.set_rng_state(torch.tensor(list(device_state), dtype=torch.uint8))
    except Exception as error:
        raise PilotExecutionError("falha ao restaurar RNG do checkpoint") from error


def _audit_id(result: ExtractionAuditResult) -> str:
    budget = f"targets-{result.target_count:03d}"
    if result.scenario == "B0":
        return f"B0-{budget}-round-000"
    return (
        f"{result.scenario}-k{result.auxiliary_weight_units:02d}-"
        f"{budget}-round-{result.round_id:03d}"
    )


def _validate_metadata(metadata: object) -> FederatedCheckpointMetadata:
    if not isinstance(metadata, FederatedCheckpointMetadata):
        raise PilotExecutionError("metadados do checkpoint são inválidos")
    sha_values = (
        metadata.config_sha256,
        metadata.victim_dataset_sha256,
        metadata.baseline_model_sha256,
        metadata.baseline_audit_sha256,
        metadata.model_state_sha256,
        metadata.auxiliary_schedule_sha256,
        metadata.auxiliary_values_sha256,
        metadata.canonical_template_sha256,
        metadata.round_result_sha256,
    )
    expected_targets = (1, 5, 20, 200) if metadata.round_id == 20 else (20,)
    if (
        metadata.schema_version != FEDERATED_CHECKPOINT_SCHEMA_VERSION
        or metadata.scenario not in {"F0", "F1"}
        or metadata.experiment_seed != 101
        or metadata.auxiliary_weight_units != 1
        or type(metadata.round_id) is not int
        or not 1 <= metadata.round_id <= 20
        or not isinstance(metadata.model_provenance, ModelProvenance)
        or any(not _is_sha256(value) for value in sha_values)
        or tuple(marker.target_count for marker in metadata.audit_markers)
        != expected_targets
        or metadata.seed_derivation
        != "sha256_domain_separated_from_single_experiment_seed"
    ):
        raise PilotExecutionError("metadados do checkpoint divergem do piloto")
    for marker in metadata.audit_markers:
        if (
            marker.schema_version != FEDERATED_CHECKPOINT_SCHEMA_VERSION
            or marker.target_count not in expected_targets
            or not isinstance(marker.audit_id, str)
            or not marker.audit_id
            or any(
                not _is_sha256(value)
                for value in (
                    marker.result_sha256,
                    marker.generation_schedule_sha256,
                    marker.model_state_sha256,
                )
            )
            or marker.model_state_sha256 != metadata.model_state_sha256
        ):
            raise PilotExecutionError("marcador de auditoria é incompatível")
    return metadata


def build_federated_checkpoint_metadata(
    *,
    round_result: FedAvgRoundResult,
    audits: Sequence[ExtractionAuditResult],
    config_sha256: str,
    baseline_model_sha256: str,
    baseline_audit_sha256: str,
    canonical_template_sha256: str,
) -> FederatedCheckpointMetadata:
    resolved_audits = tuple(sorted(audits, key=lambda result: result.target_count))
    round_payload = round_result.as_safe_dict()
    markers = []
    for audit in resolved_audits:
        validate_extraction_audit_result(audit)
        if (
            audit.scenario != round_result.scenario
            or audit.experiment_seed != round_result.experiment_seed
            or audit.round_id != round_result.round_id
            or audit.auxiliary_weight_units != round_result.auxiliary_weight_units
            or audit.model_state_sha256 != round_result.final_model_sha256
            or audit.model_provenance != round_result.model_provenance
        ):
            raise PilotExecutionError("auditoria diverge da rodada do checkpoint")
        markers.append(
            CheckpointAuditMarker(
                target_count=audit.target_count,
                audit_id=_audit_id(audit),
                result_sha256=_sha256(_canonical_json_bytes(audit.as_safe_dict())),
                generation_schedule_sha256=audit.generation_schedule_sha256,
                model_state_sha256=audit.model_state_sha256,
            )
        )
    return _validate_metadata(
        FederatedCheckpointMetadata(
            scenario=round_result.scenario,
            experiment_seed=round_result.experiment_seed,
            auxiliary_weight_units=round_result.auxiliary_weight_units,
            round_id=round_result.round_id,
            config_sha256=config_sha256,
            victim_dataset_sha256=round_result.victim_dataset_sha256,
            baseline_model_sha256=baseline_model_sha256,
            baseline_audit_sha256=baseline_audit_sha256,
            model_state_sha256=round_result.final_model_sha256,
            auxiliary_schedule_sha256=round_result.auxiliary_schedule_sha256,
            auxiliary_values_sha256=round_result.auxiliary_values_sha256,
            canonical_template_sha256=canonical_template_sha256,
            round_result_sha256=_sha256(_canonical_json_bytes(round_payload)),
            audit_markers=tuple(markers),
            model_provenance=round_result.model_provenance,
        )
    )


def _metadata_from_dict(value: Mapping[str, Any]) -> FederatedCheckpointMetadata:
    expected = frozenset(field.name for field in fields(FederatedCheckpointMetadata))
    if frozenset(value) != expected:
        raise PilotExecutionError("metadados do checkpoint possuem chaves inválidas")
    provenance_raw = value.get("model_provenance")
    markers_raw = value.get("audit_markers")
    if not isinstance(provenance_raw, Mapping) or not isinstance(markers_raw, list):
        raise PilotExecutionError("metadados do checkpoint possuem tipos inválidos")
    provenance_keys = frozenset(field.name for field in fields(ModelProvenance))
    marker_keys = frozenset(field.name for field in fields(CheckpointAuditMarker))
    if frozenset(provenance_raw) != provenance_keys or any(
        not isinstance(marker, Mapping) or frozenset(marker) != marker_keys
        for marker in markers_raw
    ):
        raise PilotExecutionError("metadados do checkpoint possuem estrutura inválida")
    try:
        payload = dict(value)
        payload["model_provenance"] = ModelProvenance(**provenance_raw)
        payload["audit_markers"] = tuple(
            CheckpointAuditMarker(**marker) for marker in markers_raw
        )
        metadata = FederatedCheckpointMetadata(**payload)
    except (TypeError, ValueError) as error:
        raise PilotExecutionError("metadados do checkpoint são inválidos") from error
    return _validate_metadata(metadata)


def _checkpoint_artifact_sha256(directory: Path) -> str:
    digest = hashlib.sha256(b"federated-checkpoint-artifact/v1\0")
    for name in sorted(_CHECKPOINT_FILES):
        raw = (directory / name).read_bytes()
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def save_federated_checkpoint(
    target_directory: Path,
    model_bundle: LoadedModelBundle,
    metadata: FederatedCheckpointMetadata,
    round_result: FedAvgRoundResult,
) -> LoadedFederatedCheckpoint:
    """Publica um checkpoint completo por staging e renomeação exclusiva."""

    resolved_metadata = _validate_metadata(metadata)
    _validate_checkpoint_model(model_bundle)
    target = Path(target_directory)
    if ".." in target.parts:
        raise PilotExecutionError("caminho do checkpoint é inválido")
    try:
        validate_storage_component(target.name, "checkpoint_id")
    except Exception as error:
        raise PilotExecutionError("identidade do checkpoint é inválida") from error
    if target.exists():
        raise FileExistsError("checkpoint já existe")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise PilotExecutionError("diretório de checkpoints é inválido")
    os.chmod(parent, 0o700)
    if (
        round_result.final_model_sha256 != resolved_metadata.model_state_sha256
        or round_result.as_safe_dict().get("scenario") != resolved_metadata.scenario
        or _sha256(_canonical_json_bytes(round_result.as_safe_dict()))
        != resolved_metadata.round_result_sha256
    ):
        raise PilotExecutionError("resultado da rodada diverge do checkpoint")
    try:
        current_hash = fingerprint_model_parameters(model_bundle)
    except Exception as error:
        raise PilotExecutionError("falha ao identificar modelo do checkpoint") from error
    if current_hash != resolved_metadata.model_state_sha256:
        raise PilotExecutionError("modelo diverge do checkpoint solicitado")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        os.chmod(staging, 0o700)
        rng_payload = _capture_rng_state(model_bundle)
        _write_exclusive(
            staging / "round_result.json",
            _canonical_json_bytes(round_result.as_safe_dict()),
        )
        _write_exclusive(
            staging / "rng_state.json",
            _canonical_json_bytes(rng_payload),
        )
        try:
            from safetensors.torch import save_model

            save_model(model_bundle.model, str(staging / "model.safetensors"))
            os.chmod(staging / "model.safetensors", 0o600)
        except Exception as error:
            raise PilotExecutionError("falha ao serializar pesos do checkpoint") from error
        file_manifest = {
            name: {
                "sha256": _sha256((staging / name).read_bytes()),
                "size": (staging / name).stat().st_size,
            }
            for name in ("model.safetensors", "rng_state.json", "round_result.json")
        }
        metadata_payload = {
            "schema_version": FEDERATED_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint": resolved_metadata.as_safe_dict(),
            "files": file_manifest,
        }
        _write_exclusive(staging / "metadata.json", _canonical_json_bytes(metadata_payload))
        if target.exists():
            raise FileExistsError("checkpoint já existe")
        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    artifact_sha256 = _checkpoint_artifact_sha256(target)
    return LoadedFederatedCheckpoint(
        metadata=resolved_metadata,
        round_result_payload=round_result.as_safe_dict(),
        artifact_sha256=artifact_sha256,
    )


def load_federated_checkpoint(
    source_directory: Path,
    model_bundle: LoadedModelBundle,
    *,
    expected_scenario: str,
    expected_round_id: int,
    expected_config_sha256: str,
    expected_victim_dataset_sha256: str,
    expected_baseline_model_sha256: str,
    expected_baseline_audit_sha256: str,
) -> LoadedFederatedCheckpoint:
    """Valida integralmente e só então restaura pesos e RNG no bundle."""

    source = Path(source_directory)
    if ".." in source.parts:
        raise PilotExecutionError("caminho do checkpoint é inválido")
    _validate_checkpoint_model(model_bundle)
    if (
        source.is_symlink()
        or not source.is_dir()
        or frozenset(path.name for path in source.iterdir()) != _CHECKPOINT_FILES
        or any(path.is_symlink() or not path.is_file() for path in source.iterdir())
    ):
        raise PilotExecutionError("estrutura do checkpoint é inválida")
    try:
        metadata_raw = (source / "metadata.json").read_bytes()
        round_raw = (source / "round_result.json").read_bytes()
        rng_raw = (source / "rng_state.json").read_bytes()
    except OSError as error:
        raise PilotExecutionError("checkpoint é inacessível") from error
    metadata_wrapper = _load_json(metadata_raw)
    if (
        frozenset(metadata_wrapper) != frozenset({"schema_version", "checkpoint", "files"})
        or metadata_wrapper.get("schema_version") != FEDERATED_CHECKPOINT_SCHEMA_VERSION
        or _canonical_json_bytes(metadata_wrapper) != metadata_raw
    ):
        raise PilotExecutionError("manifesto do checkpoint é inválido")
    checkpoint_raw = metadata_wrapper.get("checkpoint")
    files_raw = metadata_wrapper.get("files")
    if not isinstance(checkpoint_raw, Mapping) or not isinstance(files_raw, Mapping):
        raise PilotExecutionError("manifesto do checkpoint possui tipos inválidos")
    metadata = _metadata_from_dict(checkpoint_raw)
    if (
        metadata.scenario != expected_scenario
        or metadata.round_id != expected_round_id
        or metadata.config_sha256 != expected_config_sha256
        or metadata.victim_dataset_sha256 != expected_victim_dataset_sha256
        or metadata.baseline_model_sha256 != expected_baseline_model_sha256
        or metadata.baseline_audit_sha256 != expected_baseline_audit_sha256
        or metadata.model_provenance != model_bundle.provenance
    ):
        raise PilotExecutionError("checkpoint pertence a outra trajetória")
    expected_file_names = frozenset(
        {"model.safetensors", "rng_state.json", "round_result.json"}
    )
    if frozenset(files_raw) != expected_file_names:
        raise PilotExecutionError("manifesto de arquivos do checkpoint é inválido")
    for name in expected_file_names:
        entry = files_raw.get(name)
        path = source / name
        if (
            not isinstance(entry, Mapping)
            or frozenset(entry) != frozenset({"sha256", "size"})
            or not _is_sha256(entry.get("sha256"))
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or path.stat().st_size != entry["size"]
            or _sha256(path.read_bytes()) != entry["sha256"]
        ):
            raise PilotExecutionError("arquivo do checkpoint diverge do manifesto")
    round_payload = _load_json(round_raw)
    rng_payload = _load_json(rng_raw)
    if (
        _canonical_json_bytes(round_payload) != round_raw
        or _canonical_json_bytes(rng_payload) != rng_raw
        or _sha256(round_raw) != metadata.round_result_sha256
        or round_payload.get("final_model_sha256") != metadata.model_state_sha256
        or round_payload.get("scenario") != metadata.scenario
        or round_payload.get("round_id") != metadata.round_id
    ):
        raise PilotExecutionError("conteúdo do checkpoint é incompatível")
    _decode_rng_state(rng_payload)
    try:
        snapshot = capture_model_parameter_snapshot(model_bundle)
        previous_rng = _capture_rng_state(model_bundle)
    except Exception as error:
        raise PilotExecutionError("falha ao preparar restauração do checkpoint") from error
    try:
        from safetensors.torch import load_model

        missing, unexpected = load_model(
            model_bundle.model,
            source / "model.safetensors",
            strict=True,
            device=model_bundle.provenance.device,
        )
        if missing or unexpected:
            raise PilotExecutionError("pesos do checkpoint estão incompletos")
        if fingerprint_model_parameters(model_bundle) != metadata.model_state_sha256:
            raise PilotExecutionError("fingerprint restaurado diverge do checkpoint")
        _validate_checkpoint_model(model_bundle)
        _restore_rng_state(rng_payload, model_bundle)
    except Exception as error:
        try:
            restore_model_parameter_snapshot(model_bundle, snapshot)
            _restore_rng_state(previous_rng, model_bundle)
        except Exception as restore_error:
            raise PilotExecutionError(
                "checkpoint falhou e o modelo não pôde ser restaurado"
            ) from restore_error
        if isinstance(error, PilotExecutionError):
            raise
        raise PilotExecutionError("falha ao carregar checkpoint") from error
    return LoadedFederatedCheckpoint(
        metadata=metadata,
        round_result_payload=round_payload,
        artifact_sha256=_checkpoint_artifact_sha256(source),
    )


__all__ = [
    "build_federated_checkpoint_metadata",
    "load_federated_checkpoint",
    "save_federated_checkpoint",
]
