"""Perfil operacional fail-closed para a réplica RTX PRO 6000 Blackwell."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .configuration import ConfigurationError, load_yaml_mapping
from .semantic_pilot_storage import canonical_json_bytes, read_safe_json


EXECUTION_RUNTIME_PROFILE_SCHEMA_VERSION = "execution-runtime-profile/v1"
RUNTIME_MANIFEST_SCHEMA_VERSION = "execution-runtime-manifest/v1"
RTX_PROFILE_ID = "rtxpro6000-blackwell-cu128-v1"
RTX_OUTPUT_NAMESPACE = "execution-profiles/rtxpro6000-blackwell-cu128-v1"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
EXPECTED_TORCH_VERSION = "2.7.1+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_CUDA_ARCH = "sm_120"
EXPECTED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_SCIENTIFIC_CONFIG_SHA256 = "ad3407a8be18fe5a3341ce6dbbdfa2e52ad69babc208c3fd41b6b378d10ce7cc"
EXPECTED_MAIN_CONFIG_SHA256 = "f4e55ba5cda848cd5bfcbd47a0520219fe042747d132563e576eba9e87d21e4a"
EXPECTED_DEPENDENCIES = {
    "Faker": "40.36.0",
    "huggingface-hub": "0.33.4",
    "jsonschema": "4.25.0",
    "opacus": "1.6.0",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.2",
    "transformers": "4.53.2",
}
EXPECTED_REPRODUCIBILITY_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class ExecutionRuntimeError(ValueError):
    """O runtime não corresponde ao perfil operacional pinado."""


@dataclass(frozen=True, slots=True)
class ExecutionRuntimeSpec:
    schema_version: str
    profile_id: str
    profile_config_sha256: str
    scientific_config_path: Path
    scientific_config_sha256: str
    main_config_path: Path
    main_config_sha256: str
    output_namespace: str
    python_version: str
    torch_version: str
    torch_cuda_version: str
    required_cuda_arch: str
    gpu_name: str
    compute_capability: tuple[int, int]
    minimum_vram_gib: int
    visible_gpu_count: int
    bf16_required: bool
    dependencies: tuple[tuple[str, str], ...]
    reproducibility: tuple[tuple[str, str], ...]

    @property
    def output_parts(self) -> tuple[str, ...]:
        return tuple(self.output_namespace.split("/"))


@dataclass(frozen=True, slots=True)
class ExecutionRuntimeFingerprint:
    profile_id: str
    profile_config_sha256: str
    scientific_config_sha256: str
    main_config_sha256: str
    python_version: str
    torch_version: str
    torch_cuda_version: str
    cuda_architectures: tuple[str, ...]
    gpu_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    bf16_supported: bool
    driver_version: str
    dependencies: tuple[tuple[str, str], ...]
    reproducibility: tuple[tuple[str, str], ...]
    runtime_sha256: str

    def as_safe_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_config_sha256": self.profile_config_sha256,
            "scientific_config_sha256": self.scientific_config_sha256,
            "main_config_sha256": self.main_config_sha256,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_architectures": list(self.cuda_architectures),
            "gpu_name": self.gpu_name,
            "compute_capability": list(self.compute_capability),
            "total_memory_bytes": self.total_memory_bytes,
            "bf16_supported": self.bf16_supported,
            "driver_version": self.driver_version,
            "dependencies": dict(self.dependencies),
            "reproducibility": dict(self.reproducibility),
            "runtime_sha256": self.runtime_sha256,
        }


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ExecutionRuntimeError("arquivo pinado do perfil está ausente") from error


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ExecutionRuntimeError(f"{label} possui campos inválidos")


def load_execution_runtime_spec(path: Path) -> ExecutionRuntimeSpec:
    config_path = Path(path)
    try:
        raw = load_yaml_mapping(config_path)
    except ConfigurationError as error:
        raise ExecutionRuntimeError("perfil de runtime é inválido") from error
    _strict_keys(
        raw,
        {
            "schema_version", "profile_id", "scientific_config",
            "scientific_config_sha256", "main_config", "main_config_sha256",
            "output_namespace", "python_version", "torch_version",
            "torch_cuda_version", "required_cuda_arch", "hardware",
            "dependencies", "reproducibility",
        },
        "perfil de runtime",
    )
    hardware = raw.get("hardware")
    dependencies = raw.get("dependencies")
    reproducibility = raw.get("reproducibility")
    if not all(isinstance(value, dict) for value in (hardware, dependencies, reproducibility)):
        raise ExecutionRuntimeError("perfil de runtime possui seção inválida")
    _strict_keys(
        hardware,
        {"gpu_name", "compute_capability", "minimum_vram_gib", "visible_gpu_count", "bf16_required"},
        "hardware do runtime",
    )
    _strict_keys(dependencies, set(EXPECTED_DEPENDENCIES), "dependências do runtime")
    _strict_keys(
        reproducibility,
        {"cuda_cublas_workspace_config", "hf_hub_offline", "transformers_offline", "tokenizers_parallelism"},
        "reprodutibilidade do runtime",
    )
    capability = hardware.get("compute_capability")
    if not (
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in capability)
    ):
        raise ExecutionRuntimeError("compute capability do runtime é inválida")
    scientific_name = raw.get("scientific_config")
    main_name = raw.get("main_config")
    if scientific_name != "refined-defense-pilot-v1.yaml" or main_name != "main-v5.yaml":
        raise ExecutionRuntimeError("configuração científica do runtime diverge")
    scientific_path = config_path.parent / scientific_name
    main_path = config_path.parent / main_name
    expected_repro = {
        "cuda_cublas_workspace_config": EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "tokenizers_parallelism": "false",
    }
    fixed = (
        raw.get("schema_version") == EXECUTION_RUNTIME_PROFILE_SCHEMA_VERSION
        and raw.get("profile_id") == RTX_PROFILE_ID
        and raw.get("scientific_config_sha256") == EXPECTED_SCIENTIFIC_CONFIG_SHA256
        and raw.get("main_config_sha256") == EXPECTED_MAIN_CONFIG_SHA256
        and raw.get("output_namespace") == RTX_OUTPUT_NAMESPACE
        and raw.get("python_version") == "3.12"
        and raw.get("torch_version") == EXPECTED_TORCH_VERSION
        and raw.get("torch_cuda_version") == EXPECTED_TORCH_CUDA_VERSION
        and raw.get("required_cuda_arch") == EXPECTED_CUDA_ARCH
        and hardware.get("gpu_name") == EXPECTED_GPU_NAME
        and tuple(capability) == (12, 0)
        and hardware.get("minimum_vram_gib") == 90
        and hardware.get("visible_gpu_count") == 1
        and hardware.get("bf16_required") is True
        and dependencies == EXPECTED_DEPENDENCIES
        and reproducibility == expected_repro
        and _sha256_file(scientific_path) == EXPECTED_SCIENTIFIC_CONFIG_SHA256
        and _sha256_file(main_path) == EXPECTED_MAIN_CONFIG_SHA256
    )
    if not fixed:
        raise ExecutionRuntimeError("perfil de runtime diverge da receita fixada")
    return ExecutionRuntimeSpec(
        schema_version=raw["schema_version"],
        profile_id=raw["profile_id"],
        profile_config_sha256=_sha256_file(config_path),
        scientific_config_path=scientific_path,
        scientific_config_sha256=raw["scientific_config_sha256"],
        main_config_path=main_path,
        main_config_sha256=raw["main_config_sha256"],
        output_namespace=raw["output_namespace"],
        python_version=raw["python_version"],
        torch_version=raw["torch_version"],
        torch_cuda_version=raw["torch_cuda_version"],
        required_cuda_arch=raw["required_cuda_arch"],
        gpu_name=hardware["gpu_name"],
        compute_capability=tuple(capability),
        minimum_vram_gib=hardware["minimum_vram_gib"],
        visible_gpu_count=hardware["visible_gpu_count"],
        bf16_required=hardware["bf16_required"],
        dependencies=tuple(sorted(EXPECTED_DEPENDENCIES.items())),
        reproducibility=tuple(sorted(EXPECTED_REPRODUCIBILITY_ENV.items())),
    )


def _query_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as error:
        raise ExecutionRuntimeError("driver NVIDIA não pôde ser validado") from error
    values = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if len(values) != 1:
        raise ExecutionRuntimeError("inventário do driver NVIDIA é inválido")
    value = next(iter(values))
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is None:
        raise ExecutionRuntimeError("versão do driver NVIDIA é inválida")
    return value


def capture_execution_runtime(
    spec: ExecutionRuntimeSpec,
    *,
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
    package_version: Callable[[str], str] = importlib.metadata.version,
    driver_query: Callable[[], str] = _query_driver_version,
    python_version: str | None = None,
) -> ExecutionRuntimeFingerprint:
    env = os.environ if environ is None else environ
    for name, expected in spec.reproducibility:
        if env.get(name) != expected:
            raise ExecutionRuntimeError("ambiente determinístico ou offline é inválido")
    actual_python = python_version or f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
    if actual_python != spec.python_version:
        raise ExecutionRuntimeError("versão do Python diverge do perfil")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise ExecutionRuntimeError("PyTorch do perfil está ausente") from error
    if str(torch_module.__version__) != spec.torch_version:
        raise ExecutionRuntimeError("versão do PyTorch diverge do perfil")
    if str(torch_module.version.cuda) != spec.torch_cuda_version:
        raise ExecutionRuntimeError("build CUDA do PyTorch diverge do perfil")
    try:
        available = bool(torch_module.cuda.is_available())
        count = int(torch_module.cuda.device_count())
        architectures = tuple(sorted(set(torch_module.cuda.get_arch_list())))
        gpu_name = str(torch_module.cuda.get_device_name(0))
        capability = tuple(int(value) for value in torch_module.cuda.get_device_capability(0))
        total_memory = int(torch_module.cuda.get_device_properties(0).total_memory)
        bf16 = bool(torch_module.cuda.is_bf16_supported())
    except Exception as error:
        raise ExecutionRuntimeError("hardware CUDA não pôde ser validado") from error
    if not available or count != spec.visible_gpu_count:
        raise ExecutionRuntimeError("quantidade de GPUs CUDA visíveis diverge do perfil")
    if spec.required_cuda_arch not in architectures:
        raise ExecutionRuntimeError("arquitetura CUDA requerida não está no PyTorch")
    if gpu_name != spec.gpu_name or capability != spec.compute_capability:
        raise ExecutionRuntimeError("GPU visível diverge do perfil")
    if total_memory < spec.minimum_vram_gib * 1024**3:
        raise ExecutionRuntimeError("memória da GPU é inferior ao perfil")
    if spec.bf16_required and not bf16:
        raise ExecutionRuntimeError("BF16 não está disponível no runtime")
    dependency_values = []
    for name, expected in spec.dependencies:
        try:
            actual = package_version(name)
        except Exception as error:
            raise ExecutionRuntimeError("dependência do runtime está ausente") from error
        if actual != expected:
            raise ExecutionRuntimeError("versão de dependência diverge do perfil")
        dependency_values.append((name, actual))
    driver = driver_query()
    unsigned = {
        "profile_id": spec.profile_id,
        "profile_config_sha256": spec.profile_config_sha256,
        "scientific_config_sha256": spec.scientific_config_sha256,
        "main_config_sha256": spec.main_config_sha256,
        "python_version": actual_python,
        "torch_version": str(torch_module.__version__),
        "torch_cuda_version": str(torch_module.version.cuda),
        "cuda_architectures": list(architectures),
        "gpu_name": gpu_name,
        "compute_capability": list(capability),
        "total_memory_bytes": total_memory,
        "bf16_supported": bf16,
        "driver_version": driver,
        "dependencies": dict(dependency_values),
        "reproducibility": dict(spec.reproducibility),
    }
    digest = hashlib.sha256(
        b"execution-runtime-fingerprint/v1\0" + canonical_json_bytes(unsigned)
    ).hexdigest()
    return ExecutionRuntimeFingerprint(
        profile_id=spec.profile_id,
        profile_config_sha256=spec.profile_config_sha256,
        scientific_config_sha256=spec.scientific_config_sha256,
        main_config_sha256=spec.main_config_sha256,
        python_version=actual_python,
        torch_version=str(torch_module.__version__),
        torch_cuda_version=str(torch_module.version.cuda),
        cuda_architectures=architectures,
        gpu_name=gpu_name,
        compute_capability=capability,
        total_memory_bytes=total_memory,
        bf16_supported=bf16,
        driver_version=driver,
        dependencies=tuple(dependency_values),
        reproducibility=spec.reproducibility,
        runtime_sha256=digest,
    )


def runtime_output_root(base_output_root: Path, spec: ExecutionRuntimeSpec) -> Path:
    base = Path(base_output_root)
    if ".." in base.parts or base.is_symlink() or (base.exists() and not base.is_dir()):
        raise ExecutionRuntimeError("raiz operacional de saída é inválida")
    parts = spec.output_parts
    if parts != ("execution-profiles", RTX_PROFILE_ID):
        raise ExecutionRuntimeError("namespace operacional é inválido")
    return base.joinpath(*parts)


def runtime_manifest_payload(
    spec: ExecutionRuntimeSpec, fingerprint: ExecutionRuntimeFingerprint
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "profile_id": spec.profile_id,
        "output_namespace": spec.output_namespace,
        "runtime": fingerprint.as_safe_dict(),
    }


def validate_runtime_manifest(
    path: Path,
    spec: ExecutionRuntimeSpec,
    fingerprint: ExecutionRuntimeFingerprint,
) -> dict[str, Any]:
    try:
        payload = read_safe_json(path)
    except Exception as error:
        raise ExecutionRuntimeError("manifesto do runtime está ausente ou inválido") from error
    if payload != runtime_manifest_payload(spec, fingerprint):
        raise ExecutionRuntimeError("runtime da retomada diverge do manifesto")
    return payload


def publish_runtime_manifest(
    output_root: Path,
    spec: ExecutionRuntimeSpec,
    fingerprint: ExecutionRuntimeFingerprint,
) -> Path:
    root = runtime_output_root(output_root, spec)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or not root.is_dir():
        raise ExecutionRuntimeError("diretório do perfil de execução é inválido")
    os.chmod(root, 0o700)
    target = root / "runtime_manifest.json"
    payload = runtime_manifest_payload(spec, fingerprint)
    if target.exists():
        validate_runtime_manifest(target, spec, fingerprint)
        return target
    handle, name = tempfile.mkstemp(prefix=".runtime-manifest-", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(canonical_json_bytes(payload))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, target)
        except FileExistsError:
            validate_runtime_manifest(target, spec, fingerprint)
        if target.is_symlink() or not target.is_file():
            raise ExecutionRuntimeError("manifesto do runtime não foi publicado")
        os.chmod(target, 0o600)
        return target
    except ExecutionRuntimeError:
        raise
    except OSError as error:
        raise ExecutionRuntimeError("falha ao publicar manifesto do runtime") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "EXECUTION_RUNTIME_PROFILE_SCHEMA_VERSION",
    "RUNTIME_MANIFEST_SCHEMA_VERSION",
    "RTX_PROFILE_ID",
    "RTX_OUTPUT_NAMESPACE",
    "ExecutionRuntimeError",
    "ExecutionRuntimeFingerprint",
    "ExecutionRuntimeSpec",
    "capture_execution_runtime",
    "load_execution_runtime_spec",
    "publish_runtime_manifest",
    "runtime_manifest_payload",
    "runtime_output_root",
    "validate_runtime_manifest",
]
