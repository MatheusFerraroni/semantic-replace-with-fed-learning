#!/usr/bin/env bash

set -Eeuo pipefail

readonly TARGET_ENV=".venv-rtxpro6000-cu128"
readonly STAGING_PREFIX=".venv-rtxpro6000-cu128.staging-"

command -v git >/dev/null 2>&1 || { echo "erro: git não está disponível" >&2; exit 2; }
readonly REPOSITORY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPOSITORY_ROOT" || "$PWD" != "$REPOSITORY_ROOT" ]]; then
  echo "erro: execute o preparador na raiz do repositório" >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "erro: prepare o ambiente no headnode, fora de uma alocação Slurm" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "erro: .venv/bin/python é necessário como Python 3.12 de bootstrap" >&2
  exit 2
fi
if [[ "$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "erro: o Python de bootstrap deve ser 3.12" >&2
  exit 2
fi

validate_environment() {
  local python_path="$1"
  "$python_path" -m pip check
  "$python_path" -c '
import importlib.metadata
import sys
import torch

expected = {
    "Faker": "40.36.0",
    "huggingface-hub": "0.33.4",
    "jsonschema": "4.25.0",
    "opacus": "1.6.0",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.2",
    "transformers": "4.53.2",
}
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == "2.7.1+cu128"
assert torch.version.cuda == "12.8"
arch_flags = torch._C._cuda_getArchFlags().split()
assert "sm_120" in arch_flags
for name, version in expected.items():
    assert importlib.metadata.version(name) == version
print("status: ambiente RTX PRO 6000 CUDA 12.8 validado")
print("python:", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("torch:", torch.__version__)
print("cuda_build:", torch.version.cuda)
print("sm_120: presente")
'
}

if [[ -e "$TARGET_ENV" || -L "$TARGET_ENV" ]]; then
  if [[ -L "$TARGET_ENV" || ! -d "$TARGET_ENV" || ! -x "$TARGET_ENV/bin/python" ]]; then
    echo "erro: ambiente RTX existente é inválido" >&2
    exit 2
  fi
  validate_environment "$TARGET_ENV/bin/python"
  exit 0
fi

readonly STAGING="${STAGING_PREFIX}$$"
if [[ -e "$STAGING" || -L "$STAGING" ]]; then
  echo "erro: staging do ambiente já existe" >&2
  exit 2
fi
cleanup() {
  if [[ -d "$STAGING" && ! -L "$STAGING" ]]; then
    rm -rf -- "$STAGING"
  fi
}
trap cleanup EXIT

.venv/bin/python -m venv "$STAGING"
"$STAGING/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.1+cu128"
"$STAGING/bin/python" -m pip install -e '.[model,dp]'
validate_environment "$STAGING/bin/python"

if [[ -e "$TARGET_ENV" || -L "$TARGET_ENV" ]]; then
  echo "erro: ambiente RTX foi criado concorrentemente; valide-o em nova execução" >&2
  exit 2
fi
mv -- "$STAGING" "$TARGET_ENV"
trap - EXIT
echo "status: ambiente RTX PRO 6000 publicado em $TARGET_ENV"
