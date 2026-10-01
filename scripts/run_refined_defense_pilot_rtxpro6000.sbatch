#!/usr/bin/env bash
#SBATCH --partition=rtxpro6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --dependency=singleton
#SBATCH --no-requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "erro: use preflight|start|resume <seed>" >&2
  exit 2
fi
readonly MODE="$1"
readonly SEED="$2"
case "$MODE" in
  preflight|start|resume) ;;
  *) echo "erro: use preflight|start|resume <seed>" >&2; exit 2 ;;
esac
case "$SEED" in
  101) EXPECTED_JOB_NAME="refined-defense-rtxpro6000-s101-v1" ;;
  361506353) EXPECTED_JOB_NAME="refined-defense-rtxpro6000-s361506353-v1" ;;
  *) echo "erro: seed deve ser 101 ou 361506353" >&2; exit 2 ;;
esac

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NAME:-}" ]]; then
  echo "erro: launcher exige execução pelo Slurm" >&2
  exit 2
fi
command -v srun >/dev/null 2>&1 || { echo "erro: srun não está disponível" >&2; exit 2; }
command -v git >/dev/null 2>&1 || { echo "erro: git não está disponível" >&2; exit 2; }
if [[ "$SLURM_JOB_NAME" != "$EXPECTED_JOB_NAME" ]]; then
  echo "erro: job name deve identificar exatamente runtime e seed" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "erro: GPU Slurm não foi alocada" >&2
  exit 2
fi

readonly REPOSITORY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPOSITORY_ROOT" || "$PWD" != "$REPOSITORY_ROOT" ]]; then
  echo "erro: submeta o launcher a partir da raiz do repositório" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "erro: worktree possui alterações rastreadas" >&2
  exit 2
fi
readonly RTX_PYTHON=".venv-rtxpro6000-cu128/bin/python"
if [[ ! -x "$RTX_PYTHON" ]]; then
  echo "erro: ambiente RTX PRO 6000 preparado está ausente" >&2
  exit 2
fi
if [[ "$($RTX_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "erro: Python 3.12 é obrigatório" >&2
  exit 2
fi
"$RTX_PYTHON" -m pip check

readonly ARTIFACT_DIRECTORY="$REPOSITORY_ROOT/artifacts/models/ae3238fde6675942cac5"
if [[ ! -d "$ARTIFACT_DIRECTORY" ]]; then
  echo "erro: artefato refinado preparado está ausente" >&2
  exit 2
fi
readonly PROFILE_ROOT="$REPOSITORY_ROOT/outputs/execution-profiles/rtxpro6000-blackwell-cu128-v1"
readonly RUN_ID="refined-defense-forum-tech-seed-${SEED}-v1"
readonly RUN_DIRECTORY="$PROFILE_ROOT/runs/$RUN_ID"
if [[ "$MODE" == "resume" && ! -d "$RUN_DIRECTORY" ]]; then
  echo "erro: resume exige uma execução RTX existente para a seed" >&2
  exit 2
fi

ARGS=(
  -m federated_leakage.run_refined_defense_pilot_rtxpro6000
  --runtime-config configs/refined-runtime-rtxpro6000-cu128-v1.yaml
  --seed "$SEED"
  --device cuda
  --cache-dir artifacts/huggingface
  --model-artifact-dir "$ARTIFACT_DIRECTORY"
  --output-root outputs
)
case "$MODE" in
  preflight) ARGS+=(--preflight-only) ;;
  start) ARGS+=(--fresh) ;;
  resume) ;;
esac

echo "job_id: $SLURM_JOB_ID"
echo "hostname: $(hostname)"
echo "commit: $(git rev-parse HEAD)"
echo "modo: $MODE"
echo "seed: $SEED"
echo "perfil_runtime: rtxpro6000-blackwell-cu128-v1"
echo "dispositivo: cuda"

exec srun --ntasks=1 "$RTX_PYTHON" "${ARGS[@]}"
