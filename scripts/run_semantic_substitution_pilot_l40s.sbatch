#!/usr/bin/env bash
#SBATCH --partition=l40s
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
  101) EXPECTED_JOB_NAME="semantic-substitution-s101-v1" ;;
  361506353) EXPECTED_JOB_NAME="semantic-substitution-s361506353-v1" ;;
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
  echo "erro: job name deve identificar exatamente a seed" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "erro: GPU Slurm não foi alocada" >&2
  exit 2
fi

REPOSITORY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPOSITORY_ROOT" || "$PWD" != "$REPOSITORY_ROOT" ]]; then
  echo "erro: submeta o launcher a partir da raiz do repositório" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "erro: worktree possui alterações rastreadas" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "erro: ambiente .venv do projeto está ausente" >&2
  exit 2
fi
if [[ "$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
  echo "erro: Python 3.12 é obrigatório" >&2
  exit 2
fi
.venv/bin/python -m pip check
.venv/bin/python -c '
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)
' || { echo "erro: a alocação deve expor exatamente uma GPU CUDA" >&2; exit 2; }

RUN_ID="semantic-substitution-upstream-seed-${SEED}-v1"
RUN_DIRECTORY="$REPOSITORY_ROOT/outputs/runs/$RUN_ID"
if [[ "$MODE" == "resume" && ! -d "$RUN_DIRECTORY" ]]; then
  echo "erro: resume exige uma execução existente para a seed" >&2
  exit 2
fi
ARGS=(
  -m federated_leakage.run_semantic_substitution_pilot
  --config configs/semantic-substitution-pilot-v1.yaml
  --seed "$SEED"
  --device cuda
  --cache-dir artifacts/huggingface
  --output-root outputs
  --run-id "$RUN_ID"
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
echo "dispositivo: cuda"

exec srun --ntasks=1 .venv/bin/python "${ARGS[@]}"
