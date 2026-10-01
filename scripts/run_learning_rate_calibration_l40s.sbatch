#!/usr/bin/env bash
#SBATCH --job-name=memorization-calibration-greedy-lr-v4
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --dependency=singleton
#SBATCH --no-requeue

set -Eeuo pipefail

usage() {
    echo "uso: sbatch scripts/run_learning_rate_calibration_l40s.sbatch preflight|start|resume" >&2
}

fail() {
    echo "erro: $1" >&2
    exit 1
}

[[ $# -eq 1 ]] || { usage; exit 2; }
readonly CALIBRATION_MODE="$1"
case "$CALIBRATION_MODE" in
    preflight) readonly -a MODE_ARGUMENTS=(--preflight-only) ;;
    start) readonly -a MODE_ARGUMENTS=(--fresh) ;;
    resume) readonly -a MODE_ARGUMENTS=() ;;
    *) usage; exit 2 ;;
esac

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "o launcher deve ser executado pelo Slurm"
command -v srun >/dev/null 2>&1 || fail "srun não está disponível"
command -v git >/dev/null 2>&1 || fail "git não está disponível"

readonly CURRENT_DIRECTORY="$(pwd -P)"
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || \
    fail "o diretório atual não pertence a um repositório Git"
readonly PROJECT_ROOT
readonly RESOLVED_PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
[[ "$CURRENT_DIRECTORY" == "$RESOLVED_PROJECT_ROOT" ]] || \
    fail "execute o launcher a partir da raiz do repositório"
[[ -f configs/memorization-calibration-v4.yaml && -f configs/main-v2.yaml ]] || \
    fail "as configurações da calibração estão ausentes"
if ! git diff --quiet || ! git diff --cached --quiet; then
    fail "o worktree contém alterações rastreadas"
fi

readonly PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON_EXECUTABLE" ]] || fail "o ambiente .venv não está disponível"
readonly PYTHON_VERSION="$($PYTHON_EXECUTABLE -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_VERSION" == "3.12" ]] || fail "o launcher exige Python 3.12"
"$PYTHON_EXECUTABLE" -m pip check || fail "as dependências estão inconsistentes"
"$PYTHON_EXECUTABLE" -c '
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() == 1 else 1)
' || fail "a alocação deve expor exatamente uma GPU CUDA"

readonly CALIBRATION_CONFIG=configs/memorization-calibration-v4.yaml
readonly CALIBRATION_CACHE=artifacts/huggingface
readonly CALIBRATION_OUTPUT_ROOT=outputs
readonly CALIBRATION_SEED=101
readonly CALIBRATION_RUN_ID=memorization-calibration-greedy-lr-seed-101-v4
readonly CALIBRATION_RUN_DIRECTORY="$PROJECT_ROOT/$CALIBRATION_OUTPUT_ROOT/runs/$CALIBRATION_RUN_ID"
readonly GIT_COMMIT="$(git rev-parse --verify HEAD)"
if [[ "$CALIBRATION_MODE" == "resume" && ! -d "$CALIBRATION_RUN_DIRECTORY" ]]; then
    fail "resume exige a execução oficial existente"
fi
if [[ "$CALIBRATION_MODE" == "start" && -e "$CALIBRATION_RUN_DIRECTORY" ]]; then
    fail "start exige que a execução oficial ainda não exista"
fi

"$PYTHON_EXECUTABLE" -c '
import sys
from pathlib import Path

try:
    from federated_leakage.calibration_contracts import (
        load_memorization_calibration_spec_from_config,
    )
    spec = load_memorization_calibration_spec_from_config(Path(sys.argv[1]))
    valid = (
        spec.experiment_seed == int(sys.argv[2])
        and spec.default_run_id == sys.argv[3]
        and spec.fixed_repetitions == 160
        and tuple(item.learning_rate_millionths for item in spec.learning_rate_arms)
        == (10, 30, 100, 300)
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
' "$CALIBRATION_CONFIG" "$CALIBRATION_SEED" "$CALIBRATION_RUN_ID" || \
    fail "a configuração não corresponde à calibração oficial"

printf 'job_id: %s\n' "$SLURM_JOB_ID"
printf 'hostname: %s\n' "$(hostname)"
printf 'commit: %s\n' "$GIT_COMMIT"
printf 'modo: %s\n' "$CALIBRATION_MODE"
printf 'dispositivo: cuda\n'

readonly -a CALIBRATION_ARGUMENTS=(
    -m federated_leakage.run_memorization_calibration
    --config "$CALIBRATION_CONFIG"
    --device cuda
    --cache-dir "$CALIBRATION_CACHE"
    --output-root "$CALIBRATION_OUTPUT_ROOT"
    --run-id "$CALIBRATION_RUN_ID"
    "${MODE_ARGUMENTS[@]}"
)

exec srun --ntasks=1 "$PYTHON_EXECUTABLE" "${CALIBRATION_ARGUMENTS[@]}"
