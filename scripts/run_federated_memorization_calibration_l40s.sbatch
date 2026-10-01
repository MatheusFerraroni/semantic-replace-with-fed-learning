#!/usr/bin/env bash
#SBATCH --job-name=federated-memorization-calibration-v1
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
    echo "uso: sbatch scripts/run_federated_memorization_calibration_l40s.sbatch preflight|start|resume" >&2
}

fail() {
    echo "erro: $1" >&2
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

readonly CALIBRATION_MODE="$1"
case "$CALIBRATION_MODE" in
    preflight)
        readonly -a MODE_ARGUMENTS=(--preflight-only)
        ;;
    start)
        readonly -a MODE_ARGUMENTS=(--fresh)
        ;;
    resume)
        readonly -a MODE_ARGUMENTS=()
        ;;
    *)
        usage
        exit 2
        ;;
esac

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "o launcher deve ser executado pelo Slurm"
command -v srun >/dev/null 2>&1 || fail "srun não está disponível"
command -v git >/dev/null 2>&1 || fail "git não está disponível"

CURRENT_DIRECTORY="$(pwd -P)"
readonly CURRENT_DIRECTORY
PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || \
    fail "o diretório atual não pertence a um repositório Git"
readonly PROJECT_ROOT
RESOLVED_PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)" || \
    fail "a raiz do repositório não está acessível"
readonly RESOLVED_PROJECT_ROOT
[[ "$CURRENT_DIRECTORY" == "$RESOLVED_PROJECT_ROOT" ]] || \
    fail "execute o launcher a partir da raiz do repositório"
[[ -f configs/federated-memorization-calibration-v1.yaml ]] || \
    fail "a configuração oficial da calibração está ausente"

if ! git diff --quiet || ! git diff --cached --quiet; then
    fail "o worktree contém alterações rastreadas"
fi

readonly PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON_EXECUTABLE" ]] || fail "o ambiente Python local .venv não está disponível"
readonly PYTHON_VERSION="$($PYTHON_EXECUTABLE -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_VERSION" == "3.12" ]] || fail "o launcher exige Python 3.12"

"$PYTHON_EXECUTABLE" -m pip check || fail "as dependências Python estão inconsistentes"
"$PYTHON_EXECUTABLE" -c '
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(1)
' || fail "a alocação deve expor exatamente uma GPU CUDA"

readonly CALIBRATION_RUN_ID=federated-memorization-calibration-seed-101-v1
readonly CALIBRATION_DEVICE=cuda
readonly CALIBRATION_CONFIG=configs/federated-memorization-calibration-v1.yaml
readonly CALIBRATION_CACHE=artifacts/huggingface
readonly CALIBRATION_OUTPUT_ROOT=outputs
readonly CALIBRATION_RUN_DIRECTORY="$PROJECT_ROOT/$CALIBRATION_OUTPUT_ROOT/runs/$CALIBRATION_RUN_ID"
readonly GIT_COMMIT="$(git rev-parse --verify HEAD)"

if [[ "$CALIBRATION_MODE" == "resume" && ! -d "$CALIBRATION_RUN_DIRECTORY" ]]; then
    fail "resume exige uma execução existente para o run_id oficial"
fi

"$PYTHON_EXECUTABLE" -c '
import sys
from pathlib import Path
from federated_leakage.federated_exposure_contracts import load_federated_exposure_spec_from_config
try:
    spec = load_federated_exposure_spec_from_config(Path(sys.argv[1]))
    valid = (
        spec.experiment_seed == 101
        and spec.scenario == "F0"
        and spec.learning_rate_millionths == 30
        and tuple(arm.victim_repetition_multiplier for arm in spec.arms) == (1, 2, 4)
    )
except Exception:
    valid = False
raise SystemExit(0 if valid else 1)
' "$CALIBRATION_CONFIG" || fail "a configuração não corresponde à calibração oficial"

printf 'job_id: %s\n' "$SLURM_JOB_ID"
printf 'hostname: %s\n' "$(hostname)"
printf 'commit: %s\n' "$GIT_COMMIT"
printf 'modo: %s\n' "$CALIBRATION_MODE"
printf 'dispositivo: %s\n' "$CALIBRATION_DEVICE"

readonly -a CALIBRATION_ARGUMENTS=(
    -m federated_leakage.run_federated_memorization_calibration
    --config "$CALIBRATION_CONFIG"
    --device "$CALIBRATION_DEVICE"
    --cache-dir "$CALIBRATION_CACHE"
    --output-root "$CALIBRATION_OUTPUT_ROOT"
    --run-id "$CALIBRATION_RUN_ID"
    "${MODE_ARGUMENTS[@]}"
)

exec srun --ntasks=1 "$PYTHON_EXECUTABLE" "${CALIBRATION_ARGUMENTS[@]}"
