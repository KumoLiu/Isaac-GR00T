#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=gr00t-n1.7-g1-eval
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t/%x-%j.err

set -euo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/Isaac-GR00T}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/gr00t-n1.7-cu128.sqsh}"
DATASET_PATH="${DATASET_PATH:-${USER_BASE}/datasets/pick_trocar_teleop_success_train}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_CACHE_ROOT}/token}"
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_BASE}/checkpoints/gr00t_ft}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1}"
TRAJ_ID="${TRAJ_ID:-0}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"

for path in "$REPO_ROOT" "$CONTAINER_IMAGE" "$DATASET_PATH" "$HF_TOKEN_FILE"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT/open_loop_eval" "$HF_CACHE_ROOT"

MOUNTS="${REPO_ROOT}:/workspace/Isaac-GR00T,${DATASET_PATH}:/data/g1"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/hf-cache,${OUTPUT_ROOT}:/outputs"
MOUNTS="${MOUNTS},${HF_TOKEN_FILE}:/run/secrets/hf_token:ro"

srun --ntasks=1 --gpus=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/Isaac-GR00T \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        export HF_HOME=/hf-cache
        export HF_TOKEN="$(< /run/secrets/hf_token)"
        export PYTHONPATH="/workspace/Isaac-GR00T${PYTHONPATH:+:${PYTHONPATH}}"
        hf auth whoami
        python scripts/cluster/validate_lerobot_dataset.py --dataset-path /data/g1
        python gr00t/eval/open_loop_eval.py \
            --model-path "'"$MODEL_PATH"'" \
            --dataset-path /data/g1 \
            --embodiment-tag "'"$EMBODIMENT_TAG"'" \
            --traj-ids "'"$TRAJ_ID"'" \
            --execution-horizon "'"$EXECUTION_HORIZON"'" \
            --save-plot-path "/outputs/open_loop_eval/traj_'"$TRAJ_ID"'.jpeg"
    '
