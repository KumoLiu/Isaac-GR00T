#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=gr00t-hf-upload
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.err

# Upload a GR00T checkpoint into nvidia/s2r_models_dev as a named folder:
#   https://huggingface.co/nvidia/s2r_models_dev/tree/main/yunl/sft/g1/pick_trocar/<folder>
# huggingface_hub upload_folder sends *contents* of CHECKPOINT_PATH, so this
# script always appends HF_FOLDER_NAME and never dumps files onto pick_trocar/.
#
# Token must be able to WRITE that repo. Same file as training:
#   $USER_BASE/cache/huggingface/token
#
# Required:
#   CHECKPOINT_PATH  local checkpoint dir (config.json + weights)
#
# Optional:
#   HF_REPO          default nvidia/s2r_models_dev
#   HF_PREFIX        default yunl/sft/g1/pick_trocar
#   HF_FOLDER_NAME   remote folder under prefix (default: experiment name, or
#                    the checkpoint directory name)
#   WEIGHTS_ONLY     1 = skip optimizer/scheduler/RNG (default 1)
#
# Example:
#   sbatch --export=CHECKPOINT_PATH=/lustre/.../g1_pick_trocar_room_30k_bs32_lr1e-4/976128/checkpoint-27000 \
#     scripts/cluster/upload_hf.slurm
# Lands at:
#   .../tree/main/yunl/sft/g1/pick_trocar/g1_pick_trocar_room_30k_bs32_lr1e-4

set -euo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/gr00t-n1.7-cu128.sqsh}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_CACHE_ROOT}/token}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the checkpoint directory}"
HF_REPO="${HF_REPO:-nvidia/s2r_models_dev}"
HF_PREFIX="${HF_PREFIX:-yunl/sft/g1/pick_trocar}"
WEIGHTS_ONLY="${WEIGHTS_ONLY:-1}"
CREATE_REPO="${CREATE_REPO:-0}"

CHECKPOINT_LEAF="$(basename "${CHECKPOINT_PATH}")"
if [[ -z "${HF_FOLDER_NAME:-}" ]]; then
    # .../<exp>/<chain-id>/checkpoint-N  ->  <exp>
    if [[ "${CHECKPOINT_LEAF}" == checkpoint-* ]]; then
        HF_FOLDER_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    else
        HF_FOLDER_NAME="${CHECKPOINT_LEAF}"
    fi
fi
HF_PATH_IN_REPO="${HF_PREFIX}/${HF_FOLDER_NAME}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload ${HF_FOLDER_NAME} from ${SLURM_JOB_ID:-manual}}"

for path in "$CONTAINER_IMAGE" "$HF_TOKEN_FILE" "$CHECKPOINT_PATH"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
    echo "CHECKPOINT_PATH does not look like a checkpoint (no config.json): ${CHECKPOINT_PATH}" >&2
    exit 1
fi

echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "HF repo:      ${HF_REPO}"
echo "Remote dir:   ${HF_PATH_IN_REPO}"
echo "URL:          https://huggingface.co/${HF_REPO}/tree/main/${HF_PATH_IN_REPO}"
echo "Weights only: ${WEIGHTS_ONLY}"

MOUNTS="${HF_CACHE_ROOT}:/root/.cache/huggingface,${CHECKPOINT_PATH}:/ckpt:ro"

srun --ntasks=1 --gpus=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --no-container-mount-home \
    --export=HF_REPO="${HF_REPO}",HF_PATH_IN_REPO="${HF_PATH_IN_REPO}",WEIGHTS_ONLY="${WEIGHTS_ONLY}",CREATE_REPO="${CREATE_REPO}",COMMIT_MESSAGE="${COMMIT_MESSAGE}" \
    bash -lc '
        set -euo pipefail
        export PATH="/opt/gr00t-venv/bin:${PATH}"
        export HF_HOME=/root/.cache/huggingface
        export HF_TOKEN="$(< "${HF_HOME}/token")"
        hf auth whoami
        python - <<'"'"'PY'"'"'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO"]
path_in_repo = os.environ.get("HF_PATH_IN_REPO") or "yunl/sft/g1/pick_trocar"
weights_only = os.environ.get("WEIGHTS_ONLY", "1") == "1"
create_repo = os.environ.get("CREATE_REPO", "0") == "1"
commit_message = os.environ.get("COMMIT_MESSAGE", "Upload checkpoint")

ignore = None
if weights_only:
    ignore = [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "rng_state_*.pth",
        "training_args.bin",
        "**/optimizer.pt",
        "**/scheduler.pt",
        "**/rng_state*",
    ]

api = HfApi()
if create_repo:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"Uploading /ckpt -> {repo_id}:{path_in_repo} (weights_only={weights_only})")
api.upload_folder(
    folder_path="/ckpt",
    repo_id=repo_id,
    repo_type="model",
    path_in_repo=path_in_repo,
    ignore_patterns=ignore,
    commit_message=commit_message,
)
print("Upload complete:", f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
PY
    '
