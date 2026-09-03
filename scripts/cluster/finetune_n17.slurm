#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --job-name=gr00t_n17_ft
#SBATCH -o /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.out
#SBATCH -e /lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.err
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"210","reason":"model_loading","description":"GR00T N1.7 model + dataset stats loading can take tens of minutes"}}'

# GR00T N1.7 fine-tune on this cluster (Isaac-GR00T + Pyxis).
#
# Download data with scripts/cluster/download_g1_dataset.slurm first. This job
# only checks that the LeRobot root exists; it does not call Hugging Face.
#
# Defaults match examples/finetune.sh. Differences vs the rl2real
# gr00t_finetune_n17.slurm reference (override with sbatch --export):
#   MAX_STEPS 10000 vs 30000
#   SAVE_STEPS 1000 vs 4000
#   DATALOADER_NUM_WORKERS 4 vs 16
#   USE_WANDB 0 vs optional WANDB_API_KEY
#   BASE_MODEL nvidia/GR00T-N1.7-3B (HF id, cache on Lustre) vs a local edited copy
#   HF_HUB_OFFLINE 0 vs 1
# Same as both official and the reference: lr=1e-4, global_batch_size=32,
# tune_llm=false, tune_visual=false, warmup_ratio=0.05, weight_decay=1e-5.
#
# Checkpoints land in:
#   $CHECKPOINT_ROOT/<EXP_NAME>/<CHAIN_ID>/
# EXP_NAME is built as <prefix>_<cam>_<steps>_bs<batch>_lr<lr> unless you
# already put that hyperparameter tag in EXP_NAME. Example:
#   g1_pick_trocar_room_10k_bs32_lr1e-4
#
# Usage:
#   sbatch --gpus=1 scripts/cluster/finetune_n17.slurm
#   sbatch --gpus=1 --export=ALL,MAX_STEPS=30000,SAVE_STEPS=3000 \
#       scripts/cluster/finetune_n17.slurm
#   bash scripts/cluster/submit_compare_exps.sh

set -uo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
PROJECT_LOG="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft"

REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/Isaac-GR00T}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/gr00t-n1.7-cu128.sqsh}"
DATASET_ROOT="${DATASET_ROOT:-${USER_BASE}/datasets}"
DATASET_DIR_NAME="${DATASET_DIR_NAME:-pick_trocar_teleop_success_train}"

# DATASET_DIR_NAME is the source of truth. A pre-existing DATASET_PATH (from
# sbatch --export=ALL or an old SSH `export`) is ignored unless its last
# component already matches DATASET_DIR_NAME.
_leaf="${DATASET_ROOT}/${DATASET_DIR_NAME}"
_nested="${DATASET_ROOT}/s2r_data_dev/yunl/real/g1/${DATASET_DIR_NAME}"
if [[ -z "${DATASET_PATH:-}" || "$(basename "${DATASET_PATH}")" != "${DATASET_DIR_NAME}" ]]; then
    if [[ -f "${_leaf}/meta/info.json" ]]; then
        DATASET_PATH="${_leaf}"
    elif [[ -f "${_nested}/meta/info.json" ]]; then
        DATASET_PATH="${_nested}"
    else
        DATASET_PATH="${_nested}"
    fi
fi
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${USER_BASE}/checkpoints/gr00t_ft}"
EXP_PREFIX="${EXP_PREFIX:-g1_pick_trocar}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_WANDB="${USE_WANDB:-0}"
TUNE_LLM="${TUNE_LLM:-false}"
TUNE_VISUAL="${TUNE_VISUAL:-false}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${REPO_ROOT}/examples/g1-dex3/g1_dex3_config.py}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

if (( MAX_STEPS % 1000 == 0 )); then
    STEPS_TAG="$((MAX_STEPS / 1000))k"
else
    STEPS_TAG="${MAX_STEPS}"
fi
LR_TAG="$(printf '%s' "${LEARNING_RATE}" | tr '[:upper:]' '[:lower:]')"
HYPER_TAG="${STEPS_TAG}_bs${GLOBAL_BATCH_SIZE}_lr${LR_TAG}"

MODALITY_STEM="$(basename "${MODALITY_CONFIG_PATH}" .py)"
case "${MODALITY_STEM}" in
    g1_dex3_head_h25_config) CAM_TAG="headh25" ;;
    g1_dex3_head_config) CAM_TAG="head" ;;
    g1_dex3_config) CAM_TAG="room" ;;
    *) CAM_TAG="" ;;
esac

if [[ -z "${EXP_NAME:-}" ]]; then
    if [[ -n "${CAM_TAG}" ]]; then
        EXP_NAME="${EXP_PREFIX}_${CAM_TAG}_${HYPER_TAG}"
    else
        EXP_NAME="${EXP_PREFIX}_${HYPER_TAG}"
    fi
elif [[ "${EXP_NAME}" != *"${HYPER_TAG}"* ]]; then
    EXP_NAME="${EXP_NAME}_${HYPER_TAG}"
fi

CHAIN_ID="${CHAIN_ID:-${SLURM_JOB_ID}}"
RUN_COUNT="${RUN_COUNT:-1}"
MAX_RUNS="${MAX_RUNS:-8}"
INNER_TIMEOUT="${INNER_TIMEOUT:-3.9h}"

CKPT_PARENT="${CHECKPOINT_ROOT}/${EXP_NAME}"
CKPT_DIR="${CKPT_PARENT}/${CHAIN_ID}"
mkdir -p "${CKPT_DIR}" "${PROJECT_LOG}" "${HF_CACHE_ROOT}" "${DATASET_ROOT}"

for required in "${REPO_ROOT}" "${CONTAINER_IMAGE}" "${HF_CACHE_ROOT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required path: ${required}" >&2
        exit 1
    fi
done

IFS=':' read -ra _HOST_DATASETS <<< "${DATASET_PATH}"
for host_ds in "${_HOST_DATASETS[@]}"; do
    if [[ ! -f "${host_ds}/meta/info.json" ]]; then
        echo "Dataset is not ready: ${host_ds} (no ${host_ds}/meta/info.json)" >&2
        echo "Resolved from DATASET_DIR_NAME=${DATASET_DIR_NAME}" >&2
        echo "Set DATASET_DIR_NAME to the leaf folder name, or pass a full DATASET_PATH" >&2
        echo "whose basename matches DATASET_DIR_NAME." >&2
        exit 1
    fi
done

if [[ ! -f "${MODALITY_CONFIG_PATH}" ]]; then
    echo "Missing modality config: ${MODALITY_CONFIG_PATH}" >&2
    exit 1
fi

RESUME_FROM_CHECKPOINT=0
if compgen -G "${CKPT_DIR}/checkpoint-*" > /dev/null; then
    RESUME_FROM_CHECKPOINT=1
fi

MOUNTS="${REPO_ROOT}:/workspace/Isaac-GR00T"
MOUNTS="${MOUNTS},${HF_CACHE_ROOT}:/root/.cache/huggingface"
MOUNTS="${MOUNTS},${CKPT_PARENT}:/workspace/logs"
MOUNTS="${MOUNTS},${MODALITY_CONFIG_PATH}:/workspace/configs/modality_config.py:ro"

CONTAINER_DATASET_PATH=""
for i in "${!_HOST_DATASETS[@]}"; do
    MOUNTS="${MOUNTS},${_HOST_DATASETS[$i]}:/data/dataset_${i}"
    if [[ -n "${CONTAINER_DATASET_PATH}" ]]; then
        CONTAINER_DATASET_PATH="${CONTAINER_DATASET_PATH}:/data/dataset_${i}"
    else
        CONTAINER_DATASET_PATH="/data/dataset_${i}"
    fi
done

HOST_DATASET_PATH="${DATASET_PATH}"

echo "=== GR00T N1.7 fine-tune ${RUN_COUNT}/${MAX_RUNS} job=${SLURM_JOB_ID} chain=${CHAIN_ID} ==="
echo "Exp name:   ${EXP_NAME}"
echo "Repo:       ${REPO_ROOT}"
echo "Image:      ${CONTAINER_IMAGE}"
echo "Data:       ${HOST_DATASET_PATH}"
echo "Output:     ${CKPT_DIR}"
echo "GPUs:       ${NUM_GPUS}"
echo "Max steps:  ${MAX_STEPS}"
echo "Batch:      ${GLOBAL_BATCH_SIZE}"
echo "LR:         ${LEARNING_RATE}"
echo "Save every: ${SAVE_STEPS}"
echo "Resume:     ${RESUME_FROM_CHECKPOINT}"
echo "Embodiment: ${EMBODIMENT_TAG}"
echo "Modality:   ${MODALITY_CONFIG_PATH}"

export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all
unset CUDA_VISIBLE_DEVICES

FINETUNE_EXTRA=()
if [[ "${TUNE_LLM}" == "true" ]]; then
    FINETUNE_EXTRA+=(--tune-llm)
fi
if [[ "${TUNE_VISUAL}" == "true" ]]; then
    FINETUNE_EXTRA+=(--tune-visual)
fi
if [[ "${RESUME_FROM_CHECKPOINT}" == "1" ]]; then
    FINETUNE_EXTRA+=(--resume-from-checkpoint)
fi

set +e
timeout "${INNER_TIMEOUT}" srun --ntasks=1 --gpus="${NUM_GPUS}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace/Isaac-GR00T \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        export PATH="/opt/gr00t-venv/bin:${PATH}"
        export VIRTUAL_ENV=/opt/gr00t-venv
        export PYTHONPATH="/workspace/Isaac-GR00T${PYTHONPATH:+:${PYTHONPATH}}"
        export HF_HOME=/root/.cache/huggingface
        export HF_HUB_OFFLINE="'"${HF_HUB_OFFLINE}"'"
        export TRANSFORMERS_OFFLINE="'"${TRANSFORMERS_OFFLINE}"'"
        export TORCH_NCCL_ENABLE_MONITORING=0
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
        export PYTORCH_ALLOC_CONF=expandable_segments:True
        export CUDA_HOME=/usr/local/cuda
        if [[ -f "${HF_HOME}/token" ]]; then
            export HF_TOKEN="$(< "${HF_HOME}/token")"
        fi
        python -c "import torch, gr00t; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        if [[ -f scripts/cluster/validate_lerobot_dataset.py ]]; then
            python scripts/cluster/validate_lerobot_dataset.py --dataset-path /data/dataset_0
        else
            echo "Skipping scripts/cluster/validate_lerobot_dataset.py (not in this checkout)"
        fi
        NUM_GPUS="'"${NUM_GPUS}"'" \
        MASTER_PORT="'"${MASTER_PORT}"'" \
        MAX_STEPS="'"${MAX_STEPS}"'" \
        GLOBAL_BATCH_SIZE="'"${GLOBAL_BATCH_SIZE}"'" \
        SAVE_STEPS="'"${SAVE_STEPS}"'" \
        LEARNING_RATE="'"${LEARNING_RATE}"'" \
        DATALOADER_NUM_WORKERS="'"${DATALOADER_NUM_WORKERS}"'" \
        USE_WANDB="'"${USE_WANDB}"'" \
        bash examples/finetune.sh \
            --base-model-path "'"${BASE_MODEL}"'" \
            --dataset-path "'"${CONTAINER_DATASET_PATH}"'" \
            --embodiment-tag "'"${EMBODIMENT_TAG}"'" \
            --modality-config-path /workspace/configs/modality_config.py \
            --output-dir /workspace/logs \
            --experiment-name "'"${CHAIN_ID}"'" \
            -- '"${FINETUNE_EXTRA[*]}"'
    '
EXIT_CODE=$?
set -e

echo "Training wrapper exited with ${EXIT_CODE}"

if [[ "${EXIT_CODE}" -eq 124 && "${RUN_COUNT}" -lt "${MAX_RUNS}" ]]; then
    NEXT_RUN=$((RUN_COUNT + 1))
    echo "Inner timeout; resubmitting run ${NEXT_RUN}/${MAX_RUNS} chain=${CHAIN_ID}"
    sbatch --gpus="${NUM_GPUS}" \
        --export=ALL,\
CHAIN_ID="${CHAIN_ID}",\
RUN_COUNT="${NEXT_RUN}",\
MAX_RUNS="${MAX_RUNS}",\
DATASET_PATH="${HOST_DATASET_PATH}",\
CHECKPOINT_ROOT="${CHECKPOINT_ROOT}",\
CONTAINER_IMAGE="${CONTAINER_IMAGE}",\
REPO_ROOT="${REPO_ROOT}",\
EXP_NAME="${EXP_NAME}",\
EXP_PREFIX="${EXP_PREFIX}",\
NUM_GPUS="${NUM_GPUS}",\
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}",\
MAX_STEPS="${MAX_STEPS}",\
SAVE_STEPS="${SAVE_STEPS}",\
LEARNING_RATE="${LEARNING_RATE}",\
TUNE_LLM="${TUNE_LLM}",\
TUNE_VISUAL="${TUNE_VISUAL}",\
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}",\
EMBODIMENT_TAG="${EMBODIMENT_TAG}",\
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH}",\
BASE_MODEL="${BASE_MODEL}",\
HF_CACHE_ROOT="${HF_CACHE_ROOT}",\
HF_HUB_OFFLINE="${HF_HUB_OFFLINE}",\
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}",\
USE_WANDB="${USE_WANDB}",\
WANDB_API_KEY="${WANDB_API_KEY:-}" \
        "${REPO_ROOT}/scripts/cluster/finetune_n17.slurm"
    exit 0
fi

exit "${EXIT_CODE}"
