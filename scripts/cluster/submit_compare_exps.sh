#!/usr/bin/env bash
# Submit the README-aligned G1 pick-trocar comparison sweep.
#
# Official recipe (README + examples/finetune.sh):
#   lr=1e-4, global_batch_size=32, projector+DiT only, a few thousand steps.
# Real G1 data is larger than the SO100 demo (2000 steps), so the anchor is 10k.
#
#   A  10k bs32 lr1e-4   official default (anchor)
#   B  30k bs32 lr1e-4   longer schedule (does 10k under-train?)
#   C  10k bs64 lr1e-4   maximize batch size (README training tip)
#   D  10k bs32 lr2e-5   lower lr (more stable on the same 10k budget)
#
# Checkpoints:
#   $USER_BASE/checkpoints/gr00t_ft/g1_pick_trocar_<cam>_<steps>_bs<bs>_lr<lr>/<jobid>/
#
# Usage:
#   bash scripts/cluster/submit_compare_exps.sh
#   DRY_RUN=1 bash scripts/cluster/submit_compare_exps.sh
#   GPUS=1 DATASET_DIR_NAME=g1_pick_trocar_rollout_all_260_headcam_train \
#     MODALITY_CONFIG_PATH=$PWD/examples/g1-dex3/g1_dex3_head_config.py \
#     bash scripts/cluster/submit_compare_exps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM="${SCRIPT_DIR}/finetune_n17.slurm"
GPUS="${GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Let finetune_n17.slurm build EXP_NAME from steps/batch/lr.
if [[ "${KEEP_EXP_NAME:-0}" != "1" ]]; then
    unset EXP_NAME
fi

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
if [[ -z "${DATASET_PATH:-}" ]]; then
    for candidate in \
        "${USER_BASE}/datasets/s2r_data_dev/yunl/real/g1/pick_trocar_teleop_success_train" \
        "${USER_BASE}/datasets/pick_trocar_teleop_success_train"; do
        if [[ -f "${candidate}/meta/info.json" ]]; then
            DATASET_PATH="${candidate}"
            break
        fi
    done
    DATASET_PATH="${DATASET_PATH:-${USER_BASE}/datasets/s2r_data_dev/yunl/real/g1/pick_trocar_teleop_success_train}"
fi
echo "DATASET_PATH=${DATASET_PATH}"

submit() {
    local job_name="$1"
    shift
    echo "sbatch --gpus=${GPUS} --job-name=${job_name} --export=ALL,$* ${SLURM}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        sbatch --gpus="${GPUS}" --job-name="${job_name}" --export=ALL,"$*" "${SLURM}"
    fi
}

echo "Comparison sweep (1 GPU unless GPUS is set). Expected EXP_NAME tags:"
echo "  A  <prefix>_<cam>_10k_bs32_lr1e-4"
echo "  B  <prefix>_<cam>_30k_bs32_lr1e-4"
echo "  C  <prefix>_<cam>_10k_bs64_lr1e-4"
echo "  D  <prefix>_<cam>_10k_bs32_lr2e-5"

submit "g1_10k_bs32" \
    MAX_STEPS=10000,SAVE_STEPS=1000,GLOBAL_BATCH_SIZE=32,LEARNING_RATE=1e-4,NUM_GPUS="${GPUS}",DATASET_PATH="${DATASET_PATH}"

submit "g1_30k_bs32" \
    MAX_STEPS=30000,SAVE_STEPS=3000,GLOBAL_BATCH_SIZE=32,LEARNING_RATE=1e-4,NUM_GPUS="${GPUS}",DATASET_PATH="${DATASET_PATH}"

submit "g1_10k_bs64" \
    MAX_STEPS=10000,SAVE_STEPS=1000,GLOBAL_BATCH_SIZE=64,LEARNING_RATE=1e-4,NUM_GPUS="${GPUS}",DATASET_PATH="${DATASET_PATH}"

submit "g1_10k_lr2e-5" \
    MAX_STEPS=10000,SAVE_STEPS=1000,GLOBAL_BATCH_SIZE=32,LEARNING_RATE=2e-5,NUM_GPUS="${GPUS}",DATASET_PATH="${DATASET_PATH}"
