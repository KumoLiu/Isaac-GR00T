#!/usr/bin/env bash
# 50k-step finetune on episodes [0, 190), 4 GPUs (DDP).
set -euo pipefail
cd /localhome/local-yunl/Isaac-GR00T

export CUDA_HOME="${CUDA_HOME:-/localhome/local-yunl/cuda_home_stub}"
export PATH="$CUDA_HOME/bin:$PATH"

FFMPEG_ROOT="${FFMPEG_ROOT:-/localhome/local-yunl/ffmpeg4_libs}"
TORCH_LIB="$(pwd)/.venv/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="\
${FFMPEG_ROOT}/usr/lib/x86_64-linux-gnu/blas:\
${FFMPEG_ROOT}/usr/lib/x86_64-linux-gnu/lapack:\
${FFMPEG_ROOT}/usr/lib/x86_64-linux-gnu/openblas-pthread:\
${FFMPEG_ROOT}/usr/lib/x86_64-linux-gnu:\
${TORCH_LIB}\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 MASTER_PORT=29502 \
MAX_STEPS=50000 SAVE_STEPS=5000 USE_WANDB=0 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /localhome/local-yunl/g1_pick_trocar_200_train0_190 \
  --modality-config-path examples/g1-dex3/g1_dex3_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --use-ddp \
  --output-dir /localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/50k
