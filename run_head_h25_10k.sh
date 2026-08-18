#!/usr/bin/env bash
# Continue the 30k G1 head-camera policy for 10k steps with horizon 25.
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

BASE_MODEL="/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/30k/checkpoint-30000"
OUTPUT_DIR="/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/head_h25_10k"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/train.log") 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 MASTER_PORT=29525 \
MAX_STEPS=10000 SAVE_STEPS=1000 USE_WANDB=0 \
uv run bash examples/finetune.sh \
  --base-model-path "$BASE_MODEL" \
  --dataset-path /localhome/local-yunl/g1_pick_trocar_200_train0_190 \
  --modality-config-path examples/g1-dex3/g1_dex3_head_h25_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --use-ddp \
  --output-dir "$OUTPUT_DIR" \
  -- \
  --learning_rate 2e-5
