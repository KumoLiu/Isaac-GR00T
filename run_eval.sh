#!/usr/bin/env bash
# Open-loop eval for 10k / 30k / 50k G1 Dex3 finetunes.
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

# Register NEW_EMBODIMENT modality (needed when processor looks up tags)
# open_loop_eval loads modality from checkpoint processor; still import for safety.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

NAME="$1"          # 10k|30k|50k
CKPT="$2"
GPU="$3"
OUT="/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/eval/${NAME}"
mkdir -p "$OUT"

echo "[$(date)] eval ${NAME} on GPU ${GPU} -> ${OUT}"
CUDA_VISIBLE_DEVICES="$GPU" uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path /localhome/local-yunl/g1_pick_trocar_200 \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "$CKPT" \
  --traj-ids 0 190 195 199 \
  --execution-horizon 16 \
  --steps 400 \
  --modality-keys left_arm right_arm left_hand right_hand \
  --save-plot-path "$OUT/plot.jpeg" \
  2>&1 | tee "$OUT/eval.log"

echo "[$(date)] done ${NAME}"
