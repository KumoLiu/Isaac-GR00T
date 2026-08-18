#!/usr/bin/env bash
# Wait for GPUs 0-3, then start the horizon-25 fine-tuning run.
set -euo pipefail

OUTPUT_DIR="/localhome/local-yunl/Isaac-GR00T/g1_pick_trocar_200_finetune/head_h25_10k"
mkdir -p "$OUTPUT_DIR"

while nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits |
    awk -F, '$1 + 0 < 4 && $2 + 0 > 2000 { busy = 1 } END { exit busy ? 0 : 1 }'; do
    printf '%s GPUs 0-3 are busy; retrying in 5 minutes.\n' "$(date --iso-8601=seconds)" |
        tee -a "$OUTPUT_DIR/queue.log"
    sleep 300
done

printf '%s GPUs 0-3 are available; starting fine-tuning.\n' \
    "$(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/queue.log"
exec bash /localhome/local-yunl/Isaac-GR00T/run_head_h25_10k.sh
