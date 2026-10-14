#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
source /lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013/source/env.sh
export NVIDIA_VISIBLE_DEVICES=void
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
exec srun --container-image="$USER_ROOT/docker/gr00t-n1.7-cu128.sqsh" \
  --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$EXP_ROOT/code" \
  --container-env="$CONTAINER_ENV,EXP_ROOT,HF_HUB_DOWNLOAD_TIMEOUT,HF_XET_HIGH_PERFORMANCE,HF_HUB_DISABLE_PROGRESS_BARS,PYTHONUNBUFFERED" \
  --no-container-mount-home --no-container-entrypoint \
  /opt/gr00t-venv/bin/python "$EXP_ROOT/source/upload_models.py" "$@"
