#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
source /lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013/source/env.sh
export NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec srun --container-image="$USER_ROOT/docker/gr00t-n1.7-cu128.sqsh" \
  --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$EXP_ROOT/code" \
  --container-env="$CONTAINER_ENV,OMP_NUM_THREADS,MKL_NUM_THREADS" --no-container-mount-home --no-container-entrypoint \
  /opt/gr00t-venv/bin/python "$EXP_ROOT/source/prepare_open_loop.py"
