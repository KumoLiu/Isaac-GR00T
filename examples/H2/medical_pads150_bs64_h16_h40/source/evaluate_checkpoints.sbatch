#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source /lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013/source/env.sh
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec srun --container-image="$USER_ROOT/docker/gr00t-n1.7-cu128.sqsh" \
  --container-mounts="$CONTAINER_MOUNTS" --container-workdir="$EXP_ROOT/code" \
  --container-env="$CONTAINER_ENV,SLURM_ARRAY_TASK_ID,SLURM_JOB_ID,OMP_NUM_THREADS,MKL_NUM_THREADS" --no-container-mount-home --no-container-entrypoint \
  /opt/gr00t-venv/bin/python "$EXP_ROOT/source/evaluate_checkpoints.py"
