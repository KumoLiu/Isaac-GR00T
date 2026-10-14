#!/usr/bin/env bash
#SBATCH --account=healthcareeng_isaac
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=04:00:00
set -euo pipefail
source /lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013/source/env.sh
exec python3 "$EXP_ROOT/source/chain.py" "$@"
