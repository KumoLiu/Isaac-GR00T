#!/bin/bash
#SBATCH -A healthcareeng_isaac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --job-name=gr00t-g1-download
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_isaac/logs/gr00t_ft/%x-%j.err

# Download one or more Hugging Face dataset subsets into
# $USER_BASE/datasets/<leaf-name>. HF prefixes (s2r_data_dev/yunl/real/g1/...)
# are not kept on disk.
#
# Single dataset (default):
#   sbatch scripts/cluster/download_g1_dataset.slurm
#
# Several datasets, comma or space separated. Leaf name defaults to basename:
#   sbatch --export=ALL,DATASET_SUBDIRS="yunl/real/g1/foo,yunl/real/g1/bar" \
#     scripts/cluster/download_g1_dataset.slurm
#
# Custom leaf names, same order as DATASET_SUBDIRS:
#   sbatch --export=ALL,\
# DATASET_SUBDIRS="yunl/real/g1/pick_trocar_teleop_success_train yunl/real/g1/pick_trocar_teleop_success_val",\
# DATASET_DIR_NAMES="g1_pick_trocar_rollout_all_260_headcam_train g1_pick_trocar_rollout_all_260_headcam_val" \
#     scripts/cluster/download_g1_dataset.slurm

set -euo pipefail

USER_BASE="${USER_BASE:-/lustre/fsw/portfolios/healthcareeng/users/${USER}}"
REPO_ROOT="${REPO_ROOT:-${USER_BASE}/code/Isaac-GR00T}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${USER_BASE}/docker/gr00t-n1.7-cu128.sqsh}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${USER_BASE}/cache/huggingface}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HF_CACHE_ROOT}/token}"
DATASET_ROOT="${DATASET_ROOT:-${USER_BASE}/datasets}"
DATASET_REPO="${DATASET_REPO:-nvidia/s2r_data_dev}"
DATASET_SUBDIR="${DATASET_SUBDIR:-yunl/real/g1/pick_trocar_teleop_success_train}"

split_list() {
    local raw="${1:-}"
    raw="${raw//,/ }"
    raw="${raw//;/ }"
    raw="${raw//$'\n'/ }"
    local -n _out=$2
    _out=()
    local item
    # shellcheck disable=SC2086
    for item in ${raw}; do
        _out+=("${item}")
    done
}

split_list "${DATASET_SUBDIRS:-${DATASET_SUBDIR}}" SUBDIRS
if [[ "${#SUBDIRS[@]}" -eq 0 ]]; then
    echo "DATASET_SUBDIRS/DATASET_SUBDIR is empty" >&2
    exit 1
fi

if [[ -n "${DATASET_DIR_NAMES:-${DATASET_DIR_NAME:-}}" ]]; then
    split_list "${DATASET_DIR_NAMES:-${DATASET_DIR_NAME}}" DIR_NAMES
    if [[ "${#DIR_NAMES[@]}" -ne "${#SUBDIRS[@]}" ]]; then
        echo "DATASET_DIR_NAMES count (${#DIR_NAMES[@]}) != DATASET_SUBDIRS count (${#SUBDIRS[@]})" >&2
        exit 1
    fi
else
    DIR_NAMES=()
    for subdir in "${SUBDIRS[@]}"; do
        DIR_NAMES+=("$(basename "${subdir}")")
    done
fi

for path in "$REPO_ROOT" "$CONTAINER_IMAGE" "$HF_TOKEN_FILE"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
mkdir -p "$HF_CACHE_ROOT" "$DATASET_ROOT"

PENDING_SUBDIRS=()
PENDING_NAMES=()
for i in "${!SUBDIRS[@]}"; do
    dest="${DATASET_ROOT}/${DIR_NAMES[$i]}"
    if [[ -f "${dest}/meta/info.json" ]]; then
        echo "Already present, skip: ${dest}"
        continue
    fi
    PENDING_SUBDIRS+=("${SUBDIRS[$i]}")
    PENDING_NAMES+=("${DIR_NAMES[$i]}")
done

if [[ "${#PENDING_SUBDIRS[@]}" -eq 0 ]]; then
    echo "All requested datasets are already present under ${DATASET_ROOT}"
    exit 0
fi

DATASET_SUBDIRS=$(IFS=,; echo "${PENDING_SUBDIRS[*]}")
DATASET_DIR_NAMES=$(IFS=,; echo "${PENDING_NAMES[*]}")
export DATASET_REPO DATASET_SUBDIRS DATASET_DIR_NAMES

echo "Will download ${#PENDING_SUBDIRS[@]} dataset(s) from ${DATASET_REPO}:"
for i in "${!PENDING_SUBDIRS[@]}"; do
    echo "  ${PENDING_SUBDIRS[$i]} -> ${DATASET_ROOT}/${PENDING_NAMES[$i]}"
done

MOUNTS="${REPO_ROOT}:/workspace/Isaac-GR00T,${HF_CACHE_ROOT}:/root/.cache/huggingface,${DATASET_ROOT}:/datasets"

srun --ntasks=1 --gpus=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace/Isaac-GR00T \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        export PATH="/opt/gr00t-venv/bin:${PATH}"
        export HF_HOME=/root/.cache/huggingface
        if [[ -f "${HF_HOME}/token" ]]; then
            export HF_TOKEN="$(< "${HF_HOME}/token")"
        fi
        hf auth whoami

        IFS="," read -ra SUBDIRS <<< "${DATASET_SUBDIRS}"
        IFS="," read -ra NAMES <<< "${DATASET_DIR_NAMES}"
        if [[ "${#SUBDIRS[@]}" -ne "${#NAMES[@]}" ]]; then
            echo "Internal error: subdir/name count mismatch" >&2
            exit 1
        fi

        STAGE="/datasets/.hf-stage-${SLURM_JOB_ID}"
        rm -rf "${STAGE}"
        mkdir -p "${STAGE}"

        INCLUDE_ARGS=()
        for subdir in "${SUBDIRS[@]}"; do
            INCLUDE_ARGS+=(--include "${subdir}/**")
        done
        hf download "${DATASET_REPO}" \
            --repo-type dataset \
            "${INCLUDE_ARGS[@]}" \
            --local-dir "${STAGE}"

        for i in "${!SUBDIRS[@]}"; do
            SRC="${STAGE}/${SUBDIRS[$i]}"
            DEST="/datasets/${NAMES[$i]}"
            if [[ ! -d "${SRC}" ]]; then
                echo "Download did not produce ${SRC}" >&2
                exit 1
            fi
            rm -rf "${DEST}"
            mv "${SRC}" "${DEST}"
            python scripts/cluster/validate_lerobot_dataset.py --dataset-path "${DEST}"
            echo "Saved dataset at ${DEST}"
        done
        rm -rf "${STAGE}"
    '
