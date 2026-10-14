# H2 medical-pad fine-tuning: 150 episodes, batch 64, H16/H40

This directory records the six independent experiments trained on October 13–14,
2026. The branch starts from the actual training checkout,
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`.

The experiment used the unmodified native `examples/finetune.sh` and
`gr00t/experiment/launch_finetune.py`. Its custom modality configuration and
Slurm controllers lived outside that checkout in the experiment's `source/`
directory. They are collected here for review and reproduction, with Python
formatting adjusted to the repository style. `original_source_sha256.json`
records their original cluster hashes before formatting.

## Dataset

- Repository: `nvidia/s2r_data_dev`
- Subfolder: `yunl/real/h2/medical_pads_150_telelop`
- Revision: `630d510259189257e0753046a3b63d0526018d7d`
- Format: LeRobot v2.1; 150 episodes, 48,671 frames, 30 FPS.
- Training split: all supplied episodes, `train=0:150`.
- Task text: `unfold the medical nursing pad`.
- Observation: one head RGB camera, joint positions, and task language.

The downloader verified all 318 files against Git blob or LFS SHA-256 hashes.
`source/prepare_data.py` checks episode/frame indices, finite 58-dimensional
state/action arrays, timestamps, and video decoding. It prepares separate H16
and H40 data directories and recomputes both ordinary and relative-action
statistics from these 150 training episodes.

## Actual data configuration

The training configuration is [source/h2_head_config.py](source/h2_head_config.py).
`MEDICAL_ACTION_HORIZON=16` or `40` selects the action horizon. The embodiment
is `EmbodimentTag.NEW_EMBODIMENT` (`new_embodiment`).

| Modality | Keys | Delta indices | Representation |
| --- | --- | --- | --- |
| Video | `high` | `[0]` | Current RGB frame from `observation.images.cam_left_high` |
| State | `left_arm`, `right_arm`, `left_hand`, `right_hand` | `[0]` | Current absolute joint angles |
| Action | Same four joint groups | H16: `0..15`; H40: `0..39` | Relative arm joints, absolute hand joints |
| Language | `annotation.human.task_description` | `[0]` | Task text from `task_index` |

The raw state and action columns use this order, with radians as their units:

| Group | Slice in `observation.state` / `action` | Dimensions | Action representation |
| --- | --- | --- | --- |
| `left_arm` | `[0:7]` | 7 | `RELATIVE`, `NON_EEF`, `DEFAULT` |
| `right_arm` | `[7:14]` | 7 | `RELATIVE`, `NON_EEF`, `DEFAULT` |
| `left_hand` | `[14:36]` | 22 | `ABSOLUTE`, `NON_EEF`, `DEFAULT` |
| `right_hand` | `[36:58]` | 22 | `ABSOLUTE`, `NON_EEF`, `DEFAULT` |

Each arm's entire action chunk is expressed relative to that arm's current
state. Only the current camera/state observation is supplied. Tactile and wrist
camera inputs are absent from this configuration.

The processor configurations serialized by the actual final checkpoints are
included verbatim as [checkpoint_processor_h16.json](checkpoint_processor_h16.json)
and [checkpoint_processor_h40.json](checkpoint_processor_h40.json).
Within each horizon, the 10k, 30k, and 80k checkpoints have identical processor
and statistics hashes. Inspect `processor_kwargs.modality_configs.new_embodiment`
for the H2 configuration; other entries retain the base model's embodiment
configurations.

### Normalization, images, and padding

- `use_percentiles=true`, `use_mean_std=false`, `clip_outliers=true`.
  State and absolute hand actions use `q01/q99` scaling to `[-1, 1]`.
  Native relative arm processing replaces those normalization parameters with
  the horizon-specific `relative_action` statistics and uses their `min/max`
  bounds, independently at each chunk offset. These statistics have shapes
  `(16, 7)` or `(40, 7)` for each arm.
- Raw RGB resolution is 640 × 480. Albumentations uses shortest edge 256 and a
  0.95 fractional crop, followed by shortest-edge resize to 256. Training uses
  a random crop and color jitter: brightness 0.3, contrast 0.4, saturation 0.5,
  hue 0.08. Rotation is 0; letterboxing is disabled. Evaluation uses a center
  crop. `image_crop_size=[230,230]` and `image_target_size=[256,256]` remain in
  the saved config as fallback settings; the shortest-edge/fraction settings
  take precedence in the active Albumentations pipeline.
- `apply_sincos_state_encoding=false`, `exclude_state=false`, and training
  state dropout is 0.2.
- Native model capacity remains 40 action steps and 132 state/action dimensions.
  The processor pads 58 dimensions to 132 and H16 chunks to 40 steps. The action
  mask excludes both padded dimensions and padded time steps from training.
  H16 therefore trains on 16 actual action steps.

## Training and files

All runs start independently from `nvidia/GR00T-N1.7-3B`, revision
`2fc962b973bccdd5d8ce4f67cc63b264d6886495`. Global batch is 64 across eight GPUs:
8 per GPU, accumulation 1. Each run uses learning rate 1e-4, 5% warmup, and a
cosine schedule spanning its own target of 10,000, 30,000, or 80,000 optimizer
steps. LLM and vision are frozen. Full DeepSpeed ZeRO-2 checkpoints are saved
every 500 steps; inference weights are archived every 5,000 steps and at the
final checkpoint.

| File | Purpose |
| --- | --- |
| `experiment_plan.json` | Exact experiment settings and pinned dataset/code revisions |
| `source/h2_head_config.py` | H16/H40 modality configuration |
| `source/download_hf.py`, `source/prepare_data.py` | Verified download and horizon-specific data preparation |
| `source/official_runtime.py`, `source/chain.py`, `source/submit_experiments.py` | Native launcher, checkpoint validation, and Slurm continuation |
| `source/*.sbatch`, `source/env.sh` | Original cluster resource requests and environment |
| `source/prepare_open_loop.py`, `source/evaluate_checkpoints.py` | Native-loader observation cache and checkpoint evaluation |
| `source/plot_loss.py`, `source/report_open_loop.py` | Loss/evaluation plots and reports |
| `source/upload_models.py`, `source/upload_models.sbatch` | Final inference-bundle upload and hash verification |
| `open_loop_summary.json` | Final and lowest-training-MSE checkpoint metrics for all six runs |

The original cluster experiment root was
`/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013`.
The archived Slurm files retain that layout and the original environment paths.
For another experiment, adapt those paths and create a fresh output directory.
The Slurm controllers require their separate `code/` checkout to remain clean
at the pinned native commit; they intentionally validate that before training.

### Native training example

Run from the repository root in the GR00T environment, with an already downloaded
and verified dataset (`DOWNLOAD_VERIFIED.json` is produced by `download_hf.py`):

```bash
MEDICAL_EXAMPLE="$PWD/examples/H2/medical_pads150_bs64_h16_h40"
MEDICAL_RAW_DATA="/path/to/medical_pads_150_telelop"
MEDICAL_PREP_DATA="/path/to/new_experiment/data"
MEDICAL_BASE_MODEL="/path/to/GR00T-N1.7-3B"
MEDICAL_RUN_OUTPUT="/path/to/new_experiment/h16_s10000"
export PYTHONPATH="$PWD:$MEDICAL_EXAMPLE/source${PYTHONPATH:+:$PYTHONPATH}"

python "$MEDICAL_EXAMPLE/source/prepare_data.py" \
  --source "$MEDICAL_RAW_DATA" --output "$MEDICAL_PREP_DATA"

MEDICAL_ACTION_HORIZON=16 NUM_GPUS=8 GLOBAL_BATCH_SIZE=64 \
MAX_STEPS=10000 SAVE_STEPS=500 USE_WANDB=0 DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path "$MEDICAL_BASE_MODEL" \
  --dataset-path "$MEDICAL_PREP_DATA/h16/train" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MEDICAL_EXAMPLE/source/h2_head_config.py" \
  --output-dir "$MEDICAL_RUN_OUTPUT" \
  --shortest-image-edge 256 --crop-fraction 0.95
```

Use `MEDICAL_ACTION_HORIZON=40` with the prepared `h40/train` directory for H40.
Set `MAX_STEPS=30000` or `80000` and a fresh output path for the other runs.

## Evaluation and published weights

All 72 retained checkpoint steps were evaluated on the 150 training episodes
(48,671 frames) using native `Gr00tPolicy.get_action`, execution stride 16,
4 denoising steps, inference batch 32, and seed `42 + trajectory ID`, reset at
each checkpoint. Native single-trajectory parity checks passed for H16 and H40.
These results measure training-set fit: every episode was used for training.
They do not measure held-out generalization or closed-loop robot success.

The six final inference bundles are in
[nvidia/s2r_models_dev/yunl/sft/h2/medical_pad](https://huggingface.co/nvidia/s2r_models_dev/tree/main/yunl/sft/h2/medical_pad),
published at revision `ec81488469e06fbf9b80b7ac5ba3b753f502ee1e`.
Directory names follow
`h2_medical_pad_no_tactile_<10|30|80>k_bs64_h<16|40>_lr1e-4`.
Each bundle includes the final weights, native processor/statistics/config,
embodiment mapping, and experiment/evaluation metadata. All 54 uploaded files
were verified, including SHA-256 checks for all 12 safetensors shards.
