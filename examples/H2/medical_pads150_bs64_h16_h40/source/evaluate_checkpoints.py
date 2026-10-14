"""Batched native GR00T open-loop evaluation on all supplied training trajectories."""

import json
import math
import os
from pathlib import Path
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import open_loop_eval
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_20261014/open_loop_train150"
GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
BATCH = 32
STRIDE = 16


def observation(cache, metadata, language_key, begin, end):
    return {
        "video": {"high": cache["images"][begin:end]},
        "state": {key: cache[key][begin:end] for key in GROUPS},
        "language": {language_key: [[text] for text in metadata["texts"][begin:end]]},
    }


def cache_trajectory(trajectory_id):
    directory = OUT / "cache" / f"traj_{trajectory_id:03d}"
    data = {
        "images": np.load(directory / "images.npy", mmap_mode="r"),
        "gt": np.load(directory / "gt.npy"),
        "state": np.load(directory / "state.npy"),
    }
    data.update({key: np.load(directory / f"state_{key}.npy", mmap_mode="r") for key in GROUPS})
    return data, json.loads((directory / "meta.json").read_text())


def metrics(gt, pred, hold, groups):
    error = gt.astype(np.float64) - pred
    hold_error = gt.astype(np.float64) - hold
    return {
        "frames": len(gt),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "hold_state_mse": float(np.mean(hold_error**2)),
        "per_joint_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "groups": {
            key: {
                "mse": float(np.mean(error[:, start:end] ** 2)),
                "rmse": float(np.sqrt(np.mean(error[:, start:end] ** 2))),
                "mae": float(np.mean(np.abs(error[:, start:end]))),
                "hold_state_rmse": float(np.sqrt(np.mean(hold_error[:, start:end] ** 2))),
            }
            for key, (start, end) in groups.items()
        },
    }


def native_parity(policy, horizon):
    loader = LeRobotEpisodeLoader(
        ROOT / f"data/h{horizon}/train",
        policy.get_modality_config(),
        decoder_kwargs={"num_ffmpeg_threads": 1},
    )
    captured = {}
    original = open_loop_eval.plot_trajectory_results

    def capture(**kwargs):
        captured["gt"] = kwargs["gt_action_across_time"]
        captured["pred"] = kwargs["pred_action_across_time"]

    open_loop_eval.plot_trajectory_results = capture
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        native_mse, _ = open_loop_eval.evaluate_single_trajectory(
            policy, loader, 0, EmbodimentTag.NEW_EMBODIMENT, steps=STRIDE, execution_horizon=STRIDE
        )
    finally:
        open_loop_eval.plot_trajectory_results = original
    cache, metadata = cache_trajectory(0)
    torch.manual_seed(42)
    np.random.seed(42)
    action, _ = policy.get_action(observation(cache, metadata, policy.language_key, 0, 1))
    predicted = np.concatenate([action[key][0, :STRIDE] for key in GROUPS], axis=-1)
    assert np.array_equal(captured["gt"], cache["gt"][:STRIDE])
    assert np.allclose(captured["pred"], predicted, rtol=1e-5, atol=1e-5)
    return {
        "trajectory_id": 0,
        "frames": STRIDE,
        "native_mse": float(native_mse),
        "max_prediction_difference": float(np.max(np.abs(captured["pred"] - predicted))),
        "passed": True,
    }


def main():
    started = time.time()
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    item = json.loads((ROOT / "analysis_20261014/evaluation_manifest.json").read_text())[
        "checkpoints"
    ][index]
    output = OUT / "results" / item["run_id"] / f"checkpoint-{item['checkpoint_step']}"
    if (output / "summary.json").exists():
        print("ALREADY_COMPLETE", item["run_id"], item["checkpoint_step"], flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    ready = json.loads((OUT / "CACHE_READY.json").read_text())
    assert (
        ready["episodes"] == 150
        and ready["frames"] == 48671
        and ready["execution_horizon"] == STRIDE
    )
    policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, item["model_path"], device="cuda:0")
    policy.model.action_head.num_inference_timesteps = 4
    modalities = policy.get_modality_config()
    PolicyHorizonSpec.from_modality_config(modalities, n_action_steps=STRIDE)
    assert len(modalities["action"].delta_indices) == item["horizon"]
    assert modalities["video"].modality_keys == ["high"]
    parity = native_parity(policy, item["horizon"]) if index in [0, 1] else None
    records, all_gt, all_pred, all_hold = [], [], [], []
    for trajectory_id in range(150):
        cache, metadata = cache_trajectory(trajectory_id)
        torch.manual_seed(42 + trajectory_id)
        np.random.seed(42 + trajectory_id)
        chunks = []
        for begin in range(0, len(metadata["decisions"]), BATCH):
            end = min(begin + BATCH, len(metadata["decisions"]))
            action, _ = policy.get_action(
                observation(cache, metadata, policy.language_key, begin, end)
            )
            predicted = np.concatenate([action[key][:, :STRIDE] for key in GROUPS], axis=-1)
            assert predicted.shape == (end - begin, STRIDE, 58) and np.isfinite(predicted).all()
            chunks.append(predicted.reshape(-1, 58))
        predicted = np.concatenate(chunks)[: metadata["frames"]]
        gt, state = cache["gt"], cache["state"]
        hold = state[(np.arange(len(gt)) // STRIDE) * STRIDE]
        row = {"trajectory_id": trajectory_id, **metrics(gt, predicted, hold, ready["groups"])}
        assert math.isfinite(row["mse"])
        records.append(row)
        all_gt.append(gt)
        all_pred.append(predicted)
        all_hold.append(hold)
        np.savez_compressed(
            output / f"traj_{trajectory_id:03d}.npz",
            gt=gt,
            pred=predicted,
            state=state,
            hold_state_baseline=hold,
        )
        if (trajectory_id + 1) % 10 == 0:
            (output / "progress.json").write_text(
                json.dumps(
                    {
                        "episodes": len(records),
                        "total_episodes": 150,
                        "elapsed_seconds": time.time() - started,
                    }
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "checkpoint": item["checkpoint_step"],
                        "episodes": len(records),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    (output / "episodes.json").write_text(json.dumps(records, indent=2) + "\n")
    pooled = metrics(
        np.concatenate(all_gt), np.concatenate(all_pred), np.concatenate(all_hold), ready["groups"]
    )
    assert pooled["frames"] == 48671
    summary = {
        **item,
        "dataset": ready["dataset"],
        "dataset_role": "training_set_fit_evaluation",
        "method": "Official Gr00tPolicy.get_action on cached native-loader observations; ground-truth state/video at every 16-frame inference point; batched inference, decoded joint actions.",
        "execution_horizon": STRIDE,
        "prediction_horizon": item["horizon"],
        "model_action_capacity": policy.processor.max_action_horizon,
        "denoising_steps": 4,
        "inference_batch_size": BATCH,
        "seed": "42 + trajectory_id, reset identically for every checkpoint",
        "episodes": 150,
        "mean_episode_mse": float(np.mean([r["mse"] for r in records])),
        "mean_episode_mae": float(np.mean([r["mae"] for r in records])),
        "mean_episode_hold_state_mse": float(np.mean([r["hold_state_mse"] for r in records])),
        "mean_episode_group_mse": {
            key: float(np.mean([r["groups"][key]["mse"] for r in records])) for key in GROUPS
        },
        "pooled": pooled,
        "joint_names": ready["joint_names"],
        "all_predictions_finite": True,
        "native_single_chunk_parity": parity,
        "elapsed_seconds": time.time() - started,
        "completed_unix": time.time(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
        "visible_cuda_devices": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    print(
        "COMPLETE",
        json.dumps(
            {
                "run_id": item["run_id"],
                "step": item["checkpoint_step"],
                "mse": summary["mean_episode_mse"],
                "seconds": summary["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
