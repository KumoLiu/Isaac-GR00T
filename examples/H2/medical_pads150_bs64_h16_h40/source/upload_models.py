#!/usr/bin/env python3
"""Publish the six final inference bundles and verify Hub object hashes."""

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time


ROOT = Path(
    os.environ.get(
        "EXP_ROOT",
        "/lustre/fsw/portfolios/healthcareeng/users/yunl/outputs/gr00t/medical_pads150_bs64_h16_h40_20261013",
    )
)
OUT = ROOT / "upload_20261014"
STAGE = OUT / "staging"
REPO = "nvidia/s2r_models_dev"
PREFIX = "yunl/sft/h2/medical_pad"
CORE_FILES = [
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]


def emit(event, **kwargs):
    print(
        json.dumps(
            {
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                **kwargs,
            }
        ),
        flush=True,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prepare():
    plan = json.loads((ROOT / "experiment_plan.json").read_text())
    evaluation = json.loads((ROOT / "analysis_20261014/open_loop_report.json").read_text())
    reports = {item["run_id"]: item for item in evaluation["runs"]}
    manifest = {
        "repo": REPO,
        "prefix": PREFIX,
        "selection": "final_checkpoint",
        "models": [],
        "files": [],
    }
    for run in plan["runs"]:
        run_id, horizon, step = run["run_id"], run["action_horizon"], run["max_optimizer_steps"]
        source = ROOT / "runs" / run_id / "inference_checkpoints" / f"checkpoint-{step}"
        assert json.loads((ROOT / "runs" / run_id / "DONE.json").read_text())["global_step"] == step
        index = json.loads((source / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"].values()) == set(CORE_FILES[-2:])
        processor = json.loads((source / "processor_config.json").read_text())
        assert processor["processor_kwargs"]["modality_configs"]["new_embodiment"]["action"][
            "delta_indices"
        ] == list(range(horizon))
        name = f"h2_medical_pad_no_tactile_{step // 1000}k_bs64_h{horizon}_lr1e-4"
        folder = STAGE / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in CORE_FILES:
            src, dst = source / filename, folder / filename
            assert src.is_file()
            if dst.exists():
                assert os.path.samefile(src, dst), f"Unexpected staged file: {dst}"
            else:
                os.link(src, dst)
        report = reports[run_id]
        final = {k: v for k, v in report["final_checkpoint"].items() if k != "model_path"}
        best = {
            k: v for k, v in report["best_training_fit_checkpoint"].items() if k != "model_path"
        }
        metadata = {
            "run_id": run_id,
            "checkpoint_selection": "final_checkpoint",
            "checkpoint_step": step,
            "action_horizon": horizon,
            "model_internal_action_capacity": 40,
            "base_model": "nvidia/GR00T-N1.7-3B",
            "base_model_revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
            "groot_code_commit": plan["code_commit"],
            "embodiment_tag": "new_embodiment",
            "dataset": {
                "repo_id": plan["hf_repo"],
                "subfolder": plan["hf_subfolder"],
                "revision": plan["hf_revision"],
                "episodes": plan["training_episodes"],
                "frames": plan["training_frames"],
                "fps": plan["fps"],
                "split_policy": plan["split_policy"],
            },
            "training": {
                k: plan[k]
                for k in (
                    "global_batch",
                    "per_gpu_batch",
                    "gpus_per_run",
                    "gradient_accumulation_steps",
                    "learning_rate",
                    "warmup_ratio",
                    "scheduler",
                    "camera",
                    "tactile",
                    "vision_frozen",
                    "llm_frozen",
                )
            },
            "open_loop_evaluation": {
                "dataset_role": evaluation["dataset_role"],
                "episodes": evaluation["episodes_per_checkpoint"],
                "frames": evaluation["frames_per_checkpoint"],
                "execution_horizon": evaluation["execution_horizon"],
                "denoising_steps": evaluation["denoising_steps"],
                "inference_batch_size": 32,
                "seed": "42 + trajectory ID, reset for each checkpoint",
                "final_checkpoint": final,
                "best_training_fit_checkpoint_comparison": best,
                "interpretation_limit": evaluation["interpretation_limit"],
            },
        }
        write_json(folder / "experiment.json", metadata)
        readme = f"""# H2 medical pad: horizon {horizon}, {step:,} steps, global batch 64

Final checkpoint from an independent fine-tune of `nvidia/GR00T-N1.7-3B`.
This folder follows the existing `h2_medical_pad_no_tactile_<steps>k_bs<batch>_lr<lr>` naming convention, adding `h{horizon}` to distinguish the action horizon.

| Setting | Value |
| --- | --- |
| Checkpoint selection | Final checkpoint at {step:,} optimizer steps |
| Global training batch | 64 (8 GPUs × 8 per GPU, accumulation 1) |
| Action horizon | {horizon} frames |
| Internal model action capacity | 40; the processor sets the actual horizon |
| Learning rate | 1e-4, cosine schedule, 5% warmup |
| Dataset | [medical_pads_150_telelop](https://huggingface.co/datasets/nvidia/s2r_data_dev/tree/{plan["hf_revision"]}/{plan["hf_subfolder"]}) |
| Training data | 150 episodes, 48,671 frames, 30 FPS |
| Observation camera | `observation.images.cam_left_high` |
| State/action | 58 joint angles, radians: left/right arms (7 + 7), hands (22 + 22) |
| Action representation | Relative arms, absolute hands; no tactile |
| Frozen components | LLM and vision backbone |
| Embodiment tag | `new_embodiment` (`EmbodimentTag.NEW_EMBODIMENT`) |
| GR00T source commit | `{plan["code_commit"]}` |

## Open-loop evaluation

The final checkpoint has mean-episode MSE **{final["mean_episode_mse"]:.10g} rad²** and mean-episode MAE **{final["mean_episode_mae"]:.10g} rad**.
Evaluation used all 150 training episodes and 48,671 frames, a common execution stride of 16, and 4 denoising steps.
These are training-set fitting metrics. All episodes were used for training; they do not measure held-out generalization or closed-loop success.
For comparison, the lowest training-set mean-episode MSE in this run was {best["mean_episode_mse"]:.10g} at step {best["checkpoint_step"]:,}; the uploaded weights are the final checkpoint specified above.

## Files and loading

The two safetensors shards contain the complete model weights. The index, model configuration, processor configuration, normalization statistics, and embodiment mapping are included for native GR00T inference.
Training optimizer states and intermediate checkpoints remain on the cluster.

Download this folder to a local directory using `huggingface_hub.snapshot_download` with `allow_patterns="{PREFIX}/{name}/*"`, then use that downloaded subdirectory as `model_path` with the native `Gr00tPolicy` and `EmbodimentTag.NEW_EMBODIMENT`.
The GR00T environment and its Cosmos-Reason2-2B processor dependency are required.
`experiment.json` records the exact data revision, training settings, checkpoint selection, and evaluation results.
"""
        (folder / "README.md").write_text(readme)
        manifest["models"].append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "checkpoint_step": step,
                "folder_name": name,
                "path_in_repo": f"{PREFIX}/{name}",
                "source_checkpoint": str(source),
                "url": f"https://huggingface.co/{REPO}/tree/main/{PREFIX}/{name}",
            }
        )
        for path in sorted(folder.iterdir()):
            stat = path.stat()
            manifest["files"].append(
                {
                    "path_in_repo": f"{PREFIX}/{name}/{path.name}",
                    "staged_path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    assert len(manifest["models"]) == 6 and len(manifest["files"]) == 54
    emit("prepared", models=6, files=54, bytes=sum(item["size"] for item in manifest["files"]))
    return manifest


def hash_file(item):
    path = Path(item["staged_path"])
    sha256 = hashlib.sha256()
    git_sha1 = (
        hashlib.sha1(f"blob {item['size']}\0".encode()) if item["size"] < 10_000_000 else None
    )
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha256.update(chunk)
            if git_sha1 is not None:
                git_sha1.update(chunk)
    stat = path.stat()
    assert stat.st_size == item["size"] and stat.st_mtime_ns == item["mtime_ns"]
    return {
        **item,
        "sha256": sha256.hexdigest(),
        "git_blob_sha1": git_sha1.hexdigest() if git_sha1 else None,
    }


def object_record(entry):
    lfs = getattr(entry, "lfs", None)
    return {
        "path": entry.path,
        "size": entry.size,
        "blob_id": entry.blob_id,
        "sha256": lfs.sha256 if lfs else None,
    }


def list_files(api, revision):
    from huggingface_hub.hf_api import RepoFile

    return {
        entry.path: object_record(entry)
        for entry in api.list_repo_tree(
            REPO, path_in_repo=PREFIX, recursive=True, revision=revision, repo_type="model"
        )
        if isinstance(entry, RepoFile)
    }


def matches(item, remote):
    if not remote or item["size"] != remote["size"]:
        return False
    return (
        item["sha256"] == remote["sha256"]
        if remote["sha256"]
        else item["git_blob_sha1"] == remote["blob_id"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    import huggingface_hub
    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    emit("authenticated", user=who["name"], huggingface_hub_version=huggingface_hub.__version__)
    manifest = prepare()
    cached_path = OUT / "upload_manifest.json"
    cached = json.loads(cached_path.read_text()) if cached_path.exists() else {}
    cached_files = {item["path_in_repo"]: item for item in cached.get("files", [])}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for i, item in enumerate(manifest["files"]):
            previous = cached_files.get(item["path_in_repo"])
            if (
                previous
                and all(previous.get(k) == item[k] for k in ("size", "mtime_ns"))
                and previous.get("sha256")
            ):
                manifest["files"][i] = previous
            else:
                futures[pool.submit(hash_file, item)] = i
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            manifest["files"][i] = future.result()
            write_json(cached_path, manifest)
            emit(
                "hashed",
                path=manifest["files"][i]["path_in_repo"],
                size=manifest["files"][i]["size"],
            )
    write_json(cached_path, manifest)
    base_revision = api.repo_info(REPO, repo_type="model", revision="main").sha
    before = list_files(api, base_revision)
    expected = {item["path_in_repo"]: item for item in manifest["files"]}
    new_dirs = [model["path_in_repo"] + "/" for model in manifest["models"]]
    for path, entry in before.items():
        if any(path.startswith(prefix) for prefix in new_dirs):
            assert path in expected and matches(expected[path], entry), (
                f"Existing destination differs: {path}"
            )
    write_json(OUT / "preflight.json", {"revision": base_revision, "files": list(before.values())})
    already_uploaded = all(
        matches(item, before.get(item["path_in_repo"])) for item in manifest["files"]
    )
    if not args.verify_only and not already_uploaded:
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(45):
                emit("upload_in_progress", files=54, models=6)

        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            kwargs = {
                "repo_id": REPO,
                "repo_type": "model",
                "folder_path": str(STAGE),
                "path_in_repo": PREFIX,
                "revision": "main",
                "commit_message": "Add six H2 medical-pad GR00T N1.7 final checkpoints (bs64, h16/h40, 10k/30k/80k)",
                "commit_description": "Full inference bundles for medical_pads_150_telelop (150 episodes). Includes native processor/statistics and training-set open-loop results. Global batch 64, LR 1e-4. Final optimizer-step checkpoints; existing bs32 models are retained.",
            }
            signature = inspect.signature(api.upload_folder).parameters
            if "num_threads" in signature:
                kwargs["num_threads"] = 4
            if "parent_commit" in signature:
                kwargs["parent_commit"] = base_revision
            emit("upload_start", revision=base_revision)
            info = api.upload_folder(**kwargs)
            commit = {"oid": info.oid, "commit_url": info.commit_url}
            write_json(OUT / "commit.json", commit)
            emit("upload_committed", **commit)
        finally:
            stop.set()
            worker.join(timeout=1)
    revision = api.repo_info(REPO, repo_type="model", revision="main").sha
    after = list_files(api, revision)
    verified = []
    for item in manifest["files"]:
        assert matches(item, after.get(item["path_in_repo"])), (
            f"Hash verification failed: {item['path_in_repo']}"
        )
        verified.append(
            {
                "path_in_repo": item["path_in_repo"],
                "size": item["size"],
                "sha256": item["sha256"],
                "verification": "lfs_sha256"
                if after[item["path_in_repo"]]["sha256"]
                else "git_blob_sha1",
            }
        )
    retained = [path for path in before if not any(path.startswith(prefix) for prefix in new_dirs)]
    for path in retained:
        assert after.get(path) == before[path], f"Previously existing file changed: {path}"
    result = {
        "status": "complete",
        "repo": REPO,
        "prefix": PREFIX,
        "revision": revision,
        "selection": "final_checkpoint",
        "models": manifest["models"],
        "model_count": 6,
        "verified_files": verified,
        "file_count": len(verified),
        "total_bytes": sum(item["size"] for item in verified),
        "existing_files_preserved": len(retained),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(OUT / "verified_upload.json", result)
    emit(
        "verified",
        revision=revision,
        models=6,
        files=len(verified),
        bytes=result["total_bytes"],
        existing_files_preserved=len(retained),
    )


if __name__ == "__main__":
    main()
