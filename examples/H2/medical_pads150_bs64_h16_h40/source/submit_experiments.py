"""Submit six fresh runs only after the requested dataset passed preparation."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time

from chain import submit
from official_runtime import AUX_REF, AUX_REVISION, BASE, IMAGE, ROOT, atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, required=True)
    args = parser.parse_args()
    assert args.global_batch > 0 and args.global_batch % 8 == 0
    pin = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == pin
    assert not subprocess.check_output(
        ["git", "-C", str(ROOT / "code"), "status", "--porcelain"], text=True
    ).strip()
    assert AUX_REF.read_text().strip() == AUX_REVISION
    assert (BASE / "config.json").is_file() and IMAGE.is_file()
    manifests = {}
    for horizon in [16, 40]:
        dataset = ROOT / f"data/h{horizon}/train"
        manifest = json.loads((dataset.parent / "manifest.json").read_text())
        assert manifest["episodes"] == 150 and manifest["frames"] == 48671
        assert manifest["action_horizon"] == horizon and manifest["video_decode_checks"]
        for name, expected in manifest["statistics_sha256"].items():
            assert hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest() == expected
        manifests[str(horizon)] = manifest
    with (ROOT / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = ROOT / "submission.json"
        if previous.exists():
            record = json.loads(previous.read_text())
            assert record["global_batch"] == args.global_batch
            if len(record["runs"]) == 6:
                print(json.dumps(record), flush=True)
                return
        else:
            record = {
                "requested_date": "2026-10-13",
                "experiment_root": str(ROOT),
                "code_commit": pin,
                "hf_revision": manifests["16"]["hf_revision"],
                "global_batch": args.global_batch,
                "per_gpu_batch": args.global_batch // 8,
                "gpus_per_run": 8,
                "gradient_accumulation_steps": 1,
                "fresh_initialization": True,
                "training_episodes": 150,
                "training_frames": 48671,
                "automatic_evaluation": False,
                "evaluation_reason": "The supplied dataset assigns all episodes to training; no validation split was requested.",
                "automatic_continuation": True,
                "runs": [],
            }
        submitted = {entry["run_id"] for entry in record["runs"]}
        for horizon in [16, 40]:
            for steps in [10000, 30000, 80000]:
                name = f"h{horizon}_s{steps}"
                if name in submitted:
                    continue
                run = ROOT / "runs" / name
                run.mkdir(parents=True, exist_ok=True)
                if (run / "initial_job.json").exists() or list(
                    (run / "model").glob("checkpoint-*")
                ):
                    raise RuntimeError(
                        f"Existing launch requires inspection before resubmission: {name}"
                    )
                spec = {
                    "horizon": horizon,
                    "steps": steps,
                    "global_batch": args.global_batch,
                    "per_gpu_batch": args.global_batch // 8,
                    "gpus_per_run": 8,
                    "gradient_accumulation_steps": 1,
                    "save_steps": 500,
                    "max_segments": 512,
                    "automatic_evaluation": False,
                    "fresh_finetune_from_base": True,
                    "training_episodes": 150,
                    "training_frames": 48671,
                    "hf_revision": manifests[str(horizon)]["hf_revision"],
                    "batch_policy": "Global batch explicitly passed to the official finetune.sh launcher; per-GPU = global / 8.",
                    "entrypoint": "official examples/finetune.sh -> launch_finetune.py",
                    "backend": "DeepSpeed ZeRO-2",
                    "model_internal_action_horizon": 40,
                    "learning_rate": 0.0001,
                    "scheduler": "cosine independently spanning this run's max_steps",
                    "warmup_ratio": 0.05,
                    "camera": "observation.images.cam_left_high",
                    "tactile": False,
                }
                atomic(run / "spec.json", spec)
                job = submit(name)
                entry = {"run_id": name, "job_id": job, "submitted_unix": time.time(), "spec": spec}
                atomic(run / "initial_job.json", entry)
                record["runs"].append(entry)
                atomic(previous, record)
                print(json.dumps({"run_id": name, "job_id": job}), flush=True)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
