"""Slurm chaining around the unmodified official finetune.sh entry point."""

import argparse
import fcntl
import json
import os
import re
import subprocess
import time

from official_runtime import (
    ROOT,
    atomic,
    latest,
    mark_checkpoints,
    quarantine_incomplete,
    run_native,
)


def clean_environment():
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SLURM_", "SBATCH_"))
        or k.startswith("SLURM_CONF")
        or k == "SLURM_CLUSTERS"
    }


def submit(run_id, segment=1, dependency=None, dependency_kind="afterany"):
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    args = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--job-name=mp150-{run_id}",
        f"--output={run}/slurm-%j.out",
        f"--error={run}/slurm-%j.err",
        f"--chdir={ROOT}",
    ]
    if json.loads((run / "spec.json").read_text()).get("smoke"):
        args.append("--time=00:20:00")
    if dependency:
        args.append(f"--dependency={dependency_kind}:{dependency}")
    args += [
        str(ROOT / "source/train_segment.sbatch"),
        run_id,
        "--segment",
        str(segment),
    ]
    answer = subprocess.check_output(args, env=clean_environment(), text=True).strip().split(";")[0]
    assert answer.isdigit(), answer
    return answer


def submit_evaluation(run_id, job_id):
    run = ROOT / "runs" / run_id
    args = [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--dependency=afterany:{job_id}",
        f"--job-name=mp150-eval-{run_id}",
        f"--output={run}/eval-slurm-%j.out",
        f"--error={run}/eval-slurm-%j.err",
        str(ROOT / "source/evaluate_run.sbatch"),
        run_id,
    ]
    answer = subprocess.check_output(args, env=clean_environment(), text=True).strip().split(";")[0]
    assert answer.isdigit(), answer
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--segment", type=int, default=1)
    a = p.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", a.run_id)
    run = ROOT / "runs" / a.run_id
    spec = json.loads((run / "spec.json").read_text())
    if spec.get("global_batch") is None:
        selection = json.loads((ROOT / "batch_selection.json").read_text())
        assert selection["backend"] == "official DeepSpeed ZeRO-2"
        spec["global_batch"] = selection["global_batch"]
        atomic(run / "spec.json", spec)
    job_id = os.environ["SLURM_JOB_ID"]
    with (run / "chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((run / n).exists() for n in ["STOP", "FAILED.json", "DONE.json"]):
            return
        mark_checkpoints(run, spec)
        quarantine_incomplete(run)
        step, checkpoint = latest(run)
        if a.segment > 1 and checkpoint is None:
            raise RuntimeError("Continuation has no complete checkpoint; refusing a fresh restart")
        if a.segment > spec.get("max_segments", 512):
            raise RuntimeError("Segment limit reached")
        details = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True)
        elapsed = re.search(r"RunTime=(\d+):(\d+):(\d+)", details)
        spent = sum(int(x) * k for x, k in zip(elapsed.groups(), [3600, 60, 1])) if elapsed else 0
        deadline = time.time() + 3 * 3600 + 40 * 60 - spent
        next_file = run / f"successor-{job_id}.json"
        if next_file.exists():
            child = json.loads(next_file.read_text())["job_id"]
        else:
            child = submit(a.run_id, a.segment + 1, job_id)
            atomic(next_file, {"job_id": child, "parent": job_id, "segment": a.segment + 1})
        atomic(
            run / "chain_status.json",
            {
                "status": "running",
                "job_id": job_id,
                "successor_job_id": child,
                "segment": a.segment,
                "resumed_step": step,
                "max_steps": spec["steps"],
                "deadline_unix": deadline,
                "entrypoint": "official examples/finetune.sh -> launch_finetune.py",
            },
        )
        result = run_native(
            spec,
            run,
            resume=checkpoint is not None,
            deadline=deadline,
            handoff_step=step + spec.get("segment_chunk_steps", 0)
            if spec.get("segment_chunk_steps")
            else 0,
        )
        after, path = latest(run)
        if (result["exit_code"] != 0 and not result["planned_handoff"]) or after <= step:
            atomic(
                run / "FAILED.json",
                {
                    **result,
                    "reason": "native training failed or no checkpoint progress",
                },
            )
            subprocess.run(["scancel", child], check=False)
            raise SystemExit(1)
        complete = after >= spec["steps"]
        if spec.get("smoke") and a.segment >= spec.get("smoke_segments", 2):
            complete = True
        if complete:
            subprocess.run(["scancel", child], check=False)
            record = {
                "global_step": after,
                "job_id": job_id,
                "segment": a.segment,
                "checkpoint": str(path),
                "completed_unix": time.time(),
                "resumed_step": step,
                "native_total_steps": spec["steps"],
            }
            if not spec.get("smoke") and spec.get("automatic_evaluation", True):
                record["evaluation_job_id"] = submit_evaluation(a.run_id, job_id)
            atomic(run / "DONE.json", record)
            atomic(run / "chain_status.json", {"status": "completed", **record})
        else:
            atomic(
                run / "chain_status.json",
                {
                    "status": "continuation_queued",
                    "job_id": job_id,
                    "successor_job_id": child,
                    "global_step": after,
                    "segment": a.segment,
                },
            )
        print("SEGMENT_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
