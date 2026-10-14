"""Run untouched official entry points and inspect checkpoints outside training."""

import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
USER_ROOT = ROOT.parents[2]
ALIAS_ROOT = Path("/lustre/fsw/portfolios/healthcareeng/users/yunl")
BASE = (
    USER_ROOT
    / "cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
)
IMAGE = USER_ROOT / "docker/gr00t-n1.7-cu128.sqsh"
AUX_REF = USER_ROOT / "cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/refs/main"
AUX_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def zip_complete(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return False


def safetensor_complete(path):
    try:
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        return path.stat().st_size == 8 + size + end
    except (OSError, ValueError, KeyError, struct.error):
        return False


def inspect_checkpoint(path, world_size=8, require_optimizer=True):
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        step = int(path.name.split("-")[-1])
        if state["global_step"] != step:
            return None
        for name in [
            "config.json",
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ]:
            json.loads((path / name).read_text())
        weights = list(path.glob("*.safetensors"))
        if not weights or not all(safetensor_complete(p) for p in weights):
            return None
        rng = list(path.glob("rng_state*.pth"))
        if require_optimizer and (len(rng) != world_size or not all(zip_complete(p) for p in rng)):
            return None
        ds = path / f"global_step{step}"
        optimizer = list(ds.glob("*_optim_states.pt"))
        models = list(ds.glob("*_model_states.pt"))
        if require_optimizer:
            if len(optimizer) != world_size or not models:
                return None
            if not all(zip_complete(p) for p in optimizer + models):
                return None
            if (path / "latest").read_text().strip() != ds.name:
                return None
        return {
            "global_step": step,
            "max_steps": state["max_steps"],
            "world_size": world_size,
            "optimizer_shards": len(optimizer),
            "rng_shards": len(rng),
            "optimizer_scheduler_rng_saved": require_optimizer,
            "backend": "official DeepSpeed ZeRO-2",
            "complete_unix": time.time(),
        }
    except (OSError, ValueError, KeyError):
        return None


def archive_inference(run, checkpoint):
    target = run / "inference_checkpoints" / checkpoint.name
    target.mkdir(parents=True, exist_ok=True)
    for source in checkpoint.iterdir():
        if source.is_file() and source.suffix in [".json", ".safetensors"]:
            dst = target / source.name
            if not dst.exists():
                os.link(source, dst)


def mark_checkpoints(run, spec):
    for path in (run / "model").glob("checkpoint-*"):
        if (path / "COMPLETE.json").exists():
            record = json.loads((path / "COMPLETE.json").read_text())
        else:
            record = inspect_checkpoint(path, require_optimizer=not spec.get("probe"))
            if record is None:
                continue
            record["job_id"] = os.environ.get("SLURM_JOB_ID")
            atomic(path / "COMPLETE.json", record)
        if not spec.get("probe") and (
            record["global_step"] % 5000 == 0 or record["global_step"] >= spec["steps"]
        ):
            archive_inference(run, path)


def latest(run):
    candidates = []
    for path in (run / "model").glob("checkpoint-*/COMPLETE.json"):
        obj = json.loads(path.read_text())
        candidates.append((obj["global_step"], path.parent))
    return max(candidates, default=(0, None), key=lambda x: x[0])


def quarantine_incomplete(run):
    for path in (run / "model").glob("checkpoint-*"):
        if not (path / "COMPLETE.json").exists():
            dst = run / "incomplete_checkpoints" / f"{path.name}-{time.time_ns()}"
            dst.parent.mkdir(exist_ok=True)
            path.rename(dst)


def gpu_memory():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = [[float(x.strip()) / 1024 for x in line.split(",")] for line in out.splitlines()]
    return max(r[0] for r in rows), min(r[1] for r in rows)


def official_command(spec, run, resume=False):
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "51d4c89f72fda44cbf77285c6a8114b52676b8a1", commit
    assert not subprocess.check_output(
        ["git", "-C", str(ROOT / "code"), "status", "--porcelain"], text=True
    ).strip()
    assert AUX_REF.read_text().strip() == AUX_REVISION, "Backbone cache revision changed"
    env = dict(os.environ)
    env.update(
        NUM_GPUS="8",
        MAX_STEPS=str(spec["steps"]),
        SAVE_STEPS=str(spec.get("save_steps", 500)),
        GLOBAL_BATCH_SIZE=str(spec["global_batch"]),
        USE_WANDB="0",
        DATALOADER_NUM_WORKERS="4",
        MEDICAL_ACTION_HORIZON=str(spec["horizon"]),
    )
    keys = [
        "NUM_GPUS",
        "MAX_STEPS",
        "SAVE_STEPS",
        "GLOBAL_BATCH_SIZE",
        "USE_WANDB",
        "DATALOADER_NUM_WORKERS",
        "MEDICAL_ACTION_HORIZON",
    ]
    command = [
        "srun",
        "--kill-on-bad-exit=1",
        f"--container-image={IMAGE}",
        f"--container-mounts={USER_ROOT}:{USER_ROOT},{ALIAS_ROOT}:{ALIAS_ROOT}",
        f"--container-workdir={ROOT}/code",
        "--container-env=" + env["CONTAINER_ENV"] + "," + ",".join(keys),
        "--no-container-mount-home",
        "--no-container-entrypoint",
        "bash",
        "-c",
        'export PATH=/opt/gr00t-venv/bin:$PATH; exec bash examples/finetune.sh "$@"',
        "bash",
        "--base-model-path",
        str(BASE),
        "--dataset-path",
        str(ROOT / f"data/h{spec['horizon']}/train"),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(ROOT / "source/h2_head_config.py"),
        "--output-dir",
        str(run / "model"),
        "--shortest-image-edge",
        "256",
        "--crop-fraction",
        "0.95",
    ]
    if resume:
        command.append("--resume-from-checkpoint")
    if spec.get("probe"):
        command.append("--save-only-model")
    return command, env


def run_native(spec, run, resume=False, deadline=0, handoff_step=0):
    command, env = official_command(spec, run, resume)
    job = os.environ["SLURM_JOB_ID"]
    atomic(
        run / f"command-{job}.json",
        {
            "argv": command,
            "official_finetune_sh": str(ROOT / "code/examples/finetune.sh"),
            "official_launch_finetune_py": str(ROOT / "code/gr00t/experiment/launch_finetune.py"),
            "environment": {
                k: env[k]
                for k in [
                    "NUM_GPUS",
                    "MAX_STEPS",
                    "SAVE_STEPS",
                    "GLOBAL_BATCH_SIZE",
                    "DATALOADER_NUM_WORKERS",
                    "MEDICAL_ACTION_HORIZON",
                ]
            },
        },
    )
    log_path = run / f"train-{job}.log"
    peak, total, planned = 0.0, 0.0, False
    observed_step, _ = latest(run)
    interval = 1 if spec.get("probe") else 5
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if AUX_REF.read_text().strip() != AUX_REVISION:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=90)
                raise RuntimeError(
                    "Backbone cache revision changed; refusing to train with different artifacts"
                )
            mark_checkpoints(run, spec)
            step, _ = latest(run)
            used, total = gpu_memory()
            peak = max(peak, used)
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size - 32000))
                tail = stream.read().decode(errors="replace")
            matches = [
                int(a)
                for a, b in re.findall(r"\b(\d+)/(\d+)\s+\[", tail)
                if int(b) == spec["steps"]
            ]
            observed_step = max(observed_step, step, *matches)
            atomic(
                run / "progress.json",
                {
                    "global_step": observed_step,
                    "last_complete_step": step,
                    "max_steps": spec["steps"],
                    "job_id": job,
                    "updated_unix": time.time(),
                    "peak_gpu_used_GiB": peak,
                    "total_gpu_GiB": total,
                },
            )
            if (handoff_step and step >= handoff_step) or (deadline and time.time() >= deadline):
                planned = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(interval)
        code = process.wait()
    mark_checkpoints(run, spec)
    step, checkpoint = latest(run)
    result = {
        "exit_code": code,
        "planned_handoff": planned,
        "global_step": step,
        "max_steps": spec["steps"],
        "job_id": job,
        "resumed": resume,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "peak_gpu_used_GiB": peak,
        "total_gpu_GiB": total,
    }
    atomic(run / f"segment-{job}.json", result)
    atomic(run / "segment_result.json", result)
    return result
