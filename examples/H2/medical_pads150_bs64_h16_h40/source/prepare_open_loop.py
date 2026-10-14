"""Cache exact native-loader observations at a common 16-frame evaluation stride."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from h2_head_config import make_h2_head_config
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_20261014/open_loop_train150"
GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
STRIDE = 16


def prepare(trajectory_id):
    destination = OUT / "cache" / f"traj_{trajectory_id:03d}"
    if (destination / "meta.json").exists():
        return json.loads((destination / "meta.json").read_text())
    modalities = make_h2_head_config(16)
    source = ROOT / "data/h16/train"
    loader = LeRobotEpisodeLoader(source, modalities, decoder_kwargs={"num_ffmpeg_threads": 1})
    trajectory = loader[trajectory_id]
    decisions = list(range(0, len(trajectory), STRIDE))
    observation_modalities = {k: v for k, v in modalities.items() if k != "action"}
    states = {k: [] for k in GROUPS}
    images, texts = [], []
    for step in decisions:
        native = extract_step_data(
            trajectory, step, observation_modalities, EmbodimentTag.NEW_EMBODIMENT
        )
        images.append(np.asarray(native.images["high"]))
        texts.append(native.text)
        for key in GROUPS:
            states[key].append(native.states[key])
    gt = np.concatenate([np.stack(trajectory[f"action.{key}"]) for key in GROUPS], axis=-1)
    all_state = np.concatenate([np.stack(trajectory[f"state.{key}"]) for key in GROUPS], axis=-1)
    assert gt.shape == all_state.shape == (len(trajectory), 58)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "images.npy", np.stack(images))
    np.save(destination / "gt.npy", gt)
    np.save(destination / "state.npy", all_state)
    for key in GROUPS:
        np.save(destination / f"state_{key}.npy", np.stack(states[key]))
    record = {
        "trajectory_id": trajectory_id,
        "frames": len(trajectory),
        "decisions": decisions,
        "texts": texts,
    }
    temporary = destination / "meta.json.tmp"
    temporary.write_text(json.dumps(record) + "\n")
    temporary.replace(destination / "meta.json")
    return record


def main():
    started = time.time()
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(prepare, i) for i in range(150)]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 10 == 0:
                print(
                    json.dumps({"cached_episodes": len(records), "seconds": time.time() - started}),
                    flush=True,
                )
    assert len(records) == 150 and sum(r["frames"] for r in records) == 48671
    info = json.loads((ROOT / "data/h16/train/meta/info.json").read_text())
    names = info["features"]["action"]["names"][0]
    ready = {
        "dataset": str(ROOT / "data/h16/train"),
        "dataset_role": "training_set_fit_evaluation",
        "episodes": 150,
        "frames": 48671,
        "execution_horizon": STRIDE,
        "groups": {
            "left_arm": [0, 7],
            "right_arm": [7, 14],
            "left_hand": [14, 36],
            "right_hand": [36, 58],
        },
        "joint_names": names,
        "source": "Official LeRobotEpisodeLoader + extract_step_data, unchanged observations and joint ordering.",
        "completed_unix": time.time(),
        "preparation_seconds": time.time() - started,
    }
    (OUT / "CACHE_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(ready), flush=True)


if __name__ == "__main__":
    main()
