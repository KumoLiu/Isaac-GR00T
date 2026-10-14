"""Validate all 150 supplied episodes and prepare horizon-specific training data."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats
from h2_head_config import make_h2_head_config
import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    download = json.loads((source / "DOWNLOAD_VERIFIED.json").read_text())
    assert download["all_checksums_verified"] and download["files"] == 318
    info = json.loads((source / "meta/info.json").read_text())
    modality = json.loads((source / "meta/modality.json").read_text())
    episodes = [
        json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()
    ]
    assert info["total_episodes"] == len(episodes) == 150
    assert info["total_frames"] == sum(row["length"] for row in episodes) == 48671
    assert info["fps"] == 30 and info["splits"] == {"train": "0:150"}
    camera = "observation.images.cam_left_high"
    columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    arrays = []
    video_checks = []
    frame_offset = 0
    for episode_index, episode in enumerate(episodes):
        assert episode["episode_index"] == episode_index
        relative = info["data_path"].format(
            episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index
        )
        frame = pd.read_parquet(source / relative, columns=columns)
        assert len(frame) == episode["length"]
        assert np.all(frame["episode_index"].to_numpy() == episode_index)
        assert np.array_equal(frame["frame_index"].to_numpy(), np.arange(len(frame)))
        assert np.array_equal(
            frame["index"].to_numpy(), np.arange(frame_offset, frame_offset + len(frame))
        )
        for key in ["action", "observation.state"]:
            values = np.stack(frame[key])
            assert values.shape == (len(frame), 58) and np.isfinite(values).all()
        timestamps = frame["timestamp"].to_numpy()
        assert np.allclose(timestamps, np.arange(len(frame)) / 30, atol=1e-4)
        video = source / info["video_path"].format(
            episode_chunk=episode_index // info["chunks_size"],
            episode_index=episode_index,
            video_key=camera,
        )
        assert video.is_file()
        if episode_index in [0, 74, 127, 128, 149]:
            decoder = VideoDecoder(str(video), device="cpu", num_ffmpeg_threads=1)
            decoded = decoder.get_frames_at(indices=[0, len(frame) // 2, len(frame) - 1]).data
            assert tuple(decoded.shape) == (3, 3, 480, 640)
            video_checks.append(
                {"episode_index": episode_index, "decoded_shape": list(decoded.shape)}
            )
            del decoder, decoded
        arrays.append((episode_index, relative, frame, video))
        frame_offset += len(frame)

    for horizon in [16, 40]:
        MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value] = make_h2_head_config(horizon)
        target = args.output / f"h{horizon}" / "train"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite training data: {target}")
        (target / "meta").mkdir(parents=True)
        for episode_index, relative, frame, video in arrays:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False)
            video_relative = info["video_path"].format(
                episode_chunk=episode_index // info["chunks_size"],
                episode_index=episode_index,
                video_key=camera,
            )
            video_destination = target / video_relative
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            video_destination.symlink_to(video)
        filtered_info = deepcopy(info)
        filtered_info["features"] = {key: info["features"][key] for key in columns + [camera]}
        (target / "meta/info.json").write_text(json.dumps(filtered_info, indent=2) + "\n")
        (target / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
        for name in ["episodes.jsonl", "tasks.jsonl", "episode_quality.jsonl", "episode_map.csv"]:
            shutil.copy2(source / "meta" / name, target / "meta" / name)
        generate_stats(target)
        generate_rel_stats(target, EmbodimentTag.NEW_EMBODIMENT)
        result = {
            "source": str(source),
            "hf_revision": download["revision"],
            "action_horizon": horizon,
            "episodes": 150,
            "frames": frame_offset,
            "fps": 30,
            "camera": camera,
            "tactile": False,
            "split": "all 150 supplied episodes, train=0:150 as supplied",
            "statistics_source": "this dataset's 150 training episodes only",
            "video_decode_checks": video_checks,
            "statistics_sha256": {
                name: hashlib.sha256((target / "meta" / name).read_bytes()).hexdigest()
                for name in ["stats.json", "relative_stats.json"]
            },
        }
        (target.parent / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
