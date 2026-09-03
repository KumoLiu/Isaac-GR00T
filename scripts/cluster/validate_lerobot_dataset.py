#!/usr/bin/env python3
"""Validate the GR00T-specific essentials of a local LeRobot v2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "observation.state",
    "action",
    "timestamp",
    "episode_index",
    "index",
    "task_index",
    "next.reward",
    "next.done",
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    meta = dataset_path / "meta"

    for relative_path in (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
    ):
        require_file(dataset_path / relative_path)

    modality_path = meta / "modality.json"
    with modality_path.open() as file:
        modality = json.load(file)
    for section in ("state", "action", "video"):
        if not modality.get(section):
            raise ValueError(f"{modality_path} has no '{section}' configuration")

    parquet_files = sorted((dataset_path / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_path / 'data'}")

    schema = pq.read_schema(parquet_files[0])
    missing = REQUIRED_COLUMNS.difference(schema.names)
    if missing:
        raise ValueError(
            f"{parquet_files[0]} is missing required columns: {', '.join(sorted(missing))}"
        )

    annotation_keys = modality.get("annotation", {})
    missing_annotations = {
        f"annotation.{key}" for key in annotation_keys if f"annotation.{key}" not in schema.names
    }
    if missing_annotations:
        raise ValueError(
            "modality.json annotation keys are absent from the first parquet schema: "
            + ", ".join(sorted(missing_annotations))
        )

    video_directories = [
        path for path in (dataset_path / "videos").glob("chunk-*/observation.images.*") if path.is_dir()
    ]
    if not video_directories:
        raise FileNotFoundError(f"No LeRobot video directories found under {dataset_path / 'videos'}")

    print(f"Dataset: {dataset_path}")
    print(f"First episode: {parquet_files[0]}")
    print(f"Parquet columns: {', '.join(schema.names)}")
    print(f"Modality state keys: {', '.join(modality['state'])}")
    print(f"Modality action keys: {', '.join(modality['action'])}")
    print(f"Modality video keys: {', '.join(modality['video'])}")
    print("GR00T LeRobot structural validation passed.")


if __name__ == "__main__":
    main()
