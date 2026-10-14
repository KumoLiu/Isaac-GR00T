"""Aggregate every available checkpoint and plot training-set open-loop errors."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "evaluation_manifest.json").read_text())
    rows = []
    for item in manifest["checkpoints"]:
        path = (
            args.directory
            / "open_loop_train150/results"
            / item["run_id"]
            / f"checkpoint-{item['checkpoint_step']}"
            / "summary.json"
        )
        if not path.exists():
            raise RuntimeError(
                f"Missing evaluation result: {item['run_id']} / {item['checkpoint_step']}"
            )
        result = json.loads(path.read_text())
        assert result["episodes"] == 150 and result["pooled"]["frames"] == 48671
        assert result["all_predictions_finite"] and result["execution_horizon"] == 16
        row = {
            "run_id": item["run_id"],
            "horizon": item["horizon"],
            "target_steps": item["target_steps"],
            "model_path": item["model_path"],
            "checkpoint_step": item["checkpoint_step"],
            "mean_episode_mse": result["mean_episode_mse"],
            "mean_episode_mae": result["mean_episode_mae"],
            "pooled_mse": result["pooled"]["mse"],
            "mean_episode_hold_state_mse": result["mean_episode_hold_state_mse"],
            **{f"{key}_mse": value for key, value in result["mean_episode_group_mse"].items()},
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["horizon"], r["target_steps"], r["checkpoint_step"]))
    assert len(rows) == manifest["count"] == 72
    assert len({(r["run_id"], r["checkpoint_step"]) for r in rows}) == 72
    baseline = rows[0]["mean_episode_hold_state_mse"]
    assert np.allclose(
        [r["mean_episode_hold_state_mse"] for r in rows], baseline, rtol=1e-10, atol=1e-12
    )
    summaries = []
    for run_id in sorted({r["run_id"] for r in rows}):
        group = [r for r in rows if r["run_id"] == run_id]
        best = min(group, key=lambda r: r["mean_episode_mse"])
        final = next(r for r in group if r["checkpoint_step"] == r["target_steps"])
        summaries.append(
            {
                "run_id": run_id,
                "evaluated_checkpoints": len(group),
                "best_training_fit_checkpoint": best,
                "final_checkpoint": final,
                "final_mse_change_from_earliest_available": 1
                - final["mean_episode_mse"] / group[0]["mean_episode_mse"],
            }
        )
    report = {
        "dataset_role": "training_set_fit_evaluation",
        "episodes_per_checkpoint": 150,
        "frames_per_checkpoint": 48671,
        "checkpoints_evaluated": len(rows),
        "execution_horizon": 16,
        "denoising_steps": 4,
        "mean_episode_hold_state_mse": baseline,
        "runs": summaries,
        "interpretation_limit": "All 150 episodes were used for training. These errors measure training-set fit, not held-out generalization or closed-loop success.",
        "all_checkpoints": rows,
    }
    (args.directory / "open_loop_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.directory / "open_loop_all_checkpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = {10000: "#2563eb", 30000: "#ea580c", 80000: "#059669"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), constrained_layout=True, sharey="row")
    for column, horizon in enumerate([16, 40]):
        for target in [10000, 30000, 80000]:
            group = [r for r in rows if r["horizon"] == horizon and r["target_steps"] == target]
            steps = [r["checkpoint_step"] for r in group]
            for index, metric in enumerate(["mean_episode_mse", "mean_episode_mae"]):
                axes[index, column].plot(
                    steps,
                    [r[metric] for r in group],
                    marker="o",
                    markersize=3.3,
                    lw=1.6,
                    color=colors[target],
                    label=f"{target // 1000}k-step run",
                )
        axes[0, column].axhline(
            baseline, color="#52525b", ls="--", lw=1, label="Hold current state baseline"
        )
        axes[0, column].set(
            title=f"H{horizon}: open-loop action MSE", ylabel="Mean episode MSE (rad²)"
        )
        axes[1, column].set(
            title=f"H{horizon}: open-loop action MAE", ylabel="Mean episode MAE (rad)"
        )
        for ax in axes[:, column]:
            ax.set(xlabel="Checkpoint optimizer step", yscale="log", xlim=(0, 82000))
            ax.grid(alpha=0.22)
        axes[0, column].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "GR00T N1.7 | 72 checkpoints | TRAINING SET open-loop evaluation",
        fontsize=15,
        weight="bold",
    )
    fig.supxlabel(
        "150 training episodes (48,671 frames) per checkpoint; 16-frame inference stride; 4 denoising steps; identical seeds.",
        fontsize=9,
    )
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(args.directory / f"open_loop_curves.{suffix}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {"checkpoints_evaluated": len(rows), "baseline_mse": baseline, "runs": summaries},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
