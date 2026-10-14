"""Plot full, deduplicated trainer histories and summarize late training behavior."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def smooth(values, window=20):
    sums = np.cumsum(np.r_[0.0, values])
    starts = np.maximum(0, np.arange(len(values)) - window + 1)
    return (sums[np.arange(len(values)) + 1] - sums[starts]) / (np.arange(len(values)) + 1 - starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.directory / "loss_history.json").read_text())
    colors = {10000: "#2563eb", 30000: "#ea580c", 80000: "#059669"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), constrained_layout=True)
    summaries = []
    for run in data["runs"]:
        step = np.array([r["step"] for r in run["history"]])
        loss = np.array([r["loss"] for r in run["history"]])
        assert np.isfinite(loss).all() and (loss > 0).all()
        assert np.array_equal(step, np.arange(10, run["max_steps"] + 1, 10))
        horizon = run["horizon"]
        target = run["max_steps"]
        column = 0 if horizon == 16 else 1
        color = colors[target]
        moving = smooth(loss)
        axes[0, column].plot(step, loss, color=color, alpha=0.075, lw=0.65, rasterized=True)
        axes[0, column].plot(step, moving, color=color, lw=1.9, label=f"{target // 1000}k steps")
        tail = step >= 0.8 * target
        axes[1, column].plot(100 * step[tail] / target, moving[tail], color=color, lw=1.9)
        last = loss[step > target - 1000]
        previous = loss[(step > target - 2000) & (step <= target - 1000)]
        last20 = loss[step > 0.8 * target]
        previous20 = loss[(step > 0.6 * target) & (step <= 0.8 * target)]
        summaries.append(
            {
                "run_id": run["run_id"],
                "horizon": horizon,
                "max_steps": target,
                "loss_records": len(loss),
                "first_1000_mean": float(loss[step <= 1000].mean()),
                "last_logged_loss": float(loss[-1]),
                "last_1000_mean": float(last.mean()),
                "last_1000_median": float(np.median(last)),
                "last_1000_std": float(last.std()),
                "previous_1000_mean": float(previous.mean()),
                "last_1000_relative_improvement": float(1 - last.mean() / previous.mean()),
                "last_20pct_mean": float(last20.mean()),
                "previous_20pct_mean": float(previous20.mean()),
                "last_20pct_relative_improvement": float(1 - last20.mean() / previous20.mean()),
                "nonfinite_records": 0,
            }
        )
    for column, horizon in enumerate([16, 40]):
        axes[0, column].set(
            title=f"H{horizon}: complete training",
            xlabel="Optimizer step",
            ylabel="Training loss (log scale)",
            xlim=(0, 80000),
            yscale="log",
            ylim=(0.002, 2),
        )
        axes[0, column].legend(frameon=False, loc="upper right")
        axes[0, column].ticklabel_format(axis="x", style="plain")
        axes[1, column].set(
            title=f"H{horizon}: last 20% of each run",
            xlabel="Progress within each run (%)",
            ylabel="Training loss (log scale)",
            xlim=(80, 100),
            yscale="log",
            ylim=(0.002, 0.06),
        )
        for ax in axes[:, column]:
            ax.grid(alpha=0.22, which="major")
    fig.suptitle("GR00T N1.7 | medical pads 150 | global batch 64", fontsize=15, weight="bold")
    fig.supxlabel(
        "Bold curves: trailing 200-step mean. Faint curves: raw logged loss. Each run has its own cosine schedule.",
        fontsize=9,
    )
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(args.directory / f"loss_curves.{suffix}", dpi=180)
    plt.close(fig)
    (args.directory / "loss_summary.json").write_text(
        json.dumps(
            {
                "runs": summaries,
                "method": "Full final trainer_state history; logged every 10 optimizer steps; smoothing over 200 steps; tail means over 1000 steps.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
