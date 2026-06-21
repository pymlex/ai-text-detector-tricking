from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import MONITORING_DIR, PLOTS_DIR


def plot_training_summary(monitoring_dir: Path | None = None, plots_dir: Path | None = None) -> Path:
    """Build summary figures from saved train and validation histories."""
    monitoring_dir = monitoring_dir or MONITORING_DIR
    plots_dir = plots_dir or PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_path = monitoring_dir / "train_history.csv"
    valid_path = monitoring_dir / "validation_history.csv"
    if not train_path.exists() and not valid_path.exists():
        legacy_logs = sorted(monitoring_dir.glob("monitor_step_*.json"))
        if not legacy_logs:
            return plots_dir / "training_summary.png"

    train_frame = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    valid_frame = pd.read_csv(valid_path) if valid_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    if not train_frame.empty and "loss" in train_frame.columns:
        train_plot = train_frame.dropna(subset=["loss"])
        if "step" not in train_plot.columns:
            train_plot = train_plot.reset_index().rename(columns={"index": "step"})
        axes[0].plot(train_plot["step"], train_plot["loss"], marker="o", markersize=3)
        axes[0].set_ylabel("DPO loss")
        axes[0].set_title("Training history")
        axes[0].grid(alpha=0.5)

        if "rewards/accuracies" in train_plot.columns:
            axes[1].plot(
                train_plot["step"],
                train_plot["rewards/accuracies"],
                marker="o",
                markersize=3,
                color="tab:green",
            )
            axes[1].set_ylabel("Reward accuracy")
            axes[1].grid(alpha=0.5)

    if not valid_frame.empty:
        axes[2].plot(
            valid_frame["step"],
            valid_frame["mean_probability"],
            marker="o",
            color="tab:red",
            label="mean_prob",
        )
        axes[2].plot(
            valid_frame["step"],
            valid_frame["mean_logit"],
            marker="s",
            color="tab:orange",
            label="mean_logit",
        )
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Detector score")
        axes[2].legend()
        axes[2].grid(alpha=0.5)

    fig.tight_layout()
    output_path = plots_dir / "training_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
