from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import MONITORING_DIR, PLOTS_DIR


def plot_training_summary(monitoring_dir: Path | None = None, plots_dir: Path | None = None) -> Path:
    monitoring_dir = monitoring_dir or MONITORING_DIR
    plots_dir = plots_dir or PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_path = monitoring_dir / "train_history.csv"
    valid_path = monitoring_dir / "validation_history.csv"
    if not train_path.exists() and not valid_path.exists():
        return plots_dir / "training_summary.png"

    train_frame = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    valid_frame = pd.read_csv(valid_path) if valid_path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    if not train_frame.empty and "loss" in train_frame.columns:
        train_plot = train_frame.dropna(subset=["loss"])
        if not train_plot.empty:
            x_values = train_plot["epoch"] if "epoch" in train_plot.columns else train_plot.index
            x_label = "Epoch" if "epoch" in train_plot.columns else "Step"
            axes[0].plot(x_values, train_plot["loss"], marker="o", markersize=3)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel("DPO loss")
            axes[0].set_title("Training loss")
            axes[0].grid(alpha=0.5)

    if not valid_frame.empty:
        axes[1].plot(
            valid_frame["step"],
            valid_frame["mean_probability"],
            marker="o",
            color="tab:blue",
        )
        axes[1].set_ylabel("Mean AI probability")
        axes[1].set_title("Validation mean detector probability")
        axes[1].grid(alpha=0.5)
        axes[2].plot(
            valid_frame["step"],
            valid_frame["mean_logit"],
            marker="s",
            color="tab:orange",
        )
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Mean logit")
        axes[2].set_title("Validation mean detector logit")
        axes[2].grid(alpha=0.5)

    fig.tight_layout()
    output_path = plots_dir / "training_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
