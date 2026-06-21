from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.collect import AnalysisSnapshot
from plotting.evaluation_plots import plot_evaluation_summary, plot_score_distributions
from utils.paths import ANALYSIS_DIR, MONITORING_DIR, PLOTS_DIR, PREFERENCES_DIR


def _analysis_plot_dir() -> Path:
    path = PLOTS_DIR / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_logit_probe_summary() -> Path | None:
    probe_path = PREFERENCES_DIR / "logit_margin_probe.csv"
    if not probe_path.exists():
        return None
    frame = pd.read_csv(probe_path)
    output_dir = _analysis_plot_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(frame["abs_logit_diff"], bins=40)
    axes[0].set_title(f"Absolute logit gap, n={len(frame)}")
    axes[0].set_xlabel("|logit_0 - logit_1|")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.5)

    chosen = frame["chosen_logit"].to_numpy(dtype=np.float64)
    rejected = frame["rejected_logit"].to_numpy(dtype=np.float64)
    bin_edges = np.linspace(min(chosen.min(), rejected.min()), max(chosen.max(), rejected.max()), 41)
    axes[1].hist(chosen, bins=bin_edges, alpha=0.6, label="chosen")
    axes[1].hist(rejected, bins=bin_edges, alpha=0.6, label="rejected")
    axes[1].set_title("Chosen vs rejected logits")
    axes[1].set_xlabel("Detector logit")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "logit_probe_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_and_monitor(snapshot: AnalysisSnapshot) -> Path | None:
    train_path = MONITORING_DIR / "train_history.csv"
    valid_path = MONITORING_DIR / "validation_history.csv"
    if not train_path.exists() and not valid_path.exists():
        return None

    output_dir = _analysis_plot_dir()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    if train_path.exists():
        train_frame = pd.read_csv(train_path).dropna(subset=["loss"])
        if not train_frame.empty:
            x_values = train_frame["epoch"] if "epoch" in train_frame.columns else train_frame.index
            x_label = "Epoch" if "epoch" in train_frame.columns else "Step"
            axes[0].plot(x_values, train_frame["loss"], marker="o", markersize=3)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel("DPO loss")
            axes[0].set_title("Training loss")
            axes[0].grid(alpha=0.5)

    if valid_path.exists():
        valid_frame = pd.read_csv(valid_path)
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
    path = output_dir / "training_monitor_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_analysis_plots(snapshot: AnalysisSnapshot) -> list[Path]:
    plotters = [
        plot_logit_probe_summary,
        lambda: plot_training_and_monitor(snapshot),
        plot_evaluation_summary,
        plot_score_distributions,
    ]
    paths: list[Path] = []
    for plotter in plotters:
        result = plotter()
        if result is not None:
            paths.append(result)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ANALYSIS_DIR / "analysis_plots.json"
    manifest.write_text(
        json.dumps([str(path) for path in paths], indent=2),
        encoding="utf-8",
    )
    return paths
