from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.collect import AnalysisSnapshot
from utils.paths import ANALYSIS_DIR, METRICS_DIR, MONITORING_DIR, PLOTS_DIR, PREFERENCES_DIR


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
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    if train_path.exists():
        train_frame = pd.read_csv(train_path).dropna(subset=["loss"])
        if not train_frame.empty:
            steps = train_frame["step"] if "step" in train_frame.columns else train_frame.index
            axes[0].plot(steps, train_frame["loss"], marker="o", markersize=3)
            axes[0].set_ylabel("DPO loss")
            axes[0].set_title("Training loss")
            axes[0].grid(alpha=0.5)
            if "rewards/accuracies" in train_frame.columns:
                reward_frame = train_frame.dropna(subset=["rewards/accuracies"])
                if not reward_frame.empty:
                    reward_steps = (
                        reward_frame["step"]
                        if "step" in reward_frame.columns
                        else reward_frame.index
                    )
                    axes[1].plot(
                        reward_steps,
                        reward_frame["rewards/accuracies"],
                        marker="o",
                        markersize=3,
                        color="tab:green",
                    )
            axes[1].set_ylabel("Reward accuracy")
            axes[1].set_title("Reward accuracy")
            axes[1].grid(alpha=0.5)

    if valid_path.exists():
        valid_frame = pd.read_csv(valid_path)
        if not valid_frame.empty:
            axes[2].plot(
                valid_frame["step"],
                valid_frame["mean_probability"],
                marker="o",
                color="tab:blue",
            )
            axes[2].set_ylabel("Mean AI probability")
            axes[2].set_title("Validation mean detector probability")
            axes[2].grid(alpha=0.5)
            axes[3].plot(
                valid_frame["step"],
                valid_frame["mean_logit"],
                marker="s",
                color="tab:orange",
            )
            axes[3].set_xlabel("Training step")
            axes[3].set_ylabel("Mean logit")
            axes[3].set_title("Validation mean detector logit")
            axes[3].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "training_monitor_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_evaluation_summary() -> Path | None:
    report_path = METRICS_DIR / "evaluation_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = ["validation", "test"]
    x = np.arange(len(labels))
    width = 0.55

    output_dir = _analysis_plot_dir()
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    prob_values = [float(report[split]["mean_probability"]) for split in labels]
    axes[0].bar(x, prob_values, width=width, color="tab:blue")
    axes[0].set_ylabel("Mean AI probability")
    axes[0].set_title("Mean detector probability on paraphrases")
    axes[0].set_xticks(x, labels)
    axes[0].grid(alpha=0.5)

    logit_values = [float(report[split]["mean_logit"]) for split in labels]
    axes[1].bar(x, logit_values, width=width, color="tab:orange")
    axes[1].set_ylabel("Mean logit")
    axes[1].set_title("Mean detector logit on paraphrases")
    axes[1].set_xticks(x, labels)
    axes[1].grid(alpha=0.5)

    mcc_values = [float(report[split]["mcc"]) for split in labels]
    f1_values = [float(report[split]["f1"]) for split in labels]
    roc_values = [
        float(report[split]["roc_auc"]) if report[split]["roc_auc"] is not None else np.nan
        for split in labels
    ]
    metric_width = width / 3
    axes[2].bar(x - metric_width, mcc_values, width=metric_width, label="MCC")
    axes[2].bar(x, f1_values, width=metric_width, label="F1")
    axes[2].bar(x + metric_width, roc_values, width=metric_width, label="ROC-AUC")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Classification metrics, label AI, threshold 0.5")
    axes[2].set_xticks(x, labels)
    axes[2].legend()
    axes[2].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "evaluation_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_distributions() -> Path | None:
    paths = {
        "validation": METRICS_DIR / "final_validation_scores.csv",
        "test": METRICS_DIR / "final_test_scores.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items() if path.exists()}
    if not frames:
        return None

    output_dir = _analysis_plot_dir()
    split_names = list(frames.keys())
    fig, axes = plt.subplots(2, len(split_names), figsize=(6 * len(split_names), 8), sharex="col")
    if len(split_names) == 1:
        axes = np.array(axes).reshape(2, 1)

    for column_index, split_name in enumerate(split_names):
        frame = frames[split_name]
        axes[0, column_index].hist(frame["detector_logit"], bins=50, color="tab:orange")
        axes[0, column_index].set_title(f"{split_name}, logit, n={len(frame)}")
        axes[0, column_index].set_ylabel("Count")
        axes[0, column_index].grid(alpha=0.5)
        axes[1, column_index].hist(frame["detector_probability"], bins=50, color="tab:blue")
        axes[1, column_index].set_title(f"{split_name}, probability, n={len(frame)}")
        axes[1, column_index].set_xlabel("Detector score")
        axes[1, column_index].set_ylabel("Count")
        axes[1, column_index].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "score_distributions.png"
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
