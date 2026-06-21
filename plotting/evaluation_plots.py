from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import METRICS_DIR, PLOTS_DIR


def _analysis_plot_dir() -> Path:
    path = PLOTS_DIR / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_evaluation_json(filename: str) -> dict | None:
    report_path = METRICS_DIR / filename
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def plot_evaluation_summary() -> Path | None:
    final_report = _load_evaluation_json("evaluation_report.json")
    base_report = _load_evaluation_json("base_evaluation_report.json")
    if final_report is None and base_report is None:
        return None

    splits = ["validation", "test"]
    x = np.arange(len(splits))
    bar_width = 0.35
    output_dir = _analysis_plot_dir()
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    def _split_values(report: dict, field: str) -> list[float]:
        values = []
        for split in splits:
            raw = report[split][field]
            values.append(float(raw) if raw is not None else np.nan)
        return values

    for report, label, offset, color in (
        (base_report, "base", -bar_width / 2, "tab:gray"),
        (final_report, "fine-tuned", bar_width / 2, "tab:blue"),
    ):
        if report is None:
            continue
        axes[0].bar(
            x + offset,
            _split_values(report, "mean_probability"),
            width=bar_width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset,
            _split_values(report, "mean_logit"),
            width=bar_width,
            label=label,
            color=color,
        )
        axes[2].bar(
            x + offset,
            _split_values(report, "mcc"),
            width=bar_width,
            label=f"{label}, MCC",
            color=color,
            alpha=0.85,
        )

    axes[0].set_ylabel("Mean AI probability")
    axes[0].set_title("Mean detector probability on paraphrases")
    axes[0].set_xticks(x, splits)
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    axes[1].set_ylabel("Mean logit")
    axes[1].set_title("Mean detector logit on paraphrases")
    axes[1].set_xticks(x, splits)
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    axes[2].set_ylabel("MCC")
    axes[2].set_title("MCC, label AI, threshold 0.5")
    axes[2].set_xticks(x, splits)
    axes[2].legend()
    axes[2].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "evaluation_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _load_score_frame(tag: str, split_name: str) -> pd.DataFrame | None:
    path = METRICS_DIR / f"{tag}_{split_name}_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_score_distributions() -> Path | None:
    splits = ["validation", "test"]
    final_frames = {split: _load_score_frame("final", split) for split in splits}
    base_frames = {split: _load_score_frame("base", split) for split in splits}
    if all(frame is None for frame in final_frames.values()) and all(
        frame is None for frame in base_frames.values()
    ):
        return None

    output_dir = _analysis_plot_dir()
    fig, axes = plt.subplots(2, len(splits), figsize=(6 * len(splits), 8), sharex="col")
    if len(splits) == 1:
        axes = np.array(axes).reshape(2, 1)

    for column_index, split_name in enumerate(splits):
        final_frame = final_frames[split_name]
        base_frame = base_frames[split_name]
        logit_values = [
            frame["detector_logit"].to_numpy(dtype=np.float64)
            for frame in (final_frame, base_frame)
            if frame is not None
        ]
        prob_values = [
            frame["detector_probability"].to_numpy(dtype=np.float64)
            for frame in (final_frame, base_frame)
            if frame is not None
        ]
        if not logit_values:
            continue

        logit_edges = np.linspace(
            min(value.min() for value in logit_values),
            max(value.max() for value in logit_values),
            51,
        )
        prob_edges = np.linspace(
            min(value.min() for value in prob_values),
            max(value.max() for value in prob_values),
            51,
        )

        if base_frame is not None:
            axes[0, column_index].hist(
                base_frame["detector_logit"],
                bins=logit_edges,
                alpha=0.55,
                label="base",
                color="tab:gray",
            )
            axes[1, column_index].hist(
                base_frame["detector_probability"],
                bins=prob_edges,
                alpha=0.55,
                label="base",
                color="tab:gray",
            )
        if final_frame is not None:
            axes[0, column_index].hist(
                final_frame["detector_logit"],
                bins=logit_edges,
                alpha=0.55,
                label="fine-tuned",
                color="tab:orange",
            )
            axes[1, column_index].hist(
                final_frame["detector_probability"],
                bins=prob_edges,
                alpha=0.55,
                label="fine-tuned",
                color="tab:blue",
            )

        sample_count = len(final_frame) if final_frame is not None else len(base_frame)
        axes[0, column_index].set_title(f"{split_name}, logit, n={sample_count}")
        axes[0, column_index].set_ylabel("Count")
        axes[0, column_index].legend()
        axes[0, column_index].grid(alpha=0.5)
        axes[1, column_index].set_title(f"{split_name}, probability, n={sample_count}")
        axes[1, column_index].set_xlabel("Detector score")
        axes[1, column_index].set_ylabel("Count")
        axes[1, column_index].legend()
        axes[1, column_index].grid(alpha=0.5)

    fig.tight_layout()
    path = output_dir / "score_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
