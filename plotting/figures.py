from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_summary(monitoring_dir: Path, plots_dir: Path) -> Path:
    """Build a summary figure from validation monitoring logs."""
    logs = sorted(monitoring_dir.glob("monitor_step_*.json"))
    if not logs:
        return plots_dir / "training_summary.png"

    rows = []
    for log_path in logs:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        rows.append(payload)

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(frame["step"], frame["mean_probability"], marker="o")
    axes[0].set_ylabel("Mean AI probability")
    axes[0].set_title("Validation detector score during DPO")
    axes[0].grid(alpha=0.5)

    axes[1].plot(frame["step"], frame["mean_logit"], marker="o", color="tab:orange")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Mean logit")
    axes[1].grid(alpha=0.5)

    fig.tight_layout()
    output_path = plots_dir / "training_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
