from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from trl import DPOTrainer

from utils.paths import MONITORING_DIR


def save_training_history(
    trainer: DPOTrainer,
    validation_history: list[dict],
    output_dir: Path | None = None,
) -> Path:
    """Persist train and validation histories for downstream plotting."""
    output_dir = output_dir or MONITORING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_logs = list(trainer.state.log_history)
    train_json = output_dir / "train_history.json"
    train_csv = output_dir / "train_history.csv"
    train_json.write_text(json.dumps(train_logs, indent=2), encoding="utf-8")
    pd.DataFrame(train_logs).to_csv(train_csv, index=False)

    valid_json = output_dir / "validation_history.json"
    valid_csv = output_dir / "validation_history.csv"
    valid_json.write_text(json.dumps(validation_history, indent=2), encoding="utf-8")
    pd.DataFrame(validation_history).to_csv(valid_csv, index=False)

    combined_path = output_dir / "training_history.json"
    combined_path.write_text(
        json.dumps({"train": train_logs, "validation": validation_history}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved train history: {train_csv}", flush=True)
    print(f"Saved validation history: {valid_csv}", flush=True)
    print(f"Saved combined history: {combined_path}", flush=True)
    return combined_path
