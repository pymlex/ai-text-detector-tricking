from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import DETECTION_THRESHOLD, MODEL_ID, TEXT_COLUMN
from data.prepare import load_filtered_splits
from detector.scoring import OculusDetector
from evaluation.metrics import (
    build_evaluation_report,
    save_confusion_matrix,
    save_evaluation_report,
)
from generation.paraphrase import generate_paraphrases
from utils.paths import METRICS_DIR, PLOTS_DIR, ensure_result_dirs


def _plot_probability_hist(
    split: str,
    probabilities: np.ndarray,
    output_dir: Path,
) -> Path:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(probabilities, bins=70)
    axis.set_title(f"Detector AI probability on {split} paraphrases")
    axis.set_xlabel("AI probability")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.5)
    fig.tight_layout()
    path = output_dir / f"detector_probability_hist_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_confusion_matrix(split: str, matrix_path: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(matrix_path, index_col=0)
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", ax=axis, cbar=False)
    axis.set_title(f"Confusion matrix on {split}")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    fig.tight_layout()
    path = output_dir / f"confusion_matrix_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _collect_split_scores(
    model_path: str,
    output_tag: str,
    splits: dict,
    detector: OculusDetector,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    split_probs: dict[str, np.ndarray] = {}
    split_logits: dict[str, np.ndarray] = {}

    for split_name in ("validation", "test"):
        texts = splits[split_name][TEXT_COLUMN]
        paraphrases = [
            group[0]
            for group in generate_paraphrases(
                model=model,
                tokenizer=tokenizer,
                original_texts=texts,
                device=device,
                num_samples=1,
            )
        ]
        logits = np.array(detector.batch_logits(paraphrases), dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        split_probs[split_name] = probabilities
        split_logits[split_name] = logits

        frame = pd.DataFrame(
            {
                "original_text": texts,
                "paraphrase": paraphrases,
                "detector_logit": logits,
                "detector_probability": probabilities,
            }
        )
        csv_path = METRICS_DIR / f"{output_tag}_{split_name}_scores.csv"
        frame.to_csv(csv_path, index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return split_probs, split_logits


def evaluate_model(
    model_path: str | Path,
    device: torch.device | None = None,
    output_tag: str = "final",
) -> Path:
    """Generate paraphrases on validation and test, then score with Oculus."""
    ensure_result_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(model_path)

    splits = load_filtered_splits()
    detector = OculusDetector(device=device)

    split_probs, split_logits = _collect_split_scores(
        model_path,
        output_tag,
        splits,
        detector,
        device,
    )

    for split_name in ("validation", "test"):
        _plot_probability_hist(split_name, split_probs[split_name], PLOTS_DIR)

    report = build_evaluation_report(
        split_probs["validation"],
        split_logits["validation"],
        split_probs["test"],
        split_logits["test"],
        DETECTION_THRESHOLD,
    )
    report_path = save_evaluation_report(report, METRICS_DIR)
    for split_name in ("validation", "test"):
        matrix_csv = save_confusion_matrix(
            split_name,
            split_probs[split_name],
            DETECTION_THRESHOLD,
            METRICS_DIR,
        )
        _plot_confusion_matrix(split_name, matrix_csv, PLOTS_DIR)

    base_probs, base_logits = _collect_split_scores(
        MODEL_ID,
        "base",
        splits,
        detector,
        device,
    )
    base_report = build_evaluation_report(
        base_probs["validation"],
        base_logits["validation"],
        base_probs["test"],
        base_logits["test"],
        DETECTION_THRESHOLD,
    )
    base_report_path = METRICS_DIR / "base_evaluation_report.json"
    base_report_path.write_text(base_report.model_dump_json(indent=2), encoding="utf-8")

    print(report.model_dump_json(indent=2))
    print(f"Saved evaluation report: {report_path}")
    print(f"Saved base evaluation report: {base_report_path}")
    return report_path
