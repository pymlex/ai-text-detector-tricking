from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from schemas.evaluation import EvaluationReport, SplitMetrics


def compute_split_metrics(
    split: str,
    probabilities: np.ndarray,
    logits: np.ndarray,
    threshold: float,
) -> SplitMetrics:
    """Compute detector classification metrics against AI label 1."""
    labels = np.ones_like(probabilities, dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    return SplitMetrics(
        split=split,
        n_samples=int(len(probabilities)),
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        mcc=float(matthews_corrcoef(labels, predictions)),
        roc_auc=float(roc_auc_score(labels, probabilities)),
        mean_logit=float(logits.mean()),
        mean_probability=float(probabilities.mean()),
    )


def save_confusion_matrix(
    split: str,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> Path:
    """Persist confusion matrix values for one split."""
    labels = np.ones_like(probabilities, dtype=np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    frame = pd.DataFrame(
        matrix,
        index=["true_human", "true_ai"],
        columns=["pred_human", "pred_ai"],
    )
    path = output_dir / f"confusion_matrix_{split}.csv"
    frame.to_csv(path)
    return path


def build_evaluation_report(
    validation_probs: np.ndarray,
    validation_logits: np.ndarray,
    test_probs: np.ndarray,
    test_logits: np.ndarray,
    threshold: float,
) -> EvaluationReport:
    """Aggregate validation and test metrics."""
    return EvaluationReport(
        validation=compute_split_metrics(
            "validation",
            validation_probs,
            validation_logits,
            threshold,
        ),
        test=compute_split_metrics("test", test_probs, test_logits, threshold),
        threshold=threshold,
    )


def save_evaluation_report(report: EvaluationReport, output_dir: Path) -> Path:
    """Write evaluation metrics to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_preference_dataset(path: Path | None = None) -> Dataset:
    """Load the fixed DPO preference dataset from disk."""
    default_path = Path("results/preferences/dpo_hf_dataset")
    dataset_path = path or default_path
    return Dataset.load_from_disk(str(dataset_path))
