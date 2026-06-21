from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from constants import PREFERENCE_LOGIT_MARGIN
from schemas.evaluation import EvaluationReport, SplitMetrics
from utils.paths import (
    ANALYSIS_DIR,
    CARDS_DIR,
    METRICS_DIR,
    MONITORING_DIR,
    PLOTS_DIR,
    PREFERENCES_DIR,
)


class LogitProbeStats(BaseModel):
    n_samples: int
    mean_abs_logit_diff: float
    median_abs_logit_diff: float
    p25_abs_logit_diff: float
    p75_abs_logit_diff: float
    max_abs_logit_diff: float


class PreferenceStats(BaseModel):
    train_rows: int
    pairs_built: int
    pairs_skipped_tie: int
    pairs_skipped_empty: int
    logit_margin_threshold: float


class TrainingMonitorStats(BaseModel):
    n_monitor_samples: int | None
    step0_mean_probability: float | None
    final_mean_probability: float | None
    step0_mean_logit: float | None
    final_mean_logit: float | None
    final_loss: float | None
    final_reward_accuracy: float | None


class AnalysisSnapshot(BaseModel):
    preference_stats: PreferenceStats
    logit_probe: LogitProbeStats | None
    training_monitor: TrainingMonitorStats
    evaluation: EvaluationReport | None


def _load_logit_probe_stats() -> LogitProbeStats | None:
    probe_path = PREFERENCES_DIR / "logit_margin_probe.csv"
    if not probe_path.exists():
        return None
    frame = pd.read_csv(probe_path)
    gaps = frame["abs_logit_diff"].to_numpy(dtype=np.float64)
    return LogitProbeStats(
        n_samples=int(len(gaps)),
        mean_abs_logit_diff=float(gaps.mean()),
        median_abs_logit_diff=float(np.median(gaps)),
        p25_abs_logit_diff=float(np.percentile(gaps, 25)),
        p75_abs_logit_diff=float(np.percentile(gaps, 75)),
        max_abs_logit_diff=float(gaps.max()),
    )


def _load_preference_stats() -> PreferenceStats:
    stats_path = PREFERENCES_DIR / "build_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return PreferenceStats(
        train_rows=int(payload["train_rows"]),
        pairs_built=int(payload["pairs_built"]),
        pairs_skipped_tie=int(payload["pairs_skipped_tie"]),
        pairs_skipped_empty=int(payload["pairs_skipped_empty"]),
        logit_margin_threshold=PREFERENCE_LOGIT_MARGIN,
    )


def _load_training_monitor_stats() -> TrainingMonitorStats:
    valid_path = MONITORING_DIR / "validation_history.csv"
    train_path = MONITORING_DIR / "train_history.csv"
    valid_frame = pd.read_csv(valid_path) if valid_path.exists() else pd.DataFrame()
    train_frame = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()

    n_monitor = None
    step0_prob = None
    step0_logit = None
    final_prob = None
    final_logit = None
    if not valid_frame.empty and "n_samples" in valid_frame.columns:
        n_monitor = int(valid_frame.iloc[0]["n_samples"])
    if not valid_frame.empty:
        step0_prob = float(valid_frame.iloc[0]["mean_probability"])
        step0_logit = float(valid_frame.iloc[0]["mean_logit"])
        final_prob = float(valid_frame.iloc[-1]["mean_probability"])
        final_logit = float(valid_frame.iloc[-1]["mean_logit"])

    final_loss = None
    final_reward_accuracy = None
    if not train_frame.empty and "loss" in train_frame.columns:
        train_plot = train_frame.dropna(subset=["loss"])
        if not train_plot.empty:
            final_loss = float(train_plot.iloc[-1]["loss"])
        if "rewards/accuracies" in train_frame.columns:
            reward_plot = train_frame.dropna(subset=["rewards/accuracies"])
            if not reward_plot.empty:
                final_reward_accuracy = float(reward_plot.iloc[-1]["rewards/accuracies"])

    return TrainingMonitorStats(
        n_monitor_samples=n_monitor,
        step0_mean_probability=step0_prob,
        final_mean_probability=final_prob,
        step0_mean_logit=step0_logit,
        final_mean_logit=final_logit,
        final_loss=final_loss,
        final_reward_accuracy=final_reward_accuracy,
    )


def _load_evaluation_report() -> EvaluationReport | None:
    report_path = METRICS_DIR / "evaluation_report.json"
    if not report_path.exists():
        return None
    return EvaluationReport.model_validate_json(report_path.read_text(encoding="utf-8"))


def collect_analysis_snapshot() -> AnalysisSnapshot:
    """Aggregate metrics and statistics from saved artefacts."""
    return AnalysisSnapshot(
        preference_stats=_load_preference_stats(),
        logit_probe=_load_logit_probe_stats(),
        training_monitor=_load_training_monitor_stats(),
        evaluation=_load_evaluation_report(),
    )


def save_analysis_snapshot(snapshot: AnalysisSnapshot) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "analysis_snapshot.json"
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return path
