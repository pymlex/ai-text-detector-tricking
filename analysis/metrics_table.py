from __future__ import annotations

from analysis.collect import AnalysisSnapshot
from schemas.evaluation import EvaluationReport, SplitMetrics, format_optional_float


def _format_row(model_label: str, split_name: str, metrics: SplitMetrics) -> str:
    return (
        f"| {model_label} | {split_name} | {metrics.n_samples} | "
        f"{metrics.mean_probability:.4f} | {metrics.mean_logit:.4f} | "
        f"{metrics.accuracy:.4f} | {metrics.mcc:.4f} | "
        f"{format_optional_float(metrics.roc_auc)} | {metrics.f1:.4f} |"
    )


def _format_report_rows(model_label: str, report: EvaluationReport) -> list[str]:
    return [
        _format_row(model_label, "validation", report.validation),
        _format_row(model_label, "test", report.test),
    ]


def render_evaluation_metrics_table(snapshot: AnalysisSnapshot) -> str:
    if snapshot.evaluation is None and snapshot.base_evaluation is None:
        return "_Evaluation metrics pending. Run `python main.py --step evaluate` first._"

    header = (
        "| Model | Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows: list[str] = [header]
    if snapshot.base_evaluation is not None:
        rows.extend(_format_report_rows("base", snapshot.base_evaluation))
    if snapshot.evaluation is not None:
        rows.extend(_format_report_rows("fine-tuned", snapshot.evaluation))
    return "\n".join(rows)
