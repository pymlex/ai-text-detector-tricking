from __future__ import annotations

from pydantic import BaseModel, Field


class SplitMetrics(BaseModel):
    split: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    roc_auc: float | None
    mean_logit: float
    mean_probability: float


def format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


class EvaluationReport(BaseModel):
    validation: SplitMetrics
    test: SplitMetrics
    threshold: float = Field(default=0.5)
