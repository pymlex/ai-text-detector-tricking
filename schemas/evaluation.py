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
    roc_auc: float
    mean_logit: float
    mean_probability: float


class EvaluationReport(BaseModel):
    validation: SplitMetrics
    test: SplitMetrics
    threshold: float = Field(default=0.5)
