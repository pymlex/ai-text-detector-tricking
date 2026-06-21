from __future__ import annotations

from pydantic import BaseModel, Field


class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    chosen_logit: float
    rejected_logit: float
    chosen_probability: float
    rejected_probability: float
    original_text: str


class PreferenceBuildStats(BaseModel):
    train_rows: int
    pairs_built: int
    pairs_skipped_tie: int
    pairs_skipped_empty: int
