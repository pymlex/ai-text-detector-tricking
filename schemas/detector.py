from __future__ import annotations

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class DetectResponse(BaseModel):
    logits: list[float]
    probabilities: list[float]
