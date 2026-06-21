from __future__ import annotations

import torch

from constants import PREFERENCE_LOGIT_MARGIN
from schemas.preferences import PreferencePair


def logit_abs_margin(logits: list[float]) -> float:
    """Return absolute logit gap between two paraphrases."""
    return abs(logits[0] - logits[1])


def select_preference_pair(
    original_text: str,
    prompt: str,
    paraphrases: list[str],
    logits: list[float],
) -> PreferencePair | None:
    """Rank two paraphrases by detector logit and apply the logit margin filter."""
    if len(paraphrases) < 2:
        return None
    if not paraphrases[0].strip() or not paraphrases[1].strip():
        return None
    if logit_abs_margin(logits) < PREFERENCE_LOGIT_MARGIN:
        return None
    if logits[0] <= logits[1]:
        chosen_idx, rejected_idx = 0, 1
    else:
        chosen_idx, rejected_idx = 1, 0
    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.sigmoid(logit_tensor).numpy().tolist()
    return PreferencePair(
        prompt=prompt,
        chosen=paraphrases[chosen_idx],
        rejected=paraphrases[rejected_idx],
        chosen_logit=logits[chosen_idx],
        rejected_logit=logits[rejected_idx],
        chosen_probability=probabilities[chosen_idx],
        rejected_probability=probabilities[rejected_idx],
        original_text=original_text,
    )
