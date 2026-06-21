from __future__ import annotations

from constants import PARAPHRASE_PROMPT_TEMPLATE


def build_paraphrase_prompt(original_text: str) -> str:
    """Build the Spanish paraphrase instruction for one abstract."""
    return PARAPHRASE_PROMPT_TEMPLATE.format(original_text=original_text)
