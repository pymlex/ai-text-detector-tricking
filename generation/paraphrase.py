from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from constants import GENERATION_BATCH_SIZE, GENERATION_TEMPERATURE, MAX_NEW_TOKENS
from generation.prompts import build_paraphrase_prompt


def _build_chat_inputs(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    return tokenizer(rendered, return_tensors="pt", padding=True).to(device)


def generate_paraphrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    original_texts: list[str],
    device: torch.device,
    num_samples: int = 2,
    show_progress: bool = True,
) -> list[list[str]]:
    """Generate ``num_samples`` paraphrases per source text."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs_per_text: list[list[str]] = [[] for _ in original_texts]
    batch_range = range(0, len(original_texts), GENERATION_BATCH_SIZE)
    for sample_idx in range(num_samples):
        iterator = batch_range
        if show_progress:
            iterator = tqdm(
                batch_range,
                desc=f"paraphrase sample {sample_idx + 1}/{num_samples}",
            )
        for start in iterator:
            chunk = original_texts[start : start + GENERATION_BATCH_SIZE]
            prompts = [build_paraphrase_prompt(text) for text in chunk]
            inputs = _build_chat_inputs(tokenizer, prompts, device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=GENERATION_TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row_idx, row in enumerate(generated):
                decoded = tokenizer.decode(row[input_len:], skip_special_tokens=True).strip()
                outputs_per_text[start + row_idx].append(decoded)
    return outputs_per_text
