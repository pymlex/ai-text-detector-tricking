---
license: gpl-3.0
language:
- es
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- dpo
- ai-detection
- paraphrase
library_name: transformers
---

# Qwen2.5-0.5B-Human

DPO fine-tune of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) that paraphrases Spanish academic abstracts to reduce AI-detection scores from [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual).

## Training objective

Preference pairs are built before optimisation. For each train abstract, two base-model paraphrases are ranked by detector AI probability. DPO with beta = 0.1 increases the relative log-probability of the lower-scoring paraphrase.

## Intended use

Research on detector robustness and red-teaming of AI-generated text classifiers under controlled laboratory settings. Not for academic misconduct or deceptive publishing.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "pymlex/Qwen2.5-0.5B-Human"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

original = "Texto académico en español."
prompt = (
    "Parafrasea el siguiente texto manteniendo el significado, "
    "sin añadir comentarios ni explicaciones adicionales. "
    f"Texto original: {original}"
)
messages = [{"role": "user", "content": prompt}]
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
)
answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## Evaluation setup

Hardware target: NVIDIA RTX 5090, Ubuntu Jupyter, CUDA 13.0+, bf16 training and inference. Validation and test abstracts come from [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es). Metrics treat generated paraphrases as AI class with label 1 and apply threshold 0.5 on detector probability.

## Source code

The full pipeline is published on [GitHub](https://github.com/pymlex/ai-text-detector-tricking).

## Citation

If you found this model useful, please cite it as:

```bibtex
@software{zyukov2026aitexttricking,
  author = {Zyukov, Alex},
  title = {ai-text-detector-tricking: DPO fine-tuning against multilingual AI text detectors},
  year = {2026},
  url = {https://github.com/pymlex/ai-text-detector-tricking},
  publisher = {GitHub},
  organization = {pymlex}
}
```

```bibtex
@article{nicks2024detectors,
  title = {Language Model Detectors Are Easily Optimized Against},
  author = {Nicks, Cameron and others},
  year = {2024},
  url = {https://arxiv.org/abs/2406.07490}
}
```

The project is under GPL-3.0 license.
