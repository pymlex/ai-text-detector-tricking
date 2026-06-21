---
license: gpl-3.0
language:
- es
task_categories:
- text-generation
tags:
- dpo
- ai-detection
- paraphrase
---

# Spanish DPO Preference Pairs for Detector Evasion

Preference pairs for DPO fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` against the Oculus multilingual AI text detector on Spanish academic abstracts.

## Dataset construction

For each abstract in the train split of [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es), two paraphrases are sampled from the base instruct model with temperature 0.7. Each paraphrase is scored by [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual). The lower detector logit becomes `chosen`, the higher becomes `rejected`. Pairs below the logit margin threshold are discarded.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `prompt` | string | Spanish paraphrase instruction with the source abstract |
| `chosen` | string | Paraphrase with lower detector logit |
| `rejected` | string | Paraphrase with higher detector logit |

## Source code

The full pipeline is published on [GitHub](https://github.com/pymlex/ai-text-detector-tricking).

## Citation

If you found this dataset useful, please cite it as:

```bibtex
@software{zyukov2026aitexttricking,
  author = {Zyukov, Alex},
  title = {DPO Fine-Tuning Against Multilingual AI Text Detectors},
  year = {2026},
  url = {https://github.com/pymlex/ai-text-detector-tricking},
  publisher = {GitHub},
  organization = {pymlex}
}
```

The project is under GPL-3.0 license.
