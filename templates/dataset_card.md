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

# ai-generated-texts

Preference pairs for DPO fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` against the Oculus multilingual AI text detector on Spanish academic abstracts.

## Dataset construction

For each abstract in the train split of [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es), two paraphrases are sampled from the base instruct model with temperature $0.7$. Each paraphrase is scored by [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual). The lower AI probability becomes `chosen`, the higher becomes `rejected`. Pairs with probability margin below $0.05$ are discarded.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `prompt` | string | Spanish paraphrase instruction with the source abstract |
| `chosen` | string | Paraphrase with lower detector AI probability |
| `rejected` | string | Paraphrase with higher detector AI probability |

## Source code

[github.com/pymlex/ai-text-detector-tricking](https://github.com/pymlex/ai-text-detector-tricking)

## Citation

If you found this dataset useful, please cite it as:

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

The project is under GPL-3.0 license.
