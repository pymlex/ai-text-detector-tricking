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

## Dataset size

| Statistic | Count |
| --- | ---: |
| Train abstracts processed | 8891 |
| DPO pairs retained | 6396 |
| Pairs skipped by logit margin | 2495 |
| Empty paraphrase pairs | 0 |

Logit margin threshold: $|z_1-z_2| \ge 1$.

## Dataset construction

For each abstract in the train split of [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es), two paraphrases are sampled from the base instruct model with temperature $0.7$. Each paraphrase is scored by [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual). The lower detector logit becomes `chosen`, the higher becomes `rejected`.

## Logit margin calibration probe

Probe size: 512 paraphrase pairs from the train split. Mean $|z_1-z_2|$: 2.4227. Median: 1.7992. 75th percentile: 3.3554. Maximum gap: 11.2376.

![Logit margin histogram](https://huggingface.co/datasets/pymlex/ai-generated-texts/resolve/main/assets/logit_margin_probe_hist.png)

![Chosen vs rejected logits](https://huggingface.co/datasets/pymlex/ai-generated-texts/resolve/main/assets/logit_chosen_rejected_hist.png)

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `prompt` | string | Spanish paraphrase instruction with the source abstract |
| `chosen` | string | Paraphrase with lower detector logit |
| `rejected` | string | Paraphrase with higher detector logit |

## Source code

[github.com/pymlex/ai-text-detector-tricking](https://github.com/pymlex/ai-text-detector-tricking)

## Citation

If you found this dataset useful, please cite it as:

```bibtex
@dataset{zyukov2026aigeneratedtexts,
  author = {Zyukov, Alex},
  title = {ai-generated-texts: DPO preference pairs for Spanish abstract paraphrase},
  year = {2026},
  url = {https://huggingface.co/datasets/pymlex/ai-generated-texts},
  publisher = {Hugging Face}
}
```

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

```bibtex
@article{nicks2024detectors,
  title = {Language Model Detectors Are Easily Optimized Against},
  author = {Nicks, Cameron and Chua, Jeremy and Liu, Stephen and others},
  year = {2024},
  eprint = {2406.07490},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2406.07490}
}
```

```bibtex
@misc{oculus2026,
  title = {Oculus 2.0 Multilingual AI Text Detector},
  author = {danibor},
  year = {2026},
  url = {https://huggingface.co/danibor/oculus-v2.0-multilingual}
}
```

```bibtex
@misc{flaglab2025abstracts,
  title = {Academic Knowledge Abstracts Spanish},
  author = {Flaglab},
  year = {2025},
  url = {https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es}
}
```

The project is under GPL-3.0 license.
