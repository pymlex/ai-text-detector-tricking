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

## Overview

Preference pairs for optimisation come from [pymlex/ai-generated-texts](https://huggingface.co/datasets/pymlex/ai-generated-texts). The corpus is [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es). For each train abstract, two base-model paraphrases are ranked by Oculus logit. DPO with beta = 0.1 increases the relative log-probability of the lower-logit completion. Retained pairs: 6396 from 8891 train abstracts with absolute logit gap at least 1.

## Intended use

Research on detector robustness and red-teaming of AI-generated text classifiers under controlled laboratory settings. Not for academic misconduct or deceptive publishing.

## Evaluation setup

Hardware: NVIDIA RTX 5090, Ubuntu Jupyter, CUDA 13.0+, bf16 training and inference. Post-training evaluation generates one paraphrase per validation and test abstract with the base and fine-tuned models, scores each output with Oculus, and treats label 1 as AI-generated at threshold 0.5 on detector probability.

During DPO, mean validation AI probability on a 276-text subset moved from 0.6740 at step 0 to 0.2437 at the last monitor step (-0.4303).

## Results

| Model | Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | validation | 1107 | 0.6550 | 1.4619 | 0.6712 | 0.0000 | n/a | 0.8032 |
| base | test | 1112 | 0.6532 | 1.5581 | 0.6655 | 0.0000 | n/a | 0.7991 |
| fine-tuned | validation | 1107 | 0.2264 | -2.0100 | 0.1716 | 0.0000 | n/a | 0.2930 |
| fine-tuned | test | 1112 | 0.2391 | -1.8733 | 0.1835 | 0.0000 | n/a | 0.3100 |

Lower mean probability and MCC near zero indicate weaker detector response on model paraphrases under the AI-positive labelling convention.

![Evaluation summary](https://huggingface.co/pymlex/Qwen2.5-0.5B-Human/resolve/main/assets/evaluation_summary.png)

![Score distributions](https://huggingface.co/pymlex/Qwen2.5-0.5B-Human/resolve/main/assets/score_distributions.png)

![Training monitor](https://huggingface.co/pymlex/Qwen2.5-0.5B-Human/resolve/main/assets/training_monitor_analysis.png)

## Source code

The full pipeline is published on [GitHub](https://github.com/pymlex/ai-text-detector-tricking).

## Citation

If you found this model useful, please cite it as:

```bibtex
@misc{zyukov2026qwenhuman,
  author = {Zyukov, Alex},
  title = {Qwen2.5-0.5B-Human: DPO fine-tune against Oculus detector},
  year = {2026},
  url = {https://huggingface.co/pymlex/Qwen2.5-0.5B-Human}
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
