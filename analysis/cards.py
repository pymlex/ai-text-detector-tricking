from __future__ import annotations

from analysis.citations import (
    CITATION_FOOTER,
    DATASET_CITATION,
    FLAGLAB_CITATION,
    MODEL_CITATION,
    NICKS_CITATION,
    OCULUS_CITATION,
    PROJECT_CITATION,
)
from analysis.collect import AnalysisSnapshot
from constants import HF_DATASET_REPO, HF_MODEL_REPO, MODEL_ID
from schemas.evaluation import format_optional_float


def _metrics_table(snapshot: AnalysisSnapshot) -> str:
    if snapshot.evaluation is None:
        return "_Evaluation metrics pending. Run `python main.py --step evaluate` first._"
    report = snapshot.evaluation
    return "\n".join(
        [
            "| Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| validation | {report.validation.n_samples} | "
            f"{report.validation.mean_probability:.4f} | {report.validation.mean_logit:.4f} | "
            f"{report.validation.accuracy:.4f} | {report.validation.mcc:.4f} | "
            f"{format_optional_float(report.validation.roc_auc)} | {report.validation.f1:.4f} |",
            f"| test | {report.test.n_samples} | "
            f"{report.test.mean_probability:.4f} | {report.test.mean_logit:.4f} | "
            f"{report.test.accuracy:.4f} | {report.test.mcc:.4f} | "
            f"{format_optional_float(report.test.roc_auc)} | {report.test.f1:.4f} |",
        ]
    )


def render_model_card(snapshot: AnalysisSnapshot) -> str:
    pref = snapshot.preference_stats
    monitor = snapshot.training_monitor
    monitor_text = ""
    if monitor.step0_mean_probability is not None and monitor.final_mean_probability is not None:
        delta = monitor.final_mean_probability - monitor.step0_mean_probability
        n_texts = monitor.n_monitor_samples or "N"
        monitor_text = (
            f"During DPO, mean validation AI probability on a {n_texts}-text subset moved from "
            f"{monitor.step0_mean_probability:.4f} at step 0 to "
            f"{monitor.final_mean_probability:.4f} at the last monitor step "
            f"({delta:+.4f})."
        )

    return f"""---
license: gpl-3.0
language:
- es
base_model: {MODEL_ID}
tags:
- dpo
- ai-detection
- paraphrase
library_name: transformers
---

# Qwen2.5-0.5B-Human

DPO fine-tune of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) that paraphrases Spanish academic abstracts to reduce AI-detection scores from [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual).

## Overview

Preference pairs for optimisation come from [{HF_DATASET_REPO}](https://huggingface.co/datasets/{HF_DATASET_REPO}). The corpus is [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es). For each train abstract, two base-model paraphrases are ranked by Oculus logit. DPO with beta = 0.1 increases the relative log-probability of the lower-logit completion. Retained pairs: {pref.pairs_built} from {pref.train_rows} train abstracts with absolute logit gap at least {pref.logit_margin_threshold:g}.

## Intended use

Research on detector robustness and red-teaming of AI-generated text classifiers under controlled laboratory settings. Not for academic misconduct or deceptive publishing.

## Evaluation setup

Hardware: NVIDIA RTX 5090, Ubuntu Jupyter, CUDA 13.0+, bf16 training and inference. Post-training evaluation generates one paraphrase per validation and test abstract, scores each output with Oculus, and treats label 1 as AI-generated at threshold 0.5 on detector probability.

{monitor_text}

## Results

{_metrics_table(snapshot)}

Lower mean probability and MCC near zero indicate weaker detector response on model paraphrases under the AI-positive labelling convention.

![Evaluation summary](https://huggingface.co/{HF_MODEL_REPO}/resolve/main/assets/evaluation_summary.png)

![Training monitor](https://huggingface.co/{HF_MODEL_REPO}/resolve/main/assets/training_monitor_analysis.png)

## Source code

The full pipeline is published on [GitHub](https://github.com/pymlex/ai-text-detector-tricking).

## Citation

If you found this model useful, please cite it as:

{MODEL_CITATION}

{PROJECT_CITATION}

{NICKS_CITATION}

{OCULUS_CITATION}

{FLAGLAB_CITATION}

{CITATION_FOOTER}
"""


def render_dataset_card(snapshot: AnalysisSnapshot) -> str:
    pref = snapshot.preference_stats
    probe = snapshot.logit_probe
    probe_block = ""
    if probe is not None:
        probe_block = f"""
## Logit margin calibration probe

Probe size: {probe.n_samples} paraphrase pairs from the train split. Mean absolute logit gap: {probe.mean_abs_logit_diff:.4f}. Median: {probe.median_abs_logit_diff:.4f}. 75th percentile: {probe.p75_abs_logit_diff:.4f}. Maximum gap: {probe.max_abs_logit_diff:.4f}.

![Logit margin histogram](https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/assets/logit_margin_probe_hist.png)

![Chosen vs rejected logits](https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/assets/logit_chosen_rejected_hist.png)
"""

    return f"""---
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

Preference pairs for DPO fine-tuning of `{MODEL_ID}` against the Oculus multilingual AI text detector on Spanish academic abstracts. Repository id: [{HF_DATASET_REPO}](https://huggingface.co/datasets/{HF_DATASET_REPO}).

## Dataset size

| Statistic | Count |
| --- | ---: |
| Train abstracts processed | {pref.train_rows} |
| DPO pairs retained | {pref.pairs_built} |
| Pairs skipped by logit margin | {pref.pairs_skipped_tie} |
| Empty paraphrase pairs | {pref.pairs_skipped_empty} |

Logit margin threshold: absolute gap at least {pref.logit_margin_threshold:g}.

## Dataset construction

For each abstract in the train split of [Flaglab/academic-knowledge-abstracts-es](https://huggingface.co/datasets/Flaglab/academic-knowledge-abstracts-es), two paraphrases are sampled from the base instruct model with temperature 0.7. Each paraphrase is scored by [danibor/oculus-v2.0-multilingual](https://huggingface.co/danibor/oculus-v2.0-multilingual). The lower detector logit becomes `chosen`, the higher becomes `rejected`.
{probe_block}
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

{DATASET_CITATION}

{PROJECT_CITATION}

{NICKS_CITATION}

{OCULUS_CITATION}

{FLAGLAB_CITATION}

{CITATION_FOOTER}
"""
