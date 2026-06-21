# Results analysis

## Preference dataset

Train abstracts: 8891. DPO pairs retained: 6396. Pairs skipped by logit margin |z1 - z2| < 1: 2495. Empty paraphrases: 0.

## Logit margin probe

Probe size: 512. Mean |z1 - z2|: 2.4227. Median: 1.7992. IQR: [0.8724, 3.3554]. Maximum gap: 11.2376.

![Logit probe summary](../plots/analysis/logit_probe_summary.png)

## Training monitor

Validation subset mean AI probability at step 0: 0.6740. At final logged step: 0.2437. Change: -0.4303.

Final logged DPO loss: 0.2556.

![Training monitor analysis](../plots/analysis/training_monitor_analysis.png)

## Evaluation

Paraphrases from the base and fine-tuned models are scored by Oculus. Ground-truth label is AI-generated. Threshold on detector probability: 0.5.

| Model | Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | validation | 1107 | 0.6550 | 1.4619 | 0.6712 | 0.0000 | n/a | 0.8032 |
| base | test | 1112 | 0.6532 | 1.5581 | 0.6655 | 0.0000 | n/a | 0.7991 |
| fine-tuned | validation | 1107 | 0.2264 | -2.0100 | 0.1716 | 0.0000 | n/a | 0.2930 |
| fine-tuned | test | 1112 | 0.2391 | -1.8733 | 0.1835 | 0.0000 | n/a | 0.3100 |

Lower mean probability and MCC near zero indicate weaker detector response on model paraphrases under the AI-positive labelling convention.

![Evaluation summary](../plots/analysis/evaluation_summary.png)

![Score distributions](../plots/analysis/score_distributions.png)
