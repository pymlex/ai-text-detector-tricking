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

## Final evaluation

Paraphrases from the fine-tuned model are scored by Oculus. Ground-truth label is AI-generated. Threshold on detector probability: 0.5.

| Split | n | mean prob | mean logit | accuracy | MCC | ROC-AUC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 1107 | 0.2252 | -2.0261 | 0.1752 | 0.0000 | n/a | 0.2982 |
| test | 1112 | 0.2413 | -1.8549 | 0.1862 | 0.0000 | n/a | 0.3139 |

Lower mean probability and MCC near zero indicate weaker detector response on model paraphrases under the AI-positive labelling convention.

![Evaluation summary](../plots/analysis/evaluation_summary.png)

![Score distributions](../plots/analysis/score_distributions.png)
