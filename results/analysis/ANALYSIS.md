# Results analysis

## Preference dataset

Train abstracts: 8891. DPO pairs retained: 6396. Pairs skipped by logit margin $|z_1-z_2|<1$: 2495. Empty paraphrases: 0.

## Logit margin probe

Probe size: 512. Mean $|z_1-z_2|$: 2.4227. Median: 1.7992. IQR: [0.8724, 3.3554]. Maximum gap: 11.2376.

![Logit probe summary](../plots/analysis/logit_probe_summary.png)

## Training monitor

Validation subset mean AI probability at step 0: 0.6740. At final logged step: 0.2437. Change: -0.4303.

Final logged DPO loss: 0.2556.

![Training monitor analysis](../plots/analysis/training_monitor_analysis.png)
