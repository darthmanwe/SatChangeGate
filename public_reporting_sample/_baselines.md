# Learned baselines vs the rule gate

Fitted on 1351 tiles from 14 train cities; evaluated on 621 tiles from 10 held-out cities. Positive prevalence on test: 0.586 (the precision a coin-flip classifier would reach).

All models see exactly the features the gate uses, on the same split.

| Model | Average precision | ROC AUC | Precision @ gate's recall |
|---|---|---|---|
| rule_gate_confidence | 0.719 | 0.728 | 0.784 |
| logistic_regression | 0.802 | 0.747 | 0.787 |
| gradient_boosting | 0.861 | 0.805 | 0.877 |

The shipped rule gate operates at recall 0.513, precision 0.804. The final column asks what each model achieves at that same recall, which is the like-for-like comparison.

![Precision-recall curves](pr_curves.png)

Average precision is the summary statistic here rather than F1: it is threshold-free, so it compares the ranking each model produces instead of one arbitrarily chosen cut.
