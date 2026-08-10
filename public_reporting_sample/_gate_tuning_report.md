# Gate tuning report

Fitted on the **train** split: 1351 tiles across
14 cities. Held-out cities (never seen during tuning):
brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia.

Evaluated 2880 threshold combinations, selecting
on balanced accuracy.

| | Baseline | Tuned |
|---|---|---|
| Balanced accuracy | 0.514 | 0.543 |
| Recall | 0.284 | 0.411 |
| Precision | 0.484 | 0.517 |
| Specificity | 0.744 | 0.675 |
| F1 | 0.358 | 0.458 |

**These are in-sample figures.** They describe the fit, not the expected
performance. Run `satchangegate eval --split test` for the out-of-sample result,
which is the only number that should be quoted.

Tuned thresholds are written to `tuned_thresholds.yaml`; copy them into
`src/satchangegate/configs/thresholds.yaml` to adopt them.
