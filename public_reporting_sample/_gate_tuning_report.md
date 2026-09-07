# Gate tuning report

Fitted on the **train** split: 1351 tiles across
14 cities. Held-out cities (never seen during tuning):
brasilia, chongqing, dubai, lasvegas, milano, montpellier, norcia, rio, saclay_w, valencia.

Evaluated 8640 threshold combinations, selecting
on balanced accuracy.

| | Baseline | Tuned |
|---|---|---|
| Balanced accuracy | 0.546 | 0.546 |
| Recall | 0.429 | 0.429 |
| Precision | 0.519 | 0.519 |
| Specificity | 0.664 | 0.664 |
| F1 | 0.469 | 0.469 |

**These are in-sample figures.** They describe the fit, not the expected
performance. Run `satchangegate eval --split test` for the out-of-sample result,
which is the only number that should be quoted.

Tuned thresholds are written to `tuned_thresholds.yaml`; copy them into
`src/satchangegate/configs/thresholds.yaml` to adopt them.
