# Gate evaluation — `test` split

621 tiles across 10 cities (345 change / 189 no-change).

| Metric | Value | 95% CI |
|---|---|---|
| Recall | 0.513 | 0.460-0.565 |
| Precision | 0.804 | 0.747-0.852 |
| Specificity | 0.772 | 0.708-0.827 |
| F1 | 0.626 | — |
| Balanced accuracy | 0.643 | — |

Confusion matrix: TP 177 · FP 43 · FN 168 · TN 146

Gate filtered 64.6% of tiles before any VLM call (220 candidates, 87 rejected by Tier 0 quality checks).
