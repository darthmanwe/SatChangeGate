# Gate evaluation — `test` split

621 tiles across 10 cities (345 change / 189 no-change).

| Metric | Value | 95% CI |
|---|---|---|
| Recall | 0.516 | 0.463-0.568 |
| Precision | 0.805 | 0.748-0.852 |
| Specificity | 0.772 | 0.708-0.827 |
| F1 | 0.629 | — |
| Balanced accuracy | 0.644 | — |

Confusion matrix: TP 178 · FP 43 · FN 167 · TN 146

Scored on 9 of 10 cities in this split.

| Reduction | Count | Share |
|---|---|---|
| Refused by Tier 0 (unusable imagery) | 87 | 14.0% of all tiles |
| Filtered by the gate (judged unchanged) | 313 | 58.6% of assessable tiles |
| Forwarded to the VLM | 221 | 35.6% of all tiles |

Total reduction before any API call: 64.4%.

## Pixel level

F1 0.261 - IoU 0.150 - recall 0.257 - precision 0.265 (n=2401155 observed pixels).

This is a triage gate, not a segmentation model. The number is here so the claim is reproducible, not because it is competitive with supervised deep models on this benchmark.
