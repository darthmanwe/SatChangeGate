# OPTIMUS Eval Summary

**Tar group:** images/148.tar
**Labeled pairs evaluated:** 4

## Gate metrics

| Gate | Count |
|------|-------|
| no_change | 2 |
| candidate_change | 2 |
| low_quality | 0 |

- VLM candidate rate: **50.0%**
- Filtered before VLM: **50.0%**
- Calls avoided (no_change + low_quality): **2**

## Pair-level vs OPTIMUS labels (0=no change, 1=change)

- Confusion: TP=2 FP=0 FN=0 TN=2
- Precision: **1.000**
- Recall: **1.000**
- F1: **1.000**
- Specificity (no-change correct): **1.000**

## Per-tile features

| tile | gt | gate | changed_% | ndvi_d | ndbi_d | ssim | phash |
|------|----|------|-----------|--------|--------|------|-------|
| optimus_2275_3578 | 1 | candidate_change | 74.89 | 0.1459 | 0.104 | 0.3883 | 10 |
| optimus_4083_2811 | 1 | candidate_change | 75.29 | 0.135 | 0.077 | 0.3361 | 33 |
| optimus_4472_3910 | 0 | no_change | 86.09 | 0.074 | 0.0535 | 0.1635 | 27 |
| optimus_4495_4756 | 0 | no_change | 23.07 | 0.0239 | 0.0151 | 0.7652 | 3 |