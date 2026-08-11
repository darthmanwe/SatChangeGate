# End-to-end funnel — `test` split

621 tiles, seed 42.

## Funnel

| Stage | Count | Share |
|---|---|---|
| Input | 621 | 100% |
| Filtered by classical gate | 401 | 64.57% |
| Sent to VLM | 100 | 16.1% |

## Cost (measured from API token usage)


- Actually spent: **$1.2941** (290826 in / 28107 out tokens, 100 calls at $0.012941 each)
- Projected for all 220 gate candidates: $2.847
- Review-everything counterfactual: $8.0363
- Saving attributable to the gate: $5.1893 (64.57%)

> This run was budget-capped, so actual spend is lower than the projection. The saving above is attributed to the gate only.

## Gate quality

n=534 (345 change / 189 no-change). Recall 0.513, precision 0.804, specificity 0.772, F1 0.626.

## VLM verdicts

- likely_artifact: 10
- real_change: 68
- uncertain: 22
