# End-to-end funnel — `test` split

150 tiles, seed 42.

## Funnel

| Stage | Count | Share |
|---|---|---|
| Input | 150 | 100% |
| Filtered by classical gate | 98 | 65.33% |
| Sent to VLM | 20 | 13.33% |

## Cost (measured from API token usage)

- Actually spent: **$0.2706** (58822 in / 6278 out tokens, 20 calls at $0.013532 each)
- Projected for all 52 gate candidates: $0.7037
- Review-everything counterfactual: $2.0298
- Saving attributable to the gate: $1.3261 (65.33%)

> This run was budget-capped, so actual spend is lower than the projection. The saving above is attributed to the gate only.

## Gate quality

n=133 (76 change / 57 no-change). Recall 0.474, precision 0.692, specificity 0.719, F1 0.562.

## VLM verdicts

- likely_artifact: 3
- real_change: 12
- uncertain: 5
