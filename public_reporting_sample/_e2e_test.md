# End-to-end funnel — `test` split

621 tiles, seed 42, sampling `stratified`, Batch API.

## Funnel

| Stage | Count | Share |
|---|---|---|
| Input | 621 | 100% |
| Refused by Tier 0 (unusable imagery) | 87 | 14.01% of input |
| Filtered by the gate (judged unchanged) | 313 | 58.61% of assessable |
| Forwarded as candidates | 221 | 35.59% of input |
| Actually sent to the VLM | 100 | 16.1% of input |

Total reduction before any API call: **64.41%**.

## Verification coverage

9 of 9 cities with candidates had at least one verified.

| City | Candidates | Verified |
|---|---|---|
| brasilia | 11 | 5 |
| chongqing | 49 | 22 |
| dubai | 33 | 15 |
| lasvegas | 70 | 32 |
| milano | 9 | 4 |
| montpellier | 7 | 3 |
| norcia | 12 | 5 |
| rio | 15 | 7 |
| valencia | 15 | 7 |

## Cost (measured from API token usage)


- Actually spent: **$0.4689** (289259 in / 35934 out tokens, 100 calls at $0.004689 each, 100 batched)
- Projected for all 221 gate candidates: $1.0363
- Review-everything counterfactual: $2.912
- Saving attributable to the gate: $1.8757 (64.41%)

> savings_pct equals the share of pairs the funnel did not forward; with one flat per-call price it restates the candidate rate rather than measuring anything further. batch_saving_pct is independent.

- Same candidates priced synchronously: $2.0727
- **Saving attributable to batching: $1.0363 (50.0%)** — independent of the filter rate.

> This run was budget-capped, so actual spend is lower than the projection. The saving above is attributed to the gate only.

## Gate quality

n=534 (345 change / 189 no-change). Recall 0.516, precision 0.805, specificity 0.772, F1 0.629.

## VLM verdicts

- likely_artifact: 16
- real_change: 60
- uncertain: 24
