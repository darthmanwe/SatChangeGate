# What the VLM tier added — `test` split

100 verification calls across 9 cities, 0 errors, 100 batched.

| | Precision | 95% CI |
|---|---|---|
| Gate alone | 0.850 (85/100) | 0.767-0.907 |
| Gate + VLM | 0.983 (59/60) | 0.911-0.997 |

Both are measured on the verified subset only, which is the only set where the two tiers can be compared like for like. The gate-alone figure here is therefore not the split-wide gate precision — compare it against `_eval_test.json` rather than substituting one for the other.

| Second-tier behaviour | Rate | 95% CI |
|---|---|---|
| Rejected the gate's errors | 0.933 (14/15) | 0.702-0.988 |
| Retained real change | 0.694 (59/85) | 0.590-0.782 |

## Verification coverage

| City | Calls |
|---|---|
| brasilia | 5 |
| chongqing | 22 |
| dubai | 15 |
| lasvegas | 32 |
| milano | 4 |
| montpellier | 3 |
| norcia | 5 |
| rio | 7 |
| valencia | 7 |

## Change types reported

- construction: 40
- flooding: 2
- no_meaningful_change: 8
- uncertain: 20
- vegetation_clearing: 20
- vegetation_regrowth: 3
- weather_artifact: 7

## Cost

$0.4689 total, $0.004689 per verification.
Priced synchronously the same calls would have cost $0.9379.
