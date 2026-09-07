# What a budget buys — `test` split

534 assessable tiles (345 contain change). Verification costs $0.004689 per tile (measured from the recorded run ledger), so reviewing every tile would cost $2.5039.

| Budget | Calls it buys | Threshold | Flagged | Recall | Precision | Spend |
|---|---|---|---|---|---|---|
| $0.25 | 53 | 0.482 | 53 | 0.099 | 0.641 | $0.2485 |
| $0.50 | 106 | 0.433 | 106 | 0.220 | 0.717 | $0.4970 |
| $1.00 | 213 | 0.317 | 213 | 0.487 | 0.789 | $0.9988 |
| $2.00 | 426 | 0.177 | 426 | 0.881 | 0.714 | $1.9975 |
| $5.00 | 534 | 0.060 | 534 | 1.000 | 0.646 | $2.5039 |

Read this as the demand curve for review capacity: each row is the best recall that budget can reach, and the threshold that reaches it. Precision rises as the budget falls, because a tighter threshold keeps only the gate's most confident calls -- which is the trade a spend cap actually makes, stated rather than implied.

The confidence intervals that belong on these numbers are in `_eval_test.json`; they are omitted here because the table's purpose is choosing an operating point, not publishing one.
