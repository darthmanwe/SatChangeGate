# Relative radiometric normalization: A/B

Same features, same `test` split, the only difference being whether t2 is gain/offset-matched to t1 by PIF before anything else runs.

| | Normalization off | on | delta |
|---|---|---|---|
| Gate F1 | 0.6290 | 0.6197 | -0.0093 |
| Gate precision | 0.8054 | 0.7892 | -0.0162 |
| rule_gate_confidence AP | 0.7118 | 0.7171 | +0.0053 |
| logistic_regression AP | 0.8457 | 0.7469 | -0.0988 |
| gradient_boosting AP | 0.8593 | 0.8449 | -0.0144 |

Negative deltas mean normalization hurt. The linear model loses most, which is the expected shape: on a multi-year pair a large share of ground has genuinely changed, so the invariant population the PIF fit is drawn from is contaminated, and matching the two scenes compresses exactly the scene-wide spectral difference the models were using. A textbook correction that is right in general and wrong here.

> This artifact was committed from 0.3.0 with no producing command. That is the same class of defect as a headline number nobody could regenerate, and a negative result is not exempt from it.
