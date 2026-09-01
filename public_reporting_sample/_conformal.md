# Conformal risk control on the gate threshold

Target: miss at most **20%** of real change, with **90%** confidence.

## Calibration

Calibrated on 419 tiles (179 positive) from 4 training cities: abudhabi, mumbai, nantes, pisa.

- Threshold lambda = **0.200**
- Calibration false-negative rate: 0.117 (upper bound 0.198 <= 0.20)
- Flags 87.1% of calibration tiles

## The falsifier: held-out cities

Observed false-negative rate at that threshold: **0.177** (recall 0.823), flagging 74.0% of tiles.

Overall, the guarantee **held** on the held-out split.

| City | n | positives | recall | FNR | within alpha |
|---|---|---|---|---|---|
| brasilia | 39 | 23 | 1.000 | 0.000 | yes |
| chongqing | 85 | 65 | 0.908 | 0.092 | yes |
| dubai | 104 | 69 | 0.623 | 0.377 | **no** |
| lasvegas | 126 | 105 | 0.829 | 0.171 | yes |
| milano | 54 | 17 | 0.412 | 0.588 | **no** |
| montpellier | 41 | 27 | 1.000 | 0.000 | yes |
| norcia | 15 | 8 | 1.000 | 0.000 | yes |
| rio | 27 | 19 | 0.947 | 0.053 | yes |
| valencia | 43 | 12 | 1.000 | 0.000 | yes |

The guarantee held on 7 of 9 held-out cities with positives. Exchangeability across geographies is exactly what a city-level split breaks, so a failure here is a property of the assumption, not a bug in the procedure.

> Exchangeability between calibration and deployment tiles. A geographic split violates this, so the guarantee is conditional and is tested per city rather than asserted.
