# SatChangeGate

A cost-controlled review funnel for satellite change detection: cheap deterministic
preprocessing and a classical gate filter the stream, and expensive vision-model
review runs only on what survives.

[![CI](https://github.com/darthmanwe/SatChangeGate/actions/workflows/ci.yml/badge.svg)](https://github.com/darthmanwe/SatChangeGate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

![Gated review funnel](docs/figures/funnel.png)

## Quickstart

No dataset download and no API key required:

```bash
make setup
make demo
```

That runs the full funnel on committed synthetic fixtures. For the real
benchmark (~513 MB, checksum-verified, no registration required):

```bash
make data     # download 13-band Sentinel-2 OSCD
make eval     # out-of-sample evaluation on held-out cities
```

Or with Docker, which avoids the GDAL/OpenCV install entirely:

```bash
docker build -t satchangegate . && docker run --rm satchangegate --help
```

## Results

Measured on held-out OSCD cities, never seen during threshold tuning.
Disjointness between the tuning and evaluation splits is asserted at runtime,
not assumed.

| Metric | Value | 95% CI |
|---|---|---|
| Precision | **0.805** | 0.748 – 0.852 |
| Recall | 0.516 | 0.463 – 0.568 |
| Specificity | 0.772 | 0.708 – 0.827 |
| F1 | 0.629 | — |
| Balanced accuracy | 0.644 | — |

n = 534 scored tiles (345 change / 189 no-change) from 621 total.
Confusion matrix: TP 178 · FP 43 · FN 167 · TN 146.

**Scored on 9 of the 10 held-out cities, not 10.** The remaining 87 tiles are
all from `saclay_w`, whose 2.04 px co-registration error fails Tier 0.
`valid_observation` is a scene-level flag, so a city either passes or vanishes
entirely — there is no partial credit. Earlier versions of this table said "10
held-out cities", which was true of the tile index and false of the metrics
computed from it.

**Three different things reduce the workload, and they are not the same claim:**

| Reduction | Count | Share |
|---|---|---|
| Refused by Tier 0 (unusable imagery) | 87 | 14.0% of all tiles |
| Filtered by the gate (judged unchanged) | 313 | 58.6% of assessable tiles |
| Forwarded to the VLM | 221 | 41.4% of assessable |

**Total reduction before any API call: 64.4%.** This used to be published as a
single "the gate filters 64.6%", which credited the gate with 87
co-registration failures it had no part in. Tier 0 *refuses to judge*; the gate
*judges and finds nothing*. `satchangegate eval` now reports all three.

**The operating point is high precision, moderate recall.** When the gate flags
something it is right about 80% of the time, and it misses about half of all
change. That is a real trade — but it is no longer the only operating point on
offer; see the guarantee below.

### A recall guarantee, and where it breaks

"It misses about half of all change" is honest and unactionable: an operator who
needs to miss no more than 20% has no way to ask for it. Conformal risk control
(Learn-then-Test with a Hoeffding bound) turns the threshold into a claim with a
confidence attached — calibrated on four training cities, then **tested against
cities it never saw**:

```
satchangegate conformal --alpha 0.20 --delta 0.10
```

| | Value |
|---|---|
| Target | miss ≤ 20% of real change, with 90% confidence |
| Calibrated threshold | λ = 0.200 (419 tiles, 179 positive, from abudhabi / mumbai / nantes / pisa) |
| Calibration FNR | 0.117, upper bound **0.198 ≤ 0.20** ✓ |
| **Held-out FNR** | **0.177 → recall 0.823**, flagging 74.0% of tiles |

The guarantee **held overall**. It held on **7 of 9** held-out cities, and
failed on two:

| City | Recall at λ | Within α |
|---|---|---|
| brasilia, montpellier, norcia, valencia | 1.000 | yes |
| rio | 0.947 | yes |
| chongqing | 0.908 | yes |
| lasvegas | 0.829 | yes |
| **dubai** | **0.623** | no |
| **milano** | **0.412** | no |

That failure is the point, not a defect. A conformal guarantee assumes
calibration and deployment data are exchangeable, and a *geographic* split
breaks exactly that assumption: dubai and milano do not look like abudhabi,
mumbai, nantes and pisa. The procedure is sound; the assumption is what fails,
and it fails in a way this repo can measure and name. Anyone deploying to a new
AOI should read those two rows as the expected behaviour, not the exception.

The price of that recall is spend: λ = 0.200 forwards 74% of tiles rather than
41%. Which brings the obvious question.

### What a budget buys

```
satchangegate operating-points --split test --budget-usd 1.00
```

A cost-control gate is chosen by its operating point, so the repo now ships the
demand curve rather than a single point. Each row is the best recall a budget
can reach and the threshold that reaches it:

| Budget | Calls | Threshold | Recall | Precision |
|---|---|---|---|---|
| $0.25 | 53 | 0.482 | 0.099 | 0.641 |
| $0.50 | 106 | 0.433 | 0.220 | 0.717 |
| $1.00 | 213 | 0.317 | 0.487 | 0.789 |
| $2.00 | 426 | 0.177 | 0.881 | 0.714 |
| $5.00 | 534 | 0.060 | 1.000 | 0.646 |

Prices come from the measured batch rate ($0.004689 per verification), so the
whole held-out split can be reviewed for $2.50. Precision peaks in the middle:
too tight a threshold keeps only high-confidence tiles and starves recall, too
loose a one admits everything and precision decays toward the 0.586 base rate.
That is the trade a spend cap actually makes, stated rather than implied — and
the $2.00 row lands close to where the conformal threshold does, from an
entirely different direction.

![Detection on a held-out city](docs/figures/detection_lasvegas.png)

Las Vegas, a held-out test city. The detection panel is the despeckled change
mask; comparing it against ground truth shows the shape of the trade — the
flagged regions are largely real, and a good deal of real change is not flagged.

Reproduce: `make data && make tune && make eval`. Sample outputs are committed
under [`public_reporting_sample/`](public_reporting_sample/).

### Is a hand-written gate the right choice?

The gate is six rules over eighteen numeric features. The obvious question is
why those features are not simply handed to a classifier, so the repo answers it
rather than leaving it open — same features, same split, same leakage assertion:

| Model | Average precision | ROC AUC | Precision @ the gate's recall |
|---|---|---|---|
| Rule gate (confidence swept) | 0.715 | 0.719 | 0.775 |
| Logistic regression | 0.846 | 0.783 | 0.856 |
| **Gradient boosting** | **0.859** | **0.801** | **0.883** |

**Gradient boosting still wins** — +0.14 average precision over the rules, and
+11 points of precision at the same recall. That is the honest result, and it is
reported rather than buried.

The more interesting movement is in the middle row. Adding eight directional and
shape features (signed index tails, an NDBI-up-NDVI-down score, magnitude
percentiles, connected-component size and compactness) moved **logistic
regression from 0.802 to 0.846 AP** while leaving gradient boosting effectively
unchanged (0.861 → 0.859). The information was always there; boosting had been
recovering it through interactions, and the new features simply make it linearly
accessible. The rule gate's own ranking did not benefit (0.719 → 0.715, well
inside noise at n = 621) — three thresholds cannot use what a regression can.

**The learned scorer is now something you can actually run.** It used to be
fitted inside the report function and discarded, so the invitation to "swap it
in" described an operation that did not exist:

```
satchangegate fit-scorer --split train     # writes the model + a model card
```

Then set `scorer.kind: learned` in `thresholds.yaml`. The artifact records the
feature list, the sklearn version, and the cities it saw; loading it against a
changed feature vector raises rather than scoring quietly. The rules remain the
default, for reasons worth stating plainly: they are inspectable (every decision
returns the rule that fired), they add no runtime dependency, and there is no
model artifact to version, retrain, or drift. For a spend-control filter those
properties are worth real money. The measured price of that choice is the 0.14
AP above.

![Precision-recall curves](docs/figures/pr_curves.png)

The curve is the more useful artifact than any single number, because a
cost-control gate is chosen by its operating point. Two things it shows that a
headline metric hides: the gate has genuine signal in the high-precision regime,
and **above roughly 0.8 recall no method here beats chance** — if you need to
catch nearly all change on this benchmark, a classical gate is the wrong tool.

### What the funnel is worth

Measured on a live run over the full held-out split — 621 tiles, **100
verification calls, 0 errors**, `claude-sonnet-5` via the Batch API, seed 42,
**stratified across every city with candidates**. Cost comes from the token usage
the API returned, not from an estimate.

| Stage | Count | Share |
|---|---|---|
| Input tiles | 621 | 100% |
| Refused by Tier 0 | 87 | 14.0% of input |
| Filtered by the gate | 313 | 58.6% of assessable |
| Reached the VLM | 221 candidates (100 verified) | 41.4% of assessable |

**Spend: $0.4689 actual, $0.004689 per verification.** The same 100 calls
submitted synchronously would have cost $0.9379, so batching saved **50.0%** —
a saving that is *independent* of the gate's filter rate, which matters because
the gate's own "saving" is not:

> With a single per-call price, "saved 64.4%" is algebraically
> `1 − candidates/tiles`. It restates the candidate rate rather than measuring
> anything further. The repo now says so in `_e2e_test.json` instead of letting
> the number look like two findings.

Verifying all 221 candidates projects to $1.036, against $2.912 to review all
621 — and reviewing only the 534 assessable tiles costs $2.504.

The second stage pays for itself on precision:

| | Precision | 95% CI |
|---|---|---|
| Gate alone | 0.850 (85/100) | 0.767 – 0.907 |
| **Gate + VLM** | **0.983 (59/60)** | 0.911 – 0.997 |

The VLM rejected 14 of the 15 tiles the gate forwarded in error (0.933, CI
0.702 – 0.988) while retaining 59 of 85 true changes (0.694, CI 0.590 – 0.782).
It also labels what it sees: 40 construction, 20 vegetation clearing, 8 no
meaningful change, 7 weather artifact, 3 vegetation regrowth, 2 flooding, 20
uncertain.

**This replaces an earlier run that was not a sample.** That run described itself
as "100 sampled, seed 42" but was an alphabetical truncation: it verified three
cities exhaustively, 7 of a fourth city's 70 candidates, and none of the
remaining five. Its sub-sample was easier than the split it claimed to represent.
On a properly stratified sample the architecture claim survives and the second
tier looks *better* on precision (0.971 → 0.983) and **worse on retention**
(0.759 → 0.694) — the honest shape of a precision filter, which is that it costs
recall. Every figure above is regenerated by a command:

```bash
satchangegate e2e --split test --vlm --batch --sample stratified --max-vlm-calls 100
satchangegate vlm-report --split test
```

`--max-vlm-calls` caps spend across resumes and is enforced *before* submission;
`--batch` writes its batch id to disk the moment it submits, so a crash between
submitting and collecting costs time rather than money — a rerun reattaches to
the live batch instead of buying it twice.

## What this is not

Honest framing matters more here than a good-looking number, so:

- **The VLM numbers rest on 100 calls from one split.** Enough for the intervals
  quoted, not enough to be a production SLA, and all from a single benchmark.
- **This is a triage gate, not a segmentation model.** At the pixel level it
  reaches **F1 0.261, IoU 0.150** on the held-out split, against roughly
  0.50–0.60 for supervised deep models on OSCD. It is not competitive with them
  and is not trying to be — it is a ~10 ms/tile filter that decides whether to
  spend money. Reproduce with `satchangegate eval --split test --pixel-metrics`.
  (An earlier version of this README quoted F1 ≈ 0.13 here. That number came
  from a code comment recording a *train*-split figure, and no command in the
  repo computed it. There is now a command, and the real held-out number is
  roughly twice what was claimed.)
- **OSCD labels only *urban* change.** Seasonal, agricultural, and hydrological
  change across a multi-year gap is spectrally enormous and labelled negative, so
  a magnitude-based detector keys on the wrong signal. No single feature
  separates the classes — the best directional one (NDBI up, NDVI down) reaches
  only AUC ≈ 0.56. What the baseline comparison shows is that this is a *feature
  combination* problem rather than a dead end: the same eighteen features reach
  ROC AUC 0.802 once a gradient-boosted model is allowed to combine them. The rules
  leave that on the table, which is the honest reading of the table above.
- **Cloud filtering rarely fires on OSCD**, because the dataset curators picked
  clear scenes. What Tier 0 actually catches here is **co-registration failure**:
  saclay_e (4.21 px) and saclay_w (2.04 px) exceed the 1.5 px tolerance and are
  correctly withheld from the gate.
- **Thresholds are dataset-specific.** They were fitted on 14 cities. Any new
  AOI needs its own validation — and the conformal per-city table above shows
  what that means concretely: the same threshold that misses 0% of change in
  brasilia misses 59% in milano.
- **Refitting found nothing better than what was already shipped.** The latest
  sweep evaluated 8,640 combinations across seven axes and matched the incumbent
  thresholds exactly (in-sample balanced accuracy 0.546 either way). The gate is
  at a local optimum for this rule family on this data; further gains need a
  different model, not a finer grid.
- **AlphaEarth embeddings cannot be evaluated on this benchmark.** Google's
  Satellite Embedding dataset starts in 2017, and 23 of OSCD's 24 cities have a
  first acquisition in 2015 or 2016 — only `chongqing` qualifies, so bitemporal
  embedding change is measurable on **1 of 24 pairs (4.2%)**. Run
  `satchangegate embedding-coverage` to reproduce that table. What ships is the
  single-date variant, and a clearly-labelled contaminated probe; see
  `src/satchangegate/data/embeddings.py` for why the distinction matters.

## Corrections

This repo's differentiator is measurement integrity, not accuracy — so when a
published claim turns out not to meet that bar, saying so is part of the
product. A full audit in August 2026 found six that did not:

| Claim as published | What was actually true |
|---|---|
| "100 verification calls, seed 42" from 220 candidates | An alphabetical truncation, not a sample. It verified brasilia, chongqing and dubai exhaustively, 7 of lasvegas's 70 candidates, and **none of the other five cities** — 3.1 of 10. The sub-sample was measurably easier than the split it claimed to represent (gate precision 0.870 vs 0.804). Fixed by `--sample stratified`, which apportions the budget per city. |
| Gate + VLM precision 0.971 | Arithmetically correct, but derived by hand from a gitignored file. **No command in the repo could reproduce it.** `satchangegate vlm-report` now regenerates it from the run ledger. |
| `claude-sonnet-5` at $3/$15 per MTok | The rate is **$2/$10**. $3/$15 was a scheduled September 2026 increase that was cancelled and never took effect, so every dollar figure published before 2026-08-29 was ~50% too high. The percentage saving was unaffected — it is the gate's filter rate. |
| "Measured on the 10 held-out cities" | Nine. All 87 Tier-0 rejections are `saclay_w`, and the flag is scene-level. |
| "The gate filters 64.6%" | 14.0% was Tier 0 refusing to judge; the gate filtered 58.6% of what remained. |
| Pixel-level F1 ≈ 0.13 | Traced to a comment about the *train* split, with no reproducing command. The held-out figure is **0.261**. |

Two structural fixes came out of that audit rather than any single number. The
`oscd`, `e2e` and `vlm` pytest markers were declared and deselected but **no
test carried them**, so `make test-all` ran exactly the same suite as `make
test` and nothing ever touched the real 13-band imagery; there are now tests
behind those markers. And `evaluate.py` — the module that produces every
headline metric, including the `low_quality` exclusion rule — had no direct test
at all.

## Changelog

[CHANGELOG.md](CHANGELOG.md) tracks each update, with two conventions that follow
from the paragraph above: **corrections get their own subsection** — what was
claimed, what is true, why they differed — and **negative results are entries
too**, because something measured and rejected is a result.

The current release is **0.3.0** (2026-08-30), which was measurement integrity
first and gate accuracy second: six corrected claims, a risk-controlled operating
point that reports where it fails, eight new gate features, a runnable learned
scorer, batched verification at half the price, and two published negative
results. Highlights:

| | Before 0.3.0 | Now |
|---|---|---|
| Gate + VLM precision | 0.971, on 3 of 10 cities | **0.983**, on **9 of 9** cities with candidates |
| Logistic regression AP | 0.802 | **0.846** |
| Recall with a guarantee attached | not on offer | **0.823**, and the 2 cities it fails on are named |
| Measured spend, 100 calls | $1.2941 (wrong rate) | **$0.4689** (right rate, batched) |
| Offline tests | 112 | **183** |

## How it works

**Tier 0 — quality.** Cloud, snow, and cloud-shadow masks from physically
motivated tests on top-of-atmosphere reflectance, plus sub-pixel co-registration
via phase cross-correlation. Water is *reported but not treated as
contamination*: removing it would make flooding undetectable by construction.
When a source lacks the bands to compute masks, quality is reported as
`masks_assessed: false` with null fractions — unknown is represented as unknown,
never as clean.

**Optional: relative radiometric normalization.** A scene-level adaptive
threshold cancels an *additive* offset between two acquisitions. It cannot cancel
a multiplicative one, or a shift that differs between bands — which is what a
different sun angle, atmosphere and phenological state actually produce. PIF
matching fits a per-band gain and offset from the pixels that did *not* change,
iterating because identifying those pixels is the problem being solved. On
synthetic pairs it recovers a known gain and offset almost exactly, drives the
residual on unchanged ground to zero, and leaves genuine change intact.

**On OSCD it makes things worse, so it ships off by default.** Same features,
same split, the only difference being whether t2 is gain/offset-matched to t1:

| | Normalization off | on | Δ |
|---|---|---|---|
| Gate F1 | 0.629 | 0.620 | −0.009 |
| Gate precision | 0.805 | 0.789 | −0.016 |
| Logistic regression AP | 0.846 | 0.747 | **−0.099** |
| Gradient boosting AP | 0.859 | 0.845 | −0.014 |

The linear model loses most. The likely reason is that on a multi-year OSCD pair
a large share of ground has genuinely changed, so the "invariant" population the
fit is drawn from is contaminated, and matching the two scenes compresses exactly
the scene-wide spectral difference the models were using. A textbook correction
that is right in general and wrong here — reported rather than quietly dropped.

**Tier 1 — the gate.** Per-pixel change magnitude from NDVI/NDBI (NDWI over
water, where the land indices are undefined), thresholded against a robust
per-scene background (`median + k·MAD`) with an absolute floor and ceiling, then
despeckled by morphological opening and connected-component filtering. Tile-level
features — signed index deltas and their tails, SSIM, perceptual hash,
change-vector magnitude, magnitude percentiles, changed area, and the size and
compactness of the largest changed region — feed one pure `decide()` function
that returns a decision, a human-readable reason, and a confidence.

The confidence is *monotone* by construction — more evidence never moves a pair
back toward `no_change` — which is a real, tested property. It is not the same
thing as *calibrated*, and earlier versions of this README used the stronger
word. `satchangegate conformal` is what attaches an actual guarantee to it.

Four design choices are load-bearing:

- *Robust background, not a percentile.* A percentile cut assumes change is rare;
  on a scene where 14% of pixels changed it lands inside the changed population
  and suppresses exactly what it should detect.
- *An absolute floor under the adaptive cut.* Without it, a scale-free threshold
  always selects its top N% — so a pure illumination shift reads as change.
- *Land indices are undefined over water.* Deep water has SWIR reflectance around
  0.008, so a few thousandths of sensor noise swings NDBI across most of its
  range and a static lake reads as violent change.
- *Absolute value throws away the answer.* The gate computed signed NDVI and NDBI
  deltas, carried them through every structure, serialised them into every
  report — and `decide()` read only the absolute values. NDBI up while NDVI goes
  down is the physically correct signature of urban change, and erasing sign made
  construction and demolition, or clearing and regrowth, identical to the gate.
  It now has a rule that can tell them apart.

Each of those was caught by a control that fails loudly rather than by inspection.

**Tier 2 — VLM verification.** Candidates are packaged as before/after RGB plus a
heatmap overlay and sent for an independent verdict, validated server-side
against a Pydantic schema via structured outputs. The metadata the model sees is
**redacted**: it carries acquisition dates and observation quality, and
deliberately withholds the gate's verdict and every discriminative gate feature.
The VLM is the gate's independent verifier; telling it the gate's answer would
confound any agreement statistic between the two.

Two details turned out to matter more than expected, both found by running the
tier live rather than by reading the code:

- *A tile is too small to send at native size.* A 64 px crop is a 640 m window;
  the model could not resolve it and correctly answered `uncertain` at 0.30
  confidence. Sending a 3× context window with the region of interest outlined,
  upsampled to a 512 px long edge, moved the same tile to 0.55.
- *The evidence has to be real.* The tile packager was passing placeholder dates
  and a `masks_assessed: true` record with every fraction null — internally
  contradictory, and a violation of this project's own unknown-is-not-clean
  rule. It now carries the scene's actual dates, cloud/shadow/water fractions,
  registration error, and seasonal separation.

**Tier 3 — analyst report.** Markdown from the structured evidence. Without an
API key it degrades to a template that is explicitly labelled as such, and the
result JSON records `llm_called` either way.

## Commands

```bash
satchangegate download-oscd          # fetch + verify the 13-band dataset
satchangegate tiles                  # build the labelled tile index
satchangegate run --pair beirut      # one city through the full funnel
satchangegate tune --split train     # fit thresholds (test split held out)
satchangegate eval --split test      # out-of-sample metrics with CIs
satchangegate e2e --split test --vlm # funnel with measured cost
satchangegate vlm-report             # what the VLM tier added (no API calls)
satchangegate conformal              # risk-controlled threshold, then falsify it
satchangegate operating-points       # what each review budget buys
satchangegate fit-scorer             # persist the learned scorer + model card
satchangegate baselines              # rule gate vs learned models
satchangegate embedding-coverage     # what AlphaEarth can speak to here
satchangegate dev-tests              # offline control battery
satchangegate verify                 # check dataset + config
```

Useful flags: `--pixel-metrics` on `eval`, `--sample stratified|sequential` and
`--batch` on `e2e`.

Anything that can spend money defaults to not spending it; `--vlm` is opt-in and
`--max-vlm-calls` is a hard cap.

## Development

```bash
make lint type test    # ruff, mypy, 183 offline tests
make test-oscd         # the tests that need the real dataset (needs `make data`)
make cov               # coverage report
make baselines         # rule gate vs learned models (needs .[baseline])
make vlm-report        # recompute the second-tier figures from the ledger
make figures           # regenerate all three README figures
```

The test suite runs offline with no API key and no dataset: `pytest` deselects
the `oscd`, `e2e`, and `vlm` markers by default, and a session fixture unsets
`ANTHROPIC_API_KEY` so no test can authenticate by accident. CI runs lint, types,
tests on Python 3.10–3.12 across Ubuntu and Windows, a wheel-install check that
the packaged config resolves outside a source checkout, and a Docker build.

## Data

| Source | Role |
|---|---|
| OSCD (24 cities, 13-band Sentinel-2 L1C) | Primary. Auto-downloaded and checksum-verified from the [Hugging Face mirror](https://huggingface.co/datasets/hkristen/oscd). Imagery in the figures above is rendered from it; the dataset is open-access and is not redistributed here. |
| `tests/fixtures/mini_oscd` | Committed synthetic pairs so a clean clone runs end to end. |
| AlphaEarth Satellite Embedding | Optional (`pip install -e ".[embeddings]"`). 64-d annual global embeddings, CC-BY 4.0, free via Earth Engine for research use. Coverage starts 2017, which is why only a single-date feature is measurable here. |

The official train/test split is recovered from the label archives themselves and
written to `splits.json`, rather than hardcoded.

## License

MIT — see [LICENSE](LICENSE). This covers the source code only; OSCD and any
other imagery carry their own licenses, so check with the dataset providers
before commercial use.
