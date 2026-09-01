# Changelog

Notable changes to SatChangeGate, newest first. Versions follow
[semantic versioning](https://semver.org/); dates are the day the work landed.

Two conventions specific to this project:

- **Corrections get their own subsection.** When a published number turns out to
  be wrong, the changelog says what was claimed, what is true, and why the two
  differed. Measurement integrity is what this repo is for, so a silent fix
  would be the wrong kind of quiet.
- **Negative results are entries too.** Something measured and rejected is a
  result. It belongs here, not only in the commit that deleted it.

## [0.3.0] — 2026-08-30

Stage 1 of a staged advancement plan: **measurement integrity first, then gate
accuracy**. Nothing in this release chases a benchmark number — the audit that
opened it found that several headline claims did not meet the standard the
project advertises, and those come before new capability.

### Corrected

Six published claims did not hold as stated. Each now has a command that
reproduces it.

| Claimed | Actually |
|---|---|
| "100 verification calls, seed 42" from 220 candidates | An **alphabetical truncation, not a sample** — three cities exhaustively, 7 of a fourth's 70, none of the remaining five. 3.1 of 10 cities. The sub-sample was easier than the split it stood for (gate precision 0.870 vs 0.804). |
| Gate + VLM precision 0.971 | Arithmetically right, but hand-derived from a gitignored file. **No command in the repo could reproduce it.** |
| `claude-sonnet-5` priced at $3/$15 per MTok | The rate is **$2/$10**. $3/$15 was a September 2026 increase that was cancelled and never took effect, so every dollar figure published before this release was ~50% high. |
| "Measured on the 10 held-out cities" | **Nine.** All 87 Tier-0 rejections are `saclay_w`, and `valid_observation` is scene-level, so a city passes whole or vanishes whole. |
| "The gate filters 64.6%" | **14.0% was Tier 0 refusing to judge**; the gate filtered 58.6% of what remained. Refusing to judge and judging-then-finding-nothing are different claims. |
| Pixel-level F1 ≈ 0.13 | Traced to a code comment about the **train** split, with no reproducing command. The held-out figure is **0.261** (IoU 0.150) — roughly twice what was claimed. |

Two structural gaps sat behind them:

- The `oscd`, `e2e` and `vlm` pytest markers were declared and deselected but
  **no test carried any of them**, so `make test-all` ran exactly the same suite
  as `make test` and nothing ever touched real 13-band imagery.
- `evaluate.py` — the module that produces every headline metric, including the
  `low_quality` exclusion rule — had **no direct test at all**.

### Added

- **`satchangegate conformal`** — Learn-then-Test risk control with a Hoeffding
  upper confidence bound. Picks the largest threshold whose UCB on false-negative
  rate is at most α over a calibration split carved from train cities only, then
  **falsifies itself** on held-out test data, per city.
- **`satchangegate operating-points`** — budget to threshold to expected calls to
  recall, precision and spend, priced from the measured per-call cost.
- **`satchangegate fit-scorer`** — persists the gradient-boosted scorer with a
  model card, the sklearn version, the train city set, and a SHA-256 of the
  feature list. Selected by `scorer.kind: learned`; a stale artifact raises
  `StaleScorerError` rather than scoring quietly. Rules stay the default.
- **`satchangegate vlm-report`** — regenerates every second-tier figure from the
  run ledger, so the gate+VLM headline has a producer.
- **`satchangegate embedding-coverage`** — what AlphaEarth can and cannot speak
  to on this benchmark.
- **`e2e --batch`** — Message Batches submission, results keyed by `custom_id`
  and never by position. The batch id is written to disk at submit time, so a
  crash between submitting and collecting costs time rather than money: a rerun
  **reattaches to the live batch** instead of buying it twice.
- **`e2e --sample stratified`** — largest-remainder apportionment across cities
  with a floor of one call per city, applied to *candidates* rather than input
  order. This is the fix for the truncation above.
- **`eval --pixel-metrics`** — pixel confusion matrix, F1 and IoU against the
  label masks. Wires up `metrics.iou`, which had been dead code.
- **Eight gate features**: signed index tails (`ndvi_delta_p10`,
  `ndbi_delta_p90`), an `urbanization_score` (NDBI up while NDVI goes down),
  magnitude percentiles (p95, p99), and connected-component shape
  (`largest_component_px`, `n_components`, `component_fill_ratio`) — the last
  three already computed inside `despeckle()` and thrown away.
- **Two gate rules** that read the *signed* deltas the gate had been computing
  and discarding: built-up gain with vegetation loss, and a concentrated
  high-magnitude region.
- **`preprocess/radiometric.py`** — iterative pseudo-invariant-feature gain and
  offset matching. Ships off; see Measured and rejected.
- **`data/embeddings.py`** — AlphaEarth single-date landcover context, plus a
  clearly-labelled contaminated bitemporal probe that never enters a results
  table or threshold selection.
- **`ModelRate`** rate card carrying base, batch and cache-multiplier rates, so
  a batched or cache-hit call is priced rather than silently overcharged.
- **71 new tests** (112 to 183 offline, plus 4 behind the `oscd` marker).

### Changed

- `eval` and `e2e` report **three distinct reductions** — Tier-0 refusals, gate
  filtering of assessable tiles, and the total — instead of one conflated number,
  and `cities_scored` alongside `cities`.
- `_e2e_<split>.json` now states in-band that `savings_pct` is algebraically the
  candidate rate under a single flat price, and carries `batch_saving_pct`, which
  is independent of it.
- `make figures` regenerates all three README figures; the funnel diagram derives
  its counts from `_eval_test.json` rather than hardcoded string literals.
- CI uploads coverage, lint covers `scripts/`, and the Docker step's name matches
  what it runs.
- `pyproject.toml` declares the `[embeddings]` extra the README had been telling
  people to install.
- Removed the OPTIMUS README row and the `run-optimus` / `download-goes`
  references to commands that no longer exist.

### Measured and rejected

- **Threshold refitting found nothing better than what was already shipped.**
  8,640 combinations across seven axes; in-sample balanced accuracy 0.546 either
  way. The gate is at a local optimum for this rule family on this data — further
  gains need a different model, not a finer grid.
- **Radiometric normalization makes things worse on OSCD.** Logistic-regression
  AP 0.846 to 0.747, gate F1 0.629 to 0.620 — even though on synthetic pairs the
  fit recovers a known gain and offset almost exactly. On a multi-year pair too
  much ground has genuinely changed for the "invariant" population to be
  invariant. Off by default, reported rather than dropped.
- **AlphaEarth cannot be evaluated bitemporally here.** Coverage starts in 2017
  and 23 of OSCD's 24 cities have a first acquisition in 2015 or 2016, so
  embedding change is measurable on **1 of 24 pairs**. Only the single-date
  variant ships.

### Results after this release

| | Before | After |
|---|---|---|
| Gate + VLM precision | 0.971 (66/68, 3 of 10 cities) | **0.983** (59/60, **9 of 9** cities with candidates) |
| VLM retention of true change | 0.759 | 0.694 |
| Logistic regression AP | 0.802 | **0.846** |
| Gradient boosting AP | 0.861 | 0.859 |
| Recall at a risk-controlled threshold | not available | **0.823** overall, **failed on 2 of 9 cities** |
| Measured spend, 100 calls | $1.2941 (at the wrong rate) | **$0.4689** |

The retention drop is the honest shape of a precision filter measured on a
representative sample: it costs recall. The conformal per-city failure (dubai
0.623, milano 0.412) is the more interesting result — a city-level split breaks
the exchangeability the guarantee assumes, so it is reported as a property of the
assumption rather than patched away.

## [0.2.0] — 2026-08-11

- Learned baselines (logistic regression, gradient boosting) on the same
  features, same split, same runtime leakage assertion, so "why not just use a
  classifier" is answered with a number instead of a paragraph.
- Precision-recall curves and the README figures.
- The first live 100-call VLM run, with cost recorded from returned token usage
  rather than estimated.
- Spend controls after code review: a hard `--max-vlm-calls` cap, `priced` /
  `n_unpriced_calls` accounting so an unknown model cannot read as free, and
  three unreachable paths removed.

## [0.1.0] — 2026-08-10

- Rebuilt on real 13-band Sentinel-2 OSCD imagery, with a leakage-safe
  train/test split recovered from the label archives rather than hardcoded.
- Tier 0 quality masks on top-of-atmosphere reflectance, sub-pixel
  co-registration via phase cross-correlation, and the unknown-is-not-clean rule
  (`masks_assessed: false` with null fractions, never zero).
- The classical rule gate: robust `median + k*MAD` background, an absolute floor
  under the adaptive cut, NDWI over water where the land indices are undefined,
  and morphological despeckling.
- Tier 2 VLM verification with redacted metadata, and the three defects that
  running it live exposed — tiles too small to resolve at native size,
  placeholder evidence in the packager, and cost figures that were estimated
  rather than measured.
- Tier 3 analyst report, degrading to an explicitly-labelled template without an
  API key.

## [0.0.1] — 2026-05-20

Initial proof of concept: OSCD download, tiling, a first pass at the gate, and
the CLI skeleton.
