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

## [0.4.0] — 2026-09-20

A local review UI, and the corrections that had to land before it could be
honest. The ordering was the whole design: a demo is the easiest possible place
to undo what this repo is for, so a 2026-09-19 audit ran first and nothing in
`webui/` was written until every defect it found was fixed and disclosed.

The audit's headline is that **nothing published was materially wrong**. The
conformal lambda is unchanged, its bound got *tighter*, and the five published
budget rows are byte-identical after the tie fix. What changed is that those
properties are now established rather than assumed -- which is the only claim
this project has ever made for itself.

### Corrected

Four defects found by a 2026-09-19 audit run while scoping a web UI. One of them
is in a published guarantee. All four were verified against `eaad03b` before any
change was made, and all four are now covered by a failing-first test.

- **The conformal guarantee's calibration sample was not independent of the
  predictor it certified.** Learn-then-Test requires the score being calibrated
  to come from a model that was not fitted to the calibration tiles.
  `run_conformal` scored every training city with the already-fitted shipped
  thresholds and *then* drew the calibration split, so all four calibration
  cities (abudhabi, mumbai, nantes, pisa) sat inside the fourteen `tune` had
  swept. `_fit_rows` was discarded with a leading underscore because, by that
  point, there was nothing left to fit — and the disjointness assertion checked
  calibration against *test* only, never against *fit*, which was the pair that
  actually overlapped.

  What was never affected: the held-out FNR, measured on nine cities nothing had
  seen. What was not established: the 90% confidence bound attached to λ.

  The calibration cities are now withheld *before* the gate is refit, via a new
  `tune_gate.sweep_tiles`, and a three-way fit/calibration/test assertion
  replaces the two-way one. **Correcting it moved the numbers very little**: λ is
  unchanged at 0.200, the bound tightened from 0.198 to **0.186**, and held-out
  recall moved 0.823 → **0.826**. The refit differs from the shipped thresholds
  on one axis, `urbanization_score_min` (0.10 → 0.05), which the tuning
  provenance already records as a grid-order tie rather than evidence. So the
  published figures were not an artifact of the leak — which is now a
  measurement, because `conformal` reports the old contaminated result alongside
  the corrected one rather than asserting the difference is small.

- **`operating-points` could exceed the budget it was pricing.**
  `threshold_for_call_budget` returned the k-th highest score and callers apply
  it as `score >= threshold`, so a tie group straddling the budget line was
  admitted whole. `_metrics_at` counted the overshoot without preventing it.
  Confidence is rounded to four decimals and 163 of 621 held-out tiles share a
  value with another (largest group: 87), so this was live rather than
  hypothetical: sweeping 50 budgets from $0.05 to $2.50 found **three that
  overshot by one call** ($1.70, $1.80, $2.25). The five published budgets were
  not among them and their rows are unchanged. A straddling tie is now excluded
  and the unspent remainder reported as `unused_calls`. Separately, a zero
  budget serialised its threshold as `1.0` — which readmits any tile scoring
  exactly 1.0 — and is now `null`, meaning admit nothing.

- **OSCD change masks decoded correctly only from the PNG.** The GeoTIFF branch
  of `load_label_mask` applied `arr > 0` to OSCD's **1 = unchanged / 2 = changed**
  encoding, so `bercy-cm.tif` decoded to an all-ones mask — every pixel changed.
  It never fired in practice only because every city also ships `cm/cm.png`,
  which wins the candidate order in `_label_path`; the defect sat behind a lucky
  preference. Encoding is now declared rather than inferred, through a new
  `decode_label_array`, which raises on values the declared encoding does not
  define instead of guessing. The corrected TIF decode reproduces the PNG
  exactly: 0.0074 changed either way, against 1.0000 before.

- **`_verify_sequential` could lose paid work.** It accumulated every result and
  returned them after the loop, and the caller wrote them afterwards, so a crash
  on call 100 lost the 99 that had already been bought. Results are now yielded
  and written one at a time.
- **An unreadable batch manifest failed open.** `_live_batch` returned "no
  batch" on a `JSONDecodeError`, so the next run submitted afresh for work that
  may already have been in flight and paid for. It now raises
  `UnreadableBatchManifest` and stops. One existing test asserted the old
  behaviour; it has been corrected, and the correction is the point.
- **Tier 0 reported non-finite pixels as clean.** Every mask test is a
  comparison and a comparison against NaN is False, so an unobserved pixel came
  out as neither cloud nor snow nor shadow — which the code then read as valid.
  An all-NaN six-band array returned `masks_assessed: true` with
  `valid_fraction: 1.0`, which is exactly the "unknown is not clean" rule this
  project states for itself. `valid` now excludes unobserved pixels, and a scene
  with no finite readings anywhere is unassessable rather than clean. Harmless
  on OSCD, whose rasters are fully populated, and a live hole for any uploaded
  scene carrying nodata.

### Added

- **A local web UI** (`satchangegate serve`, behind a `[ui]` extra) and the
  shared `services/` layer beneath it, which the CLI now goes through too. The
  UI promises that every panel shows the command producing it; that is only
  worth anything if the command and the panel run the same code, so commands are
  rendered from the same validated request object that ran.
- **`satchangegate run-images`** — the funnel over two image files under a
  declared contract. Bands are named rather than positional, reflectance scaling
  is stated rather than guessed from dtype, footprints must genuinely overlap,
  and nodata becomes invalid rather than zero. Three-band imagery routes to a
  separate lane where the gate refuses and says which bands it lacks, returning
  structural evidence under its own result type so nothing can average the two.
- **A reservation ledger**, so a dollar cap is actually a dollar cap. Counting
  calls and multiplying by an average is not a spend control: a call is bounded
  at 4,096 output tokens with four SDK retries, and the analyst report is a
  separate paid call the count never saw. Spend is now reserved at a
  conservative upper bound *before* dispatch and settled afterwards from
  reported tokens. A model with no published rate is refused rather than priced
  at zero -- `UsageRecord.cost_usd` returning 0.0 is right for a report and a
  blank cheque for an authorisation. Money is integer micro-dollars, holds
  survive a restart, and an outcome nobody knows keeps its reservation, because
  a request that timed out may still have been served.
- **`satchangegate ab-normalize`** — the producing command
  `public_reporting_sample/_ab_normalize.json` never had. It was committed from
  0.3.0 with nothing that could regenerate it, which is the same defect class as
  a headline number with no code path; a negative result is not exempt. The
  regenerated artifact reproduces the committed one to four decimals apart from
  the rule-gate AP, which moved exactly as the `urbanization_score_min` adoption
  predicted.
- **`--thresholds`** on every scoring operation, so the threshold playground's
  export is a file something can actually consume rather than a suggestion.
- A batch-manifest viewer, and a scorer-parity report that states plainly which
  code paths honour `scorer.kind` and which always run the rules — `eval`
  dispatches, `run` and `e2e` call `decide` directly, and comparing across the
  two would be comparing different models.
- `--stride` and `--pos-min-fraction` on `tiles`, and `--overwrite` on `e2e`.

### Changed

- **`urbanization_score_min` raised from a tie to a choice: 0.10 -> 0.05.** The
  0.3.0 sweep reported 0.05 only because it comes first in the grid and scored
  identically to 0.10, so the code default was kept and the provenance comment
  said as much. The conformal correction above gave that tie a tiebreaker: a
  calibrated guarantee needs a gate fitted without the calibration cities, and
  that refit -- nine cities instead of fourteen -- independently chose 0.05.
  Keeping 0.10 would have meant publishing a guarantee certifying a gate the repo
  does not ship.

  **Adoption is free on accuracy.** It moves zero gate decisions on the held-out
  split; the confusion matrix is identical (TP 178 / FP 43 / FN 167 / TN 146,
  precision 0.8054, recall 0.5159, F1 0.6290) and the funnel candidate set is
  unchanged, so the 100 paid verifications remain valid. What moves is
  `gate_confidence` on 136 of 621 tiles, all upward and by at most 0.058, which
  shifts two published tables: the operating-point thresholds and the rule gate's
  own PR curve (AP 0.715 -> 0.712, ROC AUC 0.719 -> 0.714). Both regenerated.

  `satchangegate conformal` now reports `matches_shipped_thresholds: true`, and
  says plainly that its contaminated-vs-corrected comparison is degenerate as a
  result -- what had been contaminated was the *provenance* of those thresholds,
  not their values.

The common shape is worth naming: three of the four were invisible because
something else masked them — a file-preference order, a dataset with no nodata,
a score distribution that happened not to tie at the published budgets. A test
at a comfortable value passes in all three cases. The tests added here sit at
the boundary instead, which is the lesson the 0.3.0 review already recorded
about `--max-vlm-calls` and did not generalise far enough.

### Corrected earlier, after the 0.3.0 tag-equivalent

- **The 0.3.0 rule-count fix was incomplete.** "three documented rules" was
  corrected in `baseline.py` but a second occurrence in
  `features/classical.py`'s module docstring was missed, so the file that
  *defines* the rules still described three of them. The changelog entry claimed
  the fix generally. Now corrected there too, with a note that two of the six
  arrived once the gate began reading its own signed deltas.
- The 0.3.0 entry recorded the ROC AUC correction as "0.802". The value shipped
  is **0.801**, matching the generated baselines table; 0.802 was a rounding of
  0.8015 that disagreed with the artifact beside it. The entry now says 0.801.

Both are small, and both are the drift this project exists to catch: a
correction that names one occurrence of a stale figure and leaves another, and a
changelog line quoting a number the repo does not print.

### Measured outcomes

| | Before | Now |
|---|---|---|
| Conformal lambda | 0.200 | **0.200**, and now independently calibrated |
| Calibration bound (alpha 0.20) | 0.198 | **0.186** |
| Held-out FNR / recall | 0.177 / 0.823 | **0.174 / 0.826** |
| Gate precision / recall / F1 | 0.805 / 0.516 / 0.629 | unchanged |
| Rule gate AP | 0.715 | 0.712 (confidence moved; decisions did not) |
| Budgets that overshoot their cap | 3 of 50 swept | **0** |
| OSCD GeoTIFF label decode | 1.0000 changed | **0.0074**, matching the PNG exactly |
| Offline tests | 185 | **361** |
| CLI commands | 14 | **17** |

Two things this release deliberately did *not* do. It did not unify the scorer
dispatch -- `eval` honours `scorer.kind` while `run` and `e2e` call `decide`
directly -- so the capability endpoint names which paths apply rather than
pretending the toggle is global. And it did not wire GOES or Earth Engine, whose
prerequisites do not exist on the machine this was built on; shipping a button
for an unverified capability is not a feature.

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

### Documentation

- **`CHANGELOG.md`** — this file. Previously there was none; version history
  lived only in commit messages.
- **README `Corrections` table** — the six claims above, each with the real
  number and the command that reproduces it.
- **README `Changelog` section** — a condensed version history and a
  before/after outcome table, so a reader can see what moved without opening
  this file.
- Per-city allocation of the stratified sample published in the funnel section,
  which is what actually substantiates the word "stratified".
- Pixel-level metrics published as precision / recall / F1 / IoU / specificity
  over 2.40 M observed pixels, rather than a single F1 with no provenance.
- Stale figures corrected in the prose: "eleven numeric features" (eighteen),
  "ROC AUC 0.805" (0.801, matching the generated table), "five rules" and
  "three documented rules" (six), and an audit table introduced as five rows
  that had six.
- The `thresholds.yaml` provenance comment still carried the conflated "gate
  filters 64.6%" claim in the very file that produces the number. Corrected.

### Fixed before release

Found by reviewing this release's own diff, in code that had not shipped:

- **`--max-vlm-calls` was not a hard cap.** The stratified sampler's floor of one
  call per city was applied unconditionally, so a budget below the number of
  cities returned one tile *per city* instead: `--max-vlm-calls 1` selected 9
  calls on the held-out split. The budget now dominates the floor and spends on
  the largest contributors. No published figure changes — the documented run used
  a budget of 100 and its per-city allocation is byte-identical — but a user
  trying a single call would have paid for nine.
  The cap test that existed asserted only at budget = 13 against a five-city
  pool, which is the safe side of the boundary; it now sweeps every budget.
- **`pixel_metrics()` scored F1 and IoU over different pixel populations** — the
  confusion matrix was restricted to observed pixels while IoU was computed from
  the raw arrays, so masked-out cloud reached one number and not the other. Both
  now derive from the same masked matrix. The published pixel figures come from
  `run_eval`'s accumulator, which was already consistent, so they are unaffected.

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
