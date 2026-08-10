# SatChangeGate

A cost-controlled review funnel for satellite change detection: cheap deterministic
preprocessing and a classical gate filter the stream, and expensive vision-model
review runs only on what survives.

[![CI](https://github.com/darthmanwe/SatChangeGate/actions/workflows/ci.yml/badge.svg)](https://github.com/darthmanwe/SatChangeGate/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

```
Every satellite pair
    → Tier 0: quality — cloud/snow/shadow/water masks, co-registration
    → Tier 1: classical gate — spectral indices, SSIM, pHash, change vector
         ├─ no_change / low_quality  →  archived, no API spend
         └─ candidate_change         →  Tier 2: VLM verification
                                      →  Tier 3: LLM analyst report
```

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

Measured on the **10 held-out OSCD cities**, which are never seen during
threshold tuning. Disjointness between the tuning and evaluation splits is
asserted at runtime, not assumed.

| Metric | Value | 95% CI |
|---|---|---|
| Precision | **0.804** | 0.747 – 0.852 |
| Recall | 0.513 | 0.460 – 0.565 |
| Specificity | 0.772 | 0.708 – 0.827 |
| F1 | 0.626 | — |
| Balanced accuracy | 0.643 | — |

n = 534 scored tiles (345 change / 189 no-change) from 621 total; 87 were
rejected by Tier 0 quality checks and are reported separately rather than
counted as negatives. Confusion matrix: TP 177 · FP 43 · FN 168 · TN 146.

**The operating point is high precision, moderate recall.** The gate filters
**64.6%** of tiles before any API call, and when it does flag something it is
right about 80% of the time. It also misses about half of all change. For a
spend-control layer in front of human or model review that is usually the right
trade — but it is a trade, and it should be stated as one rather than buried.

Reproduce: `make data && make tune && make eval`. Sample outputs are committed
under [`public_reporting_sample/`](public_reporting_sample/).

### What the funnel is worth

`satchangegate e2e --split test --vlm --max-vlm-calls 20` runs the funnel with
live verification and reports cost derived from the token usage the API actually
returned — input/output tokens, dollars per pair, and the review-everything
counterfactual. The committed sample is a gate-only run ($0.00), because
publishing a cost figure that nobody can reproduce without a key is how the
previous version of this README got into trouble.

## What this is not

Honest framing matters more here than a good-looking number, so:

- **This is a triage gate, not a segmentation model.** At the pixel level it
  reaches F1 ≈ 0.13, against roughly 0.50–0.60 for supervised deep models on
  OSCD. It is not competitive with them and is not trying to be — it is a
  ~10 ms/tile filter that decides whether to spend money.
- **OSCD labels only *urban* change.** Seasonal, agricultural, and hydrological
  change across a multi-year gap is spectrally enormous and labelled negative.
  A magnitude-based detector keys on exactly the wrong signal; even directional
  built-up features (NDBI up, NDVI down) separate the classes at only AUC ≈ 0.56
  at tile level. This bounds how well *any* classical gate can score on this
  benchmark, and it is why the numbers above are what they are.
- **Cloud filtering rarely fires on OSCD**, because the dataset curators picked
  clear scenes. What Tier 0 actually catches here is **co-registration failure**:
  saclay_e (4.21 px) and saclay_w (2.04 px) exceed the 1.5 px tolerance and are
  correctly withheld from the gate.
- **Thresholds are dataset-specific.** They were fitted on 14 cities. Any new
  AOI needs its own validation.

## How it works

**Tier 0 — quality.** Cloud, snow, and cloud-shadow masks from physically
motivated tests on top-of-atmosphere reflectance, plus sub-pixel co-registration
via phase cross-correlation. Water is *reported but not treated as
contamination*: removing it would make flooding undetectable by construction.
When a source lacks the bands to compute masks, quality is reported as
`masks_assessed: false` with null fractions — unknown is represented as unknown,
never as clean.

**Tier 1 — the gate.** Per-pixel change magnitude from NDVI/NDBI (NDWI over
water, where the land indices are undefined), thresholded against a robust
per-scene background (`median + k·MAD`) with an absolute floor and ceiling, then
despeckled by morphological opening and connected-component filtering. Tile-level
features — signed index deltas, SSIM, perceptual hash, change-vector magnitude,
changed area — feed one pure `decide()` function that returns a decision, a
human-readable reason, and a calibrated confidence.

Three design choices are load-bearing:

- *Robust background, not a percentile.* A percentile cut assumes change is rare;
  on a scene where 14% of pixels changed it lands inside the changed population
  and suppresses exactly what it should detect.
- *An absolute floor under the adaptive cut.* Without it, a scale-free threshold
  always selects its top N% — so a pure illumination shift reads as change.
- *Land indices are undefined over water.* Deep water has SWIR reflectance around
  0.008, so a few thousandths of sensor noise swings NDBI across most of its
  range and a static lake reads as violent change.

Each of those was caught by a control that fails loudly rather than by inspection.

**Tier 2 — VLM verification.** Candidates are packaged as before/after RGB plus a
heatmap overlay and sent for an independent verdict, validated server-side
against a Pydantic schema via structured outputs. The metadata the model sees is
**redacted**: it carries acquisition dates and observation quality, and
deliberately withholds the gate's verdict and every discriminative gate feature.
The VLM is the gate's independent verifier; telling it the gate's answer would
confound any agreement statistic between the two.

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
satchangegate dev-tests              # offline control battery
satchangegate verify                 # check dataset + config
```

Anything that can spend money defaults to not spending it; `--vlm` is opt-in and
`--max-vlm-calls` is a hard cap.

## Development

```bash
make lint type test    # ruff, mypy, 101 offline tests
make cov               # coverage report
```

The test suite runs offline with no API key and no dataset: `pytest` deselects
the `oscd`, `e2e`, and `vlm` markers by default, and a session fixture unsets
`ANTHROPIC_API_KEY` so no test can authenticate by accident. CI runs lint, types,
tests on Python 3.10–3.12 across Ubuntu and Windows, a wheel-install check that
the packaged config resolves outside a source checkout, and a Docker build.

## Data

| Source | Role |
|---|---|
| OSCD (24 cities, 13-band Sentinel-2 L1C) | Primary. Auto-downloaded and checksum-verified from the [Hugging Face mirror](https://huggingface.co/datasets/hkristen/oscd). |
| `tests/fixtures/mini_oscd` | Committed synthetic pairs so a clean clone runs end to end. |
| OPTIMUS | Optional secondary track. RGB-only (TCI), so indices from it are flagged `rgb_only` and are not physically calibrated. |

The official train/test split is recovered from the label archives themselves and
written to `splits.json`, rather than hardcoded.

## License

MIT — see [LICENSE](LICENSE). This covers the source code only; OSCD and any
other imagery carry their own licenses, so check with the dataset providers
before commercial use.
