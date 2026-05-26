# SatChangeGate

Preprocessing-first satellite temporal change detection PoC.

## Executive summary

### The idea

Organizations monitoring land, infrastructure, or insurance exposure from satellite imagery face a cost problem: **running a vision-language model (VLM) or human review on every image pair does not scale**. Most pairs are unchanged, cloudy, or differ only because of season, sun angle, or atmospheric effects—not because something on the ground changed.

SatChangeGate is a **gated review pipeline**: cheap, deterministic preprocessing and a classical change gate filter the stream first; **expensive VLM and LLM steps run only on candidates** that survive that filter.

### How it is envisioned

The target operating model is a three-tier funnel:

```
Every satellite pair
    → Tier 0: quality + masks (cloud, snow, shadow, water)
    → Tier 1: classical gate (SSIM, spectral deltas, change area)
         ├─ no_change / low_quality  →  archive, no API spend
         └─ candidate_change         →  Tier 2: VLM verification
                                      →  Tier 3: LLM analyst report (optional)
```

**Design principles:**

- **Preprocessing-first** — normalize, mask, and score quality before any model inference.
- **Gate before VLM** — the classical layer exists to cut API cost and latency, not to replace final judgment.
- **VLM as verifier, not detector** — the VLM receives a curated package (before/after RGB, heatmap, metadata) and classifies *real change vs artifact*, with artifact-risk fields.
- **Human-in-the-loop by default** — outputs are analyst-ready; the system reduces volume, not accountability.

Longer term, the same gate can front Sentinel-2 STAC feeds, operational AOIs, and batch monitoring jobs. This repo is the **proof that the funnel works on public benchmarks**, not the production platform.

### The narrow slice (what this repo actually implements)

This PoC deliberately scopes to a **single, reviewable path** so technical and risk leadership can evaluate the approach without a full geospatial platform build.

| In scope (today) | Out of scope (deferred) |
|------------------|-------------------------|
| OSCD bitemporal pairs (24 RGB cities) | Global STAC / operational ingest |
| OPTIMUS time series (sample tars 148, 425, 455) | Full 12 TB OPTIMUS corpus |
| Classical gate with tuned thresholds | Deep change-detection models |
| Anthropic VLM on **candidates only** | VLM on every pair |
| Optional LLM markdown report | Streamlit / ops UI |
| GOES ABI as **context metadata** (CONUS) | GOES-driven gate logic (planned) |
| Eval harnesses + 100-pair E2E with live VLM | Production SLA, multi-tenant auth |

**End-to-end path in one line:**

OSCD / OPTIMUS pair → align / masks / quality → classical gate → [candidates] → VLM → LLM report

### What we have shown so far

- **Gate tuning** on labeled OPTIMUS tiles: F1 **0.93**, 100% change recall, ~27% of pairs filtered before VLM on that set.
- **100-pair random E2E** (24 OSCD + 76 OPTIMUS, live VLM): **35%** filtered by gate; **64** VLM calls; **~27%** of all pairs surfaced as `real_change` vs naive review-everything.
- **Cost narrative:** two-stage funnel reduces VLM spend; VLM further splits candidates into likely artifact vs real change.

### What a CRO / CTO should still treat as open

- OSCD inputs are RGB PNG previews—not full multispectral; gate recall on OSCD change pairs is ~46% in the E2E sample (conservative filter).
- Labels and metrics are benchmark-specific; production AOIs need their own validation.
- No production controls yet (audit logging, model versioning, rate limits, data residency) beyond PoC CLI and local reports.

Detailed metrics, commands, and roadmap follow below.

### Sample reports (representative outputs)

Committed examples in [`public_reporting_sample/`](public_reporting_sample/) illustrate what the pipeline produces. They are **representative snapshots** from PoC runs. Full run artifacts stay under `data/reports/` (gitignored locally).

| Sample | What it shows |
|--------|----------------|
| [`beirut.md`](public_reporting_sample/beirut.md) | End-to-end **LLM analyst report** for OSCD Beirut after classical gate + VLM (construction change, evidence sections) |
| [`_optimus_eval_summary.md`](public_reporting_sample/_optimus_eval_summary.md) | **Gate-only eval** on labeled OPTIMUS tiles (group 148): metrics, confusion matrix, per-tile features |
| [`_dev_tests_summary.json`](public_reporting_sample/_dev_tests_summary.json) | **Dev test battery** summary (negative controls + small OPTIMUS fixtures) |

---

**Narrow slice (technical):** OSCD / OPTIMUS pairs → preprocessing + ephemeral masks → classical change gate → Anthropic VLM (candidates only) → Anthropic LLM markdown report.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev,data]"
copy .env.example .env   # set ANTHROPIC_API_KEY
```

## Data layout (`data/raw/`, gitignored)

| Path | Source |
|------|--------|
| `data/raw/oscd/` | IEEE OSCD images + merged labels (24 RGB pairs) |
| `data/raw/optimus/` | OPTIMUS eval JSON, `index.json`, `images/{N}.tar`, extracted groups |
| `data/raw/goes/` | GOES ABI NetCDF snapshots (weather context) |

Reports and eval artifacts: `data/reports/` (gitignored).

## OSCD data

```bash
satchangegate download-oscd --out data/raw/oscd
```

Download from [IEEE DataPort OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection):

1. **Images zip** — extract to `data/raw/oscd/` (`<city>/pair/img1.png`, `img2.png`)
2. **Train Labels zip** — merge `<city>/cm/cm.png`
3. **Test Labels zip** — same for test cities

The common IEEE **Images zip** is RGB PNG previews, not per-band GeoTIFFs. Full MSI may require the [official OSCD site](https://rcdaudt.github.io/oscd/) or Sentinel-2 via Medusa.

## OPTIMUS data

Do **not** `git clone` the full Hugging Face repo (~12 TB). Tar files are `images/{group_index}.tar` (~25 GB each), resolved via `index.json`.

```bash
satchangegate download-optimus --metadata-only
satchangegate download-optimus --with-index          # ~464 MB index.json
satchangegate download-optimus --dev-fixture         # 2-frame tree from OSCD PNGs
satchangegate download-optimus --allow-large-download --tile-id 4472_3910
```

OPTIMUS uses a separate loader (`src/satchangegate/data/optimus.py`) — monthly time series in numbered tars, not OSCD-style pair folders. Eval labels: **483 series** (286 no-change, 197 change) in `2024_dataset_evaluation.json`.

## Commands

```bash
# Single OSCD pair (VLM/LLM if API key set and gate = candidate_change)
satchangegate run --pair beirut --out data/reports

# Negative controls
satchangegate run --pair beirut --negative-mode identity|stable|photometric

# OPTIMUS bitemporal (first/last frame)
satchangegate run-optimus --tile-id 4472_3910

# GOES ABI context (CONUS; metadata only today)
satchangegate download-goes --when 2020-06-15T17:00:00Z
satchangegate run --pair beirut --attach-goes

# Evaluation
satchangegate eval --split all
satchangegate eval-optimus --group 148    # or 425
satchangegate e2e-random --n 100 --seed 42   # mixed OSCD+OPTIMUS, live VLM

# Dev batteries
satchangegate dev-tests
python scripts/run_dev_tests.py
python scripts/run_e2e_random.py --n 100

pytest
pytest -m "not e2e"    # skip live VLM test
```

## Architecture

```
Pair (OSCD | OPTIMUS) → align / masks / quality → classical gate
    → [no_change | low_quality]  stop (no VLM)
    → [candidate_change]         → VLM verify → LLM markdown report
```

## Findings (PoC validation)

### Classical gate tuning (OPTIMUS labeled tiles, groups 148 + 425)

Tuned on **11 labeled tiles** (real Sentinel-2 TCI, first vs last frame, no VLM):

| Metric | Before | After tuning |
|--------|--------|--------------|
| F1 | 0.71 | **0.93** |
| Recall (change) | 0.71 | **1.00** |
| Specificity (no-change) | 0.50 | **0.75** |
| Filtered before VLM | ~27% | **~27%** |

Gate logic (`configs/thresholds.yaml`, `features/classical.py`):

- Landcover-first spectral (NDVI/NDBI; NDWI only for water at high threshold)
- Phenology guard for long-span false diffs (weak spectral + low SSIM)
- High-area structural + moderate landcover paths for real change
- One remaining FP on forest phenology (`4244_2805`) — diffuse vegetation shift vs persistent built change

See `data/reports/_gate_tuning_report.md` after running eval.

### 100-pair E2E with live VLM (`e2e-random`, seed 42)

Stratified sample: **24 OSCD + 76 OPTIMUS** from a 6,024-pair pool. **99/100** completed (~9.5 min).

| Stage | Result |
|-------|--------|
| Classical gate filtered | **35%** (no VLM) |
| VLM calls | **64** |
| VLM `real_change` | **27** (~27% of all pairs) |
| VLM `likely_artifact` | **36** (~56% of candidates) |

On labeled OSCD pairs in sample (all change-positive): gate precision **1.0**, recall **0.46** — gate is conservative (saves cost, misses some real changes on RGB PNGs).

See `data/reports/_e2e_random_summary.md`.

### Other eval highlights

- **OSCD all split (24 pairs, gate only):** ~54% candidate, F1 0.70 vs 100% candidate baseline
- **Negative controls:** identity + photometric pass; stable crop still sensitive on RGB previews
- **GOES ABI:** fetch works for CONUS; attached as pipeline metadata, not yet used in gate decisions
- **pytest:** 29 tests including mocked E2E; live VLM test marked `@pytest.mark.e2e`

## Improvements in this phase

| Area | What changed |
|------|----------------|
| **OPTIMUS** | Loader, index-based tar lookup, dev fixtures, `eval-optimus`, extract with completion marker |
| **GOES** | `goes2go` integration, Windows-safe config bootstrap |
| **Negative controls** | Identity / stable crop / photometric modes for OSCD |
| **Gate** | Multi-rule decision engine tuned on OPTIMUS labels; `tune_gate.py` sweep helper |
| **E2E eval** | `e2e_random_eval.py` — stratified random pairs across OSCD + OPTIMUS with live VLM |
| **CLI** | `download-optimus`, `download-goes`, `dev-tests`, `eval-optimus`, `e2e-random` |
| **Tests** | OPTIMUS, GOES, negative controls, E2E random (mocked + optional live) |

## Future paths

1. **Gate recall on OSCD** — radiometric harmonization before metrics; true MSI bands; spatial coherence of change mask (fix forest phenology FPs like `4244_2805`)
2. **GOES in gate logic** — `low_quality` when ABI cloud fraction is high (CONUS AOIs)
3. **Labeled OPTIMUS in E2E** — stratified random sample from 483 eval tiles; gate + VLM F1 on no-change vs change
4. **LLM reports on E2E subset** — `--with-llm` on `e2e-random` for analyst markdown on `real_change` cases
5. **Cost model** — $/pair for gate-only vs gate+VLM vs naive VLM-every-pair
6. **Production deferred** — Sentinel-2 STAC, SpaceNet 7, deep models, Streamlit UI, global GOES coverage

## Investor framing (summary)

Two-stage funnel reduces expensive VLM review: **~35% filtered free** by classical gate, VLM only on candidates, VLM further separates artifacts from real change. In the 100-pair sample, **~27%** surfaced as high-confidence `real_change` vs reviewing everything. Position as a **cost-control and trust layer** for geospatial AI ops, not a replacement for human sign-off.

## License

OSCD, OPTIMUS, and downstream datasets have their own licenses. See dataset providers before commercial use.
