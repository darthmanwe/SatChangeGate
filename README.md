# SatChangeGate

Preprocessing-first satellite temporal change detection PoC.

**Narrow slice:** OSCD pairs → preprocessing + ephemeral masks → classical change gate → Anthropic VLM (candidates only) → Anthropic LLM markdown report.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env   # set ANTHROPIC_API_KEY
```

## OSCD data

```bash
satchangegate download-oscd --out data/raw/oscd
```

Download from [IEEE DataPort OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection):

1. **Images zip** — extract to `data/raw/oscd/` (creates `<city>/pair/img1.png`, `img2.png`, etc.)
2. **Train Labels zip** — merge each `<city>/cm/cm.png` into `data/raw/oscd/<city>/cm/`
3. **Test Labels zip** — same as train labels for test cities

IEEE lists three separate files (images, train labels, test labels). The images archive is described as multispectral Sentinel-2, but the common IEEE **Images zip** extract is RGB PNG previews (`pair/img1.png`); per-band GeoTIFFs (`imgs_1/B02.tif`, etc.) may require the [official OSCD site](https://rcdaudt.github.io/oscd/) or building from Sentinel-2 via Medusa.

## Commands

```bash
# Single pair (calls VLM/LLM if API key set and gate = candidate_change)
satchangegate run --pair beirut --out data/reports

# Eval on test split (skips API by default)
satchangegate eval --split test

# Run tests
pytest
```

## Architecture

```
OSCD pair → align/masks/quality → classical gate → [candidates] → VLM → LLM report
```

Deep models, SpaceNet 7, Sentinel-2 STAC, and Streamlit are deferred to later phases.

## License

OSCD and downstream datasets have their own licenses. See dataset providers before commercial use.
