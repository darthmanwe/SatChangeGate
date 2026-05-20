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

Download from [IEEE DataPort OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) and extract under `data/raw/oscd/<pair_id>/imgs_1|imgs_2/`.

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
