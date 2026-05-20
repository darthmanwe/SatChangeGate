"""Load project configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = ROOT / "configs" / "thresholds.yaml"


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_THRESHOLDS
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
