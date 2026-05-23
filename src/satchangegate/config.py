"""Load project configuration from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS = ROOT / "configs" / "thresholds.yaml"


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not override existing)."""
    env_path = path or ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    load_env_file()
    cfg_path = path or DEFAULT_THRESHOLDS
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
