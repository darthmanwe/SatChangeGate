"""Shared fixtures.

The test session deliberately does **not** load the developer's ``.env``. The
previous conftest called ``load_thresholds()``, which loaded ``.env`` into
``os.environ`` for every run — so a bare ``pytest`` on a machine with a real key
could make live, billed API calls.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from satchangegate.config import Settings

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_oscd"


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee no test can accidentally authenticate against the real API."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_VLM_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_LLM_MODEL", raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def fixture_root() -> Path:
    if not FIXTURE_ROOT.is_dir():
        pytest.skip("fixtures missing; run python scripts/make_fixtures.py")
    return FIXTURE_ROOT


@pytest.fixture
def oscd_root() -> Path:
    root = Path(os.environ.get("SCG_OSCD_ROOT", "data/raw/oscd"))
    if not (root / "splits.json").is_file():
        pytest.skip("OSCD dataset not present; run satchangegate download-oscd")
    return root


def _synthetic_bands(
    *,
    nir: float = 0.32,
    red: float = 0.035,
    green: float = 0.06,
    blue: float = 0.03,
    swir1: float = 0.15,
    swir2: float = 0.07,
    size: int = 64,
) -> dict[str, np.ndarray]:
    """Uniform reflectance patch with the six gate bands."""

    def full(value: float) -> np.ndarray:
        return np.full((size, size), value, dtype=np.float32)

    return {
        "B02": full(blue),
        "B03": full(green),
        "B04": full(red),
        "B08": full(nir),
        "B11": full(swir1),
        "B12": full(swir2),
    }


@pytest.fixture
def synthetic_bands() -> Callable[..., dict[str, np.ndarray]]:
    """Factory fixture for uniform reflectance patches.

    Exposed as a fixture rather than an importable helper: ``from
    tests.conftest import ...`` relies on the repository root being on
    sys.path, which holds under an editable install but fails on a clean CI
    runner.
    """
    return _synthetic_bands
