"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from satchangegate.config import load_thresholds
from satchangegate.preprocess.masks import EphemeralMasks


@pytest.fixture
def cfg():
    return load_thresholds()


@pytest.fixture
def synthetic_bands_vegetation():
    """Vegetation-like t1 bands."""
    h, w = 64, 64
    return {
        "B02": np.full((h, w), 0.05, dtype=np.float32),
        "B03": np.full((h, w), 0.08, dtype=np.float32),
        "B04": np.full((h, w), 0.06, dtype=np.float32),
        "B08": np.full((h, w), 0.55, dtype=np.float32),
        "B11": np.full((h, w), 0.12, dtype=np.float32),
        "B12": np.full((h, w), 0.10, dtype=np.float32),
    }


@pytest.fixture
def synthetic_bands_bare():
    """Built/bare surface t2."""
    h, w = 64, 64
    return {
        "B02": np.full((h, w), 0.25, dtype=np.float32),
        "B03": np.full((h, w), 0.22, dtype=np.float32),
        "B04": np.full((h, w), 0.28, dtype=np.float32),
        "B08": np.full((h, w), 0.18, dtype=np.float32),
        "B11": np.full((h, w), 0.35, dtype=np.float32),
        "B12": np.full((h, w), 0.30, dtype=np.float32),
    }


@pytest.fixture
def synthetic_bands_cloudy():
    """Very bright cloudy scene."""
    h, w = 64, 64
    return {
        "B02": np.full((h, w), 0.85, dtype=np.float32),
        "B03": np.full((h, w), 0.82, dtype=np.float32),
        "B04": np.full((h, w), 0.80, dtype=np.float32),
        "B08": np.full((h, w), 0.05, dtype=np.float32),
        "B11": np.full((h, w), 0.10, dtype=np.float32),
        "B12": np.full((h, w), 0.08, dtype=np.float32),
    }


@pytest.fixture
def all_valid_mask():
    h, w = 64, 64
    valid = np.ones((h, w), dtype=bool)
    return EphemeralMasks(
        cloud=np.zeros((h, w), dtype=bool),
        snow=np.zeros((h, w), dtype=bool),
        water=np.zeros((h, w), dtype=bool),
        shadow=np.zeros((h, w), dtype=bool),
        valid=valid,
    )
