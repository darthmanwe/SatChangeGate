"""Tests for ephemeral masks."""

import numpy as np

from satchangegate.preprocess.masks import compute_ephemeral_masks


def test_cloud_mask_bright_low_ndvi(cfg):
    h, w = 32, 32
    bands = {
        "B02": np.full((h, w), 0.9, dtype=np.float32),
        "B03": np.full((h, w), 0.85, dtype=np.float32),
        "B04": np.full((h, w), 0.88, dtype=np.float32),
        "B08": np.full((h, w), 0.05, dtype=np.float32),
        "B11": np.full((h, w), 0.1, dtype=np.float32),
        "B12": np.full((h, w), 0.08, dtype=np.float32),
    }
    m = compute_ephemeral_masks(bands, cfg)
    assert m.cloud_fraction > 0.5


def test_cloud_mask_not_on_vegetation(cfg):
    h, w = 32, 32
    bands = {
        "B02": np.full((h, w), 0.05, dtype=np.float32),
        "B03": np.full((h, w), 0.08, dtype=np.float32),
        "B04": np.full((h, w), 0.06, dtype=np.float32),
        "B08": np.full((h, w), 0.6, dtype=np.float32),
        "B11": np.full((h, w), 0.15, dtype=np.float32),
        "B12": np.full((h, w), 0.12, dtype=np.float32),
    }
    m = compute_ephemeral_masks(bands, cfg)
    assert m.cloud_fraction < 0.1


def test_snow_mask_high_ndsi(cfg):
    h, w = 32, 32
    bands = {
        "B02": np.full((h, w), 0.5, dtype=np.float32),
        "B03": np.full((h, w), 0.7, dtype=np.float32),
        "B04": np.full((h, w), 0.6, dtype=np.float32),
        "B08": np.full((h, w), 0.5, dtype=np.float32),
        "B11": np.full((h, w), 0.05, dtype=np.float32),
        "B12": np.full((h, w), 0.04, dtype=np.float32),
    }
    m = compute_ephemeral_masks(bands, cfg)
    assert m.snow_fraction > 0.3
