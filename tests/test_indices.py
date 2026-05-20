"""Tests for spectral indices."""

import numpy as np

from satchangegate.features.indices import ndbi, ndsi, ndvi, ndwi


def test_ndvi_high_vegetation():
    nir = np.array([0.8], dtype=np.float32)
    red = np.array([0.1], dtype=np.float32)
    v = ndvi(nir, red)[0]
    assert v > 0.6


def test_ndvi_bare_soil():
    nir = np.array([0.2], dtype=np.float32)
    red = np.array([0.2], dtype=np.float32)
    v = ndvi(nir, red)[0]
    assert abs(v) < 0.1


def test_ndbi_built_surface():
    swir = np.array([0.5], dtype=np.float32)
    nir = np.array([0.2], dtype=np.float32)
    v = ndbi(swir, nir)[0]
    assert v > 0.3


def test_ndwi_water():
    green = np.array([0.5], dtype=np.float32)
    nir = np.array([0.1], dtype=np.float32)
    v = ndwi(green, nir)[0]
    assert v > 0.3


def test_ndsi_snow():
    green = np.array([0.6], dtype=np.float32)
    swir = np.array([0.1], dtype=np.float32)
    v = ndsi(green, swir)[0]
    assert v > 0.4
