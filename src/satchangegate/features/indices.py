"""Spectral indices (pure numpy)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + _EPS)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (swir - nir) / (swir + nir + _EPS)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (green - nir) / (green + nir + _EPS)


def ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    return (green - swir) / (green + swir + _EPS)
