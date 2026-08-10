"""Normalised-difference spectral indices.

All functions take top-of-atmosphere reflectance arrays and return values in
[-1, 1]. Division is guarded: where the denominator is ~0 the result is 0 rather
than +/-inf, and non-finite inputs propagate as 0 rather than poisoning
downstream means.

Band conventions (Sentinel-2 MSI):
    B03 green, B04 red, B08 NIR (842 nm), B11 SWIR-1 (1610 nm)
"""

from __future__ import annotations

import numpy as np

# Minimum summed reflectance for a normalised difference to be meaningful.
#
# A ratio index is unstable wherever the denominator approaches zero. Deep water
# has SWIR reflectance around 0.008, so sensor noise of a few thousandths swings
# NDBI across most of its range and a perfectly static lake reads as dramatic
# change. Below this floor the index is reported as 0 (no information) rather
# than as a large spurious value.
MIN_DENOMINATOR = 0.02


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), zero wherever the denominator is too small to trust."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    num = a - b
    den = a + b
    out = np.zeros_like(num, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (den > MIN_DENOMINATOR)
    np.divide(num, den, out=out, where=ok)
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index: (NIR - Red) / (NIR + Red).

    High over healthy vegetation (~0.3-0.9), near 0 over bare soil and built-up.
    """
    return _normalised_difference(nir, red)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index: (SWIR1 - NIR) / (SWIR1 + NIR).

    Positive over impervious/built surfaces. Requires a true SWIR band; it is
    not recoverable from RGB.
    """
    return _normalised_difference(swir, nir)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI for open water: (Green - NIR) / (Green + NIR).

    Note this is the *water-body* formulation (McFeeters 1996). Gao's NDWI for
    vegetation liquid-water content uses (NIR - SWIR) and is a different index.
    """
    return _normalised_difference(green, nir)


def ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalised Difference Snow Index: (Green - SWIR1) / (Green + SWIR1).

    Snow and ice are bright in green and strongly absorbing in SWIR, so NDSI
    separates them from cloud, which is bright in both.
    """
    return _normalised_difference(green, swir)
