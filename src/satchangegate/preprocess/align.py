"""Resampling to a common grid, co-registration, and RGB rendering.

Two behaviours here were previously wrong in ways that silently corrupted every
downstream metric:

1. ``np.resize`` was used to fit mismatched bands to the reference shape. That
   flattens and tiles rather than resampling, scrambling the raster. Resampling
   now always goes through ``skimage.transform.resize``.

2. Radiometric normalisation used percentiles computed over *both* timesteps
   stacked. A large change in t2 therefore shifted the normalisation of t1, so
   change partially normalised itself away and every threshold became
   scene-relative. Bands now stay in physical reflectance; the joint stretch
   survives only on the RGB *visualisation* path, where it is cosmetic.
"""

from __future__ import annotations

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

from satchangegate.config import RGB_BANDS

# Native Sentinel-2 ground sample distances, for reference when resampling.
BANDS_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}
BANDS_60M = {"B01", "B09", "B10"}


def _reference_shape(bands: dict[str, np.ndarray]) -> tuple[int, int]:
    """Shape of the finest-resolution band present (the 10 m bands)."""
    for key in ("B02", "B03", "B04", "B08"):
        if key in bands:
            return bands[key].shape
    return next(iter(bands.values())).shape


def resample_band_to_ref(
    band_arr: np.ndarray,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Bilinearly resample a band onto the reference grid."""
    if band_arr.shape == ref_shape:
        return band_arr.astype(np.float32, copy=False)
    return resize(
        band_arr,
        ref_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=band_arr.shape[0] > ref_shape[0],
    ).astype(np.float32)


def resample_to_common_grid(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Put both timesteps on the finest common grid, preserving reflectance.

    Bands present in one timestep but not the other are dropped, with the pair
    of dicts guaranteed to share the same key set.
    """
    ref_shape = _reference_shape(bands_t1)
    shared = [b for b in bands_t1 if b in bands_t2]
    missing = sorted(set(bands_t1) ^ set(bands_t2))
    if missing:
        raise ValueError(f"Band sets differ between timesteps; unmatched: {missing}")

    out_t1 = {b: resample_band_to_ref(bands_t1[b], ref_shape) for b in shared}
    out_t2 = {b: resample_band_to_ref(bands_t2[b], ref_shape) for b in shared}
    return out_t1, out_t2


def estimate_registration_error(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    *,
    upsample_factor: int = 10,
) -> float:
    """Sub-pixel co-registration error between timesteps, in pixels.

    Uses phase cross-correlation on the red band. Previously this quantity was
    hardcoded to 0.0 while still being reported in JSON artifacts, the VLM
    metadata package, and the analyst prose ("registration error: 0.0 px").
    """
    key = "B04" if "B04" in bands_t1 else next(iter(bands_t1))
    a = np.nan_to_num(bands_t1[key], nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(bands_t2[key], nan=0.0, posinf=0.0, neginf=0.0)
    if a.shape != b.shape:
        b = resample_band_to_ref(b, a.shape)
    shift, _error, _phasediff = phase_cross_correlation(a, b, upsample_factor=upsample_factor)
    return float(np.hypot(*shift[:2]))


def apply_subpixel_shift(band: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    """Shift a band by (dy, dx) pixels using bilinear interpolation."""
    from scipy.ndimage import shift as ndshift  # local: scipy is a skimage dep

    return ndshift(band, shift_yx, order=1, mode="nearest").astype(np.float32)


def bands_to_rgb(
    bands: dict[str, np.ndarray],
    rgb_bands: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """Stack bands into an HxWx3 float array in R,G,B order."""
    names = list(rgb_bands or RGB_BANDS)
    missing = [b for b in names if b not in bands]
    if missing:
        raise KeyError(f"Cannot render RGB, missing bands: {missing}")
    return np.stack([bands[b] for b in names], axis=-1).astype(np.float32)


def stretch_for_display(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    *,
    low: float = 2.0,
    high: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint percentile stretch for *visual comparability only*.

    Applied to the pair together so the two rendered images are visually
    comparable. Deliberately confined to the display path: applying it before
    metric computation is what previously let change normalise itself away.
    """
    stack = np.concatenate([rgb_t1.ravel(), rgb_t2.ravel()])
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros_like(rgb_t1), np.zeros_like(rgb_t2)
    p_low, p_high = np.percentile(finite, [low, high])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    scale = 1.0 / (p_high - p_low)
    a = np.clip((rgb_t1 - p_low) * scale, 0.0, 1.0).astype(np.float32)
    b = np.clip((rgb_t2 - p_low) * scale, 0.0, 1.0).astype(np.float32)
    return a, b


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0,1] to 8-bit, rounding rather than truncating."""
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
