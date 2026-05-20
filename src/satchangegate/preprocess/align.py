"""Align bands to common 10m grid and radiometric normalization."""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# Bands at 20m in Sentinel-2
BANDS_20M = {"B11", "B12"}


def _reference_shape(bands: dict[str, np.ndarray]) -> tuple[int, int]:
    """Use highest-resolution band (10m) as reference."""
    for key in ("B02", "B04", "B03", "B08"):
        if key in bands:
            return bands[key].shape
    return next(iter(bands.values())).shape


def resample_band_to_ref(
    band_arr: np.ndarray,
    band_name: str,
    ref_shape: tuple[int, int],
    src_path: str | None = None,
) -> np.ndarray:
    """Resample 20m band to reference 10m shape."""
    if band_arr.shape == ref_shape:
        return band_arr
    if band_name not in BANDS_20M:
        return np.asarray(
            np.resize(band_arr, ref_shape),
            dtype=np.float32,
        )

    if src_path:
        with rasterio.open(src_path) as src:
            dst = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
            if dst.shape != ref_shape:
                dst = np.asarray(
                    np.resize(dst, ref_shape),
                    dtype=np.float32,
                )
            return dst

    from skimage.transform import resize

    return resize(
        band_arr,
        ref_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def percentile_clip(
    arr: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """Clip array to percentile range and scale to ~0-1 reflectance."""
    p_low, p_high = np.percentile(arr[np.isfinite(arr)], [low, high])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    clipped = np.clip(arr, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


def align_pair_bands(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    band_paths_t1: dict[str, str] | None = None,
    band_paths_t2: dict[str, str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resample all bands to common grid and apply radiometric clip."""
    cfg = cfg or {}
    rad = cfg.get("radiometric", {})
    low = rad.get("percentile_low", 2)
    high = rad.get("percentile_high", 98)

    ref_shape = _reference_shape(bands_t1)
    out_t1: dict[str, np.ndarray] = {}
    out_t2: dict[str, np.ndarray] = {}

    paths_t1 = band_paths_t1 or {}
    paths_t2 = band_paths_t2 or {}

    for band in bands_t1:
        b1 = bands_t1[band]
        b2 = bands_t2.get(band)
        if b2 is None:
            continue
        b1 = resample_band_to_ref(b1, band, ref_shape, paths_t1.get(band))
        b2 = resample_band_to_ref(b2, band, ref_shape, paths_t2.get(band))
        # Normalize using combined percentiles for pair consistency
        stack = np.stack([b1, b2])
        p_low, p_high = np.percentile(stack[np.isfinite(stack)], [low, high])
        if p_high <= p_low:
            p_high = p_low + 1e-6
        out_t1[band] = np.clip((b1 - p_low) / (p_high - p_low), 0, 1).astype(np.float32)
        out_t2[band] = np.clip((b2 - p_low) / (p_high - p_low), 0, 1).astype(np.float32)

    return out_t1, out_t2


def bands_to_rgb(
    bands: dict[str, np.ndarray],
    rgb_bands: list[str] | None = None,
) -> np.ndarray:
    """Stack bands into HxWx3 RGB (expects B04,B03,B02 order)."""
    rgb_bands = rgb_bands or ["B04", "B03", "B02"]
    channels = [bands[b] for b in rgb_bands]
    rgb = np.stack(channels, axis=-1)
    return np.clip(rgb, 0, 1).astype(np.float32)


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB to 8-bit for SSIM/pHash/image export."""
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
