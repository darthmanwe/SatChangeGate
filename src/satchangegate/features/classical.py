"""Classical change gate: SSIM, pHash, CVA, spectral deltas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel
from skimage.metrics import structural_similarity as ssim_fn

from satchangegate.features.indices import ndbi, ndvi, ndwi
from satchangegate.preprocess.align import bands_to_rgb, rgb_to_uint8
from satchangegate.preprocess.masks import EphemeralMasks
from satchangegate.preprocess.quality import QualityScore

GateDecision = Literal["no_change", "candidate_change", "low_quality"]


class ClassicalResult(BaseModel):
    tile_id: str
    aoi_id: str
    date_t1: str
    date_t2: str
    cloud_fraction_max: float
    snow_fraction_max: float
    registration_error_px: float
    ssim: float
    phash_distance: int
    ndvi_delta_mean: float
    ndbi_delta_mean: float
    ndwi_delta_mean: float
    changed_area_percent: float
    classical_gate: GateDecision
    cva_magnitude_mean: float = 0.0


@dataclass
class GateArtifacts:
    result: ClassicalResult
    change_mask: np.ndarray
    heatmap: np.ndarray


def _phash(rgb: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute perceptual hash (DCT-based)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    return (dct_low > med).flatten()


def phash_distance(rgb1: np.ndarray, rgb2: np.ndarray) -> int:
    h1, h2 = _phash(rgb1), _phash(rgb2)
    return int(np.sum(h1 != h2))


def compute_ssim(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    u1, u2 = rgb_to_uint8(rgb1), rgb_to_uint8(rgb2)
    return float(ssim_fn(u1, u2, channel_axis=-1, data_range=255))


def compute_cva(bands_t1: dict[str, np.ndarray], bands_t2: dict[str, np.ndarray]) -> np.ndarray:
    """Change Vector Analysis magnitude on RGB+NIR stack."""
    stack_bands = ["B02", "B03", "B04", "B08"]
    t1 = np.stack([bands_t1[b] for b in stack_bands], axis=0)
    t2 = np.stack([bands_t2[b] for b in stack_bands], axis=0)
    diff = t2 - t1
    return np.sqrt((diff**2).sum(axis=0))


def _masked_mean(delta: np.ndarray, valid: np.ndarray) -> float:
    v = valid & np.isfinite(delta)
    if not v.any():
        return 0.0
    return float(np.mean(np.abs(delta[v])))


def compute_change_mask(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    valid: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Binary change mask from spectral index deltas."""
    gcfg = cfg.get("gate", {})
    ndvi_d = ndvi(bands_t2["B08"], bands_t2["B04"]) - ndvi(bands_t1["B08"], bands_t1["B04"])
    ndbi_d = ndbi(bands_t2["B11"], bands_t2["B08"]) - ndbi(bands_t1["B11"], bands_t1["B08"])
    ndwi_d = ndwi(bands_t2["B03"], bands_t2["B08"]) - ndwi(bands_t1["B03"], bands_t1["B08"])
    cva = compute_cva(bands_t1, bands_t2)

    small = gcfg.get("index_delta_small", 0.03)
    cva_thr = gcfg.get("cva_magnitude_threshold", 0.15)

    changed = (
        (np.abs(ndvi_d) > small)
        | (np.abs(ndbi_d) > small)
        | (np.abs(ndwi_d) > small)
        | (cva > cva_thr)
    ) & valid
    return changed.astype(np.uint8)


def compute_heatmap(change_mask: np.ndarray, bands_t1: dict, bands_t2: dict) -> np.ndarray:
    """Normalized heatmap for visualization."""
    ndvi_d = np.abs(
        ndvi(bands_t2["B08"], bands_t2["B04"]) - ndvi(bands_t1["B08"], bands_t1["B04"])
    )
    ndbi_d = np.abs(
        ndbi(bands_t2["B11"], bands_t2["B08"]) - ndbi(bands_t1["B11"], bands_t1["B08"])
    )
    combined = (ndvi_d + ndbi_d) / 2.0
    combined = combined * change_mask
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined.astype(np.float32)


def classical_gate(
    pair_id: str,
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    masks: EphemeralMasks,
    quality: QualityScore,
    cfg: dict[str, Any] | None = None,
    date_t1: str = "t1",
    date_t2: str = "t2",
) -> GateArtifacts:
    """Run classical change gate and produce feature row + masks."""
    cfg = cfg or {}
    gcfg = cfg.get("gate", {})

    valid = masks.valid
    cloud_max = max(quality.cloud_fraction, 0.0)
    snow_max = max(quality.snow_fraction, 0.0)

    rgb1 = bands_to_rgb(bands_t1, cfg.get("rgb_bands"))
    rgb2 = bands_to_rgb(bands_t2, cfg.get("rgb_bands"))
    ssim_score = compute_ssim(rgb1, rgb2)
    phash_dist = phash_distance(rgb_to_uint8(rgb1), rgb_to_uint8(rgb2))

    ndvi_d = ndvi(bands_t2["B08"], bands_t2["B04"]) - ndvi(bands_t1["B08"], bands_t1["B04"])
    ndbi_d = ndbi(bands_t2["B11"], bands_t2["B08"]) - ndbi(bands_t1["B11"], bands_t1["B08"])
    ndwi_d = ndwi(bands_t2["B03"], bands_t2["B08"]) - ndwi(bands_t1["B03"], bands_t1["B08"])
    cva = compute_cva(bands_t1, bands_t2)

    change_mask = compute_change_mask(bands_t1, bands_t2, valid, cfg)
    n_valid = valid.sum()
    changed_pct = (
        float(100.0 * (change_mask & valid).sum() / n_valid) if n_valid > 0 else 0.0
    )

    ndvi_mean = _masked_mean(ndvi_d, valid)
    ndbi_mean = _masked_mean(ndbi_d, valid)
    ndwi_mean = _masked_mean(ndwi_d, valid)
    cva_mean = _masked_mean(cva, valid)

    reg_px = quality.registration_error_px

    decision: GateDecision
    if cloud_max > gcfg.get("cloud_fraction_max", 0.25) or not quality.valid_observation:
        decision = "low_quality"
    elif (
        changed_pct < gcfg.get("min_changed_area_percent", 1.0)
        and ndvi_mean < gcfg.get("ndvi_delta_mean_min", 0.05)
        and ndbi_mean < gcfg.get("ndbi_delta_mean_min", 0.05)
        and ndwi_mean < gcfg.get("ndwi_delta_mean_min", 0.05)
        and ssim_score >= gcfg.get("ssim_no_change_min", 0.95)
        and phash_dist <= gcfg.get("phash_no_change_max", 8)
    ):
        decision = "no_change"
    else:
        decision = "candidate_change"

    result = ClassicalResult(
        tile_id=pair_id,
        aoi_id=f"oscd_{pair_id}",
        date_t1=date_t1,
        date_t2=date_t2,
        cloud_fraction_max=round(cloud_max, 4),
        snow_fraction_max=round(snow_max, 4),
        registration_error_px=round(reg_px, 4),
        ssim=round(ssim_score, 4),
        phash_distance=phash_dist,
        ndvi_delta_mean=round(ndvi_mean, 4),
        ndbi_delta_mean=round(ndbi_mean, 4),
        ndwi_delta_mean=round(ndwi_mean, 4),
        changed_area_percent=round(changed_pct, 2),
        classical_gate=decision,
        cva_magnitude_mean=round(cva_mean, 4),
    )

    heatmap = compute_heatmap(change_mask, bands_t1, bands_t2)
    return GateArtifacts(result=result, change_mask=change_mask, heatmap=heatmap)


def save_heatmap_png(heatmap: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
