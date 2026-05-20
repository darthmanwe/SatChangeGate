"""Quality scoring schema (Tier 0 output)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from satchangegate.preprocess.align import bands_to_rgb
from satchangegate.preprocess.masks import EphemeralMasks, combine_pair_masks


class QualityScore(BaseModel):
    valid_observation: bool
    cloud_fraction: float
    shadow_fraction: float
    snow_fraction: float
    water_fraction: float
    registration_error_px: float = 0.0
    illumination_delta: float
    season_delta_days: int | None = None
    quality_score: float


def _illumination_delta(bands_t1: dict[str, np.ndarray], bands_t2: dict[str, np.ndarray]) -> float:
    rgb1 = bands_to_rgb(bands_t1)
    rgb2 = bands_to_rgb(bands_t2)
    mean1 = float(rgb1.mean())
    mean2 = float(rgb2.mean())
    return abs(mean1 - mean2)


def compute_quality_score(
    masks_t1: EphemeralMasks,
    masks_t2: EphemeralMasks,
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    cfg: dict[str, Any] | None = None,
    registration_error_px: float = 0.0,
) -> QualityScore:
    """Compute pair-level quality metrics."""
    cfg = cfg or {}
    qcfg = cfg.get("quality", {})
    combined = combine_pair_masks(masks_t1, masks_t2)

    cloud_frac = max(masks_t1.cloud_fraction, masks_t2.cloud_fraction)
    shadow_frac = max(masks_t1.shadow_fraction, masks_t2.shadow_fraction)
    snow_frac = max(masks_t1.snow_fraction, masks_t2.snow_fraction)
    water_frac = max(masks_t1.water_fraction, masks_t2.water_fraction)
    valid_frac = combined.valid_fraction
    illum = _illumination_delta(bands_t1, bands_t2)

    # Quality score: high valid fraction, low contamination
    contam = cloud_frac + shadow_frac + snow_frac + water_frac * 0.5
    score = float(np.clip(valid_frac * (1.0 - contam), 0.0, 1.0))

    cloud_max = qcfg.get("cloud_fraction_max", 0.25)
    min_valid = qcfg.get("min_valid_pixel_fraction", 0.5)
    min_score = qcfg.get("min_quality_score", 0.3)

    valid_obs = (
        cloud_frac <= cloud_max
        and valid_frac >= min_valid
        and score >= min_score
        and registration_error_px <= cfg.get("gate", {}).get("registration_error_px_max", 1.5)
    )

    return QualityScore(
        valid_observation=valid_obs,
        cloud_fraction=round(cloud_frac, 4),
        shadow_fraction=round(shadow_frac, 4),
        snow_fraction=round(snow_frac, 4),
        water_fraction=round(water_frac, 4),
        registration_error_px=registration_error_px,
        illumination_delta=round(illum, 4),
        season_delta_days=None,
        quality_score=round(score, 4),
    )
