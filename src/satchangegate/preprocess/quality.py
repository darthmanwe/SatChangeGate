"""Pair-level observation quality.

``quality_score`` is the fraction of pixels usable in both timesteps — the
complement of the union of every contamination mask.

Two earlier forms were wrong. The first summed the fractions of *overlapping*
masks (``cloud + shadow + snow + 0.5*water``), which double-counted any pixel
flagged twice, could exceed 1.0, and clipped the score to 0 for merely
partly-cloudy scenes. The second multiplied the valid fraction by
``1 - (1 - valid_fraction)`` — squaring it — under a comment claiming a set
union, which also made ``min_valid_pixel_fraction`` unreachable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from pydantic import BaseModel

from satchangegate.config import QualityThresholds
from satchangegate.preprocess.masks import EphemeralMasks, combine_pair_masks


class QualityScore(BaseModel):
    """Observation quality for a bitemporal pair.

    Fraction fields are ``None`` when masks could not be computed for the source
    (see ``EphemeralMasks.assessed``). ``None`` means unknown, not zero.
    """

    masks_assessed: bool
    cloud_fraction_max: float | None = None
    snow_fraction_max: float | None = None
    shadow_fraction_max: float | None = None
    water_fraction_max: float | None = None
    valid_fraction: float | None = None
    quality_score: float | None = None
    valid_observation: bool
    registration_error_px: float | None = None
    season_delta_days: int | None = None


def _parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def season_delta_days(date_t1: str | None, date_t2: str | None) -> int | None:
    """Absolute day-of-year separation, folded to at most half a year.

    Seasonality is the dominant confounder for a change gate: two images a year
    apart share a season, two six months apart do not. This field was previously
    hardcoded to None even though both dates were already parsed.
    """
    d1, d2 = _parse_iso(date_t1), _parse_iso(date_t2)
    if d1 is None or d2 is None:
        return None
    delta = abs(d1.timetuple().tm_yday - d2.timetuple().tm_yday)
    return int(min(delta, 365 - delta))


def compute_quality_score(
    masks_t1: EphemeralMasks,
    masks_t2: EphemeralMasks,
    thresholds: QualityThresholds | None = None,
    *,
    registration_error_px: float | None = None,
    date_t1: str | None = None,
    date_t2: str | None = None,
    combined: EphemeralMasks | None = None,
) -> QualityScore:
    """Score a pair's usability.

    ``combined`` may be passed to avoid recomputing the union, which the pipeline
    already needs for the change mask.
    """
    thresholds = thresholds or QualityThresholds()
    combined = combined if combined is not None else combine_pair_masks(masks_t1, masks_t2)
    season = season_delta_days(date_t1, date_t2)

    registration_ok = (
        registration_error_px is None
        or registration_error_px <= thresholds.registration_error_px_max
    )

    if not combined.assessed:
        # Unknown quality. Do not block the pipeline, but never claim the scene
        # is clean: fractions stay None and the report must say "not assessed".
        return QualityScore(
            masks_assessed=False,
            valid_observation=registration_ok,
            registration_error_px=registration_error_px,
            season_delta_days=season,
        )

    cloud = float(np.maximum(masks_t1.cloud, masks_t2.cloud).mean())
    snow = float(np.maximum(masks_t1.snow, masks_t2.snow).mean())
    shadow = float(np.maximum(masks_t1.shadow, masks_t2.shadow).mean())
    water = float(np.maximum(masks_t1.water, masks_t2.water).mean())

    # `combined.valid` is already the complement of the union of every
    # contamination mask, so the usable fraction *is* the quality score. The
    # previous form multiplied it by (1 - (1 - valid_frac)), i.e. squared it,
    # while a comment claimed it was performing a set union — and that squaring
    # made `min_valid_pixel_fraction` unreachable, since `score >= 0.30`
    # implies `valid_frac >= 0.5477` and strictly dominated the 0.50 floor.
    valid_frac = float(combined.valid.mean())
    score = valid_frac

    valid_observation = bool(
        cloud <= thresholds.cloud_fraction_max
        and valid_frac >= thresholds.min_valid_pixel_fraction
        and registration_ok
    )

    return QualityScore(
        masks_assessed=True,
        cloud_fraction_max=round(cloud, 4),
        snow_fraction_max=round(snow, 4),
        shadow_fraction_max=round(shadow, 4),
        water_fraction_max=round(water, 4),
        valid_fraction=round(valid_frac, 4),
        quality_score=round(score, 4),
        valid_observation=valid_observation,
        registration_error_px=(
            None if registration_error_px is None else round(registration_error_px, 3)
        ),
        season_delta_days=season,
    )
