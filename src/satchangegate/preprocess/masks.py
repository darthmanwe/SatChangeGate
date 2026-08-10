"""Ephemeral masks: cloud, snow, water, and cloud shadow.

These are physically-motivated heuristics on top-of-atmosphere reflectance, not
a trained cloud detector. They are honest about that, and — critically — honest
about when they cannot run at all.

Two prior behaviours are deliberately reversed here:

* Masks used to short-circuit to all-zero whenever the input was an RGB preview,
  which was every production call path. Cloud fraction was then reported as
  "0.0" and quality as "1.0" in JSON, in the VLM package, and in analyst prose —
  constants presented as measurements. Unavailable is now represented as
  unavailable (``assessed=False``), never as clean.

* Water was subtracted from the valid mask, so water pixels were excluded from
  the change mask and its denominator — while the VLM schema advertised
  ``flooding`` as a detectable change type. Water is now reported but not
  removed from validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from satchangegate.config import MaskThresholds
from satchangegate.features.indices import ndsi, ndvi, ndwi

# Bands required to compute masks at all.
REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11")


@dataclass
class EphemeralMasks:
    """Per-timestep contamination masks.

    When ``assessed`` is False the mask arrays are all-False placeholders and the
    fractions are None: the source lacked the bands needed to judge. Callers must
    not read that as a clean scene.
    """

    cloud: np.ndarray
    snow: np.ndarray
    water: np.ndarray
    shadow: np.ndarray
    valid: np.ndarray
    assessed: bool = True

    @property
    def cloud_fraction(self) -> float | None:
        return float(self.cloud.mean()) if self.assessed else None

    @property
    def snow_fraction(self) -> float | None:
        return float(self.snow.mean()) if self.assessed else None

    @property
    def water_fraction(self) -> float | None:
        return float(self.water.mean()) if self.assessed else None

    @property
    def shadow_fraction(self) -> float | None:
        return float(self.shadow.mean()) if self.assessed else None

    @property
    def valid_fraction(self) -> float | None:
        return float(self.valid.mean()) if self.assessed else None


def unassessed_masks(shape: tuple[int, int]) -> EphemeralMasks:
    """Placeholder for sources without the bands needed to compute masks."""
    empty = np.zeros(shape, dtype=bool)
    return EphemeralMasks(
        cloud=empty.copy(),
        snow=empty.copy(),
        water=empty.copy(),
        shadow=empty.copy(),
        valid=np.ones(shape, dtype=bool),
        assessed=False,
    )


def can_assess(bands: dict[str, np.ndarray]) -> bool:
    """True when every band needed for masking is present."""
    return all(b in bands for b in REQUIRED_BANDS)


def _brightness(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Mean visible reflectance."""
    return (bands["B02"] + bands["B03"] + bands["B04"]) / 3.0


def _shadow_near_cloud(
    shadow: np.ndarray,
    cloud: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Keep only dark pixels within ``radius_px`` of a cloud.

    Cloud shadow is displaced from its cloud by height*tan(solar_zenith), which
    at 10 m GSD is typically tens to hundreds of pixels. The previous fixed 7x7
    kernel could only find shadows within 3 px of a cloud, so the shadow mask was
    essentially always empty.
    """
    if radius_px <= 0 or not cloud.any():
        return np.zeros_like(shadow, dtype=bool)
    ksize = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    near = cv2.dilate(cloud.astype(np.uint8), kernel) > 0
    return shadow & near


def compute_ephemeral_masks(
    bands: dict[str, np.ndarray],
    thresholds: MaskThresholds | None = None,
) -> EphemeralMasks:
    """Compute contamination masks for one timestep.

    Returns an unassessed placeholder when the required bands are absent, rather
    than pretending the scene is clean.
    """
    thresholds = thresholds or MaskThresholds()
    if not can_assess(bands):
        shape = next(iter(bands.values())).shape
        return unassessed_masks(shape)

    green, red = bands["B03"], bands["B04"]
    nir, swir = bands["B08"], bands["B11"]

    bright = _brightness(bands)
    veg = ndvi(nir, red)
    snow_idx = ndsi(green, swir)
    water_idx = ndwi(green, nir)

    # Cloud: bright across the visible and not vegetated. Snow is also bright
    # and not vegetated, so it is excluded explicitly via NDSI — otherwise a
    # pixel can be counted as both and double-penalised in the quality score.
    snow = (snow_idx > thresholds.snow_ndsi_min) & (nir > thresholds.snow_nir_min)
    cloud = (bright > thresholds.cloud_brightness) & (veg < thresholds.cloud_ndvi_max) & ~snow

    water = water_idx > thresholds.water_ndwi_min

    dark = (nir < thresholds.shadow_nir_max) & (bright < thresholds.shadow_brightness_max)
    shadow = _shadow_near_cloud(dark, cloud, thresholds.shadow_search_radius_px)

    # Water is reported but is NOT contamination: removing it would make
    # flooding undetectable by construction.
    contaminated = cloud | snow | shadow
    valid = ~contaminated

    return EphemeralMasks(
        cloud=cloud.astype(bool),
        snow=snow.astype(bool),
        water=water.astype(bool),
        shadow=shadow.astype(bool),
        valid=valid.astype(bool),
        assessed=True,
    )


def combine_pair_masks(m1: EphemeralMasks, m2: EphemeralMasks) -> EphemeralMasks:
    """Union contamination and intersect validity across the two timesteps."""
    assessed = m1.assessed and m2.assessed
    if not assessed:
        return unassessed_masks(m1.valid.shape)
    return EphemeralMasks(
        cloud=m1.cloud | m2.cloud,
        snow=m1.snow | m2.snow,
        water=m1.water | m2.water,
        shadow=m1.shadow | m2.shadow,
        valid=m1.valid & m2.valid,
        assessed=True,
    )
