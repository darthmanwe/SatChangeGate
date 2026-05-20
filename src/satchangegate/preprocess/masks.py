"""Cloud, snow, water, and shadow masks (heuristic, no SCL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from satchangegate.features.indices import ndsi, ndvi, ndwi


@dataclass
class EphemeralMasks:
    cloud: np.ndarray
    snow: np.ndarray
    water: np.ndarray
    shadow: np.ndarray
    valid: np.ndarray

    @property
    def cloud_fraction(self) -> float:
        return float(self.cloud.mean())

    @property
    def snow_fraction(self) -> float:
        return float(self.snow.mean())

    @property
    def water_fraction(self) -> float:
        return float(self.water.mean())

    @property
    def shadow_fraction(self) -> float:
        return float(self.shadow.mean())

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def _brightness(bands: dict[str, np.ndarray]) -> np.ndarray:
    b02 = bands["B02"]
    b03 = bands["B03"]
    b04 = bands["B04"]
    return (b02 + b03 + b04) / 3.0


def compute_ephemeral_masks(
    bands: dict[str, np.ndarray],
    cfg: dict[str, Any] | None = None,
) -> EphemeralMasks:
    """Compute boolean masks for one timestep."""
    cfg = cfg or {}
    mcfg = cfg.get("masks", {})

    green = bands["B03"]
    red = bands["B04"]
    nir = bands["B08"]
    swir = bands["B11"]

    bright = _brightness(bands)
    veg_ndvi = ndvi(nir, red)
    snow_idx = ndsi(green, swir)
    water_idx = ndwi(green, nir)

    cloud = (bright > mcfg.get("cloud_brightness", 0.35)) & (
        veg_ndvi < mcfg.get("cloud_ndvi_max", 0.2)
    )

    snow = (snow_idx > mcfg.get("snow_ndsi_min", 0.4)) & (
        nir > mcfg.get("snow_nir_min", 0.25)
    )

    water = water_idx > mcfg.get("water_ndwi_min", 0.3)

    shadow = (nir < mcfg.get("shadow_nir_max", 0.15)) & (
        bright < mcfg.get("shadow_brightness_max", 0.12)
    )

    # Expand shadow near clouds
    kernel = np.ones((7, 7), np.uint8)
    cloud_dilated = cv2.dilate(cloud.astype(np.uint8), kernel) > 0
    shadow = shadow & cloud_dilated

    contam = cloud | snow | water | shadow
    valid = ~contam

    return EphemeralMasks(
        cloud=cloud.astype(bool),
        snow=snow.astype(bool),
        water=water.astype(bool),
        shadow=shadow.astype(bool),
        valid=valid.astype(bool),
    )


def combine_pair_masks(m1: EphemeralMasks, m2: EphemeralMasks) -> EphemeralMasks:
    """Union masks across t1/t2 for pair-level QA."""
    return EphemeralMasks(
        cloud=m1.cloud | m2.cloud,
        snow=m1.snow | m2.snow,
        water=m1.water | m2.water,
        shadow=m1.shadow | m2.shadow,
        valid=m1.valid & m2.valid,
    )
