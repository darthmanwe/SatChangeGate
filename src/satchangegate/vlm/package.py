"""Build the VLM candidate package (before/after/overlay + redacted metadata).

The metadata written here is deliberately *redacted*. The previous version wrote
``classical.model_dump()`` — including ``classical_gate`` and every
discriminative gate feature — into the file that is dumped verbatim into the
VLM prompt. The VLM is architecturally the gate's independent verifier, so
telling it the gate's answer confounds any agreement statistic computed between
the two. Acquisition context and observation quality are kept; the verdict and
the features that produced it are withheld.

The unredacted result is still written next to it as ``classical_full.json`` for
offline analysis; it is never sent to the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from satchangegate.config import RGB_BANDS
from satchangegate.features.classical import ClassicalResult
from satchangegate.preprocess.align import (
    bands_to_rgb,
    rgb_to_uint8,
    stretch_for_display,
)
from satchangegate.preprocess.quality import QualityScore

# Quality fields safe to show the model: they describe observing conditions,
# not the detector's conclusion.
_QUALITY_ALLOWLIST = (
    "masks_assessed",
    "cloud_fraction_max",
    "snow_fraction_max",
    "shadow_fraction_max",
    "water_fraction_max",
    "registration_error_px",
    "season_delta_days",
)


# Minimum long edge for any image sent to the vision model. A 64 px tile is a
# 640 m window at Sentinel-2 resolution; sent at native size the model has
# essentially nothing to look at and correctly answers "uncertain". Upsampling
# adds no information, but it does let the model resolve the structure that is
# present. Well under the model's resolution tier, so it costs few extra tokens.
MIN_RENDER_PX = 512


def _upscale(img_u8: np.ndarray, min_px: int = MIN_RENDER_PX) -> np.ndarray:
    """Enlarge small crops so the model can resolve them; never downscale."""
    h, w = img_u8.shape[:2]
    longest = max(h, w)
    if longest >= min_px:
        return img_u8
    scale = min_px / longest
    return cv2.resize(
        img_u8,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        # Nearest keeps pixel boundaries crisp, so the model sees the actual
        # sensor grid rather than interpolation smear.
        interpolation=cv2.INTER_NEAREST,
    )


def _save_rgb(rgb_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(_upscale(rgb_u8), cv2.COLOR_RGB2BGR))


def _save_overlay(rgb_u8: np.ndarray, heatmap: np.ndarray, path: Path) -> None:
    h = np.rint(np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    base = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(base, 0.6, coloured, 0.4, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _upscale(blended))


def redacted_metadata(quality: QualityScore, classical: ClassicalResult) -> dict:
    """Acquisition context only — no gate verdict, no gate features."""
    q = quality.model_dump()
    return {
        "aoi_id": classical.aoi_id,
        "tile_id": classical.tile_id,
        "date_t1": classical.date_t1,
        "date_t2": classical.date_t2,
        "imagery": "Sentinel-2 RGB rendering"
        + (" (RGB-only source)" if classical.rgb_only else ""),
        "observation_quality": {k: q.get(k) for k in _QUALITY_ALLOWLIST},
        "note": (
            "The third image overlays a change-intensity heatmap on the after "
            "image. Detector conclusions are intentionally withheld."
        ),
    }


def package_candidate_for_vlm(
    out_dir: Path,
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    heatmap: np.ndarray,
    quality: QualityScore,
    classical: ClassicalResult,
    *,
    rgb_bands: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Write the VLM package files; return the package directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(rgb_bands or RGB_BANDS)

    rgb1 = bands_to_rgb(bands_t1, names)
    rgb2 = bands_to_rgb(bands_t2, names)
    disp1, disp2 = stretch_for_display(rgb1, rgb2)
    u1, u2 = rgb_to_uint8(disp1), rgb_to_uint8(disp2)

    _save_rgb(u1, out_dir / "before_rgb.png")
    _save_rgb(u2, out_dir / "after_rgb.png")
    _save_overlay(u2, heatmap, out_dir / "change_overlay.png")
    cv2.imwrite(
        str(out_dir / "change_heatmap.png"),
        np.rint(np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
    )

    (out_dir / "metadata.json").write_text(
        json.dumps(redacted_metadata(quality, classical), indent=2), encoding="utf-8"
    )
    # Kept for offline analysis only; never sent to the model.
    (out_dir / "classical_full.json").write_text(
        classical.model_dump_json(indent=2), encoding="utf-8"
    )
    return out_dir
