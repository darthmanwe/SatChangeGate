"""Build VLM candidate package (before/after/heatmap + metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from satchangegate.features.classical import ClassicalResult
from satchangegate.preprocess.align import bands_to_rgb, rgb_to_uint8
from satchangegate.preprocess.quality import QualityScore


def _save_rgb_png(bands: dict[str, np.ndarray], path: Path, rgb_bands: list[str]) -> None:
    rgb = rgb_to_uint8(bands_to_rgb(bands, rgb_bands))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_heatmap_overlay(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    path: Path,
) -> None:
    h = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    rgb_bgr = cv2.cvtColor(rgb_to_uint8(rgb), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(rgb_bgr, 0.6, h_color, 0.4, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def package_candidate_for_vlm(
    out_dir: Path,
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    heatmap: np.ndarray,
    quality: QualityScore,
    classical: ClassicalResult,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Write VLM package files; return package directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or {}
    rgb_bands = cfg.get("rgb_bands", ["B04", "B03", "B02"])

    before_path = out_dir / "before_rgb.png"
    after_path = out_dir / "after_rgb.png"
    heatmap_path = out_dir / "change_heatmap.png"
    overlay_path = out_dir / "change_overlay.png"

    _save_rgb_png(bands_t1, before_path, rgb_bands)
    _save_rgb_png(bands_t2, after_path, rgb_bands)

    hm = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(heatmap_path), hm)

    rgb2 = bands_to_rgb(bands_t2, rgb_bands)
    _save_heatmap_overlay(rgb2, heatmap, overlay_path)

    metadata = {
        "aoi_id": classical.aoi_id,
        "tile_id": classical.tile_id,
        "date_t1": classical.date_t1,
        "date_t2": classical.date_t2,
        "quality": quality.model_dump(),
        "classical": classical.model_dump(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "model_prediction.json").write_text(
        json.dumps({"classical_gate": classical.classical_gate, **classical.model_dump()}, indent=2),
        encoding="utf-8",
    )
    return out_dir
