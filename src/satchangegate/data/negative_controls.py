"""Synthetic negative controls for change-detection testing."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from satchangegate.data.oscd import ImagePair, load_bands, load_label_mask

NegativeMode = Literal["identity", "stable", "photometric"]


def _crop_dict(bands: dict[str, np.ndarray], y0: int, x0: int, size: int) -> dict[str, np.ndarray]:
    y1, x1 = y0 + size, x0 + size
    return {k: v[y0:y1, x0:x1].copy() for k, v in bands.items()}


def find_stable_crop_box(
    label_mask: np.ndarray,
    min_size: int = 128,
    target_size: int = 256,
) -> tuple[int, int, int] | None:
    """Return (y0, x0, size) for a square crop inside no-change (label==0) region."""
    stable = (label_mask == 0).astype(np.uint8)
    h, w = stable.shape
    size = min(target_size, h, w)
    if size < min_size:
        return None

    best = None
    best_count = 0
    step = max(size // 4, 32)
    for y0 in range(0, h - size + 1, step):
        for x0 in range(0, w - size + 1, step):
            patch = stable[y0 : y0 + size, x0 : x0 + size]
            count = int(patch.sum())
            if count > best_count:
                best_count = count
                best = (y0, x0, size)
    if best is None or best_count < size * size * 0.95:
        return None
    return best


def apply_photometric_perturbation(
    bands: dict[str, np.ndarray],
    *,
    brightness: float = 1.18,
    gamma: float = 0.85,
    blue_shift: float = 1.05,
) -> dict[str, np.ndarray]:
    """Simulate illumination/color shift on pseudo multispectral bands."""
    out: dict[str, np.ndarray] = {}
    for name, arr in bands.items():
        x = np.clip(arr.astype(np.float32) * brightness, 0, 1)
        if name == "B02":
            x = np.clip(x * blue_shift, 0, 1)
        x = np.power(x, gamma)
        out[name] = x.astype(np.float32)
    return out


def prepare_negative_pair(
    pair: ImagePair,
    mode: NegativeMode,
    bands_list: list[str],
    *,
    crop_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """
    Load and transform an OSCD pair for negative-control testing.

    Returns (bands_t1, bands_t2, run_id_suffix).
    """
    raw_t1 = load_bands(pair.img1_dir, bands_list)
    raw_t2 = load_bands(pair.img2_dir, bands_list)

    if mode == "identity":
        return raw_t1, raw_t1, "_identity"

    if mode == "photometric":
        perturbed = apply_photometric_perturbation(raw_t1)
        return raw_t1, perturbed, "_photometric"

    # stable crop: both timesteps, region with cm==0
    label = load_label_mask(pair)
    if label is None:
        raise ValueError(f"No label mask for stable crop: {pair.pair_id}")
    if label.shape != next(iter(raw_t1.values())).shape:
        label = cv2.resize(
            label, (raw_t1["B02"].shape[1], raw_t1["B02"].shape[0]), interpolation=cv2.INTER_NEAREST
        )

    box = find_stable_crop_box(label, target_size=crop_size)
    if box is None:
        raise ValueError(f"No stable {crop_size}px crop found for {pair.pair_id}")
    y0, x0, size = box
    return (
        _crop_dict(raw_t1, y0, x0, size),
        _crop_dict(raw_t2, y0, x0, size),
        f"_stable_{y0}_{x0}_{size}",
    )
