"""On-demand layer images, derived through the pipeline's own functions.

Every layer here comes from ``features.classical`` and ``preprocess`` rather
than from a reimplementation, for the same reason the playground calls
``decide``: a second copy of the maths is a copy that drifts.

Three rendering rules, each of which exists because breaking it would flatter
or mislead:

**Signed maps are never percentile-stretched.** A diverging index delta rendered
to its own extremes makes every scene look equally dramatic. These use a fixed
symmetric range, stated in the legend, so two scenes are comparable and a quiet
one looks quiet.

**Invalid pixels are drawn, not filled.** Anything outside the valid mask is
hatched rather than coloured as though it were measured, because "unknown" and
"zero" are different and this repo says so everywhere else.

**Masks say which timestep they are.** Cloud at t1, cloud at t2 and their union
are three different pictures; a layer named just "cloud" would be a guess.

Layers are computed per scene and cached by settings fingerprint, so moving a
threshold in the UI cannot silently show a picture from a different
configuration.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from satchangegate.config import Settings

#: Fixed display range for signed index deltas. NDVI and NDBI live in [-2, 2]
#: but real change concentrates well inside this; anything beyond clips, and the
#: legend says so rather than the scale quietly moving.
SIGNED_FULL_SCALE = 0.5

#: Full scenes are large. Two in memory is enough for a before/after comparison
#: without letting a browse of 24 cities grow without bound.
_CACHE_MAX = 2


@dataclass
class SceneLayers:
    """Everything renderable for one pair, computed once."""

    city: str
    height: int
    width: int
    rgb_t1: np.ndarray
    rgb_t2: np.ndarray
    change_mask: np.ndarray
    heatmap: np.ndarray
    deltas: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    valid: np.ndarray
    scene_threshold: float | None
    fingerprint: str

    def names(self) -> list[str]:
        return [
            "rgb_t1",
            "rgb_t2",
            "change_mask",
            "heatmap",
            *(f"delta_{k}" for k in self.deltas),
            *(f"mask_{k}" for k in self.masks),
        ]


_cache: OrderedDict[tuple[str, str, str], SceneLayers] = OrderedDict()


def _fingerprint(settings: Settings) -> str:
    from satchangegate.pipeline import provenance

    return str(provenance(settings)["config_sha256"])


def compute_layers(root: Path, city: str, settings: Settings) -> SceneLayers:
    """Derive every layer for one pair, through the real pipeline functions."""
    key = (str(root), city, _fingerprint(settings))
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit

    from satchangegate.data.oscd import discover_pairs, load_bands
    from satchangegate.features.classical import (
        _index_deltas,
        compute_change_mask,
        compute_cva,
        compute_heatmap,
        scene_change_threshold,
    )
    from satchangegate.preprocess.align import (
        bands_to_rgb,
        resample_to_common_grid,
        rgb_to_uint8,
        stretch_for_display,
    )
    from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks

    pair = next((p for p in discover_pairs(root) if p.pair_id == city), None)
    if pair is None:
        raise LookupError(f"No pair named {city!r} under {root}")

    raw_t1 = load_bands(pair.img1_dir)
    raw_t2 = load_bands(pair.img2_dir)
    bands_t1, bands_t2 = resample_to_common_grid(raw_t1, raw_t2)

    masks_t1 = compute_ephemeral_masks(bands_t1, settings.masks)
    masks_t2 = compute_ephemeral_masks(bands_t2, settings.masks)
    combined = combine_pair_masks(masks_t1, masks_t2)

    deltas = _index_deltas(bands_t1, bands_t2)
    cva = compute_cva(bands_t1, bands_t2)
    threshold = scene_change_threshold(deltas, combined.valid, settings.gate, combined.water)
    change_mask = compute_change_mask(
        deltas,
        cva,
        combined.valid,
        settings.gate,
        scene_threshold=threshold,
        water=combined.water,
    )
    heatmap = compute_heatmap(change_mask, deltas)

    rgb1 = bands_to_rgb(bands_t1, settings.rgb_bands)
    rgb2 = bands_to_rgb(bands_t2, settings.rgb_bands)
    disp1, disp2 = stretch_for_display(rgb1, rgb2)

    layers = SceneLayers(
        city=city,
        height=int(combined.valid.shape[0]),
        width=int(combined.valid.shape[1]),
        rgb_t1=rgb_to_uint8(disp1),
        rgb_t2=rgb_to_uint8(disp2),
        change_mask=change_mask,
        heatmap=heatmap,
        deltas=dict(deltas),
        masks={
            "cloud_t1": masks_t1.cloud,
            "cloud_t2": masks_t2.cloud,
            "snow_t1": masks_t1.snow,
            "snow_t2": masks_t2.snow,
            "shadow_t1": masks_t1.shadow,
            "shadow_t2": masks_t2.shadow,
            "water_union": combined.water,
            "valid_union": combined.valid,
        },
        valid=combined.valid,
        scene_threshold=threshold,
        fingerprint=key[2],
    )

    _cache[key] = layers
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return layers


# ----------------------------------------------------------------- rendering


def _hatch_invalid(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark unobserved pixels with a diagonal hatch rather than a colour.

    A flat fill reads as a measurement. A hatch reads as absence, which is what
    it is.
    """
    if valid.all():
        return rgb
    h, w = valid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    stripe = ((xx + yy) % 8) < 2
    out = rgb.copy()
    out[~valid] = (28, 34, 34)
    out[(~valid) & stripe] = (78, 90, 90)
    return out


def _diverging(values: np.ndarray, valid: np.ndarray, full_scale: float) -> np.ndarray:
    """Signed data on a fixed blue-white-red ramp. Never self-normalised."""
    t = np.clip(np.nan_to_num(values) / full_scale, -1.0, 1.0)
    pos = np.clip(t, 0, 1)
    neg = np.clip(-t, 0, 1)
    r = 255 - (neg * 150)
    g = 255 - (np.maximum(pos, neg) * 190)
    b = 255 - (pos * 150)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return _hatch_invalid(rgb, valid)


def _binary(mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    rgb = np.full((*mask.shape, 3), 18, dtype=np.uint8)
    rgb[mask.astype(bool)] = colour
    return rgb


def encode_layer(layers: SceneLayers, name: str) -> tuple[bytes, dict[str, Any]]:
    """(PNG bytes, legend) for one named layer."""
    legend: dict[str, Any] = {"layer": name, "fingerprint": layers.fingerprint}

    if name == "rgb_t1":
        rgb, legend["scale"] = layers.rgb_t1, "joint 2-98 percentile stretch across the pair"
    elif name == "rgb_t2":
        rgb, legend["scale"] = layers.rgb_t2, "joint 2-98 percentile stretch across the pair"
    elif name == "change_mask":
        rgb = _binary(layers.change_mask, (255, 214, 64))
        legend["scale"] = "binary, after morphological opening and small-component removal"
        legend["scene_threshold"] = layers.scene_threshold
    elif name == "heatmap":
        grey = np.clip(np.nan_to_num(layers.heatmap), 0, 1)
        rgb = cv2.applyColorMap((grey * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = _hatch_invalid(rgb, layers.valid)
        legend["scale"] = "0 to 1 on a fixed physical full scale, not self-normalised"
    elif name.startswith("delta_"):
        key = name.removeprefix("delta_")
        if key not in layers.deltas:
            raise LookupError(f"No layer named {name!r}")
        rgb = _diverging(layers.deltas[key], layers.valid, SIGNED_FULL_SCALE)
        legend["scale"] = f"diverging, fixed range -{SIGNED_FULL_SCALE} to +{SIGNED_FULL_SCALE}"
        legend["direction"] = "blue is a decrease, red is an increase; values beyond clip"
    elif name.startswith("mask_"):
        key = name.removeprefix("mask_")
        if key not in layers.masks:
            raise LookupError(f"No layer named {name!r}")
        colours = {
            "cloud": (236, 240, 241),
            "snow": (170, 220, 255),
            "shadow": (120, 110, 160),
            "water": (64, 150, 200),
            "valid": (110, 200, 150),
        }
        base = next((c for k, c in colours.items() if key.startswith(k)), (200, 200, 200))
        rgb = _binary(layers.masks[key], base)
        legend["scale"] = "binary"
        legend["timestep"] = (
            "t1"
            if key.endswith("_t1")
            else "t2"
            if key.endswith("_t2")
            else "union of both acquisitions"
        )
    else:
        raise LookupError(f"No layer named {name!r}")

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover - cv2 encode failure is not recoverable here
        raise RuntimeError(f"Could not encode layer {name!r}")
    return buf.tobytes(), legend


def describe_layers() -> list[dict[str, str]]:
    """The layer menu, with what each one actually is."""
    return [
        {"name": "rgb_t1", "group": "imagery", "label": "RGB before"},
        {"name": "rgb_t2", "group": "imagery", "label": "RGB after"},
        {"name": "change_mask", "group": "detection", "label": "Change mask"},
        {"name": "heatmap", "group": "detection", "label": "Change heatmap"},
        {"name": "delta_ndvi", "group": "indices", "label": "ΔNDVI (vegetation)"},
        {"name": "delta_ndbi", "group": "indices", "label": "ΔNDBI (built-up)"},
        {"name": "delta_ndwi", "group": "indices", "label": "ΔNDWI (water)"},
        {"name": "mask_cloud_t1", "group": "quality", "label": "Cloud t1"},
        {"name": "mask_cloud_t2", "group": "quality", "label": "Cloud t2"},
        {"name": "mask_snow_t1", "group": "quality", "label": "Snow t1"},
        {"name": "mask_snow_t2", "group": "quality", "label": "Snow t2"},
        {"name": "mask_shadow_t1", "group": "quality", "label": "Shadow t1"},
        {"name": "mask_shadow_t2", "group": "quality", "label": "Shadow t2"},
        {"name": "mask_water_union", "group": "quality", "label": "Water (union)"},
        {"name": "mask_valid_union", "group": "quality", "label": "Valid (union)"},
    ]
