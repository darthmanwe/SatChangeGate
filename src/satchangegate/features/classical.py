"""Classical change gate: SSIM, pHash, CVA, spectral index deltas.

The decision itself lives in one pure function, ``decide``. Previously the
ladder was implemented twice — once here and once, hand-copied, inside the
threshold sweep — and the two had already drifted apart, so the sweep was
optimising a model the pipeline did not run.

The rule set is deliberately smaller than its predecessor. That version had ten
branches, but with a per-pixel change mask and no despeckling the "area" term
was true for nearly every pair and the "structural" term was true for nearly
every multi-year pair, so most branches collapsed onto a single spectral
threshold. Two of the remaining branches were carve-outs whose constants sat
between two individual observed tiles. What follows is three documented rules
with a monotonicity guarantee: adding evidence never moves a pair from
``candidate_change`` back to ``no_change``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field
from skimage.measure import label as cc_label
from skimage.metrics import structural_similarity as ssim_fn

from satchangegate.config import GateThresholds
from satchangegate.features.indices import ndbi, ndvi, ndwi
from satchangegate.preprocess.align import bands_to_rgb, rgb_to_uint8, stretch_for_display
from satchangegate.preprocess.masks import EphemeralMasks
from satchangegate.preprocess.quality import QualityScore

GateDecision = Literal["no_change", "candidate_change", "low_quality"]

# Physical scale used to normalise the heatmap. Index deltas are in [-2, 2];
# a change of this magnitude renders as full intensity. Fixing the scale means
# a faint change and a dramatic one no longer produce identical images.
HEATMAP_FULL_SCALE = 0.5


class GateFeatures(BaseModel):
    """Everything the decision function is allowed to see."""

    ssim: float
    phash_distance: int
    # Signed means preserve direction: vegetation loss vs regrowth, construction
    # vs demolition. The previous implementation returned mean(|delta|) under a
    # field named "*_delta_mean" and reports printed it with a leading '+'.
    ndvi_delta_mean: float
    ndbi_delta_mean: float
    ndwi_delta_mean: float
    ndvi_delta_abs_mean: float
    ndbi_delta_abs_mean: float
    ndwi_delta_abs_mean: float
    cva_magnitude_mean: float
    changed_area_percent: float
    valid_observation: bool = True


class ClassicalResult(BaseModel):
    tile_id: str
    aoi_id: str
    source: str = "oscd"
    date_t1: str
    date_t2: str
    masks_assessed: bool = True
    cloud_fraction_max: float | None = None
    snow_fraction_max: float | None = None
    registration_error_px: float | None = None
    season_delta_days: int | None = None
    ssim: float
    phash_distance: int
    ndvi_delta_mean: float
    ndbi_delta_mean: float
    ndwi_delta_mean: float
    ndvi_delta_abs_mean: float = 0.0
    ndbi_delta_abs_mean: float = 0.0
    ndwi_delta_abs_mean: float = 0.0
    cva_magnitude_mean: float = 0.0
    changed_area_percent: float
    classical_gate: GateDecision
    gate_reason: str = ""
    gate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # True when spectral indices were derived from RGB-only imagery and are
    # therefore not physically meaningful (OPTIMUS TCI). Never true for OSCD.
    rgb_only: bool = False


@dataclass
class GateArtifacts:
    result: ClassicalResult
    change_mask: np.ndarray
    heatmap: np.ndarray


def _phash(gray_u8: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """DCT perceptual hash.

    The median is taken over the same coefficients that are thresholded, with
    only the DC term excluded. Previously the median came from ``dct[1:, 1:]``
    while all 64 coefficients were compared against it, so the DC term (orders of
    magnitude larger than any AC term) forced bit 0 high and biased the whole
    first row and column — roughly 49 usable bits instead of 64.
    """
    side = hash_size * 4
    resized = cv2.resize(gray_u8, (side, side), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    low = dct[:hash_size, :hash_size].flatten()
    ac = low[1:]
    med = float(np.median(ac))
    bits = low > med
    bits[0] = False  # DC carries brightness, not structure
    return bits


def phash_distance(gray1_u8: np.ndarray, gray2_u8: np.ndarray) -> int:
    """Hamming distance between perceptual hashes of two 8-bit grayscale images."""
    return int(np.sum(_phash(gray1_u8) != _phash(gray2_u8)))


def compute_ssim(rgb1: np.ndarray, rgb2: np.ndarray, *, already_stretched: bool = False) -> float:
    """Structural similarity on display-stretched RGB.

    Uses the Wang et al. (2004) 11x11 Gaussian window rather than skimage's
    uniform 7x7 default, so values are comparable to published SSIM figures.

    Pass ``already_stretched`` when the caller has applied a scene-wide stretch.
    Re-stretching a small crop normalises against that crop's own 2-98 range,
    which on a uniform tile (water, bare field, car park) is sensor noise — the
    stretch then amplifies noise to full scale and SSIM collapses toward 0,
    while pHash on the scene-stretched image correctly reads 0 distance.
    """
    a, b = (rgb1, rgb2) if already_stretched else stretch_for_display(rgb1, rgb2)
    win = min(11, a.shape[0] - (1 - a.shape[0] % 2), a.shape[1] - (1 - a.shape[1] % 2))
    win = max(3, win if win % 2 == 1 else win - 1)
    return float(
        ssim_fn(
            a,
            b,
            channel_axis=-1,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=win,
        )
    )


def compute_cva(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    band_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """Change Vector Analysis magnitude across the configured bands."""
    names = list(band_names or [b for b in ("B02", "B03", "B04", "B08") if b in bands_t1])
    t1 = np.stack([bands_t1[b] for b in names], axis=0)
    t2 = np.stack([bands_t2[b] for b in names], axis=0)
    return np.sqrt(((t2 - t1) ** 2).sum(axis=0)).astype(np.float32)


def _index_deltas(
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per-pixel index deltas, computed once and reused.

    Previously NDVI/NDBI deltas were recomputed three times per pair.
    """
    return {
        "ndvi": ndvi(bands_t2["B08"], bands_t2["B04"]) - ndvi(bands_t1["B08"], bands_t1["B04"]),
        "ndbi": ndbi(bands_t2["B11"], bands_t2["B08"]) - ndbi(bands_t1["B11"], bands_t1["B08"]),
        "ndwi": ndwi(bands_t2["B03"], bands_t2["B08"]) - ndwi(bands_t1["B03"], bands_t1["B08"]),
    }


def _masked_mean(delta: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(signed mean, mean absolute) over valid finite pixels.

    Returns (0.0, 0.0) only when nothing is valid; callers should treat a fully
    masked tile as unknown rather than unchanged.
    """
    v = valid & np.isfinite(delta)
    if not v.any():
        return 0.0, 0.0
    vals = delta[v]
    return float(np.mean(vals)), float(np.mean(np.abs(vals)))


def despeckle(mask: np.ndarray, *, open_radius: int, min_size: int) -> np.ndarray:
    """Morphological opening plus small-component removal.

    Without this the change mask is a per-pixel OR of correlated thresholds, so
    isolated sensor and compression noise counts toward changed area. Observed
    changed-area values of 23-86% on no-change tiles came from this.
    """
    out = mask.astype(bool)
    if open_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    if min_size > 1 and out.any():
        labels = cc_label(out, connectivity=2)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        keep = np.isin(labels, np.flatnonzero(counts >= min_size))
        out = out & keep
    return out


def change_magnitude(
    deltas: dict[str, np.ndarray],
    water: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel change magnitude.

    NDVI and NDBI are *land* indices: over water both denominators approach zero
    (deep-water NIR and SWIR reflectance are ~0.015 and ~0.008), so sensor noise
    of a few thousandths swings them across a large range and a static lake reads
    as violent change. Where a water mask is available, land indices are replaced
    over water by the NDWI response, which is the physically meaningful signal
    there — and is what makes flooding detectable rather than masked away.
    """
    land = np.maximum(np.abs(deltas["ndvi"]), np.abs(deltas["ndbi"]))
    if water is None:
        return land
    return np.where(water, np.abs(deltas["ndwi"]), land)


def compute_change_mask(
    deltas: dict[str, np.ndarray],
    cva: np.ndarray,
    valid: np.ndarray,
    thresholds: GateThresholds,
    *,
    scene_threshold: float | None = None,
    water: np.ndarray | None = None,
) -> np.ndarray:
    """Spatially coherent binary change mask.

    With ``adaptive_threshold`` the cut tracks this scene's own background
    variability, which cancels the scene-wide radiometric offset between
    acquisitions. Pass ``scene_threshold`` to reuse a threshold derived from the
    whole scene when masking a sub-window, so tiles in the same city share one
    consistent cut.
    """
    magnitude = change_magnitude(deltas, water)
    if scene_threshold is None:
        computed = scene_change_threshold(deltas, valid, thresholds, water)
        cut = thresholds.index_delta_small if computed is None else computed
    else:
        cut = scene_threshold

    raw = ((magnitude > cut) | (cva > thresholds.cva_magnitude_threshold)) & valid
    return despeckle(
        raw,
        open_radius=thresholds.open_radius_px,
        min_size=thresholds.min_component_size_px,
    ).astype(np.uint8)


def scene_change_threshold(
    deltas: dict[str, np.ndarray],
    valid: np.ndarray,
    thresholds: GateThresholds,
    water: np.ndarray | None = None,
) -> float | None:
    """Scene-level adaptive cut, shared across that scene's tiles.

    The percentile is floored by ``min_absolute_delta``. A percentile alone is
    scale-free and therefore always selects its top N% of pixels — including on
    a scene where nothing changed at all, which made a pure illumination shift
    read as change. The floor is what makes "no change anywhere" representable.
    """
    if not thresholds.adaptive_threshold:
        return None
    magnitude = change_magnitude(deltas, water)
    pool = magnitude[valid] if valid.any() else magnitude
    if pool.size == 0:
        return thresholds.min_absolute_delta

    # Robust background estimate. A high percentile is not usable here: it
    # assumes change is rare, so on a scene where 14% of pixels changed the cut
    # lands *inside* the changed population and suppresses exactly what it
    # should detect. Median and MAD are dominated by the unchanged majority for
    # any plausible changed fraction, so the threshold tracks background
    # variability instead of the change itself.
    median = float(np.median(pool))
    mad = float(np.median(np.abs(pool - median)))
    adaptive = median + thresholds.background_sigma * 1.4826 * mad
    return float(
        np.clip(
            max(adaptive, thresholds.min_absolute_delta),
            thresholds.min_absolute_delta,
            thresholds.max_absolute_delta,
        )
    )


def compute_heatmap(
    change_mask: np.ndarray,
    deltas: dict[str, np.ndarray],
) -> np.ndarray:
    """Change-intensity heatmap on a fixed physical scale.

    Previously normalised by its own maximum, so a tile with a 0.005 delta and
    one with a 0.5 delta produced visually identical images — and this image is
    what the VLM sees as evidence.
    """
    combined = (np.abs(deltas["ndvi"]) + np.abs(deltas["ndbi"])) / 2.0
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    combined = combined * (change_mask > 0)
    return np.clip(combined / HEATMAP_FULL_SCALE, 0.0, 1.0).astype(np.float32)


def _ramp(value: float, lo: float, hi: float) -> float:
    """Linear 0->1 ramp, clamped. Used to build a monotone confidence."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def decide(
    f: GateFeatures,
    t: GateThresholds,
) -> tuple[GateDecision, str, float]:
    """Pure gate decision. The single source of truth for gate behaviour.

    Returns (decision, human-readable reason, confidence in [0, 1]).

    Evidence terms:
      spectral   - magnitude of landcover index change (NDVI/NDBI, now genuinely
                   independent measurements rather than aliases of each other)
      area       - fraction of the tile flagged by the despeckled change mask
      structural - SSIM/pHash disagreement, i.e. geometry moved
      cva        - multi-band change vector magnitude

    Monotone by construction: every rule is a conjunction of "evidence exceeds
    threshold" terms, so more evidence can only move a pair toward
    ``candidate_change``.
    """
    if not f.valid_observation:
        return "low_quality", "failed Tier 0 quality checks", 0.0

    spectral = max(f.ndvi_delta_abs_mean, f.ndbi_delta_abs_mean)
    strong_spectral = (
        f.ndvi_delta_abs_mean >= t.ndvi_strong_min or f.ndbi_delta_abs_mean >= t.ndbi_strong_min
    )
    moderate_spectral = (
        f.ndvi_delta_abs_mean >= t.ndvi_delta_mean_min
        or f.ndbi_delta_abs_mean >= t.ndbi_delta_mean_min
    )
    water_spectral = f.ndwi_delta_abs_mean >= t.ndwi_delta_mean_min
    area = f.changed_area_percent >= t.min_changed_area_percent
    structural = f.ssim < t.ssim_no_change_min or f.phash_distance > t.phash_no_change_max
    cva = f.cva_magnitude_mean >= t.cva_magnitude_mean_min

    confidence = float(
        np.clip(
            0.45 * _ramp(spectral, t.ndvi_delta_mean_min * 0.5, t.ndvi_strong_min)
            + 0.25 * _ramp(f.changed_area_percent, 0.0, t.min_changed_area_percent * 2)
            + 0.20 * _ramp(f.cva_magnitude_mean, 0.0, t.cva_magnitude_mean_min * 2)
            + 0.10 * _ramp(1.0 - f.ssim, 0.0, 1.0 - t.ssim_no_change_min),
            0.0,
            1.0,
        )
    )

    if strong_spectral:
        return "candidate_change", "strong landcover index change", confidence
    if moderate_spectral and area:
        return "candidate_change", "moderate landcover change over a large area", confidence
    if area and structural and cva:
        return "candidate_change", "structural break with multi-band change vector", confidence
    if water_spectral and area:
        return "candidate_change", "water extent change over a large area", confidence
    return "no_change", "insufficient change evidence", confidence


def classical_gate(
    pair_id: str,
    bands_t1: dict[str, np.ndarray],
    bands_t2: dict[str, np.ndarray],
    masks: EphemeralMasks,
    quality: QualityScore,
    thresholds: GateThresholds | None = None,
    *,
    date_t1: str = "t1",
    date_t2: str = "t2",
    aoi_id: str | None = None,
    source: str = "oscd",
    rgb_only: bool = False,
    cva_bands: tuple[str, ...] | None = None,
) -> GateArtifacts:
    """Run the classical gate over one aligned bitemporal pair."""
    thresholds = thresholds or GateThresholds()

    rgb1 = bands_to_rgb(bands_t1)
    rgb2 = bands_to_rgb(bands_t2)
    disp1, disp2 = stretch_for_display(rgb1, rgb2)
    gray1 = cv2.cvtColor(rgb_to_uint8(disp1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(rgb_to_uint8(disp2), cv2.COLOR_RGB2GRAY)

    ssim_val = compute_ssim(rgb1, rgb2)
    phash_d = phash_distance(gray1, gray2)

    deltas = _index_deltas(bands_t1, bands_t2)
    cva = compute_cva(bands_t1, bands_t2, cva_bands)
    valid = masks.valid

    change_mask = compute_change_mask(deltas, cva, valid, thresholds, water=masks.water)
    n_valid = int(valid.sum())
    changed_pct = float(100.0 * change_mask.sum() / n_valid) if n_valid else 0.0

    ndvi_m, ndvi_a = _masked_mean(deltas["ndvi"], valid)
    ndbi_m, ndbi_a = _masked_mean(deltas["ndbi"], valid)
    ndwi_m, ndwi_a = _masked_mean(deltas["ndwi"], valid)
    cva_mean = float(np.mean(cva[valid])) if n_valid else 0.0

    features = GateFeatures(
        ssim=ssim_val,
        phash_distance=phash_d,
        ndvi_delta_mean=ndvi_m,
        ndbi_delta_mean=ndbi_m,
        ndwi_delta_mean=ndwi_m,
        ndvi_delta_abs_mean=ndvi_a,
        ndbi_delta_abs_mean=ndbi_a,
        ndwi_delta_abs_mean=ndwi_a,
        cva_magnitude_mean=cva_mean,
        changed_area_percent=changed_pct,
        valid_observation=quality.valid_observation,
    )
    decision, reason, confidence = decide(features, thresholds)

    result = ClassicalResult(
        tile_id=pair_id,
        aoi_id=aoi_id or f"{source}_{pair_id}",
        source=source,
        date_t1=date_t1,
        date_t2=date_t2,
        masks_assessed=quality.masks_assessed,
        cloud_fraction_max=quality.cloud_fraction_max,
        snow_fraction_max=quality.snow_fraction_max,
        registration_error_px=quality.registration_error_px,
        season_delta_days=quality.season_delta_days,
        ssim=round(ssim_val, 4),
        phash_distance=phash_d,
        ndvi_delta_mean=round(ndvi_m, 4),
        ndbi_delta_mean=round(ndbi_m, 4),
        ndwi_delta_mean=round(ndwi_m, 4),
        ndvi_delta_abs_mean=round(ndvi_a, 4),
        ndbi_delta_abs_mean=round(ndbi_a, 4),
        ndwi_delta_abs_mean=round(ndwi_a, 4),
        cva_magnitude_mean=round(cva_mean, 4),
        changed_area_percent=round(changed_pct, 2),
        classical_gate=decision,
        gate_reason=reason,
        gate_confidence=round(confidence, 4),
        rgb_only=rgb_only,
    )
    return GateArtifacts(
        result=result,
        change_mask=change_mask,
        heatmap=compute_heatmap(change_mask, deltas),
    )


def save_heatmap_png(heatmap: np.ndarray, path: Path) -> None:
    """Write a heatmap as an 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.rint(np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
