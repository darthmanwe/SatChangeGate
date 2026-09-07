"""Gate evaluation on the labelled OSCD tile set.

Reports on a real confusion matrix with a genuine negative class, with n and
95% Wilson intervals on every proportion. Tuning and reporting are separated by
city: ``--split test`` never touches a city that ``--split train`` tuned on, and
that disjointness is asserted rather than assumed.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from satchangegate.config import Settings, get_settings
from satchangegate.data.oscd import discover_pairs, load_bands, load_label_mask
from satchangegate.data.tiles import Tile, build_tile_index, summarise
from satchangegate.features.classical import (
    GateFeatures,
    _index_deltas,
    _masked_mean,
    compute_change_mask,
    compute_cva,
    compute_ssim,
    decide,
    phash_distance,
    scene_change_threshold,
    tile_feature_extras,
)
from satchangegate.metrics import ConfusionMatrix, confusion_from_pairs, pixel_confusion
from satchangegate.preprocess.align import (
    bands_to_rgb,
    estimate_registration_error,
    resample_to_common_grid,
    rgb_to_uint8,
    stretch_for_display,
)
from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
from satchangegate.preprocess.quality import QualityScore, compute_quality_score
from satchangegate.preprocess.radiometric import normalize_to_reference


@dataclass
class SceneCache:
    """Whole-scene arrays computed once and sliced per tile."""

    city: str
    deltas: dict[str, np.ndarray]
    cva: np.ndarray
    valid: np.ndarray
    gray1: np.ndarray
    gray2: np.ndarray
    rgb1: np.ndarray
    rgb2: np.ndarray
    disp1: np.ndarray
    disp2: np.ndarray
    valid_observation: bool
    registration_px: float
    scene_threshold: float | None
    water: np.ndarray
    quality: QualityScore
    date_t1: str
    date_t2: str


def build_scene_cache(pair, settings: Settings) -> SceneCache:
    """Preprocess one city once; tiles are then pure slicing."""
    b1 = load_bands(pair.img1_dir, list(settings.bands))
    b2 = load_bands(pair.img2_dir, list(settings.bands))
    b1, b2 = resample_to_common_grid(b1, b2)

    # Radiometric normalization, when enabled, happens before masks and indices:
    # a gain/offset correction fitted on unchanged pixels changes what counts as
    # cloud-bright and what counts as a spectral delta, so applying it later
    # would leave the two disagreeing.
    if settings.preprocess.normalize:
        b2, _norm = normalize_to_reference(
            b1,
            b2,
            iterations=settings.preprocess.pif_iterations,
            invariant_percentile=settings.preprocess.pif_invariant_percentile,
        )

    registration_px = estimate_registration_error(b1, b2)
    m1 = compute_ephemeral_masks(b1, settings.masks)
    m2 = compute_ephemeral_masks(b2, settings.masks)
    combined = combine_pair_masks(m1, m2)
    quality = compute_quality_score(
        m1,
        m2,
        settings.quality,
        registration_error_px=registration_px,
        date_t1=pair.date_t1,
        date_t2=pair.date_t2,
        combined=combined,
    )

    rgb1, rgb2 = bands_to_rgb(b1), bands_to_rgb(b2)
    disp1, disp2 = stretch_for_display(rgb1, rgb2)
    deltas = _index_deltas(b1, b2)
    return SceneCache(
        city=pair.pair_id,
        deltas=deltas,
        cva=compute_cva(b1, b2),
        valid=combined.valid,
        gray1=cv2.cvtColor(rgb_to_uint8(disp1), cv2.COLOR_RGB2GRAY),
        gray2=cv2.cvtColor(rgb_to_uint8(disp2), cv2.COLOR_RGB2GRAY),
        rgb1=rgb1,
        rgb2=rgb2,
        disp1=disp1,
        disp2=disp2,
        valid_observation=quality.valid_observation,
        registration_px=registration_px,
        water=combined.water,
        quality=quality,
        date_t1=pair.date_t1 or "unknown",
        date_t2=pair.date_t2 or "unknown",
        scene_threshold=scene_change_threshold(
            deltas, combined.valid, settings.gate, combined.water
        ),
    )


def mask_and_features_for_tile(
    cache: SceneCache, tile: Tile, settings: Settings
) -> tuple[np.ndarray, GateFeatures]:
    """Gate features for one tile window, plus the change mask they came from.

    The mask used to be computed here and discarded once ``changed_area_percent``
    had been taken from it, which is why no pixel-level metric was reproducible.
    """
    ys, xs = tile.slice_yx
    deltas = {k: v[ys, xs] for k, v in cache.deltas.items()}
    cva = cache.cva[ys, xs]
    valid = cache.valid[ys, xs]

    change_mask = compute_change_mask(
        deltas,
        cva,
        valid,
        settings.gate,
        scene_threshold=cache.scene_threshold,
        water=cache.water[ys, xs],
    )
    n_valid = int(valid.sum())
    changed_pct = float(100.0 * change_mask.sum() / n_valid) if n_valid else 0.0

    ndvi_m, ndvi_a = _masked_mean(deltas["ndvi"], valid)
    ndbi_m, ndbi_a = _masked_mean(deltas["ndbi"], valid)
    ndwi_m, ndwi_a = _masked_mean(deltas["ndwi"], valid)

    extras = tile_feature_extras(deltas, valid, change_mask, cache.water[ys, xs])
    features = GateFeatures(
        ssim=compute_ssim(cache.disp1[ys, xs], cache.disp2[ys, xs], already_stretched=True),
        phash_distance=phash_distance(cache.gray1[ys, xs], cache.gray2[ys, xs]),
        ndvi_delta_mean=ndvi_m,
        ndbi_delta_mean=ndbi_m,
        ndwi_delta_mean=ndwi_m,
        ndvi_delta_abs_mean=ndvi_a,
        ndbi_delta_abs_mean=ndbi_a,
        ndwi_delta_abs_mean=ndwi_a,
        cva_magnitude_mean=float(np.mean(cva[valid])) if n_valid else 0.0,
        changed_area_percent=changed_pct,
        valid_observation=cache.valid_observation,
        **extras,  # type: ignore[arg-type]
    )
    return change_mask, features


def features_for_tile(cache: SceneCache, tile: Tile, settings: Settings) -> GateFeatures:
    """Compute gate features for one tile window."""
    return mask_and_features_for_tile(cache, tile, settings)[1]


def compute_tile_features(
    root: Path | None,
    tiles: Iterable[Tile],
    settings: Settings,
    *,
    pixel_accumulator: ConfusionMatrix | None = None,
) -> list[dict[str, Any]]:
    """Compute features for every tile, loading each city exactly once.

    When ``pixel_accumulator`` is supplied it is summed in place with the
    pixel-level confusion between each tile's change mask and the dataset's own
    change mask, restricted to observed pixels. It is an accumulator rather than
    a return value so the expensive per-city preprocessing is not repeated.
    """
    tiles = list(tiles)
    pairs = {p.pair_id: p for p in discover_pairs(root)}
    rows: list[dict[str, Any]] = []
    by_city: dict[str, list[Tile]] = {}
    for tile in tiles:
        by_city.setdefault(tile.city, []).append(tile)

    for city in sorted(by_city):
        pair = pairs.get(city)
        if pair is None:
            continue
        cache = build_scene_cache(pair, settings)
        label = load_label_mask(pair) if pixel_accumulator is not None else None
        for tile in by_city[city]:
            mask, feats = mask_and_features_for_tile(cache, tile, settings)
            if pixel_accumulator is not None and label is not None:
                _accumulate_pixels(pixel_accumulator, mask, label, cache, tile)
            rows.append(
                {
                    "tile_id": tile.tile_id,
                    "city": tile.city,
                    "split": tile.split,
                    "label": tile.label,
                    "change_fraction": tile.change_fraction,
                    "registration_px": round(cache.registration_px, 3),
                    **feats.model_dump(),
                }
            )
    return rows


def _accumulate_pixels(
    acc: ConfusionMatrix,
    mask: np.ndarray,
    label: np.ndarray,
    cache: SceneCache,
    tile: Tile,
) -> None:
    """Add one tile's pixel confusion into a running total.

    Skips a tile whose label grid does not match the resampled band grid rather
    than silently comparing misaligned arrays.
    """
    ys, xs = tile.slice_yx
    if label.shape != cache.valid.shape:
        return
    cm = pixel_confusion(mask.astype(bool), label[ys, xs].astype(bool), cache.valid[ys, xs])
    acc.tp += cm.tp
    acc.fp += cm.fp
    acc.fn += cm.fn
    acc.tn += cm.tn


def score_rows(
    rows: list[dict[str, Any]], settings: Settings
) -> tuple[ConfusionMatrix, list[dict]]:
    """Apply the Tier-1 decision to precomputed feature rows.

    Dispatches on ``settings.scorer.kind``. The rules are the default; the
    learned scorer is opt-in and degrades back to the rules with a warning rather
    than failing a run, because a missing or stale artifact is an operations
    problem and should not look like a modelling result.
    """
    if settings.scorer.kind == "learned":
        scored = _score_rows_learned_or_fall_back(rows, settings)
        if scored is not None:
            y_true = [int(r["label"]) for r in scored if r["gate"] != "low_quality"]
            y_pred = [
                1 if r["gate"] == "candidate_change" else 0
                for r in scored
                if r["gate"] != "low_quality"
            ]
            return confusion_from_pairs(y_true, y_pred), scored

    scored = []
    y_true, y_pred = [], []
    for row in rows:
        feats = GateFeatures.model_validate(
            {k: row[k] for k in GateFeatures.model_fields if k in row}
        )
        decision, reason, confidence = decide(feats, settings.gate)
        out = dict(row)
        out.update(
            {"gate": decision, "gate_reason": reason, "gate_confidence": round(confidence, 4)}
        )
        scored.append(out)
        # low_quality is a refusal to judge, not a negative prediction; it is
        # excluded from the classification metrics and reported separately.
        if decision == "low_quality":
            continue
        y_true.append(int(row["label"]))
        y_pred.append(1 if decision == "candidate_change" else 0)
    return confusion_from_pairs(y_true, y_pred), scored


def _score_rows_learned_or_fall_back(
    rows: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]] | None:
    """Score with the persisted model, or return None to use the rules."""
    try:
        from satchangegate.scorer import load_scorer, score_rows_learned

        artifact = load_scorer(Path(settings.scorer.path))
    except Exception as exc:
        warnings.warn(
            f"scorer.kind is 'learned' but the model could not be used "
            f"({type(exc).__name__}: {exc}). Falling back to the rule gate. "
            "Fit one with `satchangegate fit-scorer --split train`.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return score_rows_learned(rows, artifact, settings.scorer.threshold, settings)


def run_eval(
    oscd_root: Path | None = None,
    split: str = "test",
    out_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    tile_size: int | None = None,
    pixel_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate the gate on one split of the labelled tile set."""
    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"tile_size": tile_size} if tile_size else {}
    all_tiles = build_tile_index(oscd_root, **kwargs)
    tiles = [t for t in all_tiles if split == "all" or t.split == split]
    if not tiles:
        return {"error": f"no tiles for split {split!r}", "n": 0}

    pixel_acc = ConfusionMatrix() if pixel_metrics else None
    rows = compute_tile_features(oscd_root, tiles, settings, pixel_accumulator=pixel_acc)
    cm, scored = score_rows(rows, settings)

    n_low_quality = sum(1 for r in scored if r["gate"] == "low_quality")
    n_candidate = sum(1 for r in scored if r["gate"] == "candidate_change")
    n_assessable = len(scored) - n_low_quality
    n_gate_filtered = n_assessable - n_candidate
    # Cities Tier 0 was willing to judge. valid_observation is a scene-level flag,
    # so one city failing co-registration removes all of its tiles at once --
    # which is how "measured on 10 held-out cities" was true of the tile index
    # but not of the metrics computed from it.
    cities_scored = sorted({r["city"] for r in scored if r["gate"] != "low_quality"})
    summary: dict[str, Any] = {
        "split": split,
        "cities": sorted({t.city for t in tiles}),
        "cities_scored": cities_scored,
        "tile_summary": summarise(tiles),
        "n_tiles": len(scored),
        "n_low_quality": n_low_quality,
        "n_assessable": n_assessable,
        "n_candidate_change": n_candidate,
        "candidate_rate_pct": round(100.0 * n_candidate / max(1, len(scored)), 2),
        # Three distinct reductions. Tier 0 refuses to judge; the gate judges and
        # finds nothing. One blended number credited the gate with co-registration
        # failures it had no part in.
        "tier0_refused_pct": round(100.0 * n_low_quality / max(1, len(scored)), 2),
        "gate_filtered_pct": round(100.0 * n_gate_filtered / max(1, n_assessable), 2),
        "total_reduction_pct": round(100.0 * (len(scored) - n_candidate) / max(1, len(scored)), 2),
        "metrics": cm.to_dict(),
        "thresholds": settings.gate.model_dump(),
    }
    if pixel_acc is not None:
        summary["pixel_metrics"] = pixel_acc.to_dict()
        summary["pixel_metrics"]["iou"] = round(
            pixel_acc.tp / max(1, pixel_acc.tp + pixel_acc.fp + pixel_acc.fn), 4
        )
        summary["pixel_metrics"]["note"] = (
            "Pixel-level confusion between the despeckled change mask and the "
            "dataset's own change mask, over observed pixels only, accumulated "
            f"across the {split} split. This is a triage gate, not a segmentation "
            "model: the number exists so the claim is reproducible, not because it "
            "is competitive."
        )

    (out_dir / f"_eval_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_features_csv(scored, out_dir / f"_eval_{split}_features.csv")
    (out_dir / f"_eval_{split}.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _write_features_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    m = summary["metrics"]
    conf = m["confusion"]
    lines = [
        f"# Gate evaluation — `{summary['split']}` split",
        "",
        f"{summary['n_tiles']} tiles across {len(summary['cities'])} cities "
        f"({m['n_positive']} change / {m['n_negative']} no-change).",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Recall | {m['recall']:.3f} | {m['recall_ci95'][0]:.3f}-{m['recall_ci95'][1]:.3f} |",
    ]
    if m["has_negative_class"]:
        lines += [
            f"| Precision | {m['precision']:.3f} | "
            f"{m['precision_ci95'][0]:.3f}-{m['precision_ci95'][1]:.3f} |",
            f"| Specificity | {m['specificity']:.3f} | "
            f"{m['specificity_ci95'][0]:.3f}-{m['specificity_ci95'][1]:.3f} |",
            f"| F1 | {m['f1']:.3f} | — |",
            f"| Balanced accuracy | {m['balanced_accuracy']:.3f} | — |",
        ]
    else:
        lines.append(f"| Precision / F1 | undefined | {m.get('metrics_note', '')} |")
    lines += [
        "",
        f"Confusion matrix: TP {conf['tp']} · FP {conf['fp']} · FN {conf['fn']} · TN {conf['tn']}",
        "",
        f"Scored on {len(summary.get('cities_scored', summary['cities']))} of "
        f"{len(summary['cities'])} cities in this split.",
        "",
        "| Reduction | Count | Share |",
        "|---|---|---|",
        f"| Refused by Tier 0 (unusable imagery) | {summary['n_low_quality']} | "
        f"{summary.get('tier0_refused_pct', 0.0):.1f}% of all tiles |",
        f"| Filtered by the gate (judged unchanged) | "
        f"{summary.get('n_assessable', 0) - summary['n_candidate_change']} | "
        f"{summary['gate_filtered_pct']:.1f}% of assessable tiles |",
        f"| Forwarded to the VLM | {summary['n_candidate_change']} | "
        f"{summary['candidate_rate_pct']:.1f}% of all tiles |",
        "",
        f"Total reduction before any API call: {summary.get('total_reduction_pct', 0.0):.1f}%.",
    ]
    if summary.get("pixel_metrics"):
        pm = summary["pixel_metrics"]
        lines += [
            "",
            "## Pixel level",
            "",
            f"F1 {pm['f1']:.3f} - IoU {pm['iou']:.3f} - recall {pm['recall']:.3f} - "
            f"precision {pm['precision']:.3f} (n={pm['n']} observed pixels).",
            "",
            "This is a triage gate, not a segmentation model. The number is here so "
            "the claim is reproducible, not because it is competitive with supervised "
            "deep models on this benchmark.",
        ]
    return "\n".join(lines) + "\n"
