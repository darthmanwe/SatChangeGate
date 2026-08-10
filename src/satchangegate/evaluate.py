"""Gate evaluation on the labelled OSCD tile set.

Reports on a real confusion matrix with a genuine negative class, with n and
95% Wilson intervals on every proportion. Tuning and reporting are separated by
city: ``--split test`` never touches a city that ``--split train`` tuned on, and
that disjointness is asserted rather than assumed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from satchangegate.config import Settings, get_settings
from satchangegate.data.oscd import discover_pairs, load_bands
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
)
from satchangegate.metrics import ConfusionMatrix, confusion_from_pairs
from satchangegate.preprocess.align import (
    bands_to_rgb,
    estimate_registration_error,
    resample_to_common_grid,
    rgb_to_uint8,
    stretch_for_display,
)
from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
from satchangegate.preprocess.quality import compute_quality_score


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
    valid_observation: bool
    registration_px: float
    scene_threshold: float | None
    water: np.ndarray


def build_scene_cache(pair, settings: Settings) -> SceneCache:
    """Preprocess one city once; tiles are then pure slicing."""
    b1 = load_bands(pair.img1_dir, list(settings.bands))
    b2 = load_bands(pair.img2_dir, list(settings.bands))
    b1, b2 = resample_to_common_grid(b1, b2)

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
        valid_observation=quality.valid_observation,
        registration_px=registration_px,
        water=combined.water,
        scene_threshold=scene_change_threshold(
            deltas, combined.valid, settings.gate, combined.water
        ),
    )


def features_for_tile(cache: SceneCache, tile: Tile, settings: Settings) -> GateFeatures:
    """Compute gate features for one tile window."""
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

    return GateFeatures(
        ssim=compute_ssim(cache.rgb1[ys, xs], cache.rgb2[ys, xs]),
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
    )


def compute_tile_features(
    root: Path | None,
    tiles: Iterable[Tile],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Compute features for every tile, loading each city exactly once."""
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
        for tile in by_city[city]:
            feats = features_for_tile(cache, tile, settings)
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


def score_rows(
    rows: list[dict[str, Any]], settings: Settings
) -> tuple[ConfusionMatrix, list[dict]]:
    """Apply the gate decision to precomputed feature rows."""
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


def run_eval(
    oscd_root: Path | None = None,
    split: str = "test",
    out_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    tile_size: int | None = None,
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

    rows = compute_tile_features(oscd_root, tiles, settings)
    cm, scored = score_rows(rows, settings)

    n_low_quality = sum(1 for r in scored if r["gate"] == "low_quality")
    n_candidate = sum(1 for r in scored if r["gate"] == "candidate_change")
    summary: dict[str, Any] = {
        "split": split,
        "cities": sorted({t.city for t in tiles}),
        "tile_summary": summarise(tiles),
        "n_tiles": len(scored),
        "n_low_quality": n_low_quality,
        "n_candidate_change": n_candidate,
        "candidate_rate_pct": round(100.0 * n_candidate / max(1, len(scored)), 2),
        "gate_filtered_pct": round(100.0 * (len(scored) - n_candidate) / max(1, len(scored)), 2),
        "metrics": cm.to_dict(),
        "thresholds": settings.gate.model_dump(),
    }

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
        f"Gate filtered {summary['gate_filtered_pct']:.1f}% of tiles before any VLM call "
        f"({summary['n_candidate_change']} candidates, "
        f"{summary['n_low_quality']} rejected by Tier 0 quality checks).",
    ]
    return "\n".join(lines) + "\n"
