"""Threshold sweep for classical gate on OPTIMUS labeled tiles."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from satchangegate.config import load_thresholds
from satchangegate.data.optimus import (
    build_tile_group_lookup,
    default_optimus_root,
    extract_group_tar,
    get_bitemporal_frames,
    list_eval_tiles_in_group,
    load_eval_labels,
)
from satchangegate.features.classical import classical_gate
from satchangegate.preprocess.align import align_pair_bands
from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
from satchangegate.preprocess.quality import compute_quality_score


def _collect_labeled_features(
    groups: list[int],
    root: Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(root or default_optimus_root())
    cfg = load_thresholds()
    rows: list[dict[str, Any]] = []

    for gi in groups:
        try:
            extract_group_tar(gi, root)
        except FileNotFoundError:
            continue
        for tile_id, label in list_eval_tiles_in_group(gi, root):
            try:
                b1, b2, t1, t2 = get_bitemporal_frames(tile_id, root)
            except (FileNotFoundError, ValueError):
                continue
            bands_t1, bands_t2 = align_pair_bands(b1, b2, cfg=cfg)
            m1 = compute_ephemeral_masks(bands_t1, cfg, layout="rgb_png")
            m2 = compute_ephemeral_masks(bands_t2, cfg, layout="rgb_png")
            combined = combine_pair_masks(m1, m2)
            quality = compute_quality_score(m1, m2, bands_t1, bands_t2, cfg)
            art = classical_gate(
                f"optimus_{tile_id}",
                bands_t1,
                bands_t2,
                combined,
                quality,
                cfg=cfg,
                date_t1=t1,
                date_t2=t2,
            )
            row = art.result.model_dump()
            row["ground_truth"] = label
            row["tile_id"] = tile_id
            row["group_index"] = gi
            rows.append(row)
    return rows


def _predict(row: dict[str, Any], gcfg: dict[str, Any]) -> str:
    """Mirror classical_gate decision logic for sweeps."""
    cloud_max = row["cloud_fraction_max"]
    if cloud_max > gcfg.get("cloud_fraction_max", 0.25):
        return "low_quality"

    if cloud_max > gcfg.get("cloud_fraction_max", 0.25):
        return "low_quality"

    ssim_score = row["ssim"]
    phash_dist = row["phash_distance"]
    ndvi_mean = row["ndvi_delta_mean"]
    ndbi_mean = row["ndbi_delta_mean"]
    ndwi_mean = row["ndwi_delta_mean"]
    cva_mean = row["cva_magnitude_mean"]
    changed_pct = row["changed_area_percent"]

    structural_ok = (
        ssim_score >= gcfg.get("ssim_no_change_min", 0.65)
        and phash_dist <= gcfg.get("phash_no_change_max", 16)
    )
    area_thr = gcfg.get("min_changed_area_percent", 25.0)
    cva_thr = gcfg.get("cva_magnitude_mean_min", 0.28)

    landcover_spectral = (
        ndvi_mean >= gcfg.get("ndvi_delta_mean_min", 0.10)
        or ndbi_mean >= gcfg.get("ndbi_delta_mean_min", 0.08)
    )
    strong_landcover = (
        ndvi_mean >= gcfg.get("ndvi_strong_min", 0.12)
        or ndbi_mean >= gcfg.get("ndbi_strong_min", 0.10)
    )
    water_spectral = ndwi_mean >= gcfg.get("ndwi_delta_mean_min", 0.15)
    area_change = changed_pct >= area_thr
    structural_break = not structural_ok
    cva_change = cva_mean >= cva_thr

    if (
        max(ndvi_mean, ndbi_mean) < gcfg.get("phenology_spectral_max", 0.08)
        and ssim_score < gcfg.get("phenology_ssim_max", 0.30)
        and changed_pct < gcfg.get("phenology_area_max_percent", 85.0)
    ):
        return "no_change"
    if (
        ssim_score < gcfg.get("unreliable_ssim_max", 0.08)
        and max(ndvi_mean, ndbi_mean) < gcfg.get("unreliable_spectral_max", 0.10)
    ):
        return "no_change"
    if area_change and structural_break and changed_pct >= gcfg.get("high_area_structural_min", 70.0):
        if max(ndvi_mean, ndbi_mean) >= gcfg.get("high_area_spectral_min", 0.078):
            return "candidate_change"
        return "no_change"
    if strong_landcover and (area_change or structural_break):
        return "candidate_change"
    if landcover_spectral and area_change and structural_break:
        return "candidate_change"
    if landcover_spectral and cva_change and area_change:
        return "candidate_change"
    if ndbi_mean >= gcfg.get("ndbi_built_min", 0.09) and area_change and structural_break:
        return "candidate_change"
    if water_spectral and area_change and cva_change:
        return "candidate_change"
    if (
        (ndvi_mean >= gcfg.get("ndvi_moderate_min", 0.065) or ndbi_mean >= gcfg.get("ndbi_moderate_min", 0.065))
        and area_change
        and structural_break
        and changed_pct >= gcfg.get("moderate_area_min_percent", 45.0)
    ):
        return "candidate_change"
    if ndvi_mean >= gcfg.get("ndvi_strong_min", 0.12) and area_change:
        return "candidate_change"
    return "no_change"


def _score(rows: list[dict[str, Any]], gcfg: dict[str, Any]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for row in rows:
        pred = _predict(row, gcfg) == "candidate_change"
        truth = row["ground_truth"] == 1
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    n = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    filt = (tn + fn) / n if n else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "specificity": spec, "filtered_pct": filt}


def sweep(groups: list[int] | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    root = default_optimus_root()
    if groups is None:
        groups = [148, 425]
    rows = _collect_labeled_features(groups, root)
    if not rows:
        raise RuntimeError("No labeled features collected — are tars extracted?")

    base = load_thresholds()["gate"]
    grid = {
        "ndvi_delta_mean_min": [0.08, 0.10, 0.12],
        "ndbi_delta_mean_min": [0.07, 0.08, 0.09],
        "ndvi_strong_min": [0.11, 0.12, 0.14],
        "ndbi_strong_min": [0.09, 0.10, 0.11],
        "ndbi_built_min": [0.08, 0.09, 0.10],
        "min_changed_area_percent": [15.0, 25.0, 35.0],
        "ssim_no_change_min": [0.55, 0.65, 0.72],
        "cva_magnitude_mean_min": [0.22, 0.28, 0.35],
    }

    best_cfg = dict(base)
    best_score = {"f1": -1.0}
    for combo in itertools.product(*grid.values()):
        gcfg = dict(base)
        for key, val in zip(grid.keys(), combo):
            gcfg[key] = val
        s = _score(rows, gcfg)
        # Prefer F1, then specificity, then filter rate
        if (s["f1"], s["specificity"], s["filtered_pct"]) > (
            best_score.get("f1", -1),
            best_score.get("specificity", -1),
            best_score.get("filtered_pct", -1),
        ):
            best_score = s
            best_cfg.update({k: gcfg[k] for k in grid})

    return best_cfg, best_score


if __name__ == "__main__":
    cfg, metrics = sweep()
    print("Best metrics:", metrics)
    for k, v in sorted(cfg.items()):
        if k in {
            "ndvi_delta_mean_min",
            "ndbi_delta_mean_min",
            "ndvi_strong_min",
            "ndbi_strong_min",
            "ndbi_built_min",
            "min_changed_area_percent",
            "ssim_no_change_min",
            "cva_magnitude_mean_min",
            "phash_no_change_max",
        }:
            print(f"  {k}: {v}")
