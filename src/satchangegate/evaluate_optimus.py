"""Evaluation metrics for OPTIMUS labeled time series (first vs last frame)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import load_thresholds
from satchangegate.data.optimus import (
    build_tile_group_lookup,
    default_optimus_root,
    extract_group_tar,
    get_bitemporal_frames,
    list_eval_tiles_in_group,
    list_timestamps_for_tile,
    load_eval_labels,
)
from satchangegate.pipeline import run_from_bands


def _precision_recall_f1(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
    }


def run_optimus_eval(
    *,
    group_index: int | None = None,
    limit: int | None = None,
    optimus_root: Path | None = None,
    out_dir: Path | None = None,
    skip_vlm: bool = True,
    skip_llm: bool = True,
    sample_unlabeled: int = 0,
) -> dict[str, Any]:
    """
    Evaluate classical gate on OPTIMUS eval-labeled tiles from an extracted tar group.

    Labels are pair-level persistent change (0=no change, 1=change) for first vs last frame.
    """
    cfg = load_thresholds()
    root = Path(optimus_root or default_optimus_root())
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_index is None:
        lookup = build_tile_group_lookup(root)
        labels = load_eval_labels(root)
        counts: dict[int, int] = {}
        for tile_id in labels:
            gi = lookup.get(tile_id)
            if gi is not None:
                counts[gi] = counts.get(gi, 0) + 1
        if not counts:
            return {"error": "No eval tiles mapped in index.json", "n_pairs": 0}
        group_index = max(counts, key=counts.get)

    extract_group_tar(group_index, root)
    labeled = list_eval_tiles_in_group(group_index, root)
    if limit is not None:
        labeled = labeled[:limit]

    rows: list[dict[str, Any]] = []

    if not labeled:
        metrics = {
            "error": f"No eval-labeled tiles in group {group_index}. "
            f"Local tar may not contain eval set (455.tar has 0 eval tiles; try group 425 or 148).",
            "n_pairs": 0,
            "group_index": group_index,
            "gate_counts": {},
            "pair_level": _precision_recall_f1(0, 0, 0, 0),
        }
        if sample_unlabeled > 0:
            metrics["unlabeled_sample"] = _sample_unlabeled_group(
                group_index, root, cfg, out_dir, sample_unlabeled
            )
        summary_path = out_dir / "_optimus_eval_summary.md"
        summary_path.write_text(_format_summary(metrics, rows), encoding="utf-8")
        (out_dir / "_optimus_eval_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    gate_counts = {"no_change": 0, "candidate_change": 0, "low_quality": 0}
    tp = fp = fn = tn = 0

    for tile_id, gt in labeled:
        try:
            b1, b2, t1, t2 = get_bitemporal_frames(tile_id, root)
        except (FileNotFoundError, ValueError) as e:
            rows.append(
                {
                    "tile_id": tile_id,
                    "ground_truth": gt,
                    "classical_gate": "error",
                    "error": str(e),
                }
            )
            continue

        result = run_from_bands(
            f"optimus_{tile_id}",
            b1,
            b2,
            cfg=cfg,
            out_dir=out_dir / "optimus_eval",
            date_t1=t1,
            date_t2=t2,
            skip_vlm=skip_vlm,
            skip_llm=skip_llm,
            ground_truth_label=gt,
        )
        gate = result.classical_gate
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        pred = gate == "candidate_change"
        truth = gt == 1
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1

        row = dict(result.feature_row)
        row["ground_truth"] = gt
        row["group_index"] = group_index
        rows.append(row)

    n = len([r for r in rows if r.get("classical_gate") != "error"])
    prf = _precision_recall_f1(tp, fp, fn, tn)
    vlm_candidates = gate_counts.get("candidate_change", 0)
    metrics = {
        "dataset": "optimus",
        "group_index": group_index,
        "n_pairs": n,
        "n_labeled_requested": len(labeled),
        "gate_counts": gate_counts,
        "gate_candidate_pct": round(100.0 * vlm_candidates / n, 1) if n else 0.0,
        "gate_filtered_pct": round(
            100.0 * (gate_counts.get("no_change", 0) + gate_counts.get("low_quality", 0)) / n,
            1,
        )
        if n
        else 0.0,
        "vlm_calls_avoided": gate_counts.get("no_change", 0) + gate_counts.get("low_quality", 0),
        "pair_level": prf,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    summary_path = out_dir / "_optimus_eval_summary.md"
    summary_path.write_text(_format_summary(metrics, rows), encoding="utf-8")
    (out_dir / "_optimus_eval_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if rows:
        csv_path = out_dir / "_optimus_eval_features.csv"
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    if sample_unlabeled > 0:
        metrics["unlabeled_sample"] = _sample_unlabeled_group(
            group_index, root, cfg, out_dir, sample_unlabeled
        )

    return metrics


def _sample_unlabeled_group(
    group_index: int,
    root: Path,
    cfg: dict[str, Any],
    out_dir: Path,
    n: int,
) -> dict[str, Any]:
    """Run gate on random unlabeled tiles from extracted group (distribution check)."""
    from satchangegate.data.optimus import load_index

    labels = set(load_eval_labels(root))
    index = load_index(root)
    candidates = [
        png.replace(".png", "")
        for png in index[group_index]
        if png.replace(".png", "") not in labels
    ]
    if not candidates:
        return {"n": 0}
    rng = np.random.default_rng(42)
    pick = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
    gates: dict[str, int] = {"no_change": 0, "candidate_change": 0, "low_quality": 0}
    for tile_id in pick:
        try:
            stamps = list_timestamps_for_tile(tile_id, root)
            if len(stamps) < 2:
                continue
            b1, b2, t1, t2 = get_bitemporal_frames(tile_id, root)
            r = run_from_bands(
                f"optimus_sample_{tile_id}",
                b1,
                b2,
                cfg=cfg,
                out_dir=out_dir / "optimus_sample",
                date_t1=t1,
                date_t2=t2,
                skip_vlm=True,
                skip_llm=True,
            )
            gates[r.classical_gate] = gates.get(r.classical_gate, 0) + 1
        except (FileNotFoundError, ValueError):
            continue
    return {"n": int(len(pick)), "gate_counts": gates}


def _format_summary(metrics: dict[str, Any], rows: list[dict]) -> str:
    if metrics.get("error"):
        lines = [f"# OPTIMUS Eval\n\n**Note:** {metrics['error']}\n"]
        unlabeled = metrics.get("unlabeled_sample")
        if unlabeled and unlabeled.get("n"):
            lines.extend(
                [
                    "",
                    "## Unlabeled sample (same tar)",
                    "",
                    f"- Sampled: **{unlabeled['n']}** tiles",
                    f"- Gate counts: {unlabeled.get('gate_counts')}",
                    f"- Candidate rate: **{100*unlabeled['gate_counts'].get('candidate_change',0)/unlabeled['n']:.1f}%**",
                ]
            )
        return "\n".join(lines)

    prf = metrics["pair_level"]
    conf = metrics["confusion"]
    lines = [
        "# OPTIMUS Eval Summary",
        "",
        f"**Tar group:** images/{metrics['group_index']}.tar",
        f"**Labeled pairs evaluated:** {metrics['n_pairs']}",
        "",
        "## Gate metrics",
        "",
        "| Gate | Count |",
        "|------|-------|",
    ]
    for k, v in metrics["gate_counts"].items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            f"- VLM candidate rate: **{metrics['gate_candidate_pct']}%**",
            f"- Filtered before VLM: **{metrics['gate_filtered_pct']}%**",
            f"- Calls avoided (no_change + low_quality): **{metrics['vlm_calls_avoided']}**",
            "",
            "## Pair-level vs OPTIMUS labels (0=no change, 1=change)",
            "",
            f"- Confusion: TP={conf['tp']} FP={conf['fp']} FN={conf['fn']} TN={conf['tn']}",
            f"- Precision: **{prf['precision']:.3f}**",
            f"- Recall: **{prf['recall']:.3f}**",
            f"- F1: **{prf['f1']:.3f}**",
            f"- Specificity (no-change correct): **{prf['specificity']:.3f}**",
            "",
        ]
    )
    if rows:
        lines.append("## Per-tile features")
        lines.append("")
        lines.append("| tile | gt | gate | changed_% | ndvi_d | ndbi_d | ssim | phash |")
        lines.append("|------|----|------|-----------|--------|--------|------|-------|")
        for r in rows:
            if r.get("classical_gate") == "error":
                continue
            lines.append(
                f"| {r.get('tile_id')} | {r.get('ground_truth')} | {r.get('classical_gate')} | "
                f"{r.get('changed_area_percent')} | {r.get('ndvi_delta_mean')} | "
                f"{r.get('ndbi_delta_mean')} | {r.get('ssim')} | {r.get('phash_distance')} |"
            )
    unlabeled = metrics.get("unlabeled_sample")
    if unlabeled and unlabeled.get("n"):
        lines.extend(
            [
                "",
                "## Unlabeled sample (same tar)",
                "",
                f"- Sampled: **{unlabeled['n']}** tiles",
                f"- Gate counts: {unlabeled.get('gate_counts')}",
            ]
        )
    return "\n".join(lines)
