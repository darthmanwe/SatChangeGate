"""Evaluation metrics for OSCD test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import load_thresholds
from satchangegate.data.oscd import discover_pairs, list_pairs, load_label_mask
from satchangegate.features.classical import compute_change_mask
from satchangegate.pipeline import run_pair


def _pair_has_change(label: np.ndarray | None) -> bool:
    if label is None:
        return True
    return bool(label.sum() > 0)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def run_eval(
    oscd_root: Path,
    split: str = "test",
    out_dir: Path | None = None,
    *,
    skip_vlm: bool = True,
    skip_llm: bool = True,
) -> dict[str, Any]:
    """Evaluate pipeline on OSCD split; return metrics dict."""
    cfg = load_thresholds()
    oscd_root = Path(oscd_root)
    out_dir = Path(out_dir or Path("data/reports"))
    pairs = list_pairs(oscd_root, split)
    if not pairs:
        discovered = discover_pairs(oscd_root)
        pairs = [p for p in discovered if p.split == split or split == "all"]

    n = len(pairs)
    if n == 0:
        msg = f"No OSCD pairs found under {oscd_root} for split={split}"
        summary = {"error": msg, "n_pairs": 0}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_eval_summary.md").write_text(f"# Eval\n\n{msg}\n", encoding="utf-8")
        return summary

    gate_counts = {"no_change": 0, "candidate_change": 0, "low_quality": 0}
    vlm_calls = 0
    tp = fp = fn = 0
    ious: list[float] = []
    vlm_agree = 0
    vlm_total = 0

    feature_rows: list[dict] = []

    for pair in pairs:
        result = run_pair(
            pair,
            cfg=cfg,
            out_dir=out_dir,
            skip_vlm=skip_vlm,
            skip_llm=skip_llm,
        )
        gate = result.classical_gate
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
        feature_rows.append(result.feature_row)

        if result.vlm_called:
            vlm_calls += 1

        label = load_label_mask(pair)
        gt_change = _pair_has_change(label)

        pred_change = gate == "candidate_change"
        if pred_change and gt_change:
            tp += 1
        elif pred_change and not gt_change:
            fp += 1
        elif not pred_change and gt_change:
            fn += 1

        if label is not None and gate != "low_quality":
            from satchangegate.data.oscd import load_bands
            from satchangegate.preprocess.align import align_pair_bands
            from satchangegate.preprocess.masks import compute_ephemeral_masks, combine_pair_masks

            bands_list = cfg.get("bands", ["B02", "B03", "B04", "B08", "B11", "B12"])
            raw_t1 = load_bands(pair.img1_dir, bands_list)
            raw_t2 = load_bands(pair.img2_dir, bands_list)
            b1, b2 = align_pair_bands(raw_t1, raw_t2, cfg=cfg)
            m = combine_pair_masks(
                compute_ephemeral_masks(b1, cfg),
                compute_ephemeral_masks(b2, cfg),
            )
            pred_mask = compute_change_mask(b1, b2, m.valid, cfg)
            if label.shape != pred_mask.shape:
                from skimage.transform import resize

                label = resize(
                    label,
                    pred_mask.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
            ious.append(_iou(pred_mask, label))

        if result.vlm_verdict and label is not None:
            vlm_total += 1
            vlm_real = result.vlm_verdict.vlm_verdict == "real_change"
            if vlm_real == gt_change:
                vlm_agree += 1

    prf = _precision_recall_f1(tp, fp, fn)
    naive_vlm = n
    saved = naive_vlm - vlm_calls
    reduction_pct = 100.0 * saved / naive_vlm if naive_vlm > 0 else 0.0

    metrics = {
        "n_pairs": n,
        "split": split,
        "gate_counts": gate_counts,
        "gate_candidate_pct": round(100.0 * gate_counts.get("candidate_change", 0) / n, 1) if n else 0,
        "gate_filtered_pct": round(
            100.0
            * (gate_counts.get("no_change", 0) + gate_counts.get("low_quality", 0))
            / n,
            1,
        )
        if n
        else 0,
        "vlm_calls": vlm_calls,
        "vlm_calls_saved": saved,
        "vlm_reduction_pct": round(reduction_pct, 1),
        "pair_level": prf,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "vlm_agreement_rate": round(vlm_agree / vlm_total, 4) if vlm_total else None,
    }

    summary_path = out_dir / "_eval_summary.md"
    summary_path.write_text(_format_summary(metrics, feature_rows), encoding="utf-8")
    (out_dir / "_eval_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if feature_rows:
        import csv

        csv_path = out_dir / "_eval_features.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=feature_rows[0].keys())
            writer.writeheader()
            writer.writerows(feature_rows)
    return metrics


def _format_summary(metrics: dict[str, Any], rows: list[dict]) -> str:
    prf = metrics["pair_level"]
    lines = [
        "# SatChangeGate Eval Summary",
        "",
        f"**Pairs evaluated:** {metrics['n_pairs']} ({metrics['split']})",
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
            f"- Candidate retention: **{metrics['gate_candidate_pct']}%**",
            f"- Filtered before VLM: **{metrics['gate_filtered_pct']}%**",
            "",
            "## Classical gate vs labels (pair level)",
            "",
            f"- Precision: **{prf['precision']:.3f}**",
            f"- Recall: **{prf['recall']:.3f}**",
            f"- F1: **{prf['f1']:.3f}**",
            f"- Mean IoU (pixel): **{metrics['mean_iou']}**",
            "",
            "## VLM call reduction",
            "",
            f"- VLM calls: **{metrics['vlm_calls']}** / {metrics['n_pairs']}",
            f"- Calls saved vs naive (VLM every pair): **{metrics['vlm_calls_saved']}**",
            f"- Reduction: **{metrics['vlm_reduction_pct']}%**",
            "",
            "## Thesis",
            "",
            "Preprocessing + classical gate filters pairs before expensive VLM/LLM review.",
            "",
        ]
    )
    if rows:
        lines.append("## Per-pair classical features")
        lines.append("")
        lines.append("| pair | gate | changed_% | ndvi_d | ndbi_d | ssim |")
        lines.append("|------|------|-----------|--------|--------|------|")
        for r in rows[:20]:
            lines.append(
                f"| {r.get('tile_id')} | {r.get('classical_gate')} | "
                f"{r.get('changed_area_percent')} | {r.get('ndvi_delta_mean')} | "
                f"{r.get('ndbi_delta_mean')} | {r.get('ssim')} |"
            )
    return "\n".join(lines)
