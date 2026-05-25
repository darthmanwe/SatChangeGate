"""Random mixed-source E2E evaluation with optional VLM/LLM calls."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from satchangegate.config import load_thresholds
from satchangegate.data.optimus import (
    default_optimus_root,
    get_bitemporal_frames,
    load_eval_labels,
)
from satchangegate.data.oscd import (
    ImagePair,
    default_oscd_root,
    discover_pairs,
    load_label_mask,
)
from satchangegate.pipeline import run_from_bands, run_pair

Source = Literal["oscd", "optimus"]


@dataclass
class RandomPairSpec:
    source: Source
    pair_id: str
    group_index: int | None = None
    ground_truth: int | None = None  # 1=change, 0=no-change, None=unknown
    split: str = "random"


def _pair_has_change(label: np.ndarray | None) -> bool | None:
    if label is None:
        return None
    return bool(label.sum() > 0)


def list_optimus_tiles(
    optimus_root: Path | None = None,
    groups: tuple[int, ...] = (148, 425, 455),
) -> list[tuple[int, str]]:
    """Return (group_index, tile_id) for tiles with PNGs in extracted tar groups."""
    root = Path(optimus_root or default_optimus_root())
    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for gi in groups:
        base = root / "extracted" / str(gi)
        if not base.is_dir():
            continue
        for png in base.rglob("*.png"):
            if png.parent.name != "tci":
                continue
            key = (gi, png.stem)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def build_pair_pool(
    *,
    oscd_root: Path | None = None,
    optimus_root: Path | None = None,
    optimus_groups: tuple[int, ...] = (148, 425, 455),
) -> list[RandomPairSpec]:
    """Build runnable random-pair pool from OSCD + extracted OPTIMUS groups."""
    pool: list[RandomPairSpec] = []
    oscd_root = Path(oscd_root or default_oscd_root())
    optimus_root = Path(optimus_root or default_optimus_root())

    eval_labels: dict[str, int] = {}
    try:
        eval_labels = load_eval_labels(optimus_root)
    except FileNotFoundError:
        pass

    for pair in discover_pairs(oscd_root):
        label = load_label_mask(pair)
        gt: int | None
        has_change = _pair_has_change(label)
        if has_change is None:
            gt = None
        else:
            gt = 1 if has_change else 0
        pool.append(
            RandomPairSpec(
                source="oscd",
                pair_id=pair.pair_id,
                ground_truth=gt,
                split=pair.split,
            )
        )

    for gi, tile_id in list_optimus_tiles(optimus_root, optimus_groups):
        pool.append(
            RandomPairSpec(
                source="optimus",
                pair_id=tile_id,
                group_index=gi,
                ground_truth=eval_labels.get(tile_id),
            )
        )
    return pool


def sample_random_pairs(
    n: int,
    pool: list[RandomPairSpec] | None = None,
    *,
    seed: int = 42,
    oscd_fraction: float = 0.24,
    oscd_root: Path | None = None,
    optimus_root: Path | None = None,
    optimus_groups: tuple[int, ...] = (148, 425, 455),
) -> list[RandomPairSpec]:
    if pool is None:
        pool = build_pair_pool(
            oscd_root=oscd_root,
            optimus_root=optimus_root,
            optimus_groups=optimus_groups,
        )
    if not pool:
        raise RuntimeError("Random pair pool is empty — check OSCD/OPTIMUS data paths")
    rng = random.Random(seed)
    oscd_pool = [p for p in pool if p.source == "oscd"]
    optimus_pool = [p for p in pool if p.source == "optimus"]

    n_oscd = min(len(oscd_pool), max(0, round(n * oscd_fraction))) if oscd_pool else 0
    n_oscd = min(n_oscd, n)
    n_optimus = n - n_oscd

    sample: list[RandomPairSpec] = []
    if n_oscd:
        sample.extend(rng.sample(oscd_pool, n_oscd))
    if n_optimus and optimus_pool:
        if n_optimus <= len(optimus_pool):
            sample.extend(rng.sample(optimus_pool, n_optimus))
        else:
            sample.extend(rng.choice(optimus_pool) for _ in range(n_optimus))
    rng.shuffle(sample)
    return sample


def _run_one(
    spec: RandomPairSpec,
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    oscd_root: Path,
    optimus_root: Path,
    skip_vlm: bool,
    skip_llm: bool,
    api_key: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": spec.source,
        "pair_id": spec.pair_id,
        "group_index": spec.group_index,
        "ground_truth": spec.ground_truth,
    }
    try:
        if spec.source == "oscd":
            pairs = discover_pairs(oscd_root)
            match = [p for p in pairs if p.pair_id == spec.pair_id]
            if not match:
                raise FileNotFoundError(f"OSCD pair not found: {spec.pair_id}")
            result = run_pair(
                match[0],
                cfg=cfg,
                out_dir=out_dir,
                skip_vlm=skip_vlm,
                skip_llm=skip_llm,
                api_key=api_key,
            )
        else:
            b1, b2, t1, t2 = get_bitemporal_frames(spec.pair_id, optimus_root)
            result = run_from_bands(
                f"optimus_{spec.pair_id}",
                b1,
                b2,
                cfg=cfg,
                out_dir=out_dir,
                date_t1=t1,
                date_t2=t2,
                skip_vlm=skip_vlm,
                skip_llm=skip_llm,
                api_key=api_key,
                ground_truth_label=spec.ground_truth,
            )
        row.update(result.feature_row)
        row["classical_gate"] = result.classical_gate
        row["vlm_called"] = result.vlm_called
        row["vlm_verdict"] = (
            result.vlm_verdict.vlm_verdict if result.vlm_verdict else None
        )
        row["vlm_change_type"] = (
            result.vlm_verdict.change_type if result.vlm_verdict else None
        )
        row["vlm_confidence"] = (
            result.vlm_verdict.confidence if result.vlm_verdict else None
        )
        row["error"] = None
    except Exception as e:
        row["error"] = str(e)
        row["classical_gate"] = "error"
        row["vlm_called"] = False
    return row


def run_e2e_random_eval(
    n: int = 100,
    *,
    seed: int = 42,
    oscd_root: Path | None = None,
    optimus_root: Path | None = None,
    optimus_groups: tuple[int, ...] = (148, 425, 455),
    out_dir: Path | None = None,
    skip_vlm: bool = False,
    skip_llm: bool = True,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run n random OSCD+OPTIMUS pairs through full pipeline including VLM when gated."""
    load_thresholds()  # loads .env
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not skip_vlm and not api_key:
        raise ValueError("ANTHROPIC_API_KEY required when skip_vlm=False")

    cfg = load_thresholds()
    oscd_root = Path(oscd_root or default_oscd_root())
    optimus_root = Path(optimus_root or default_optimus_root())
    out_dir = Path(out_dir or Path("data/reports"))
    run_dir = out_dir / "e2e_random"
    run_dir.mkdir(parents=True, exist_ok=True)

    pool = build_pair_pool(
        oscd_root=oscd_root,
        optimus_root=optimus_root,
        optimus_groups=optimus_groups,
    )
    sample = sample_random_pairs(
        n,
        pool,
        seed=seed,
        oscd_fraction=min(0.24, len([p for p in pool if p.source == "oscd"]) / max(n, 1)),
        oscd_root=oscd_root,
        optimus_root=optimus_root,
        optimus_groups=optimus_groups,
    )

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(sample, start=1):
        pair_out = run_dir / f"{i:03d}_{spec.source}_{spec.pair_id}"
        row = _run_one(
            spec,
            cfg=cfg,
            out_dir=pair_out,
            oscd_root=oscd_root,
            optimus_root=optimus_root,
            skip_vlm=skip_vlm,
            skip_llm=skip_llm,
            api_key=api_key,
        )
        row["sample_index"] = i
        rows.append(row)

    metrics = _aggregate_metrics(rows, n_requested=n, pool_size=len(pool), seed=seed)
    metrics["skip_vlm"] = skip_vlm
    metrics["skip_llm"] = skip_llm
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    summary_md = out_dir / "_e2e_random_summary.md"
    summary_json = out_dir / "_e2e_random_summary.json"
    summary_md.write_text(_format_summary(metrics, rows), encoding="utf-8")
    summary_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if rows:
        csv_path = out_dir / "_e2e_random_features.csv"
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    return metrics


def _aggregate_metrics(
    rows: list[dict[str, Any]],
    *,
    n_requested: int,
    pool_size: int,
    seed: int,
) -> dict[str, Any]:
    ok = [r for r in rows if r.get("classical_gate") != "error"]
    errors = len(rows) - len(ok)
    gate_counts: dict[str, int] = {}
    for r in ok:
        g = r.get("classical_gate", "unknown")
        gate_counts[g] = gate_counts.get(g, 0) + 1

    vlm_calls = sum(1 for r in ok if r.get("vlm_called"))
    n = len(ok) or 1
    filtered = gate_counts.get("no_change", 0) + gate_counts.get("low_quality", 0)

    vlm_verdicts: dict[str, int] = {}
    for r in ok:
        if r.get("vlm_called") and r.get("vlm_verdict"):
            v = r["vlm_verdict"]
            vlm_verdicts[v] = vlm_verdicts.get(v, 0) + 1

    source_counts: dict[str, int] = {}
    for r in rows:
        s = r.get("source", "?")
        source_counts[s] = source_counts.get(s, 0) + 1

    labeled = [r for r in ok if r.get("ground_truth") is not None]
    gate_tp = gate_fp = gate_fn = gate_tn = 0
    for r in labeled:
        pred = r.get("classical_gate") == "candidate_change"
        truth = r.get("ground_truth") == 1
        if pred and truth:
            gate_tp += 1
        elif pred and not truth:
            gate_fp += 1
        elif not pred and truth:
            gate_fn += 1
        else:
            gate_tn += 1

    vlm_labeled = [r for r in labeled if r.get("vlm_called") and r.get("vlm_verdict")]
    vlm_real = sum(1 for r in vlm_labeled if r["vlm_verdict"] == "real_change")
    vlm_artifact = sum(1 for r in vlm_labeled if r["vlm_verdict"] == "likely_artifact")

    def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    return {
        "n_requested": n_requested,
        "n_completed": len(ok),
        "n_errors": errors,
        "pool_size": pool_size,
        "seed": seed,
        "source_counts": source_counts,
        "gate_counts": gate_counts,
        "gate_candidate_pct": round(100.0 * gate_counts.get("candidate_change", 0) / n, 1),
        "gate_filtered_pct": round(100.0 * filtered / n, 1),
        "vlm_calls": vlm_calls,
        "vlm_reduction_pct": round(100.0 * (n - vlm_calls) / n, 1),
        "vlm_verdict_counts": vlm_verdicts,
        "vlm_real_change_pct": round(100.0 * vlm_real / len(vlm_labeled), 1) if vlm_labeled else None,
        "vlm_artifact_pct": round(100.0 * vlm_artifact / len(vlm_labeled), 1) if vlm_labeled else None,
        "labeled_pairs": len(labeled),
        "gate_vs_label": {**_prf(gate_tp, gate_fp, gate_fn), "tn": gate_tn, "tp": gate_tp, "fp": gate_fp, "fn": gate_fn},
    }


def _format_summary(metrics: dict[str, Any], rows: list[dict]) -> str:
    lines = [
        "# E2E Random Evaluation (OSCD + OPTIMUS)",
        "",
        f"**Pairs requested:** {metrics['n_requested']} | **Completed:** {metrics['n_completed']} | "
        f"**Errors:** {metrics['n_errors']}",
        f"**Pool size:** {metrics['pool_size']} | **Seed:** {metrics['seed']}",
        f"**VLM enabled:** {not metrics['skip_vlm']} | **LLM enabled:** {not metrics['skip_llm']}",
        "",
        "## Source mix",
        "",
        "| Source | Count |",
        "|--------|-------|",
    ]
    for k, v in metrics.get("source_counts", {}).items():
        lines.append(f"| {k} | {v} |")

    lines.extend(
        [
            "",
            "## Classical gate",
            "",
            "| Gate | Count |",
            "|------|-------|",
        ]
    )
    for k, v in metrics.get("gate_counts", {}).items():
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            f"- Candidate rate: **{metrics['gate_candidate_pct']}%**",
            f"- Filtered before VLM: **{metrics['gate_filtered_pct']}%**",
            f"- VLM calls: **{metrics['vlm_calls']}**",
            f"- VLM reduction vs naive-all: **{metrics['vlm_reduction_pct']}%**",
            "",
        ]
    )

    if metrics.get("vlm_verdict_counts"):
        lines.append("## VLM verdicts (candidates only)")
        lines.append("")
        for k, v in metrics["vlm_verdict_counts"].items():
            lines.append(f"- {k}: **{v}**")
        if metrics.get("vlm_real_change_pct") is not None:
            lines.append(
                f"- Real change: **{metrics['vlm_real_change_pct']}%** | "
                f"Likely artifact: **{metrics.get('vlm_artifact_pct')}%**"
            )
        lines.append("")

    if metrics.get("labeled_pairs", 0) > 0:
        gvl = metrics["gate_vs_label"]
        lines.extend(
            [
                "## Gate vs available labels",
                "",
                f"- Labeled pairs in sample: **{metrics['labeled_pairs']}**",
                f"- Precision: **{gvl['precision']:.3f}** | Recall: **{gvl['recall']:.3f}** | F1: **{gvl['f1']:.3f}**",
                f"- Confusion: TP={gvl['tp']} FP={gvl['fp']} FN={gvl['fn']} TN={gvl['tn']}",
                "",
            ]
        )

    lines.append("## Sample rows (first 15)")
    lines.append("")
    lines.append("| # | source | id | gate | VLM | verdict | gt |")
    lines.append("|---|--------|----|------|-----|---------|-----|")
    for r in rows[:15]:
        lines.append(
            f"| {r.get('sample_index')} | {r.get('source')} | {r.get('pair_id')} | "
            f"{r.get('classical_gate')} | {r.get('vlm_called')} | {r.get('vlm_verdict')} | "
            f"{r.get('ground_truth')} |"
        )
    return "\n".join(lines)
