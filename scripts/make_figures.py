"""Regenerate the figures embedded in the README.

Writes to ``docs/figures/``. The imagery panels are rendered from the OSCD
dataset, which is open-access and credited in the README; they are small
illustrative renderings, not a redistribution of the dataset.

Run: python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from satchangegate.config import get_settings  # noqa: E402
from satchangegate.data.oscd import default_oscd_root, discover_pairs, load_label_mask  # noqa: E402
from satchangegate.evaluate import build_scene_cache  # noqa: E402
from satchangegate.features.classical import compute_change_mask, compute_heatmap  # noqa: E402
from satchangegate.preprocess.align import rgb_to_uint8  # noqa: E402

FIGURES = REPO / "docs" / "figures"

INK = "#1f2933"
MUTED = "#7b8794"
FILTERED = "#b6c2cf"
CANDIDATE = "#c1440e"
ACCENT = "#2f855a"


def funnel_counts(reports_dir: Path, split: str = "test") -> dict:
    """The measured funnel counts, read from the evaluation the repo published.

    These used to be string literals in this file. Nothing cross-checked them
    against `_eval_test.json`, so refitting the thresholds would silently leave
    the README's diagram disagreeing with the README's table.
    """
    path = reports_dir / f"_eval_{split}.json"
    if not path.is_file():
        raise SystemExit(f"No {path}. Run: satchangegate eval --split {split}")
    data = json.loads(path.read_text(encoding="utf-8"))
    n_tiles = int(data["n_tiles"])
    n_low = int(data["n_low_quality"])
    n_candidate = int(data["n_candidate_change"])
    return {
        "n_tiles": n_tiles,
        "n_low_quality": n_low,
        "n_assessable": int(data.get("n_assessable", n_tiles - n_low)),
        "n_no_change": int(data.get("n_assessable", n_tiles - n_low)) - n_candidate,
        "n_candidate": n_candidate,
        "candidate_pct": float(data["candidate_rate_pct"]),
        "tier0_pct": float(data.get("tier0_refused_pct", 100.0 * n_low / max(1, n_tiles))),
        "gate_pct": float(data["gate_filtered_pct"]),
        "total_pct": float(
            data.get("total_reduction_pct", 100.0 * (n_tiles - n_candidate) / max(1, n_tiles))
        ),
        "n_cities": len(data.get("cities", [])),
        "n_cities_scored": len(data.get("cities_scored", data.get("cities", []))),
    }


def funnel_diagram(path: Path, counts: dict) -> None:
    """The three-tier funnel, annotated with the measured filter rates."""
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    stages = [
        (f"Every pair\n{counts['n_tiles']} tiles", 0.25, 1.75, "#e4e7eb", INK),
        ("Tier 0 — quality\ncloud · snow · shadow\nco-registration", 2.35, 2.15, "#dbeafe", INK),
        ("Tier 1 — classical gate\nindices · SSIM\npHash · CVA", 5.05, 2.35, "#fde8d7", INK),
        ("Tier 2 — VLM verify\nonly candidates", 7.95, 1.85, "#d7f0e2", INK),
    ]
    for label, x, w, face, text in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.35),
                w,
                1.15,
                boxstyle="round,pad=0.06",
                facecolor=face,
                edgecolor=MUTED,
                linewidth=1.0,
            )
        )
        ax.text(x + w / 2, 2.93, label, ha="center", va="center", fontsize=9, color=text)

    for x0, x1 in [(2.00, 2.35), (4.50, 5.05), (7.40, 7.95)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 2.93),
                (x1, 2.93),
                arrowstyle="-|>",
                mutation_scale=12,
                color=MUTED,
                linewidth=1.1,
            )
        )

    # Drop-outs, with the numbers actually measured on the held-out split.
    drops = [
        (3.65, f"{counts['n_low_quality']} tiles\nfailed Tier 0", FILTERED),
        (6.2, f"{counts['n_no_change']} tiles\nno_change", FILTERED),
    ]
    for x, label, colour in drops:
        ax.add_patch(
            FancyArrowPatch(
                (x, 2.32),
                (x, 1.55),
                arrowstyle="-|>",
                mutation_scale=11,
                color=colour,
                linewidth=1.1,
            )
        )
        ax.text(x, 1.28, label, ha="center", va="top", fontsize=8, color=MUTED)

    ax.text(
        8.88,
        1.55,
        f"{counts['n_candidate']} candidates\nreach the model\n({counts['candidate_pct']:.1f}%)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=CANDIDATE,
    )
    ax.add_patch(
        FancyArrowPatch(
            (8.88, 2.32),
            (8.88, 1.78),
            arrowstyle="-|>",
            mutation_scale=11,
            color=CANDIDATE,
            linewidth=1.2,
        )
    )

    ax.text(
        5.0,
        0.42,
        f"{counts['total_pct']:.1f}% resolved without an API call — "
        f"{counts['tier0_pct']:.1f}% refused by Tier 0, "
        f"{counts['gate_pct']:.1f}% of the rest filtered by the gate — "
        f"scored on {counts['n_cities_scored']} of {counts['n_cities']} held-out cities",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    ax.text(
        5.0,
        4.15,
        "Gated review funnel",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def detection_panel(path: Path, city: str = "lasvegas") -> None:
    """Before / after / detection / ground-truth for one real scene."""
    settings = get_settings()
    pairs = {p.pair_id: p for p in discover_pairs(default_oscd_root())}
    if city not in pairs:
        raise SystemExit(f"{city} not available; run: satchangegate download-oscd")
    pair = pairs[city]
    cache = build_scene_cache(pair, settings)
    label = load_label_mask(pair)

    mask = compute_change_mask(
        cache.deltas,
        cache.cva,
        cache.valid,
        settings.gate,
        scene_threshold=cache.scene_threshold,
        water=cache.water,
    )
    heat = compute_heatmap(mask, cache.deltas)
    before, after = rgb_to_uint8(cache.disp1), rgb_to_uint8(cache.disp2)

    if label is not None and label.shape != mask.shape:
        h = min(label.shape[0], mask.shape[0])
        w = min(label.shape[1], mask.shape[1])
        label, mask, heat = label[:h, :w], mask[:h, :w], heat[:h, :w]
        before, after = before[:h, :w], after[:h, :w]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), dpi=100)
    panels = [
        (before, None, f"Before — {pair.date_t1}"),
        (after, None, f"After — {pair.date_t2}"),
        (after, heat, "Detected change"),
        (after, label.astype(np.float32) if label is not None else None, "Ground truth"),
    ]
    for ax, (base, overlay, title) in zip(axes, panels, strict=True):
        ax.imshow(base)
        if overlay is not None:
            masked = np.ma.masked_where(overlay <= 0.02, overlay)
            ax.imshow(masked, cmap="autumn", alpha=0.75, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(MUTED)
            spine.set_linewidth(0.8)

    fig.suptitle(
        f"{city.title()} — Sentinel-2, {pair.split} split "
        "(detection is the despeckled change mask, not a segmentation model)",
        fontsize=10.5,
        color=INK,
        y=1.0,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


REPORTS = REPO / "data" / "reports"
SAMPLES = REPO / "public_reporting_sample"


def pr_curves(path: Path) -> None:
    """Re-plot the precision-recall curves from the committed baselines JSON.

    This figure used to be produced only as a side effect of running
    `satchangegate baselines`, into `data/reports/`, and then hand-copied into
    `docs/figures/`. `make figures` regenerated two of the three README figures
    and silently left this one stale.
    """
    from satchangegate.baseline import plot_pr_curves

    source = REPORTS / "_baselines.json"
    if not source.is_file():
        source = SAMPLES / "_baselines.json"
    if not source.is_file():
        raise SystemExit("No _baselines.json. Run: satchangegate baselines")
    plot_pr_curves(json.loads(source.read_text(encoding="utf-8")), path)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    counts = funnel_counts(REPORTS if (REPORTS / "_eval_test.json").is_file() else SAMPLES)
    funnel_diagram(FIGURES / "funnel.png", counts)
    print(f"wrote {FIGURES / 'funnel.png'}")

    try:
        pr_curves(FIGURES / "pr_curves.png")
        print(f"wrote {FIGURES / 'pr_curves.png'}")
    except SystemExit as exc:
        print(exc)
        return 1

    try:
        detection_panel(FIGURES / "detection_lasvegas.png")
        print(f"wrote {FIGURES / 'detection_lasvegas.png'}")
    except SystemExit as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
