"""Regenerate the figures embedded in the README.

Writes to ``docs/figures/``. The imagery panels are rendered from the OSCD
dataset, which is open-access and credited in the README; they are small
illustrative renderings, not a redistribution of the dataset.

Run: python scripts/make_figures.py
"""

from __future__ import annotations

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


def funnel_diagram(path: Path) -> None:
    """The three-tier funnel, annotated with the measured filter rates."""
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    stages = [
        ("Every pair\n621 tiles", 0.25, 1.75, "#e4e7eb", INK),
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
        (3.65, "87 tiles\nfailed Tier 0", FILTERED),
        (6.2, "314 tiles\nno_change", FILTERED),
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
        "220 candidates\nreach the model\n(35.4%)",
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
        "64.6% of tiles are resolved without an API call — measured on 10 held-out cities",
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


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    funnel_diagram(FIGURES / "funnel.png")
    print(f"wrote {FIGURES / 'funnel.png'}")
    try:
        detection_panel(FIGURES / "detection_lasvegas.png")
        print(f"wrote {FIGURES / 'detection_lasvegas.png'}")
    except SystemExit as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
