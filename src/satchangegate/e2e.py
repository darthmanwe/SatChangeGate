"""End-to-end funnel evaluation with measured cost.

This is the harness behind the project's central claim. It runs the full funnel
over a labelled sample, counts what the classical gate filtered, calls the VLM
only on survivors, and derives cost from the token usage the API actually
returned — rather than asserting a saving.

It replaces two earlier harnesses (``evaluate_optimus`` and
``e2e_random_eval``). Both were built on the OPTIMUS corpus, whose imagery is
distributed as 25 GB tar shards out of a 12 TB set, so nobody cloning the repo
could run either one or reproduce a single published figure. Both also drew
their thresholds and their reporting from the same tiles.

Results stream to JSONL as they complete and ``--resume`` skips finished work:
a crash at pair 99 previously discarded 99 results and all the spend that
produced them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satchangegate.config import Settings, get_settings
from satchangegate.data.oscd import discover_pairs
from satchangegate.data.tiles import Tile, build_tile_index
from satchangegate.evaluate import build_scene_cache, features_for_tile
from satchangegate.features.classical import decide
from satchangegate.metrics import ConfusionMatrix, FunnelCost, confusion_from_pairs


@dataclass
class E2EConfig:
    split: str = "test"
    n: int | None = None
    seed: int = 42
    skip_vlm: bool = True
    max_vlm_calls: int | None = None
    vlm_model: str | None = None


def sample_tiles(tiles: list[Tile], n: int | None, seed: int) -> list[Tile]:
    """Deterministic sample.

    ``tiles`` arrives sorted by tile_id from ``build_tile_index``, so the same
    seed yields the same sample on any machine. The previous sampler drew from
    an unsorted ``rglob``, which made ``--seed 42`` reproducible only within a
    single filesystem.
    """
    if n is None or n >= len(tiles):
        return list(tiles)
    return sorted(random.Random(seed).sample(tiles, n), key=lambda t: t.tile_id)


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["tile_id"]] = row
    return out


def run_e2e(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    config: E2EConfig | None = None,
    settings: Settings | None = None,
    resume: bool = False,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run the funnel over a labelled sample and report measured cost."""
    config = config or E2EConfig()
    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"_e2e_{config.split}.jsonl"

    all_tiles = build_tile_index(oscd_root)
    tiles = [t for t in all_tiles if config.split == "all" or t.split == config.split]
    tiles = sample_tiles(tiles, config.n, config.seed)

    done = _load_done(jsonl_path) if resume else {}
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    pairs = {p.pair_id: p for p in discover_pairs(oscd_root)}
    by_city: dict[str, list[Tile]] = {}
    for tile in tiles:
        by_city.setdefault(tile.city, []).append(tile)

    cost = FunnelCost(n_pairs=len(tiles))
    rows: list[dict[str, Any]] = list(done.values())
    vlm_calls = 0

    with open(jsonl_path, "a", encoding="utf-8") as sink:
        for city in sorted(by_city):
            pair = pairs.get(city)
            if pair is None:
                continue
            pending = [t for t in by_city[city] if t.tile_id not in done]
            if not pending:
                continue
            cache = build_scene_cache(pair, settings)
            for tile in pending:
                feats = features_for_tile(cache, tile, settings)
                decision, reason, confidence = decide(feats, settings.gate)
                row: dict[str, Any] = {
                    "tile_id": tile.tile_id,
                    "city": tile.city,
                    "split": tile.split,
                    "label": tile.label,
                    "gate": decision,
                    "gate_reason": reason,
                    "gate_confidence": round(confidence, 4),
                    "vlm_verdict": None,
                    "vlm_called": False,
                    "cost_usd": 0.0,
                    "error": None,
                }
                budget_left = config.max_vlm_calls is None or vlm_calls < config.max_vlm_calls
                if decision == "candidate_change" and not config.skip_vlm and budget_left:
                    verdict, record, err = _verify_tile(
                        cache, tile, feats, settings, api_key, config.vlm_model, out_dir
                    )
                    vlm_calls += 1
                    row["vlm_called"] = True
                    if err:
                        row["error"] = err
                        cost.n_errors += 1
                    if verdict is not None:
                        row["vlm_verdict"] = verdict.vlm_verdict
                        row["vlm_change_type"] = verdict.change_type
                        row["vlm_confidence"] = verdict.confidence
                    if record is not None:
                        cost.vlm_cost_usd += record.cost_usd
                        cost.input_tokens += record.input_tokens
                        cost.output_tokens += record.output_tokens
                        cost.per_call_costs.append(record.cost_usd)
                        row["cost_usd"] = round(record.cost_usd, 6)
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()

    return _summarise(rows, cost, config, settings, out_dir)


def _verify_tile(cache, tile, feats, settings, api_key, model, out_dir):
    """Package one tile and ask the VLM. Errors degrade, they do not raise."""
    from satchangegate.features.classical import ClassicalResult, compute_change_mask
    from satchangegate.vlm.client import verify_candidate

    ys, xs = tile.slice_yx
    deltas = {k: v[ys, xs] for k, v in cache.deltas.items()}
    mask = compute_change_mask(
        deltas,
        cache.cva[ys, xs],
        cache.valid[ys, xs],
        settings.gate,
        scene_threshold=cache.scene_threshold,
        water=cache.water[ys, xs],
    )
    package_dir = out_dir / "e2e_packages" / tile.tile_id
    try:
        classical = ClassicalResult(
            tile_id=tile.tile_id,
            aoi_id=f"oscd_{tile.city}",
            date_t1=cache.date_t1,
            date_t2=cache.date_t2,
            masks_assessed=cache.quality.masks_assessed,
            cloud_fraction_max=cache.quality.cloud_fraction_max,
            snow_fraction_max=cache.quality.snow_fraction_max,
            registration_error_px=cache.quality.registration_error_px,
            season_delta_days=cache.quality.season_delta_days,
            ssim=feats.ssim,
            phash_distance=feats.phash_distance,
            ndvi_delta_mean=feats.ndvi_delta_mean,
            ndbi_delta_mean=feats.ndbi_delta_mean,
            ndwi_delta_mean=feats.ndwi_delta_mean,
            changed_area_percent=feats.changed_area_percent,
            classical_gate="candidate_change",
        )
        _write_tile_package(package_dir, cache, tile, mask, cache.quality, classical)
        verdict, record = verify_candidate(package_dir, api_key=api_key, model=model)
        return verdict, record, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


# How much surrounding scene to include around the tile under evaluation.
# A bare 64 px crop gives the model no reference for what "normal" looks like
# in this scene, so it cannot tell a new building from an ordinary rooftop.
CONTEXT_FACTOR = 3


def _context_window(
    tile, height: int, width: int
) -> tuple[slice, slice, tuple[int, int, int, int]]:
    """Expanded window around a tile, plus the tile's box within that window."""
    th, tw = tile.y1 - tile.y0, tile.x1 - tile.x0
    pad_y, pad_x = (th * (CONTEXT_FACTOR - 1)) // 2, (tw * (CONTEXT_FACTOR - 1)) // 2
    y0, y1 = max(0, tile.y0 - pad_y), min(height, tile.y1 + pad_y)
    x0, x1 = max(0, tile.x0 - pad_x), min(width, tile.x1 + pad_x)
    box = (tile.y0 - y0, tile.x0 - x0, tile.y1 - y0, tile.x1 - x0)
    return slice(y0, y1), slice(x0, x1), box


def _draw_box(img_u8, box: tuple[int, int, int, int]):
    """Outline the region under evaluation so the model knows where to look."""
    import cv2

    out = img_u8.copy()
    y0, x0, y1, x1 = box
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 1)
    return out


def _write_tile_package(package_dir, cache, tile, mask, quality, classical):
    import numpy as np

    from satchangegate.features.classical import compute_heatmap
    from satchangegate.preprocess.align import rgb_to_uint8, stretch_for_display
    from satchangegate.vlm.package import _save_overlay, _save_rgb, redacted_metadata

    h, w = cache.valid.shape
    cys, cxs, box = _context_window(tile, h, w)

    # The heatmap is computed on the tile only (that is what was scored), then
    # placed inside the wider context frame so the marked box and the highlight
    # correspond.
    ys, xs = tile.slice_yx
    tile_heat = compute_heatmap(mask, {k: v[ys, xs] for k, v in cache.deltas.items()})
    heat = np.zeros(cache.rgb1[cys, cxs].shape[:2], dtype=np.float32)
    heat[box[0] : box[2], box[1] : box[3]] = tile_heat

    d1, d2 = stretch_for_display(cache.rgb1[cys, cxs], cache.rgb2[cys, cxs])
    package_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(_draw_box(rgb_to_uint8(d1), box), package_dir / "before_rgb.png")
    _save_rgb(_draw_box(rgb_to_uint8(d2), box), package_dir / "after_rgb.png")
    _save_overlay(_draw_box(rgb_to_uint8(d2), box), heat, package_dir / "change_overlay.png")

    meta = redacted_metadata(quality, classical)
    meta["region_of_interest"] = (
        "Judge only the area inside the cyan box; the surrounding scene is "
        "context for what is normal here."
    )
    (package_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _summarise(
    rows: list[dict[str, Any]],
    cost: FunnelCost,
    config: E2EConfig,
    settings: Settings,
    out_dir: Path,
) -> dict[str, Any]:
    scored = [r for r in rows if r["gate"] != "low_quality"]
    cm: ConfusionMatrix = confusion_from_pairs(
        [int(r["label"]) for r in scored],
        [1 if r["gate"] == "candidate_change" else 0 for r in scored],
    )
    cost.n_pairs = len(rows)
    cost.n_vlm_calls = sum(1 for r in rows if r["vlm_called"])
    cost.n_gate_filtered = sum(1 for r in rows if r["gate"] != "candidate_change")
    cost.n_candidates = sum(1 for r in rows if r["gate"] == "candidate_change")

    verdicts: dict[str, int] = {}
    for r in rows:
        if r.get("vlm_verdict"):
            verdicts[r["vlm_verdict"]] = verdicts.get(r["vlm_verdict"], 0) + 1

    summary = {
        "split": config.split,
        "seed": config.seed,
        "n": len(rows),
        "gate_metrics": cm.to_dict(),
        "vlm_verdicts": verdicts,
        "funnel_cost": cost.to_dict(),
        "thresholds": settings.gate.model_dump(),
    }
    (out_dir / f"_e2e_{config.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / f"_e2e_{config.split}.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(s: dict[str, Any]) -> str:
    c, m = s["funnel_cost"], s["gate_metrics"]
    lines = [
        f"# End-to-end funnel — `{s['split']}` split",
        "",
        f"{s['n']} tiles, seed {s['seed']}.",
        "",
        "## Funnel",
        "",
        "| Stage | Count | Share |",
        "|---|---|---|",
        f"| Input | {c['n_pairs']} | 100% |",
        f"| Filtered by classical gate | {c['n_gate_filtered']} | {c['gate_filtered_pct']}% |",
        f"| Sent to VLM | {c['n_vlm_calls']} | {c['vlm_call_rate_pct']}% |",
        "",
        "## Cost (measured from API token usage)",
        "",
        f"- Actually spent: **${c['cost_usd']['total']}** "
        f"({c['input_tokens']} in / {c['output_tokens']} out tokens, "
        f"{c['n_vlm_calls']} calls at ${c['cost_usd']['mean_per_vlm_call']} each)",
        f"- Projected for all {c['n_candidates']} gate candidates: "
        f"${c['projected_cost_at_gate_rate_usd']}",
        f"- Review-everything counterfactual: ${c['counterfactual_review_everything_usd']}",
        f"- Saving attributable to the gate: "
        f"${c['savings_vs_review_everything_usd']} ({c['savings_pct']}%)",
        "",
        (
            "> This run was budget-capped, so actual spend is lower than the "
            "projection. The saving above is attributed to the gate only."
            if c["vlm_budget_capped"]
            else ""
        ),
        "",
        "## Gate quality",
        "",
        f"n={m['n']} ({m['n_positive']} change / {m['n_negative']} no-change). "
        f"Recall {m['recall']:.3f}, "
        + (
            f"precision {m['precision']:.3f}, specificity {m['specificity']:.3f}, F1 {m['f1']:.3f}."
            if m["has_negative_class"]
            else "precision undefined (no negatives)."
        ),
    ]
    if s["vlm_verdicts"]:
        lines += ["", "## VLM verdicts", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(s["vlm_verdicts"].items())]
    return "\n".join(lines) + "\n"
