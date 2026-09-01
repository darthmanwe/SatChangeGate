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

**The funnel runs in two passes**, which is a change. It used to gate and verify
in one interleaved loop that stopped the moment ``--max-vlm-calls`` was reached.
Because that loop walked cities in alphabetical order, a run capped at 100 calls
verified brasilia, chongqing and dubai exhaustively, 7 of lasvegas's 70
candidates, and none of the remaining five cities — while the report described
itself as a seeded sample of 100. It was a truncation, and the sub-sample it
produced was measurably easier than the split it claimed to represent (gate
precision 0.870 against 0.804 overall). Gating every tile first, then choosing
which candidates to verify, is what makes ``--sample stratified`` possible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satchangegate.config import Settings, get_settings
from satchangegate.data.oscd import discover_pairs
from satchangegate.data.tiles import Tile, build_tile_index
from satchangegate.evaluate import build_scene_cache, mask_and_features_for_tile
from satchangegate.features.classical import decide
from satchangegate.metrics import ConfusionMatrix, FunnelCost, confusion_from_pairs

# How candidates are chosen when a budget cap cannot cover all of them.
#   stratified — proportional per city, so the verified subset has the same city
#                mix as the candidate pool it is drawn from.
#   sequential — alphabetical, verifying each city exhaustively until the budget
#                runs out. This is the historical behaviour and is kept only so
#                the earlier run can be reproduced; it is not representative.
SAMPLE_STRATEGIES = ("stratified", "sequential")


@dataclass
class E2EConfig:
    split: str = "test"
    n: int | None = None
    seed: int = 42
    skip_vlm: bool = True
    max_vlm_calls: int | None = None
    vlm_model: str | None = None
    sample: str = "stratified"
    batch: bool = False


def sample_tiles(tiles: list[Tile], n: int | None, seed: int) -> list[Tile]:
    """Deterministic sample of input tiles.

    ``tiles`` arrives sorted by tile_id from ``build_tile_index``, so the same
    seed yields the same sample on any machine. The previous sampler drew from
    an unsorted ``rglob``, which made ``--seed 42`` reproducible only within a
    single filesystem.

    Note this samples the *input*, which is a different question from which
    candidates get verified — see ``select_candidates``.
    """
    if n is None or n >= len(tiles):
        return list(tiles)
    return sorted(random.Random(seed).sample(tiles, n), key=lambda t: t.tile_id)


def select_candidates(
    candidate_ids: list[tuple[str, str]],
    budget: int | None,
    *,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[str]:
    """Choose which gate candidates to verify, given a call budget.

    ``candidate_ids`` is [(city, tile_id), ...]. Returns tile_ids, sorted.

    Under ``stratified`` each city gets a share of the budget proportional to how
    many candidates it contributed, allocated by largest remainder so the counts
    sum exactly to the budget. Every city with at least one candidate gets at
    least one call, because a city absent from the sample cannot be said to have
    been measured at all.
    """
    if strategy not in SAMPLE_STRATEGIES:
        raise ValueError(
            f"unknown sample strategy {strategy!r}; expected one of {SAMPLE_STRATEGIES}"
        )
    ids = sorted(candidate_ids, key=lambda t: (t[0], t[1]))
    if budget is None or budget >= len(ids):
        return [tile_id for _, tile_id in ids]
    if budget <= 0:
        return []
    if strategy == "sequential":
        return [tile_id for _, tile_id in ids[:budget]]

    by_city: dict[str, list[str]] = {}
    for city, tile_id in ids:
        by_city.setdefault(city, []).append(tile_id)

    total = len(ids)
    cities = sorted(by_city)
    # Largest-remainder apportionment, with a floor of one call per city so no
    # city silently drops out of the measured set.
    exact = {c: budget * len(by_city[c]) / total for c in cities}
    alloc = {c: min(len(by_city[c]), max(1, int(exact[c]))) for c in cities}

    # Reconcile to exactly `budget`: the floor can overshoot, the floor-of-one can
    # undershoot. Give to the largest remainders, take from the largest allocations.
    def remainder(c: str) -> float:
        return exact[c] - alloc[c]

    while sum(alloc.values()) > budget:
        victim = min(
            (c for c in cities if alloc[c] > 1),
            key=lambda c: (remainder(c), c),
            default=None,
        )
        if victim is None:
            break
        alloc[victim] -= 1
    while sum(alloc.values()) < budget:
        lucky = max(
            (c for c in cities if alloc[c] < len(by_city[c])),
            key=lambda c: (remainder(c), c),
            default=None,
        )
        if lucky is None:
            break
        alloc[lucky] += 1

    rng = random.Random(seed)
    chosen: list[str] = []
    for city in cities:
        pool = sorted(by_city[city])
        take = min(alloc[city], len(pool))
        chosen.extend(rng.sample(pool, take) if take < len(pool) else pool)
    return sorted(chosen)


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


@dataclass
class _GatePass:
    """Everything the first pass produces."""

    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidates: list[tuple[str, str]] = field(default_factory=list)
    packages: dict[str, Path] = field(default_factory=dict)
    candidates_by_city: dict[str, int] = field(default_factory=dict)


def _gate_pass(
    tiles: list[Tile],
    pairs: dict[str, Any],
    settings: Settings,
    done: dict[str, dict],
    out_dir: Path,
    *,
    package_candidates: bool,
) -> _GatePass:
    """Gate every pending tile, packaging candidates for later verification.

    Packaging happens here because it needs the scene cache, which is the
    expensive part of the run; deferring it would mean preprocessing each city
    twice.
    """
    result = _GatePass()
    by_city: dict[str, list[Tile]] = {}
    for tile in tiles:
        if tile.tile_id in done:
            continue
        by_city.setdefault(tile.city, []).append(tile)

    for city in sorted(by_city):
        pair = pairs.get(city)
        if pair is None:
            continue
        cache = build_scene_cache(pair, settings)
        for tile in by_city[city]:
            mask, feats = mask_and_features_for_tile(cache, tile, settings)
            decision, reason, confidence = decide(feats, settings.gate)
            result.rows[tile.tile_id] = {
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
            if decision != "candidate_change":
                continue
            result.candidates.append((tile.city, tile.tile_id))
            result.candidates_by_city[tile.city] = result.candidates_by_city.get(tile.city, 0) + 1
            if package_candidates:
                package_dir = out_dir / "e2e_packages" / tile.tile_id
                try:
                    _write_tile_package(package_dir, cache, tile, mask, cache.quality, feats)
                    result.packages[tile.tile_id] = package_dir
                except Exception as exc:  # packaging must not abort the gate pass
                    result.rows[tile.tile_id]["error"] = f"{type(exc).__name__}: {exc}"
    return result


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

    cost = FunnelCost(n_pairs=len(tiles))
    # Restore spend and call count from completed work. Without this a resumed
    # run reports "$0.00 spent, 0% saved" for calls that were genuinely paid
    # for, and --max-vlm-calls silently re-arms: three resumes at a cap of 20
    # would buy 60 calls.
    for prior in done.values():
        if not prior.get("vlm_called"):
            continue
        prior_cost = float(prior.get("cost_usd") or 0.0)
        cost.vlm_cost_usd += prior_cost
        cost.per_call_costs.append(prior_cost)
        cost.per_call_sync_costs.append(float(prior.get("cost_usd_synchronous") or prior_cost))
        if prior.get("batch"):
            cost.n_batch_calls += 1
        if prior.get("error"):
            cost.n_errors += 1
    vlm_calls = sum(1 for r in done.values() if r.get("vlm_called"))

    want_vlm = not config.skip_vlm
    gated = _gate_pass(tiles, pairs, settings, done, out_dir, package_candidates=want_vlm)

    # Budget remaining after any resumed calls. Computed before submission, not
    # checked per call afterwards: a concurrent or batched submission that checks
    # the cap while spending would overshoot it.
    remaining = None if config.max_vlm_calls is None else max(0, config.max_vlm_calls - vlm_calls)
    selected: list[str] = []
    if want_vlm and gated.candidates:
        eligible = [(c, t) for c, t in gated.candidates if t in gated.packages]
        selected = select_candidates(eligible, remaining, strategy=config.sample, seed=config.seed)

    selected_set = set(selected)
    with open(jsonl_path, "a", encoding="utf-8") as sink:
        # Everything not going to the VLM is final now.
        for tile_id in sorted(gated.rows):
            if tile_id in selected_set:
                continue
            sink.write(json.dumps(gated.rows[tile_id]) + "\n")
        sink.flush()

        if selected:
            verify = _verify_batch if config.batch else _verify_sequential
            for tile_id, (verdict, record, err) in verify(
                selected, gated.packages, api_key, config, out_dir
            ).items():
                row = gated.rows[tile_id]
                row["vlm_called"] = True
                vlm_calls += 1
                if err:
                    row["error"] = err
                    cost.n_errors += 1
                if verdict is not None:
                    row["vlm_verdict"] = verdict.vlm_verdict
                    row["vlm_change_type"] = verdict.change_type
                    row["vlm_confidence"] = verdict.confidence
                if record is not None:
                    if not record.priced:
                        cost.n_unpriced_calls += 1
                    sync_record = record.model_copy(update={"batch": False})
                    cost.vlm_cost_usd += record.cost_usd
                    cost.input_tokens += record.input_tokens
                    cost.output_tokens += record.output_tokens
                    cost.per_call_costs.append(record.cost_usd)
                    cost.per_call_sync_costs.append(sync_record.cost_usd)
                    if record.batch:
                        cost.n_batch_calls += 1
                    row["batch"] = record.batch
                    row["cost_usd"] = round(record.cost_usd, 6)
                    row["cost_usd_synchronous"] = round(sync_record.cost_usd, 6)
                sink.write(json.dumps(row) + "\n")
                sink.flush()

    rows = list(_load_done(jsonl_path).values())
    return _summarise(rows, cost, config, settings, out_dir, gated.candidates_by_city)


def _verify_sequential(
    selected: list[str],
    packages: dict[str, Path],
    api_key: str | None,
    config: E2EConfig,
    out_dir: Path,  # noqa: ARG001 - kept so both verifiers share one call shape
) -> dict[str, tuple[Any, Any, str | None]]:
    """One blocking call per candidate."""
    from satchangegate.vlm.client import verify_candidate

    out: dict[str, tuple[Any, Any, str | None]] = {}
    for tile_id in selected:
        try:
            verdict, record = verify_candidate(
                packages[tile_id], api_key=api_key, model=config.vlm_model
            )
            out[tile_id] = (verdict, record, None)
        except Exception as exc:
            out[tile_id] = (None, None, f"{type(exc).__name__}: {exc}")
    return out


def _live_batch(manifest_path: Path, selected: list[str], model: str) -> str | None:
    """Batch id from a manifest that covers exactly this work, else None."""
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("model") != model:
        return None
    if set(manifest.get("tile_ids") or []) != set(selected):
        # A different selection is different work; collecting it would attach
        # verdicts to tiles they were not computed for.
        return None
    batch_id = manifest.get("batch_id")
    return str(batch_id) if batch_id else None


def _verify_batch(
    selected: list[str],
    packages: dict[str, Path],
    api_key: str | None,
    config: E2EConfig,
    out_dir: Path,
) -> dict[str, tuple[Any, Any, str | None]]:
    """Submit every candidate as one Batch API job, at half rate.

    The batch id and its custom_id map are written to disk immediately after
    submission. A crash between submitting and collecting has already spent the
    money; without the manifest the results would be unreachable and the spend
    simply lost.
    """
    import os

    import anthropic

    from satchangegate.vlm.client import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TIMEOUT_S,
        DEFAULT_VLM_MODEL,
        build_batch_request,
        collect_batch,
        resolve_model,
        submit_batch,
    )

    model = resolve_model(config.vlm_model, "ANTHROPIC_VLM_MODEL", DEFAULT_VLM_MODEL)
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(
        api_key=key, timeout=DEFAULT_TIMEOUT_S, max_retries=DEFAULT_MAX_RETRIES
    )

    manifest_path = out_dir / f"_e2e_{config.split}_batch.json"

    # A batch already in flight for exactly this set of tiles has been paid for
    # the moment it was submitted. Reattach to it rather than submitting a second
    # one: the manifest exists precisely so that a crash between submitting and
    # collecting costs nothing but time.
    existing = _live_batch(manifest_path, selected, model)
    if existing is not None:
        return collect_batch(existing, client=client, model=model)

    requests: list[dict] = []
    failed: dict[str, tuple[Any, Any, str | None]] = {}
    for tile_id in selected:
        try:
            requests.append(
                build_batch_request(
                    tile_id, packages[tile_id], model=model, max_tokens=DEFAULT_MAX_TOKENS
                )
            )
        except Exception as exc:
            failed[tile_id] = (None, None, f"{type(exc).__name__}: {exc}")
    if not requests:
        return failed

    batch_id = submit_batch(requests, client=client)
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "model": model,
                "split": config.split,
                "tile_ids": [r["custom_id"] for r in requests],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results = collect_batch(batch_id, client=client, model=model)
    results.update(failed)
    return results


# How much surrounding scene to include around the tile under evaluation.
# A bare 64 px crop gives the model no reference for what "normal" looks like
# in this scene, so it cannot tell a new building from an ordinary rooftop.
CONTEXT_FACTOR = 3

# Cyan, in R,G,B order. The metadata tells the model to judge inside "the cyan
# box", so this constant and that sentence must not drift apart.
ROI_BOX_RGB = (0, 255, 255)


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
    # Array is R,G,B (BGR conversion happens at write time), so this is cyan.
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), ROI_BOX_RGB, 1)
    return out


def _write_tile_package(package_dir, cache, tile, mask, quality, feats):
    import numpy as np

    from satchangegate.features.classical import ClassicalResult, compute_heatmap
    from satchangegate.preprocess.align import rgb_to_uint8, stretch_for_display
    from satchangegate.vlm.package import _save_overlay, _save_rgb, redacted_metadata

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
    candidates_by_city: dict[str, int] | None = None,
) -> dict[str, Any]:
    scored = [r for r in rows if r["gate"] != "low_quality"]
    cm: ConfusionMatrix = confusion_from_pairs(
        [int(r["label"]) for r in scored],
        [1 if r["gate"] == "candidate_change" else 0 for r in scored],
    )
    cost.n_pairs = len(rows)
    cost.n_vlm_calls = sum(1 for r in rows if r["vlm_called"])
    cost.n_tier0_refused = sum(1 for r in rows if r["gate"] == "low_quality")
    cost.n_gate_filtered = sum(1 for r in rows if r["gate"] == "no_change")
    cost.n_candidates = sum(1 for r in rows if r["gate"] == "candidate_change")

    verdicts: dict[str, int] = {}
    for r in rows:
        if r.get("vlm_verdict"):
            verdicts[r["vlm_verdict"]] = verdicts.get(r["vlm_verdict"], 0) + 1

    # Per-city verification coverage. Reported because the previous harness
    # published "100 sampled" for a run that had verified three cities of ten,
    # and nothing in the output would have revealed it.
    coverage: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["gate"] != "candidate_change":
            continue
        bucket = coverage.setdefault(r["city"], {"candidates": 0, "verified": 0})
        bucket["candidates"] += 1
        if r["vlm_called"]:
            bucket["verified"] += 1
    if candidates_by_city:
        for city, n in candidates_by_city.items():
            coverage.setdefault(city, {"candidates": n, "verified": 0})

    summary = {
        "split": config.split,
        "seed": config.seed,
        "sample_strategy": config.sample,
        "batch": config.batch,
        "n": len(rows),
        "cities_with_candidates": len(coverage),
        "cities_verified": sum(1 for v in coverage.values() if v["verified"] > 0),
        "vlm_coverage_by_city": dict(sorted(coverage.items())),
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
        f"{s['n']} tiles, seed {s['seed']}, sampling `{s['sample_strategy']}`"
        + (", Batch API" if s.get("batch") else "")
        + ".",
        "",
        "## Funnel",
        "",
        "| Stage | Count | Share |",
        "|---|---|---|",
        f"| Input | {c['n_pairs']} | 100% |",
        f"| Refused by Tier 0 (unusable imagery) | {c['n_tier0_refused']} | "
        f"{c['tier0_refused_pct']}% of input |",
        f"| Filtered by the gate (judged unchanged) | {c['n_gate_filtered']} | "
        f"{c['gate_filtered_pct']}% of assessable |",
        f"| Forwarded as candidates | {c['n_candidates']} | "
        f"{round(100.0 - c['total_reduction_pct'], 2)}% of input |",
        f"| Actually sent to the VLM | {c['n_vlm_calls']} | {c['vlm_call_rate_pct']}% of input |",
        "",
        f"Total reduction before any API call: **{c['total_reduction_pct']}%**.",
        "",
        "## Verification coverage",
        "",
        f"{s['cities_verified']} of {s['cities_with_candidates']} cities with candidates "
        "had at least one verified.",
        "",
        "| City | Candidates | Verified |",
        "|---|---|---|",
    ]
    for city, v in s.get("vlm_coverage_by_city", {}).items():
        lines.append(f"| {city} | {v['candidates']} | {v['verified']} |")
    lines += [
        "",
        "## Cost (measured from API token usage)",
        "",
        (
            f"> WARNING: {c['n_unpriced_calls']} call(s) used a model with no "
            "published rate; the costs below understate actual spend."
            if not c["cost_is_complete"]
            else ""
        ),
        f"- Actually spent: **${c['cost_usd']['total']}** "
        f"({c['input_tokens']} in / {c['output_tokens']} out tokens, "
        f"{c['n_vlm_calls']} calls at ${c['cost_usd']['mean_per_vlm_call']} each"
        + (f", {c['n_batch_calls']} batched" if c["n_batch_calls"] else "")
        + ")",
        f"- Projected for all {c['n_candidates']} gate candidates: "
        f"${c['projected_cost_at_gate_rate_usd']}",
        f"- Review-everything counterfactual: ${c['counterfactual_review_everything_usd']}",
        f"- Saving attributable to the gate: "
        f"${c['savings_vs_review_everything_usd']} ({c['savings_pct']}%)",
        "",
        f"> {c['savings_pct_note']}",
        "",
    ]
    if c["n_batch_calls"]:
        lines += [
            f"- Same candidates priced synchronously: ${c['projected_cost_synchronous_usd']}",
            f"- **Saving attributable to batching: ${c['batch_saving_usd']} "
            f"({c['batch_saving_pct']}%)** — independent of the filter rate.",
            "",
        ]
    lines += [
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
