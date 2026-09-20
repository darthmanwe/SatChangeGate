"""Budget -> operating point: what a given spend buys.

A cost-control gate is chosen by its operating point, and until now the repo
shipped exactly one: whatever a grid search maximising balanced accuracy landed
on. The precision-recall curve was already being computed for the baselines
report, so the question "I have $X per run -- what recall can I get?" was one
arithmetic step away from an artifact that already existed.

The mapping is direct. Verifying a tile costs a measured amount, so a budget of
$X buys X/cost calls; the gate must therefore flag no more than that share of
tiles; that share fixes a threshold on the score; and the threshold fixes recall
and precision. Every step is measured rather than assumed -- the per-call price
comes from token usage the API returned, and the curve comes from held-out
tiles.

Corrected 2026-09-19 -- ties could exceed the budget
----------------------------------------------------

``threshold_for_call_budget`` returned the k-th highest score and callers apply
it as ``score >= threshold``, so whenever the k-th and (k+1)-th tiles shared a
score the whole tie group was admitted and the budget was exceeded.
``_metrics_at`` counted the overshoot but nothing corrected it.

Ties are not hypothetical here: gate confidence is rounded to four decimals, and
on the held-out split 163 of 621 tiles share a value with another tile, the
largest tie group holding 87. Sweeping 50 budgets from $0.05 to $2.50 found
three that overshot -- $1.70 bought 362 calls and flagged 363, $1.80 bought 383
and flagged 384, $2.25 bought 479 and flagged 480. The five budgets in the
published table were unaffected, and their rows are unchanged by this fix.

The threshold is now stepped up past any tie group that straddles the budget
line, so a reapplied threshold selects exactly what the curve reported. The
price is that a straddling tie leaves budget unspent, which is now reported as
``unused_calls`` rather than hidden. A zero budget also used to serialise its
threshold as ``1.0`` -- which readmits any tile scoring exactly 1.0 -- and is
now recorded as ``null``, meaning admit nothing.

This curve is an analytic artifact and not a spending authority: it prices calls
at a historical mean, which is a projection rather than a reservation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.metrics import wilson_interval

# Budgets to tabulate when the caller does not name one. Chosen to bracket the
# measured cost of verifying a full held-out split.
DEFAULT_BUDGETS_USD = (0.25, 0.50, 1.00, 2.00, 5.00)


@dataclass
class OperatingPoint:
    budget_usd: float
    affordable_calls: int
    # None means "admit nothing": no threshold selects this many tiles without
    # exceeding the budget. Serialising that as 1.0 readmitted any tile scoring
    # exactly 1.0, which is the opposite of what it meant.
    threshold: float | None
    n_flagged: int
    unused_calls: int
    recall: float
    precision: float
    spend_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_usd": round(self.budget_usd, 4),
            "affordable_calls": self.affordable_calls,
            "threshold": None if self.threshold is None else round(self.threshold, 4),
            "n_flagged": self.n_flagged,
            "unused_calls": self.unused_calls,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "spend_usd": round(self.spend_usd, 4),
        }


def threshold_for_call_budget(scores: np.ndarray, budget_calls: int) -> float:
    """Lowest threshold that flags no more than ``budget_calls`` tiles.

    Applied by callers as ``score >= threshold``. Ties are the whole difficulty:
    several tiles can share a score, and returning the k-th highest score admits
    the entire tie group it belongs to, which overshoots the budget. This used to
    happen, and ``_metrics_at`` reported the overshoot without preventing it.

    A tie group straddling the budget line is therefore excluded wholesale, by
    stepping the threshold up to the next distinct score. That leaves budget
    unspent -- reported as ``unused_calls`` -- and is the conservative direction.
    It also keeps the exported threshold a self-contained policy: reapplying it
    reproduces exactly the selection the curve reported, with no tie-break rule
    to carry alongside it. ``inf`` means admit nothing.
    """
    arr = np.asarray(scores, dtype=float)
    if budget_calls <= 0:
        return float("inf")
    if len(arr) == 0:
        return 0.0
    if budget_calls >= len(arr):
        return float(arr.min())
    ordered = np.sort(arr)[::-1]
    cut = float(ordered[budget_calls - 1])
    if float(ordered[budget_calls]) != cut:
        return cut
    # The budget line falls inside a tie group. Nothing below the group can be
    # admitted without taking all of it, so take only what sits strictly above.
    above = ordered[ordered > cut]
    return float(above[-1]) if len(above) else float("inf")


def _metrics_at(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[int, float, float]:
    flagged = scores >= threshold
    n_flagged = int(flagged.sum())
    n_pos = int((labels == 1).sum())
    tp = int((flagged & (labels == 1)).sum())
    recall = tp / n_pos if n_pos else 0.0
    precision = tp / n_flagged if n_flagged else 0.0
    return n_flagged, recall, precision


def curve_for_budgets(
    scores: np.ndarray,
    labels: np.ndarray,
    cost_per_call_usd: float,
    budgets_usd: tuple[float, ...] = DEFAULT_BUDGETS_USD,
) -> list[OperatingPoint]:
    """One operating point per budget."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    points: list[OperatingPoint] = []
    for budget in budgets_usd:
        calls = int(budget // cost_per_call_usd) if cost_per_call_usd > 0 else len(scores)
        calls = min(calls, len(scores))
        threshold = threshold_for_call_budget(scores, calls)
        n_flagged, recall, precision = _metrics_at(scores, labels, threshold)
        points.append(
            OperatingPoint(
                budget_usd=budget,
                affordable_calls=calls,
                threshold=float(threshold) if np.isfinite(threshold) else None,
                n_flagged=n_flagged,
                unused_calls=calls - n_flagged,
                recall=recall,
                precision=precision,
                spend_usd=n_flagged * cost_per_call_usd,
            )
        )
    return points


def run_operating_points(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    split: str = "test",
    cost_per_call_usd: float | None = None,
    budgets_usd: tuple[float, ...] = DEFAULT_BUDGETS_USD,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Tabulate what each budget buys on a held-out split.

    ``cost_per_call_usd`` defaults to the batch-rate price of a verification at
    the configured model, using the measured mean token counts from the last
    recorded run when one is available. It is an input, not a claim: pass the
    price your own deployment observes.
    """
    from satchangegate.config import get_settings
    from satchangegate.data.tiles import build_tile_index
    from satchangegate.evaluate import compute_tile_features, score_rows

    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if cost_per_call_usd is None:
        cost_per_call_usd = _measured_cost_per_call(out_dir, split)

    tiles = [t for t in build_tile_index(oscd_root) if split == "all" or t.split == split]
    rows = compute_tile_features(oscd_root, tiles, settings)
    _, scored = score_rows(rows, settings)
    scored = [r for r in scored if r["gate"] != "low_quality"]

    scores = np.array([r["gate_confidence"] for r in scored], dtype=float)
    labels = np.array([int(r["label"]) for r in scored], dtype=int)
    points = curve_for_budgets(scores, labels, cost_per_call_usd, budgets_usd)

    n_pos = int((labels == 1).sum())
    summary = {
        "split": split,
        "n_scored": len(scored),
        "n_positive": n_pos,
        "cost_per_call_usd": round(cost_per_call_usd, 6),
        "cost_source": (
            "measured from the recorded run ledger"
            if (out_dir / f"_e2e_{split}.json").is_file()
            else "published rate card, assumed token counts"
        ),
        "review_everything_usd": round(len(scored) * cost_per_call_usd, 4),
        "selection_rule": (
            "Flag every tile with score >= threshold. A tie group straddling the "
            "budget line is excluded entirely rather than admitted, so the count "
            "never exceeds affordable_calls; the shortfall is unused_calls. A null "
            "threshold means admit nothing."
        ),
        "not_a_spending_authority": (
            "Calls are priced at a historical mean, which is a projection. Use the "
            "reservation ledger to authorise spend, not this curve."
        ),
        "operating_points": [p.to_dict() for p in points],
    }
    (out_dir / "_operating_points.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_operating_points.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _measured_cost_per_call(out_dir: Path, split: str) -> float:
    """Per-call price from the last run, falling back to the published rate."""
    path = out_dir / f"_e2e_{split}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mean = float(data["funnel_cost"]["cost_usd"]["mean_per_vlm_call"])
            if mean > 0:
                return mean
        except (KeyError, ValueError, TypeError):
            pass
    # No run on disk: price a typical package (three 512 px images plus metadata
    # in, a short structured verdict out) at the batch rate.
    from satchangegate.vlm.client import DEFAULT_VLM_MODEL, PRICING_USD_PER_MTOK

    rate = PRICING_USD_PER_MTOK[DEFAULT_VLM_MODEL]
    return (2900 * rate.batch_input_usd + 280 * rate.batch_output_usd) / 1_000_000


def _render(s: dict[str, Any]) -> str:
    lines = [
        f"# What a budget buys — `{s['split']}` split",
        "",
        f"{s['n_scored']} assessable tiles ({s['n_positive']} contain change). "
        f"Verification costs ${s['cost_per_call_usd']:.6f} per tile "
        f"({s['cost_source']}), so reviewing every tile would cost "
        f"${s['review_everything_usd']}.",
        "",
        "| Budget | Calls it buys | Threshold | Flagged | Unused | Recall | Precision | Spend |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in s["operating_points"]:
        thr = "none" if p["threshold"] is None else f"{p['threshold']:.3f}"
        lines.append(
            f"| ${p['budget_usd']:.2f} | {p['affordable_calls']} | {thr} | "
            f"{p['n_flagged']} | {p['unused_calls']} | {p['recall']:.3f} | "
            f"{p['precision']:.3f} | ${p['spend_usd']:.4f} |"
        )
    lines += [
        "",
        s["selection_rule"],
        "",
        "Read this as the demand curve for review capacity: each row is the best "
        "recall that budget can reach, and the threshold that reaches it. Precision "
        "rises as the budget falls, because a tighter threshold keeps only the "
        "gate's most confident calls -- which is the trade a spend cap actually "
        "makes, stated rather than implied.",
        "",
        "The confidence intervals that belong on these numbers are in "
        f"`_eval_{s['split']}.json`; they are omitted here because the table's "
        "purpose is choosing an operating point, not publishing one.",
    ]
    return "\n".join(lines) + "\n"


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Re-exported for callers tabulating their own operating points."""
    return wilson_interval(successes, total)
