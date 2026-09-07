"""Distribution-free risk control for the gate's operating point.

The gate's weakest published sentence is that it "misses about half of all
change -- but it is a trade". That is honest and it is also unactionable: an
operator who needs to miss no more than 20% of change has no way to ask for it,
and the shipped operating point is a grid-search argmax of balanced accuracy
with nothing attached to it.

Conformal risk control replaces that with a claim of the form:

    at threshold lambda, the false-negative rate is at most alpha,
    with confidence 1 - delta

fitted on a calibration split and *then tested against a split it never saw*.
The method is Learn-then-Test (Angelopoulos et al.): sweep candidate thresholds,
compute a distribution-free upper confidence bound on the risk at each, and keep
the most permissive threshold whose bound clears alpha. Nothing here assumes the
score is calibrated, or that risk is monotone in lambda, or anything about the
score's distribution -- only that calibration and deployment data are
exchangeable.

**That last assumption is exactly what a geographic split breaks**, which is why
this module ships with the machinery to falsify itself rather than only to fit.
Tiles from a city the model has never seen are not exchangeable with tiles from
the calibration cities: land cover differs, acquisition seasons differ, and the
gate's own thresholds were fitted elsewhere. The expected result is that the
guarantee holds on some held-out cities and fails on others, and the per-city
breakdown in ``evaluate_guarantee`` is the finding, not a diagnostic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Candidate thresholds are swept on a fixed grid rather than at every observed
# score. A grid keeps the multiple-testing correction interpretable and stops the
# procedure from tuning lambda to a single calibration tile.
DEFAULT_LAMBDA_GRID = tuple(round(x, 3) for x in np.arange(0.0, 1.0001, 0.005))
DEFAULT_ALPHA = 0.20
DEFAULT_DELTA = 0.10


def hoeffding_bentkus_ucb(risk_hat: float, n: int, delta: float) -> float:
    """Upper confidence bound on a [0,1]-bounded risk.

    Hoeffding's inequality, inverted. Bentkus' bound is tighter for small n and
    the reference implementations take the minimum of the two; Hoeffding alone is
    used here because it is the conservative side of that pair -- an interval
    that is too wide makes the guarantee harder to claim, never easier, which is
    the correct direction for a number this repo intends to publish.
    """
    if n <= 0:
        return 1.0
    return min(1.0, risk_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def false_negative_rate(scores: np.ndarray, labels: np.ndarray, lam: float) -> float:
    """Share of true positives that a threshold of ``lam`` would miss."""
    positives = labels == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return 0.0
    missed = int(((scores < lam) & positives).sum())
    return missed / n_pos


@dataclass
class ConformalResult:
    """A risk-controlled threshold and the evidence for it."""

    alpha: float
    delta: float
    lam: float | None
    risk_hat: float
    risk_ucb: float
    n_cal: int
    n_cal_positive: int
    calibration_cities: tuple[str, ...]
    flagged_fraction: float
    controlled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fnr_alpha": self.alpha,
            "confidence_delta": self.delta,
            "lambda": self.lam,
            "calibration_fnr": round(self.risk_hat, 4),
            "calibration_fnr_upper_bound": round(self.risk_ucb, 4),
            "n_calibration": self.n_cal,
            "n_calibration_positive": self.n_cal_positive,
            "calibration_cities": list(self.calibration_cities),
            "flagged_fraction_at_lambda": round(self.flagged_fraction, 4),
            "risk_controlled": self.controlled,
            "method": "Learn-then-Test with a Hoeffding upper confidence bound",
            "assumption": (
                "Exchangeability between calibration and deployment tiles. A "
                "geographic split violates this, so the guarantee is conditional "
                "and is tested per city rather than asserted."
            ),
        }


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    cities: list[str] | tuple[str, ...],
    *,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> ConformalResult:
    """Largest threshold whose false-negative rate is provably at most ``alpha``.

    Larger lambda flags fewer tiles and therefore costs less, so the most
    permissive threshold that still clears the bound is the cheapest admissible
    operating point. Candidates are tested from the top down and the first one
    that clears is returned; because raising lambda can only increase the miss
    rate, that first success is also the best one.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())

    best: ConformalResult | None = None
    for lam in sorted(lambda_grid, reverse=True):
        risk = false_negative_rate(scores, labels, lam)
        ucb = hoeffding_bentkus_ucb(risk, n_pos, delta)
        if ucb <= alpha:
            best = ConformalResult(
                alpha=alpha,
                delta=delta,
                lam=float(lam),
                risk_hat=risk,
                risk_ucb=ucb,
                n_cal=len(labels),
                n_cal_positive=n_pos,
                calibration_cities=tuple(sorted(set(cities))),
                flagged_fraction=float((scores >= lam).mean()) if len(scores) else 0.0,
                controlled=True,
            )
            break

    if best is None:
        # No threshold on the grid clears the bound -- usually because alpha is
        # tighter than the sample size can certify. Say so; do not return the
        # least-bad threshold as though it were controlled.
        risk = false_negative_rate(scores, labels, 0.0)
        best = ConformalResult(
            alpha=alpha,
            delta=delta,
            lam=None,
            risk_hat=risk,
            risk_ucb=hoeffding_bentkus_ucb(risk, n_pos, delta),
            n_cal=len(labels),
            n_cal_positive=n_pos,
            calibration_cities=tuple(sorted(set(cities))),
            flagged_fraction=1.0,
            controlled=False,
        )
    return best


def evaluate_guarantee(
    result: ConformalResult,
    scores: np.ndarray,
    labels: np.ndarray,
    cities: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Test the calibrated threshold on data it never saw, per city.

    This is the falsifier. A conformal guarantee is a claim about future
    exchangeable data; held-out *cities* are not exchangeable with calibration
    cities, and the honest thing to publish is where that shows.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    lam = result.lam
    if lam is None:
        return {"error": "no controlled threshold to evaluate", "held": False}

    overall_fnr = false_negative_rate(scores, labels, lam)
    per_city: dict[str, dict[str, Any]] = {}
    for city in sorted(set(cities)):
        mask = np.array([c == city for c in cities])
        if not mask.any():
            continue
        city_labels = labels[mask]
        n_pos = int((city_labels == 1).sum())
        fnr = false_negative_rate(scores[mask], city_labels, lam)
        per_city[city] = {
            "n": int(mask.sum()),
            "n_positive": n_pos,
            "fnr": round(fnr, 4),
            "recall": round(1.0 - fnr, 4),
            # A city with no positives cannot falsify a false-negative claim.
            "held": bool(fnr <= result.alpha) if n_pos else None,
        }

    testable = [c for c, v in per_city.items() if v["held"] is not None]
    held = [c for c in testable if per_city[c]["held"]]
    return {
        "lambda": lam,
        "target_fnr_alpha": result.alpha,
        "observed_fnr": round(overall_fnr, 4),
        "observed_recall": round(1.0 - overall_fnr, 4),
        "held_overall": bool(overall_fnr <= result.alpha),
        "flagged_fraction": round(float((scores >= lam).mean()), 4) if len(scores) else 0.0,
        "cities_tested": len(testable),
        "cities_where_it_held": len(held),
        "per_city": per_city,
        "reading": (
            f"The guarantee held on {len(held)} of {len(testable)} held-out cities "
            f"with positives. Exchangeability across geographies is exactly what a "
            f"city-level split breaks, so a failure here is a property of the "
            f"assumption, not a bug in the procedure."
        ),
    }


def split_calibration(
    rows: list[dict[str, Any]],
    *,
    fraction: float = 0.3,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carve a calibration set out of the *training* cities, by city.

    Split by city, not by tile: tiles from one scene share illumination, season,
    and the scene-level adaptive threshold, so a random tile split would leak
    scene identity across the boundary and certify a threshold that had already
    seen the calibration scenes.
    """
    cities = sorted({r["city"] for r in rows})
    if len(cities) < 2:
        raise ValueError("need at least two cities to split calibration from fitting")
    rng = np.random.RandomState(seed)
    n_cal = max(1, min(len(cities) - 1, round(len(cities) * fraction)))
    cal_cities = set(rng.choice(cities, size=n_cal, replace=False).tolist())
    cal = [r for r in rows if r["city"] in cal_cities]
    fit = [r for r in rows if r["city"] not in cal_cities]
    return cal, fit


def run_conformal(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    seed: int = 42,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Calibrate on train cities, then test the guarantee on held-out cities."""
    from satchangegate.config import get_settings
    from satchangegate.data.tiles import build_tile_index
    from satchangegate.evaluate import compute_tile_features, score_rows
    from satchangegate.tune_gate import assert_disjoint

    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = build_tile_index(oscd_root)
    train_tiles = [t for t in tiles if t.split == "train"]
    test_tiles = [t for t in tiles if t.split == "test"]
    assert_disjoint({t.city for t in train_tiles}, {t.city for t in test_tiles})

    train_rows = compute_tile_features(oscd_root, train_tiles, settings)
    test_rows = compute_tile_features(oscd_root, test_tiles, settings)

    _, train_scored = score_rows(train_rows, settings)
    _, test_scored = score_rows(test_rows, settings)

    # Tier 0 refusals carry no decision, so they cannot calibrate one.
    train_scored = [r for r in train_scored if r["gate"] != "low_quality"]
    test_scored = [r for r in test_scored if r["gate"] != "low_quality"]

    cal_rows, _fit_rows = split_calibration(train_scored, seed=seed)
    assert_disjoint(
        {r["city"] for r in cal_rows},
        {r["city"] for r in test_scored},
    )

    result = calibrate(
        np.array([r["gate_confidence"] for r in cal_rows]),
        np.array([int(r["label"]) for r in cal_rows]),
        [r["city"] for r in cal_rows],
        alpha=alpha,
        delta=delta,
    )
    verdict = evaluate_guarantee(
        result,
        np.array([r["gate_confidence"] for r in test_scored]),
        np.array([int(r["label"]) for r in test_scored]),
        [r["city"] for r in test_scored],
    )

    summary = {
        "calibration": result.to_dict(),
        "held_out_test": verdict,
        "n_test": len(test_scored),
    }
    (out_dir / "_conformal.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_conformal.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(s: dict[str, Any]) -> str:
    c, v = s["calibration"], s["held_out_test"]
    lines = [
        "# Conformal risk control on the gate threshold",
        "",
        f"Target: miss at most **{c['target_fnr_alpha']:.0%}** of real change, with "
        f"**{1 - c['confidence_delta']:.0%}** confidence.",
        "",
        "## Calibration",
        "",
        f"Calibrated on {c['n_calibration']} tiles "
        f"({c['n_calibration_positive']} positive) from "
        f"{len(c['calibration_cities'])} training cities: "
        f"{', '.join(c['calibration_cities'])}.",
        "",
    ]
    if not c["risk_controlled"]:
        lines += [
            "**No threshold on the grid could be certified at this alpha.** The "
            "sample is too small to bound the risk that tightly, so no controlled "
            "operating point is published. That is the correct outcome, not a "
            "failure to report one.",
            "",
        ]
        return "\n".join(lines) + "\n"

    lines += [
        f"- Threshold lambda = **{c['lambda']:.3f}**",
        f"- Calibration false-negative rate: {c['calibration_fnr']:.3f} "
        f"(upper bound {c['calibration_fnr_upper_bound']:.3f} <= "
        f"{c['target_fnr_alpha']:.2f})",
        f"- Flags {c['flagged_fraction_at_lambda']:.1%} of calibration tiles",
        "",
        "## The falsifier: held-out cities",
        "",
        f"Observed false-negative rate at that threshold: **{v['observed_fnr']:.3f}** "
        f"(recall {v['observed_recall']:.3f}), flagging {v['flagged_fraction']:.1%} "
        "of tiles.",
        "",
        f"Overall, the guarantee **{'held' if v['held_overall'] else 'did not hold'}** "
        f"on the held-out split.",
        "",
        "| City | n | positives | recall | FNR | within alpha |",
        "|---|---|---|---|---|---|",
    ]
    for city, m in v["per_city"].items():
        held = "-" if m["held"] is None else ("yes" if m["held"] else "**no**")
        lines.append(
            f"| {city} | {m['n']} | {m['n_positive']} | {m['recall']:.3f} | "
            f"{m['fnr']:.3f} | {held} |"
        )
    lines += ["", v["reading"], "", f"> {c['assumption']}"]
    return "\n".join(lines) + "\n"
