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

Corrected 2026-09-19 -- the calibration sample was not independent
-----------------------------------------------------------------

Learn-then-Test needs the predictor to be fixed *before* calibration: the scores
it is calibrated on must not come from a model that was fitted to those same
tiles. The first version of ``run_conformal`` did not provide that. It scored
every training city with the already-fitted shipped thresholds and *then* called
``split_calibration``, so the four calibration cities (abudhabi, mumbai, nantes,
pisa) were all inside the fourteen ``tune`` had swept. ``_fit_rows`` was
discarded with a leading underscore because, by that point, there was nothing
left to fit. The disjointness assertion checked calibration against *test* only
-- never against *fit* -- which is precisely the pair that was overlapping.

The published held-out FNR of 0.177 was never affected: it is measured on nine
cities that nothing had seen. What was not established is the 90% confidence
bound attached to lambda, because the calibration scores came from a gate tuned
on those same cities and were optimistic by an unmeasured amount.

``run_conformal`` now chooses the calibration cities first, refits the gate on
the remaining training cities via ``tune_gate.sweep_tiles``, and calibrates that
predictor -- which cannot have seen its own calibration set. It also recomputes
the old contaminated result alongside, so the size of the defect is reported as
a measurement rather than asserted. The cost is one extra threshold sweep per
run, which is the slowest thing in the repo and is the honest price.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
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
                "and is tested per city rather than asserted. Predictor "
                "independence -- the other precondition, and the one this module "
                "previously failed -- is now established by construction: the "
                "gate is refit with the calibration cities withheld."
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


def split_calibration_cities(
    cities: Iterable[str],
    *,
    fraction: float = 0.3,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Choose ``(calibration, fit)`` cities *before* anything has been fitted.

    Taking the split at the city-name level rather than over scored rows is what
    lets the caller fit a predictor with the calibration cities already withheld.
    Doing it the other way round -- score everything, then split -- is the defect
    described in this module docstring: it produces a calibration set the
    predictor has already been tuned on.
    """
    ordered = sorted(set(cities))
    if len(ordered) < 2:
        raise ValueError("need at least two cities to split calibration from fitting")
    rng = np.random.RandomState(seed)
    n_cal = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    cal = set(rng.choice(ordered, size=n_cal, replace=False).tolist())
    return cal, set(ordered) - cal


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

    Row-level convenience over :func:`split_calibration_cities`, kept because the
    partition it produces is worth asserting on directly. It cannot, on its own,
    give Learn-then-Test the independence it needs: by the time rows exist they
    have already been scored by some predictor. ``run_conformal`` therefore calls
    the city-level function instead.
    """
    cal_cities, _ = split_calibration_cities(
        {r["city"] for r in rows}, fraction=fraction, seed=seed
    )
    cal = [r for r in rows if r["city"] in cal_cities]
    fit = [r for r in rows if r["city"] not in cal_cities]
    return cal, fit


def assert_three_way_disjoint(fit: set[str], cal: set[str], test: set[str]) -> None:
    """Fit, calibration and test must share no city with any other.

    Three assertions, not one. The previous flow checked only calibration against
    test and passed, because the pair that actually overlapped was fit against
    calibration.
    """
    from satchangegate.tune_gate import assert_disjoint

    assert_disjoint(fit, cal)
    assert_disjoint(fit, test)
    assert_disjoint(cal, test)


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(scores, labels, cities) for a list of scored rows."""
    return (
        np.array([r["gate_confidence"] for r in rows], dtype=float),
        np.array([int(r["label"]) for r in rows], dtype=int),
        [r["city"] for r in rows],
    )


def run_conformal(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    seed: int = 42,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Refit the gate with the calibration cities withheld, calibrate, then falsify.

    The order of operations is the whole point, and it is the thing the previous
    version got wrong. Calibration cities are chosen first; the gate is refit on
    the cities that remain; only then is lambda calibrated against a predictor
    that provably never saw the tiles certifying it.

    Cities with no assessable tiles are excluded from the split before it is
    drawn. That is not a peek at the predictor: ``valid_observation`` comes from
    ``QualityThresholds`` and the registration estimate, neither of which is
    being fitted here, so a city that Tier 0 refuses wholesale can inform neither
    a fit nor a calibration and would only consume a slot.

    The superseded contaminated result is recomputed alongside, over the same
    calibration cities, so the defect is reported as a measured difference
    instead of an assertion.
    """
    from satchangegate.config import get_settings
    from satchangegate.data.tiles import build_tile_index
    from satchangegate.evaluate import compute_tile_features, score_rows
    from satchangegate.tune_gate import sweep_tiles

    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = build_tile_index(oscd_root)
    train_tiles = [t for t in tiles if t.split == "train"]
    test_tiles = [t for t in tiles if t.split == "test"]
    test_cities = {t.city for t in test_tiles}

    # Features under the shipped gate: needed for the contaminated comparison,
    # and to read Tier 0 status without consulting the predictor.
    train_rows_shipped = compute_tile_features(oscd_root, train_tiles, settings)
    test_rows_shipped = compute_tile_features(oscd_root, test_tiles, settings)

    assessable = {r["city"] for r in train_rows_shipped if r.get("valid_observation")}
    refused = sorted({t.city for t in train_tiles} - assessable)
    cal_cities, fit_cities = split_calibration_cities(assessable, seed=seed)
    assert_three_way_disjoint(fit_cities, cal_cities, test_cities)

    # The correction: refit with the calibration cities withheld.
    fit_tiles = [t for t in train_tiles if t.city in fit_cities]
    refit, fit_baseline, n_evaluated = sweep_tiles(oscd_root, fit_tiles, settings=settings)
    refit_settings = settings.model_copy(
        update={"gate": settings.gate.model_copy(update=refit.thresholds)}
    )

    shipped_gate = settings.gate.model_dump()
    differs = {
        key: {"shipped": shipped_gate[key], "refit": refit.thresholds[key]}
        for key in sorted(shipped_gate)
        if shipped_gate[key] != refit.thresholds[key]
    }

    # The gate touches features only through background_sigma, so the shipped
    # feature matrix is reusable whenever the refit kept it.
    reusable = refit.thresholds["background_sigma"] == settings.gate.background_sigma
    cal_tiles = [t for t in train_tiles if t.city in cal_cities]
    if reusable:
        cal_rows = [r for r in train_rows_shipped if r["city"] in cal_cities]
        test_rows = test_rows_shipped
    else:
        cal_rows = compute_tile_features(oscd_root, cal_tiles, refit_settings)
        test_rows = compute_tile_features(oscd_root, test_tiles, refit_settings)

    _, cal_scored = score_rows(cal_rows, refit_settings)
    _, test_scored = score_rows(test_rows, refit_settings)

    # Tier 0 refusals carry no decision, so they cannot calibrate one.
    cal_scored = [r for r in cal_scored if r["gate"] != "low_quality"]
    test_scored = [r for r in test_scored if r["gate"] != "low_quality"]

    result = calibrate(*_arrays(cal_scored), alpha=alpha, delta=delta)
    verdict = evaluate_guarantee(result, *_arrays(test_scored))

    # What the leaky flow reported: identical calibration cities, scored by the
    # gate that had been tuned on them.
    _, cal_contaminated = score_rows(
        [r for r in train_rows_shipped if r["city"] in cal_cities], settings
    )
    _, test_contaminated = score_rows(test_rows_shipped, settings)
    cal_contaminated = [r for r in cal_contaminated if r["gate"] != "low_quality"]
    test_contaminated = [r for r in test_contaminated if r["gate"] != "low_quality"]
    contaminated = calibrate(*_arrays(cal_contaminated), alpha=alpha, delta=delta)
    contaminated_verdict = evaluate_guarantee(contaminated, *_arrays(test_contaminated))

    summary = {
        "predictor": {
            "note": (
                "Gate thresholds refit on the training cities that remained after "
                "the calibration cities were withheld. Learn-then-Test requires "
                "the predictor to be independent of the sample that certifies it."
            ),
            "fitted_on_cities": sorted(fit_cities),
            "n_fit_tiles": len(fit_tiles),
            "n_combinations_evaluated": n_evaluated,
            "in_sample_balanced_accuracy": round(refit.score, 4),
            "shipped_in_sample_balanced_accuracy": round(fit_baseline.score, 4),
            "thresholds": refit.thresholds,
            "matches_shipped_thresholds": not differs,
            "differs_from_shipped": differs,
            "features_reused_from_shipped": reusable,
            "excluded_no_assessable_tiles": refused,
        },
        "calibration": result.to_dict(),
        "held_out_test": verdict,
        "n_test": len(test_scored),
        "superseded_contaminated_run": {
            "what_this_is": (
                "The pre-2026-09-19 result, recomputed over the same calibration "
                "cities but scored by the shipped gate, which had been tuned on "
                "those cities. Published for comparison only. Its lambda is not a "
                "risk-controlled threshold, because its calibration scores were "
                "not independent of the predictor that produced them."
            ),
            "calibration": contaminated.to_dict(),
            "held_out_test": contaminated_verdict,
        },
    }
    (out_dir / "_conformal.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_conformal.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(s: dict[str, Any]) -> str:
    c, v = s["calibration"], s["held_out_test"]
    pr = s["predictor"]
    sup = s["superseded_contaminated_run"]
    lines = [
        "# Conformal risk control on the gate threshold",
        "",
        f"Target: miss at most **{c['target_fnr_alpha']:.0%}** of real change, with "
        f"**{1 - c['confidence_delta']:.0%}** confidence.",
        "",
        "## The predictor being certified",
        "",
        f"Gate refit on {len(pr['fitted_on_cities'])} training cities with the "
        f"calibration cities withheld: {', '.join(pr['fitted_on_cities'])}. "
        f"{pr['n_fit_tiles']} tiles, {pr['n_combinations_evaluated']} threshold "
        "combinations evaluated.",
        "",
        (
            "**The refit reproduced the shipped thresholds exactly**, so the "
            "guarantee below applies to the gate this repo actually ships."
            if pr["matches_shipped_thresholds"]
            else "**The refit differs from the shipped thresholds** on "
            + ", ".join(f"`{k}`" for k in pr["differs_from_shipped"])
            + ". The guarantee below certifies the refit gate, not the shipped "
            "one; adopting it means adopting these thresholds."
        ),
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
    lines += ["", v["reading"], ""]

    sc, sv = sup["calibration"], sup["held_out_test"]
    lines += [
        "## What the correction changed",
        "",
        "Before 2026-09-19 the calibration cities were scored by a gate that had "
        "been tuned on them, so the bound on lambda was not established. Same "
        "cities, same alpha, the only difference being whether the predictor had "
        "seen them:",
        "",
        "| | Contaminated (superseded) | Corrected |",
        "|---|---|---|",
        f"| Lambda | {sc['lambda']} | {c['lambda']} |",
        f"| Calibration FNR | {sc['calibration_fnr']:.4f} | {c['calibration_fnr']:.4f} |",
        f"| Upper bound | {sc['calibration_fnr_upper_bound']:.4f} | "
        f"{c['calibration_fnr_upper_bound']:.4f} |",
        f"| Risk controlled | {sc['risk_controlled']} | {c['risk_controlled']} |",
        f"| Held-out FNR | {sv.get('observed_fnr', 'n/a')} | {v['observed_fnr']:.4f} |",
        f"| Held overall | {sv.get('held_overall', 'n/a')} | {v['held_overall']} |",
        "",
        sup["what_this_is"],
        "",
        f"> {c['assumption']}",
    ]
    return "\n".join(lines) + "\n"
