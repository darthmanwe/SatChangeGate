"""Threshold tuning, fitted on train cities only.

Two properties this module now guarantees that its predecessor did not:

1. **It optimises the deployed model.** The sweep calls ``decide`` — the same
   function the pipeline calls. The previous version carried a hand-copied
   duplicate of the decision ladder that had already drifted (it dropped the
   ``valid_observation`` term and evaluated the same cloud check twice), so the
   thresholds were fitted to a model that was never run.

2. **It cannot leak.** Candidate thresholds are scored only on train-split
   tiles, and ``assert_disjoint`` fails loudly if a train city ever appears in
   the evaluation split. The previous flow swept 6,561 combinations against 11
   labelled tiles with no holdout and reported the argmax as a result.

Selection uses balanced accuracy rather than F1: F1 ignores true negatives, and
avoiding needless VLM spend on unchanged tiles is precisely what the gate is for.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satchangegate.config import Settings, get_settings
from satchangegate.data.tiles import build_tile_index
from satchangegate.evaluate import compute_tile_features, score_rows
from satchangegate.metrics import ConfusionMatrix

# Swept axes. Deliberately small: with ~1300 training tiles a large grid would
# still overfit, and every extra axis multiplies the multiple-comparison burden.
SWEEP_GRID: dict[str, Sequence[float]] = {
    "background_sigma": (2.0, 3.0, 4.5, 6.0),
    "min_changed_area_percent": (2.0, 5.0, 10.0, 20.0, 35.0),
    "ndvi_strong_min": (0.04, 0.08, 0.14, 0.25),
    "ndbi_strong_min": (0.04, 0.08, 0.14, 0.25),
    "ndvi_delta_mean_min": (0.02, 0.05, 0.10),
    "ndbi_delta_mean_min": (0.02, 0.05, 0.10),
}


@dataclass
class SweepResult:
    thresholds: dict[str, float]
    metrics: dict[str, Any]
    score: float


def assert_disjoint(train_cities: set[str], eval_cities: set[str]) -> None:
    """Fail loudly if tuning and evaluation share a city."""
    overlap = train_cities & eval_cities
    if overlap:
        raise RuntimeError(
            "Train/test leakage: these cities appear in both the tuning and "
            f"evaluation sets: {sorted(overlap)}"
        )


def _score(cm: ConfusionMatrix) -> float:
    """Balanced accuracy; 0 when the set has no negatives to be right about."""
    ba = cm.balanced_accuracy
    return 0.0 if ba is None else ba


def sweep(
    oscd_root: Path | None = None,
    *,
    settings: Settings | None = None,
    grid: dict[str, Sequence[float]] | None = None,
    split: str = "train",
    out_dir: Path | None = None,
) -> SweepResult:
    """Fit gate thresholds on one split and write a tuning report."""
    settings = settings or get_settings()
    grid = grid or SWEEP_GRID
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = build_tile_index(oscd_root)
    tiles = [t for t in all_tiles if t.split == split]
    if not tiles:
        raise RuntimeError(f"No tiles in split {split!r}")

    train_cities = {t.city for t in tiles}
    eval_cities = {t.city for t in all_tiles if t.split != split}
    assert_disjoint(train_cities, eval_cities)

    # adaptive_percentile changes the change mask itself, so it changes the
    # features. It is swept in the outer loop where features are recomputed;
    # the remaining axes only affect the decision and reuse cached features.
    grid = dict(grid)
    percentiles = list(grid.pop("background_sigma", (settings.gate.background_sigma,)))
    keys = list(grid)

    baseline_rows = compute_tile_features(oscd_root, tiles, settings)
    baseline_cm, _ = score_rows(baseline_rows, settings)
    baseline = SweepResult(
        thresholds=settings.gate.model_dump(),
        metrics=baseline_cm.to_dict(),
        score=_score(baseline_cm),
    )

    best: SweepResult | None = None
    n_evaluated = 0
    for pct in percentiles:
        feature_settings = settings.model_copy(
            update={"gate": settings.gate.model_copy(update={"background_sigma": pct})}
        )
        rows = (
            baseline_rows
            if pct == settings.gate.background_sigma
            else compute_tile_features(oscd_root, tiles, feature_settings)
        )
        for combo in itertools.product(*(grid[k] for k in keys)):
            candidate = dict(zip(keys, combo, strict=True))
            trial = feature_settings.model_copy(
                update={"gate": feature_settings.gate.model_copy(update=candidate)}
            )
            cm, _ = score_rows(rows, trial)
            n_evaluated += 1
            score = _score(cm)
            if best is None or score > best.score:
                best = SweepResult(
                    thresholds=trial.gate.model_dump(), metrics=cm.to_dict(), score=score
                )

    assert best is not None
    report = {
        "split": split,
        "n_tiles": len(tiles),
        "n_cities": len(train_cities),
        "cities": sorted(train_cities),
        "held_out_cities": sorted(eval_cities),
        "n_combinations_evaluated": n_evaluated,
        "selection_criterion": "balanced_accuracy",
        "baseline": {"score": round(baseline.score, 4), "metrics": baseline.metrics},
        "best": {"score": round(best.score, 4), "metrics": best.metrics},
        "tuned_thresholds": best.thresholds,
    }
    (out_dir / "_gate_tuning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "_gate_tuning_report.md").write_text(_render_report(report), encoding="utf-8")
    _write_threshold_yaml(best.thresholds, out_dir / "tuned_thresholds.yaml", report)
    return best


def _render_report(report: dict[str, Any]) -> str:
    b, base = report["best"]["metrics"], report["baseline"]["metrics"]
    return f"""# Gate tuning report

Fitted on the **{report["split"]}** split: {report["n_tiles"]} tiles across
{report["n_cities"]} cities. Held-out cities (never seen during tuning):
{", ".join(report["held_out_cities"])}.

Evaluated {report["n_combinations_evaluated"]} threshold combinations, selecting
on balanced accuracy.

| | Baseline | Tuned |
|---|---|---|
| Balanced accuracy | {report["baseline"]["score"]:.3f} | {report["best"]["score"]:.3f} |
| Recall | {base["recall"]:.3f} | {b["recall"]:.3f} |
| Precision | {base.get("precision") or float("nan"):.3f} | {b.get("precision") or float("nan"):.3f} |
| Specificity | {base.get("specificity") or float("nan"):.3f} | {b.get("specificity") or float("nan"):.3f} |
| F1 | {base.get("f1") or float("nan"):.3f} | {b.get("f1") or float("nan"):.3f} |

**These are in-sample figures.** They describe the fit, not the expected
performance. Run `satchangegate eval --split test` for the out-of-sample result,
which is the only number that should be quoted.

Tuned thresholds are written to `tuned_thresholds.yaml`; copy them into
`src/satchangegate/configs/thresholds.yaml` to adopt them.
"""


def _write_threshold_yaml(thresholds: dict[str, Any], path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Generated by `satchangegate tune`. In-sample fit on split "
        f"'{report['split']}' ({report['n_tiles']} tiles, {report['n_cities']} cities).",
        f"# Held out during tuning: {', '.join(report['held_out_cities'])}",
        "gate:",
    ]
    for key in sorted(thresholds):
        lines.append(f"  {key}: {thresholds[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
