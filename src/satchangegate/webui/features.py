"""The cached feature matrix, and re-scoring it through the real decision function.

Two things this module exists to get right.

**It never reimplements the gate.** Every re-score calls
``features.classical.decide`` -- the same pure function the pipeline calls. A
JavaScript copy of the decision ladder would be fast and would be exactly the
defect ``tune_gate``'s docstring records: a hand-copied duplicate that drifted,
so the tuner optimised a model the pipeline never ran.

**It knows which knobs it can honour.** Only some thresholds live inside
``decide``. Eight of them change the change *mask* -- via the scene threshold,
the magnitude cut, or despeckling -- and therefore change six of the eighteen
features the cached matrix holds. Re-scoring cached rows after moving one of
those produces a number that looks live and is wrong. They are separated here,
by name, and a request that touches one is told the matrix is stale rather than
being quietly served.

The split matters too. The held-out rows are for *reporting*; sliding thresholds
against them turns the test set into development data, and a city-disjointness
assertion cannot undo that after the fact. The default population is the training
split, and anything fitted against test is labelled as such.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from satchangegate.config import GateThresholds
from satchangegate.features.classical import GateFeatures, decide

#: Thresholds read only inside ``decide``. Moving one re-scores instantly.
DECISION_ONLY: frozenset[str] = frozenset(
    {
        "ndvi_strong_min",
        "ndbi_strong_min",
        "ndvi_delta_mean_min",
        "ndbi_delta_mean_min",
        "ndwi_delta_mean_min",
        "min_changed_area_percent",
        "urbanization_score_min",
        "ssim_no_change_min",
        "phash_no_change_max",
        "cva_magnitude_mean_min",
        "magnitude_p95_min",
        "min_largest_component_px",
    }
)

#: Thresholds that change the mask, and so change the features themselves.
#: ``scene_change_threshold`` reads the first four, ``compute_change_mask`` the
#: next two, ``despeckle`` the last two.
FEATURE_CHANGING: frozenset[str] = frozenset(
    {
        "adaptive_threshold",
        "background_sigma",
        "min_absolute_delta",
        "max_absolute_delta",
        "index_delta_small",
        "cva_magnitude_threshold",
        "open_radius_px",
        "min_component_size_px",
    }
)

#: Features the mask-changing knobs invalidate.
DERIVED_FROM_MASK = (
    "changed_area_percent",
    "magnitude_p95",
    "magnitude_p99",
    "largest_component_px",
    "n_components",
    "component_fill_ratio",
)

_INT_FIELDS = {"phash_distance", "largest_component_px", "n_components"}


@dataclass(frozen=True)
class FeatureMatrix:
    split: str
    rows: tuple[dict[str, Any], ...]
    source: str
    produced_by: str

    def __len__(self) -> int:
        return len(self.rows)


def _coerce(raw: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in GateFeatures.model_fields:
        value = raw.get(name)
        if value is None:
            continue
        if name == "valid_observation":
            out[name] = str(value).strip().lower() in ("true", "1", "yes")
        elif name in _INT_FIELDS:
            out[name] = int(float(value))
        else:
            out[name] = float(value)
    return out


@lru_cache(maxsize=8)
def load_matrix(path_str: str, split: str, source: str, produced_by: str) -> FeatureMatrix:
    rows: list[dict[str, Any]] = []
    with open(path_str, encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "tile_id": raw["tile_id"],
                    "city": raw["city"],
                    "label": int(float(raw["label"])),
                    "features": _coerce(raw),
                }
            )
    return FeatureMatrix(split=split, rows=tuple(rows), source=source, produced_by=produced_by)


def classify_overrides(overrides: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(decision_only, feature_changing) names present in an override set."""
    decision = sorted(k for k in overrides if k in DECISION_ONLY)
    changing = sorted(k for k in overrides if k in FEATURE_CHANGING)
    return decision, changing


def score(matrix: FeatureMatrix, thresholds: GateThresholds) -> dict[str, Any]:
    """Run the real ``decide`` across the matrix and summarise the outcome."""
    tp = fp = fn = tn = refused = 0
    flagged_conf: list[float] = []
    reasons: dict[str, int] = {}

    for row in matrix.rows:
        decision, reason, confidence = decide(GateFeatures(**row["features"]), thresholds)
        if decision == "low_quality":
            refused += 1
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
        flagged = decision == "candidate_change"
        if flagged:
            flagged_conf.append(confidence)
        if row["label"] == 1:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged

    scored = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall)
        else None
    )
    balanced = (
        (recall + specificity) / 2 if recall is not None and specificity is not None else None
    )
    candidates = tp + fp
    return {
        "n_total": len(matrix),
        "n_refused": refused,
        "n_scored": scored,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
        "n_candidates": candidates,
        "candidate_rate": candidates / scored if scored else 0.0,
        "gate_filtered_pct": 100.0 * (scored - candidates) / scored if scored else 0.0,
        "tier0_refused_pct": 100.0 * refused / len(matrix) if len(matrix) else 0.0,
        "total_reduction_pct": (
            100.0 * (len(matrix) - candidates) / len(matrix) if len(matrix) else 0.0
        ),
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
