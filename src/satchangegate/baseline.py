"""Learned baselines against the hand-written gate.

The gate is three documented rules over eleven numeric features, with
grid-tuned thresholds. The obvious question is why those features are not
simply fed to a classifier — so this module answers it rather than leaving it
open. Logistic regression and gradient boosting are fitted on the same train
cities the rules were tuned on, and evaluated on the same held-out cities.

The comparison is deliberately fair: identical features, identical split,
identical leakage assertion. If a learned model wins, that is the finding; if
it does not, the rules have evidence behind them instead of just prose.

It also produces the artifact a *cost* gate should ship. A single operating
point cannot answer "what if I need 80% recall?", but a precision-recall curve
can — and because ``decide`` emits a calibrated confidence, the rule-based gate
gets a curve too rather than a lone point.

Requires the optional extra: ``pip install -e ".[baseline]"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import Settings, get_settings
from satchangegate.data.tiles import build_tile_index
from satchangegate.evaluate import compute_tile_features, score_rows
from satchangegate.metrics import wilson_interval
from satchangegate.tune_gate import assert_disjoint

# Features the gate itself consumes. The learned models see exactly these, so
# neither side has an information advantage.
FEATURE_NAMES = (
    "ssim",
    "phash_distance",
    "ndvi_delta_mean",
    "ndbi_delta_mean",
    "ndwi_delta_mean",
    "ndvi_delta_abs_mean",
    "ndbi_delta_abs_mean",
    "ndwi_delta_abs_mean",
    "cva_magnitude_mean",
    "changed_area_percent",
)


@dataclass
class CurvePoint:
    threshold: float
    precision: float
    recall: float


@dataclass
class ModelResult:
    name: str
    average_precision: float
    roc_auc: float
    curve: list[CurvePoint] = field(default_factory=list)
    # Metrics at the recall the shipped rule gate achieves, so the comparison is
    # like-for-like rather than at each model's own favourite operating point.
    precision_at_gate_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "average_precision": round(self.average_precision, 4),
            "roc_auc": round(self.roc_auc, 4),
            "precision_at_gate_recall": (
                None
                if self.precision_at_gate_recall is None
                else round(self.precision_at_gate_recall, 4)
            ),
            "curve": [
                {
                    "threshold": round(p.threshold, 4),
                    "precision": round(p.precision, 4),
                    "recall": round(p.recall, 4),
                }
                for p in self.curve
            ],
        }


def _require_sklearn() -> Any:
    try:
        import sklearn
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            'Learned baselines need the optional extra: pip install -e ".[baseline]"'
        ) from exc
    return sklearn


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) from precomputed tile feature rows."""
    x = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return x, y


def _precision_at_recall(precisions: np.ndarray, recalls: np.ndarray, target: float) -> float:
    """Best precision available at or above a target recall."""
    viable = precisions[recalls >= target]
    return float(viable.max()) if viable.size else 0.0


def run_baselines(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    settings: Settings | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit learned baselines on train, evaluate on test, and emit a PR curve."""
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = build_tile_index(oscd_root)
    train_tiles = [t for t in tiles if t.split == "train"]
    test_tiles = [t for t in tiles if t.split == "test"]
    assert_disjoint({t.city for t in train_tiles}, {t.city for t in test_tiles})

    train_rows = compute_tile_features(oscd_root, train_tiles, settings)
    test_rows = compute_tile_features(oscd_root, test_tiles, settings)

    x_train, y_train = feature_matrix(train_rows)
    x_test, y_test = feature_matrix(test_rows)

    # The shipped rule gate, scored the same way, on the same rows.
    gate_cm, gate_scored = score_rows(test_rows, settings)
    gate_recall = gate_cm.recall
    gate_precision = gate_cm.precision or 0.0
    gate_scores = np.array([r["gate_confidence"] for r in gate_scored], dtype=np.float64)
    gate_labels = np.array([int(r["label"]) for r in gate_scored], dtype=np.int64)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, random_state=seed
        ),
    }

    results: list[ModelResult] = []

    # The rule gate's own confidence, swept — this is what turns a single
    # operating point into a curve a buyer can choose from.
    p, r, thr = precision_recall_curve(gate_labels, gate_scores)
    results.append(
        ModelResult(
            name="rule_gate_confidence",
            average_precision=float(average_precision_score(gate_labels, gate_scores)),
            roc_auc=float(roc_auc_score(gate_labels, gate_scores)),
            curve=[
                CurvePoint(float(t), float(pp), float(rr))
                for pp, rr, t in zip(p[:-1], r[:-1], thr, strict=True)
            ],
            precision_at_gate_recall=_precision_at_recall(p, r, gate_recall),
        )
    )

    for name, model in models.items():
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        p, r, thr = precision_recall_curve(y_test, scores)
        results.append(
            ModelResult(
                name=name,
                average_precision=float(average_precision_score(y_test, scores)),
                roc_auc=float(roc_auc_score(y_test, scores)),
                curve=[
                    CurvePoint(float(t), float(pp), float(rr))
                    for pp, rr, t in zip(p[:-1], r[:-1], thr, strict=True)
                ],
                precision_at_gate_recall=_precision_at_recall(p, r, gate_recall),
            )
        )

    prevalence = float(y_test.mean())
    summary: dict[str, Any] = {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_cities": sorted({t.city for t in train_tiles}),
        "test_cities": sorted({t.city for t in test_tiles}),
        "positive_prevalence_test": round(prevalence, 4),
        "features": list(FEATURE_NAMES),
        "shipped_gate_operating_point": {
            "recall": round(gate_recall, 4),
            "precision": round(gate_precision, 4),
            "recall_ci95": [round(v, 4) for v in wilson_interval(gate_cm.tp, gate_cm.n_positive)],
        },
        "models": [m.to_dict() for m in results],
    }
    (out_dir / "_baselines.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "_baselines.md").write_text(_render(summary), encoding="utf-8")
    plot_pr_curves(summary, out_dir / "pr_curves.png")
    return summary


def _render(summary: dict[str, Any]) -> str:
    op = summary["shipped_gate_operating_point"]
    lines = [
        "# Learned baselines vs the rule gate",
        "",
        f"Fitted on {summary['n_train']} tiles from {len(summary['train_cities'])} train cities; "
        f"evaluated on {summary['n_test']} tiles from {len(summary['test_cities'])} held-out cities. "
        f"Positive prevalence on test: {summary['positive_prevalence_test']:.3f} "
        "(the precision a coin-flip classifier would reach).",
        "",
        "All models see exactly the features the gate uses, on the same split.",
        "",
        "| Model | Average precision | ROC AUC | Precision @ gate's recall |",
        "|---|---|---|---|",
    ]
    for m in summary["models"]:
        lines.append(
            f"| {m['name']} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | "
            f"{m['precision_at_gate_recall']:.3f} |"
        )
    lines += [
        "",
        f"The shipped rule gate operates at recall {op['recall']:.3f}, precision "
        f"{op['precision']:.3f}. The final column asks what each model achieves at that "
        "same recall, which is the like-for-like comparison.",
        "",
        "![Precision-recall curves](pr_curves.png)",
        "",
        "Average precision is the summary statistic here rather than F1: it is "
        "threshold-free, so it compares the ranking each model produces instead of "
        "one arbitrarily chosen cut.",
    ]
    return "\n".join(lines) + "\n"


def plot_pr_curves(summary: dict[str, Any], path: Path) -> None:
    """Precision-recall curves for every model, with the shipped point marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)
    colours = {
        "rule_gate_confidence": "#c1440e",
        "logistic_regression": "#2b6cb0",
        "gradient_boosting": "#2f855a",
    }
    labels = {
        "rule_gate_confidence": "Rule gate (confidence swept)",
        "logistic_regression": "Logistic regression",
        "gradient_boosting": "Gradient boosting",
    }
    for m in summary["models"]:
        curve = m["curve"]
        if not curve:
            continue
        recalls = [c["recall"] for c in curve]
        precisions = [c["precision"] for c in curve]
        ax.plot(
            recalls,
            precisions,
            label=f"{labels.get(m['name'], m['name'])} (AP {m['average_precision']:.3f})",
            color=colours.get(m["name"], "#666666"),
            linewidth=1.8,
        )

    op = summary["shipped_gate_operating_point"]
    ax.scatter(
        [op["recall"]],
        [op["precision"]],
        marker="o",
        s=70,
        zorder=5,
        color="#c1440e",
        edgecolor="white",
        linewidth=1.4,
        label=f"Shipped operating point ({op['recall']:.2f}, {op['precision']:.2f})",
    )
    prevalence = summary["positive_prevalence_test"]
    ax.axhline(prevalence, linestyle=":", color="#888888", linewidth=1.2)
    ax.text(0.02, prevalence + 0.015, f"chance ({prevalence:.2f})", fontsize=8, color="#666666")

    ax.set_xlabel("Recall — share of real change forwarded for review")
    ax.set_ylabel("Precision — share of forwarded tiles that are real change")
    ax.set_title("Gate operating characteristics on held-out cities")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
