"""A persisted learned scorer, as an alternative to the rule gate.

The repo already measured that a gradient-boosted model beats the hand-written
rules on identical features and an identical split -- +0.14 average precision,
+7 points of precision at the gate's own recall -- and the README invites the
reader to "swap in the learned scorer if your deployment values accuracy over
inspectability". Until now that swap was not a supported operation: ``baseline``
fitted both models inside its report function and discarded them, and no runtime
path could load one.

This module makes the offer real without weakening the argument for keeping the
rules as the default. That argument is good and stands: the rules are
inspectable (every decision returns the rule that fired), add no runtime
dependency, and have no artifact to version, retrain, or drift. What changes is
that the trade is now the operator's to make rather than a claim they cannot
act on.

**A stale artifact must fail loudly, not score quietly.** The saved model records
the feature list it was fitted on, the sklearn version that produced it, and the
cities it saw. Loading it against a different feature vector raises. The failure
this prevents is the one this repo has already been bitten by once: a threshold
sweep optimising a decision ladder the pipeline no longer ran.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import Settings, get_settings

DEFAULT_SCORER_PATH = Path("data/models/gate_scorer.pkl")
ARTIFACT_VERSION = 1


def feature_hash(names: tuple[str, ...] | list[str]) -> str:
    """Stable fingerprint of a feature vector's identity and order."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]


class StaleScorerError(RuntimeError):
    """Raised when a saved scorer does not match the current feature vector."""


@dataclass
class ScorerArtifact:
    """A fitted model plus everything needed to know whether it still applies."""

    model: Any
    feature_names: tuple[str, ...]
    feature_hash: str
    sklearn_version: str
    train_cities: tuple[str, ...]
    n_train: int
    metrics: dict[str, Any]
    artifact_version: int = ARTIFACT_VERSION

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """P(change) for each row."""
        return np.asarray(self.model.predict_proba(x))[:, 1]

    def card(self) -> dict[str, Any]:
        """The model card written next to the artifact."""
        return {
            "artifact_version": self.artifact_version,
            "model": type(self.model).__name__,
            "sklearn_version": self.sklearn_version,
            "feature_names": list(self.feature_names),
            "feature_hash": self.feature_hash,
            "n_train": self.n_train,
            "train_cities": list(self.train_cities),
            "held_out_metrics": self.metrics,
            "intended_use": (
                "Tier-1 triage scoring for satellite change candidates, in place "
                "of the hand-written rule gate. Emits a probability, not a "
                "decision; the operating point is chosen separately."
            ),
            "limitations": [
                "Fitted on OSCD urban-change labels from the listed cities only. "
                "OSCD labels urban change and treats seasonal, agricultural and "
                "hydrological change as negative, so the scorer inherits that "
                "definition of 'change'.",
                "No feature is scene-constant, so the model cannot key on city "
                "identity -- but it has still only seen these cities. Any new AOI "
                "needs its own validation.",
                "Unlike the rule gate it returns no human-readable reason. If an "
                "operator has to justify a decision, that is a real cost.",
            ],
        }


def fit_scorer(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]] | None = None,
    *,
    feature_names: tuple[str, ...] | None = None,
    seed: int = 42,
) -> ScorerArtifact:
    """Fit the gradient-boosted scorer on precomputed feature rows."""
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            'The learned scorer needs the optional extra: pip install -e ".[baseline]"'
        ) from exc

    from satchangegate.baseline import FEATURE_NAMES, feature_matrix

    names = tuple(feature_names or FEATURE_NAMES)
    x_train, y_train = feature_matrix(train_rows)
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=seed)
    model.fit(x_train, y_train)

    metrics: dict[str, Any] = {}
    if test_rows:
        x_test, y_test = feature_matrix(test_rows)
        scores = model.predict_proba(x_test)[:, 1]
        metrics = {
            "n_test": len(test_rows),
            "average_precision": round(float(average_precision_score(y_test, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
            "test_cities": sorted({r["city"] for r in test_rows}),
        }

    return ScorerArtifact(
        model=model,
        feature_names=names,
        feature_hash=feature_hash(names),
        sklearn_version=sklearn.__version__,
        train_cities=tuple(sorted({r["city"] for r in train_rows})),
        n_train=len(train_rows),
        metrics=metrics,
    )


def save_scorer(artifact: ScorerArtifact, path: Path | None = None) -> Path:
    """Persist the scorer and write its model card alongside."""
    path = Path(path or DEFAULT_SCORER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    path.with_suffix(".card.json").write_text(
        json.dumps(artifact.card(), indent=2), encoding="utf-8"
    )
    return path


def load_scorer(path: Path | None = None) -> ScorerArtifact:
    """Load a persisted scorer, refusing one that no longer matches the features.

    ``pickle`` is used because a fitted sklearn estimator has no stable portable
    format. The artifact is one this repo wrote, under ``data/models``, and is
    never fetched from anywhere; loading an untrusted pickle would be a different
    proposition entirely.
    """
    from satchangegate.baseline import FEATURE_NAMES

    path = Path(path or DEFAULT_SCORER_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"No scorer at {path}. Fit one with `satchangegate fit-scorer --split train`."
        )
    with open(path, "rb") as fh:
        artifact = pickle.load(fh)

    if not isinstance(artifact, ScorerArtifact):
        raise StaleScorerError(f"{path} does not contain a ScorerArtifact")
    if artifact.artifact_version != ARTIFACT_VERSION:
        raise StaleScorerError(
            f"{path} was written by artifact version {artifact.artifact_version}; "
            f"this build expects {ARTIFACT_VERSION}. Refit it."
        )
    current = feature_hash(FEATURE_NAMES)
    if artifact.feature_hash != current:
        raise StaleScorerError(
            f"{path} was fitted on a different feature vector "
            f"({artifact.feature_hash} vs {current}). The gate's features have "
            "changed since; refit with `satchangegate fit-scorer --split train` "
            "rather than scoring against a model that never saw them."
        )
    return artifact


def score_rows_learned(
    rows: list[dict[str, Any]],
    artifact: ScorerArtifact,
    threshold: float,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Apply the learned scorer to feature rows, mirroring ``score_rows``' shape.

    Tier 0 still has the first and final word: a pair whose imagery is unusable
    is refused before any model sees it, exactly as under the rules. A learned
    scorer that happily scored a cloud-covered pair would be a regression in the
    one place this repo is strict.
    """
    settings = settings or get_settings()
    from satchangegate.baseline import feature_matrix

    scored: list[dict[str, Any]] = []
    scorable = [r for r in rows if r.get("valid_observation", True)]
    probs = (
        artifact.predict_proba(feature_matrix(scorable)[0])
        if scorable
        else np.zeros(0, dtype=float)
    )
    by_id = {id(r): p for r, p in zip(scorable, probs, strict=True)}

    for row in rows:
        out = dict(row)
        if not row.get("valid_observation", True):
            out.update(
                {
                    "gate": "low_quality",
                    "gate_reason": "failed Tier 0 quality checks",
                    "gate_confidence": 0.0,
                }
            )
        else:
            p = float(by_id[id(row)])
            decision = "candidate_change" if p >= threshold else "no_change"
            out.update(
                {
                    "gate": decision,
                    "gate_reason": f"learned scorer p={p:.3f} vs threshold {threshold:.3f}",
                    "gate_confidence": round(p, 4),
                }
            )
        scored.append(out)
    return scored
