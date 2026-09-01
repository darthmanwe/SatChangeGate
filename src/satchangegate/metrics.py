"""Binary-classification metrics with uncertainty.

One implementation, used by every harness. There were previously four copies of
precision/recall/F1 across the codebase, and none of them reported a sample size
or an interval — so a headline "F1 0.93" fitted to 11 tiles was presented with
the same authority as a result on thousands.

Every metric here carries n. Proportion metrics carry a Wilson score interval,
which behaves sensibly at small n and near 0 or 1 where the normal approximation
does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# 97.5th percentile of the standard normal, for a two-sided 95% interval.
_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (0.0, 1.0) when there is no data, since nothing is known.
    """
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_negative(self) -> int:
        return self.fp + self.tn

    @property
    def has_negatives(self) -> bool:
        """False when precision is structurally pinned at 1.0 and meaningless.

        The OSCD pair-level evaluation previously had zero negatives — every pair
        contains change, and unlabelled pairs were counted as positive — so `fp`
        could never increment and F1 reduced to 2r/(1+r), carrying no information
        beyond the candidate rate.
        """
        return self.n_negative > 0

    @property
    def precision(self) -> float | None:
        if not self.has_negatives:
            return None
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_positive if self.n_positive else 0.0

    @property
    def specificity(self) -> float | None:
        if not self.has_negatives:
            return None
        return self.tn / self.n_negative

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def balanced_accuracy(self) -> float | None:
        s = self.specificity
        return None if s is None else (self.recall + s) / 2

    def to_dict(self) -> dict[str, Any]:
        rec_lo, rec_hi = wilson_interval(self.tp, self.n_positive)
        out: dict[str, Any] = {
            "n": self.n,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "recall": round(self.recall, 4),
            "recall_ci95": [round(rec_lo, 4), round(rec_hi, 4)],
            "has_negative_class": self.has_negatives,
        }
        if self.has_negatives:
            prec_lo, prec_hi = wilson_interval(self.tp, self.tp + self.fp)
            spec_lo, spec_hi = wilson_interval(self.tn, self.n_negative)
            out.update(
                {
                    "precision": round(self.precision or 0.0, 4),
                    "precision_ci95": [round(prec_lo, 4), round(prec_hi, 4)],
                    "specificity": round(self.specificity or 0.0, 4),
                    "specificity_ci95": [round(spec_lo, 4), round(spec_hi, 4)],
                    "f1": round(self.f1 or 0.0, 4),
                    "balanced_accuracy": round(self.balanced_accuracy or 0.0, 4),
                }
            )
        else:
            out["precision"] = None
            out["f1"] = None
            out["metrics_note"] = (
                "No negative examples in this evaluation set: precision and F1 are "
                "undefined (precision would be pinned at 1.0 by construction). "
                "Only recall and the candidate rate are meaningful here."
            )
        return out


def confusion_from_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ConfusionMatrix:
    """Build a confusion matrix from aligned 0/1 sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} truth vs {len(y_pred)} predictions")
    cm = ConfusionMatrix()
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            cm.tp += 1
        elif not t and p:
            cm.fp += 1
        elif t and not p:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def iou(pred: Any, gt: Any) -> float:
    """Intersection over union for two boolean masks."""
    import numpy as np

    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    union = int((p | g).sum())
    if union == 0:
        return 1.0
    return float((p & g).sum() / union)


def pixel_confusion(pred: Any, gt: Any, valid: Any = None) -> ConfusionMatrix:
    """Pixel-level confusion matrix for a predicted mask against ground truth.

    ``valid`` restricts the count to observed pixels, so masked-out cloud does
    not silently accumulate true negatives and inflate specificity.
    """
    import numpy as np

    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"mask shape mismatch: {p.shape} vs {g.shape}")
    if valid is not None:
        v = np.asarray(valid).astype(bool)
        p, g = p & v, g & v
        keep = v
    else:
        keep = np.ones_like(p, dtype=bool)
    return ConfusionMatrix(
        tp=int((p & g & keep).sum()),
        fp=int((p & ~g & keep).sum()),
        fn=int((~p & g & keep).sum()),
        tn=int((~p & ~g & keep).sum()),
    )


def pixel_metrics(pred: Any, gt: Any, valid: Any = None) -> dict[str, Any]:
    """Pixel-level F1 and IoU, with the confusion matrix they derive from.

    This exists because the repo published a pixel-level F1 figure that no code
    path computed: its only trace was a comment recording a *train*-split number,
    quoted in a section about the test split. A headline number needs a command
    that reproduces it.
    """
    cm = pixel_confusion(pred, gt, valid)
    out = cm.to_dict()
    out["iou"] = round(iou(pred, gt), 4)
    return out


@dataclass
class FunnelCost:
    """Cost accounting for the review funnel.

    This is the repo's central claim, so it is measured rather than asserted:
    every figure derives from real token usage returned by the API.

    Two rejections are counted separately, because they are different claims.
    Tier 0 *refuses to judge* a pair whose imagery is unusable; the gate *judges*
    a pair and finds no change. Reporting one combined "filtered 64.6%" credited
    the gate with 87 co-registration failures it had no part in.
    """

    n_pairs: int = 0
    # Refused by Tier 0 quality checks: unusable imagery, not a gate decision.
    n_tier0_refused: int = 0
    # Judged by the gate and found unchanged. Denominator is the assessable set.
    n_gate_filtered: int = 0
    n_candidates: int = 0
    n_vlm_calls: int = 0
    n_batch_calls: int = 0
    n_errors: int = 0
    n_unpriced_calls: int = 0
    vlm_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    per_call_costs: list[float] = field(default_factory=list)
    # What each call would have cost at the synchronous rate. Batched calls bill
    # at half, so this is the only way to report a saving that is not simply the
    # candidate rate restated (see the note emitted in to_dict).
    per_call_sync_costs: list[float] = field(default_factory=list)

    @property
    def n_assessable(self) -> int:
        """Pairs Tier 0 was willing to judge."""
        return max(0, self.n_pairs - self.n_tier0_refused)

    @property
    def total_cost_usd(self) -> float:
        return self.vlm_cost_usd + self.llm_cost_usd

    @property
    def mean_cost_per_vlm_call(self) -> float:
        return (sum(self.per_call_costs) / len(self.per_call_costs)) if self.per_call_costs else 0.0

    @property
    def mean_sync_cost_per_vlm_call(self) -> float:
        costs = self.per_call_sync_costs or self.per_call_costs
        return (sum(costs) / len(costs)) if costs else 0.0

    def to_dict(self) -> dict[str, Any]:
        n = self.n_pairs or 1
        assessable = self.n_assessable or 1
        naive = self.mean_cost_per_vlm_call * self.n_pairs
        actual = self.total_cost_usd
        # What the funnel would cost if every gate candidate were verified,
        # rather than only the ones a budget cap allowed through.
        projected = self.mean_cost_per_vlm_call * self.n_candidates
        # The same candidates priced as if none had been batched. Independent of
        # the filter rate, so this is a real second measurement.
        projected_sync = self.mean_sync_cost_per_vlm_call * self.n_candidates
        return {
            "n_pairs": self.n_pairs,
            "n_errors": self.n_errors,
            # Three distinct reductions, never one blended number.
            "n_tier0_refused": self.n_tier0_refused,
            "tier0_refused_pct": round(100.0 * self.n_tier0_refused / n, 2),
            "n_assessable": self.n_assessable,
            "n_gate_filtered": self.n_gate_filtered,
            "gate_filtered_pct": round(100.0 * self.n_gate_filtered / assessable, 2),
            "total_reduction_pct": round(100.0 * (self.n_pairs - self.n_candidates) / n, 2),
            "n_vlm_calls": self.n_vlm_calls,
            "n_batch_calls": self.n_batch_calls,
            "vlm_call_rate_pct": round(100.0 * self.n_vlm_calls / n, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": {
                "vlm": round(self.vlm_cost_usd, 4),
                "llm": round(self.llm_cost_usd, 4),
                "total": round(actual, 4),
                "per_pair": round(actual / n, 6),
                "mean_per_vlm_call": round(self.mean_cost_per_vlm_call, 6),
                "mean_per_vlm_call_synchronous": round(self.mean_sync_cost_per_vlm_call, 6),
            },
            "counterfactual_review_everything_usd": round(naive, 4),
            # Attributable to the gate only. Spend actually incurred can be far
            # lower than this because of --max-vlm-calls, but a budget cap is a
            # truncated experiment, not a saving: crediting it to the gate would
            # report "97.5% saved" for a run that simply stopped early.
            "projected_cost_at_gate_rate_usd": round(projected, 4),
            "savings_vs_review_everything_usd": round(max(0.0, naive - projected), 4),
            "savings_pct": (round(100.0 * (1 - projected / naive), 2) if naive > 0 else 0.0),
            # Honest framing of the line above: with a single per-call price it is
            # algebraically 1 - n_candidates/n_pairs, i.e. the candidate rate under
            # another name. It carries independent information only once the funnel
            # has more than one price, which is what the batch figures below add.
            "savings_pct_note": (
                "savings_pct equals the share of pairs the funnel did not forward; "
                "with one flat per-call price it restates the candidate rate rather "
                "than measuring anything further. batch_saving_pct is independent."
            ),
            "projected_cost_synchronous_usd": round(projected_sync, 4),
            "batch_saving_usd": round(max(0.0, projected_sync - projected), 4),
            "batch_saving_pct": (
                round(100.0 * (1 - projected / projected_sync), 2) if projected_sync > 0 else 0.0
            ),
            "vlm_budget_capped": self.n_vlm_calls < self.n_candidates,
            "n_candidates": self.n_candidates,
            # When a model has no published rate every figure above understates
            # spend; say so rather than letting $0.00 read as free.
            "n_unpriced_calls": self.n_unpriced_calls,
            "cost_is_complete": self.n_unpriced_calls == 0,
        }
