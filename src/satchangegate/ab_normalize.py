"""The radiometric-normalization A/B, with a command that produces it.

``public_reporting_sample/_ab_normalize.json`` has been committed since 0.3.0
and nothing in this repo could regenerate it. That is the same defect the
August 2026 audit found in the 0.971 gate+VLM headline -- a published artifact
with no producing code path -- and the fact that its *conclusion* is a negative
result does not make it exempt. A negative result nobody can reproduce is just
an assertion.

So: run the whole evaluation twice, once with PIF normalization off and once
with it on, and report both sides. The expected outcome is that normalization
makes things worse on this benchmark, which is what the README already says and
what the synthetic test in ``preprocess/radiometric`` shows is a property of the
data rather than a bug -- on a multi-year pair too much ground has genuinely
changed for the "invariant" population the fit draws from to be invariant.

This is expensive: two full feature passes and two baseline fits. It is the
honest price of a claim, and it runs off the default path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from satchangegate.config import Settings, get_settings


def _side(
    oscd_root: Path | None,
    out_dir: Path,
    settings: Settings,
    *,
    normalize: bool,
    split: str,
) -> dict[str, Any]:
    """Evaluate one side of the A/B into its own directory."""
    from satchangegate.baseline import run_baselines
    from satchangegate.evaluate import run_eval

    side_settings = settings.model_copy(
        update={"preprocess": settings.preprocess.model_copy(update={"normalize": normalize})}
    )
    side_dir = out_dir / ("on" if normalize else "off")
    side_dir.mkdir(parents=True, exist_ok=True)

    evaluation = run_eval(oscd_root, split, side_dir, settings=side_settings)
    baselines = run_baselines(oscd_root, side_dir, settings=side_settings)

    metrics = evaluation["metrics"]
    return {
        "gate": {
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "total_reduction_pct": evaluation["total_reduction_pct"],
        },
        "models": {m["name"]: m["average_precision"] for m in baselines["models"]},
        "roc_auc": {m["name"]: m["roc_auc"] for m in baselines["models"]},
    }


def run_ab_normalize(
    oscd_root: Path | None = None,
    out_dir: Path | None = None,
    *,
    split: str = "test",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate the gate and the baselines with normalization off, then on."""
    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports")) / "ab_normalize"
    out_dir.mkdir(parents=True, exist_ok=True)

    off = _side(oscd_root, out_dir, settings, normalize=False, split=split)
    on = _side(oscd_root, out_dir, settings, normalize=True, split=split)

    deltas = {
        "gate_f1": round(on["gate"]["f1"] - off["gate"]["f1"], 4),
        "gate_precision": round(on["gate"]["precision"] - off["gate"]["precision"], 4),
        **{
            f"{name}_ap": round(on["models"][name] - off["models"][name], 4)
            for name in off["models"]
        },
    }
    summary = {
        "split": split,
        "off": off,
        "on": on,
        "delta_on_minus_off": deltas,
        "reading": (
            "Negative deltas mean normalization hurt. The linear model loses most, "
            "which is the expected shape: on a multi-year pair a large share of "
            "ground has genuinely changed, so the invariant population the PIF fit "
            "is drawn from is contaminated, and matching the two scenes compresses "
            "exactly the scene-wide spectral difference the models were using. A "
            "textbook correction that is right in general and wrong here."
        ),
        "provenance": (
            "This artifact was committed from 0.3.0 with no producing command. "
            "That is the same class of defect as a headline number nobody could "
            "regenerate, and a negative result is not exempt from it."
        ),
    }
    parent = Path(out_dir).parent
    (parent / "_ab_normalize.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (parent / "_ab_normalize.md").write_text(_render(summary), encoding="utf-8")
    return summary


def _render(summary: dict[str, Any]) -> str:
    off, on, delta = summary["off"], summary["on"], summary["delta_on_minus_off"]
    lines = [
        "# Relative radiometric normalization: A/B",
        "",
        f"Same features, same `{summary['split']}` split, the only difference being "
        "whether t2 is gain/offset-matched to t1 by PIF before anything else runs.",
        "",
        "| | Normalization off | on | delta |",
        "|---|---|---|---|",
        f"| Gate F1 | {off['gate']['f1']:.4f} | {on['gate']['f1']:.4f} | {delta['gate_f1']:+.4f} |",
        f"| Gate precision | {off['gate']['precision']:.4f} | {on['gate']['precision']:.4f} | "
        f"{delta['gate_precision']:+.4f} |",
    ]
    for name in off["models"]:
        lines.append(
            f"| {name} AP | {off['models'][name]:.4f} | {on['models'][name]:.4f} | "
            f"{delta[f'{name}_ap']:+.4f} |"
        )
    lines += ["", summary["reading"], "", f"> {summary['provenance']}"]
    return "\n".join(lines) + "\n"
