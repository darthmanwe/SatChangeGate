"""Second-tier reporting: what the VLM added on top of the gate.

This module exists because the repo published a gate+VLM precision figure that
no command could reproduce. ``public_reporting_sample/_e2e_vlm_calls.json`` was
the only committed artifact with no producing code path — it had been derived by
hand from a gitignored JSONL. The arithmetic was right; the reproducibility was
not, and this repo trades on the second.

Everything here reads the ledger ``e2e`` already writes and computes nothing the
run did not record, so a published number and the command that produces it
cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satchangegate.metrics import wilson_interval

# A verdict of real_change is the VLM saying "forward this". Anything else --
# likely_artifact or uncertain -- is a rejection at the second tier. `uncertain`
# is counted as a rejection rather than dropped, because a candidate the model
# could not confirm is one an analyst still has to look at, and pretending it
# never happened would flatter the tier.
FORWARDING_VERDICT = "real_change"


@dataclass
class Proportion:
    """A count with the interval that makes it quotable."""

    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        lo, hi = wilson_interval(self.successes, self.total)
        return {
            "value": round(self.value, 4),
            "successes": self.successes,
            "total": self.total,
            "ci95": [round(lo, 4), round(hi, 4)],
        }


def load_calls(jsonl_path: Path) -> list[dict[str, Any]]:
    """Every row in an e2e ledger that actually reached the VLM."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"No e2e ledger at {jsonl_path}. Run `satchangegate e2e --split <split> --vlm` first."
        )
    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("vlm_called"):
            rows.append(row)
    return sorted(rows, key=lambda r: r["tile_id"])


def analyse(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate-alone vs gate+VLM precision over the verified subset."""
    labelled = [c for c in calls if c.get("label") is not None]
    forwarded = [c for c in labelled if c.get("vlm_verdict") == FORWARDING_VERDICT]
    gate_errors = [c for c in labelled if int(c["label"]) == 0]
    true_changes = [c for c in labelled if int(c["label"]) == 1]

    gate_alone = Proportion(sum(1 for c in labelled if int(c["label"]) == 1), len(labelled))
    gate_plus_vlm = Proportion(sum(1 for c in forwarded if int(c["label"]) == 1), len(forwarded))
    rejected_errors = Proportion(
        sum(1 for c in gate_errors if c.get("vlm_verdict") != FORWARDING_VERDICT),
        len(gate_errors),
    )
    retained_changes = Proportion(
        sum(1 for c in true_changes if c.get("vlm_verdict") == FORWARDING_VERDICT),
        len(true_changes),
    )

    verdicts: dict[str, int] = {}
    change_types: dict[str, int] = {}
    coverage: dict[str, int] = {}
    for c in calls:
        if c.get("vlm_verdict"):
            verdicts[c["vlm_verdict"]] = verdicts.get(c["vlm_verdict"], 0) + 1
        if c.get("vlm_change_type"):
            change_types[c["vlm_change_type"]] = change_types.get(c["vlm_change_type"], 0) + 1
        coverage[c["city"]] = coverage.get(c["city"], 0) + 1

    total_cost = sum(float(c.get("cost_usd") or 0.0) for c in calls)
    sync_cost = sum(float(c.get("cost_usd_synchronous") or c.get("cost_usd") or 0.0) for c in calls)
    return {
        "n_calls": len(calls),
        "n_errors": sum(1 for c in calls if c.get("error")),
        "n_batched": sum(1 for c in calls if c.get("batch")),
        "cities_verified": dict(sorted(coverage.items())),
        "precision": {
            "gate_alone": gate_alone.to_dict(),
            "gate_plus_vlm": gate_plus_vlm.to_dict(),
        },
        "vlm_rejected_gate_errors": rejected_errors.to_dict(),
        "vlm_retained_true_changes": retained_changes.to_dict(),
        "verdicts": dict(sorted(verdicts.items())),
        "change_types": dict(sorted(change_types.items())),
        "cost_usd": {
            "total": round(total_cost, 4),
            "per_call": round(total_cost / len(calls), 6) if calls else 0.0,
            "synchronous_equivalent": round(sync_cost, 4),
        },
    }


def run_vlm_report(
    split: str = "test",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Regenerate the second-tier report from the e2e ledger."""
    out_dir = Path(out_dir or Path("data/reports"))
    calls = load_calls(out_dir / f"_e2e_{split}.jsonl")
    if not calls:
        return {"error": "no VLM calls in the ledger", "n_calls": 0}

    summary = analyse(calls)
    payload = {
        "note": (
            f"Regenerated by `satchangegate vlm-report --split {split}` from "
            f"_e2e_{split}.jsonl. Every figure below is recomputed from the "
            "recorded per-call ledger; none is transcribed."
        ),
        "summary": summary,
        "calls": [
            {
                k: c.get(k)
                for k in (
                    "tile_id",
                    "city",
                    "label",
                    "gate",
                    "gate_confidence",
                    "vlm_verdict",
                    "vlm_change_type",
                    "vlm_confidence",
                    "batch",
                    "cost_usd",
                )
            }
            for c in calls
        ],
    }
    (out_dir / "_e2e_vlm_calls.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "_e2e_vlm_report.md").write_text(_render(split, summary), encoding="utf-8")
    return summary


def _row(label: str, p: dict[str, Any]) -> str:
    return (
        f"| {label} | {p['value']:.3f} ({p['successes']}/{p['total']}) | "
        f"{p['ci95'][0]:.3f}-{p['ci95'][1]:.3f} |"
    )


def _render(split: str, s: dict[str, Any]) -> str:
    prec = s["precision"]
    lines = [
        f"# What the VLM tier added — `{split}` split",
        "",
        f"{s['n_calls']} verification calls across {len(s['cities_verified'])} cities, "
        f"{s['n_errors']} errors" + (f", {s['n_batched']} batched" if s["n_batched"] else "") + ".",
        "",
        "| | Precision | 95% CI |",
        "|---|---|---|",
        _row("Gate alone", prec["gate_alone"]),
        _row("Gate + VLM", prec["gate_plus_vlm"]),
        "",
        "Both are measured on the verified subset only, which is the only set where "
        "the two tiers can be compared like for like. The gate-alone figure here is "
        "therefore not the split-wide gate precision — compare it against "
        f"`_eval_{split}.json` rather than substituting one for the other.",
        "",
        "| Second-tier behaviour | Rate | 95% CI |",
        "|---|---|---|",
        _row("Rejected the gate's errors", s["vlm_rejected_gate_errors"]),
        _row("Retained real change", s["vlm_retained_true_changes"]),
        "",
        "## Verification coverage",
        "",
        "| City | Calls |",
        "|---|---|",
    ]
    lines += [f"| {c} | {n} |" for c, n in s["cities_verified"].items()]
    if s["change_types"]:
        lines += ["", "## Change types reported", ""]
        lines += [f"- {k}: {v}" for k, v in s["change_types"].items()]
    cost = s["cost_usd"]
    lines += [
        "",
        "## Cost",
        "",
        f"${cost['total']} total, ${cost['per_call']} per verification.",
    ]
    if s["n_batched"]:
        lines.append(
            f"Priced synchronously the same calls would have cost "
            f"${cost['synchronous_equivalent']}."
        )
    return "\n".join(lines) + "\n"
