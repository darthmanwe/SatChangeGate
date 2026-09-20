"""Read-only access to generated artifacts, through a registry rather than paths.

Two rules shape this module.

**No generic file serving.** The UI never hands the server a path. It names a key
from a fixed registry, and the registry resolves it. A browser that can ask for
an arbitrary file is a browser that can ask for ``.env``, and this process holds
a live API key.

**Missing is a state, not an error.** ``data/`` is gitignored, so a fresh clone
has the committed ``public_reporting_sample/`` and nothing else -- no run ledger,
no feature matrix, no evidence packages. Every panel therefore has to be able to
say "not computed yet, here is the command", which means every artifact carries
the command that produces it.

The two sources are never mixed silently. A reader asks for a key and is told
which source answered, because half a local run stitched to half a committed
sample would be a number nobody could reproduce.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Kind = Literal["json", "markdown", "csv", "jsonl", "png"]
Source = Literal["local", "sample"]

DEFAULT_REPORTS = Path("data/reports")
DEFAULT_SAMPLE = Path("public_reporting_sample")
DEFAULT_MODELS = Path("data/models")

_TILE_ID = re.compile(r"^[a-z0-9_]{1,64}$")


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    filename: str
    kind: Kind
    title: str
    produced_by: str
    description: str
    #: False for things a fresh clone genuinely cannot have.
    in_sample: bool = True
    base: Literal["reports", "models"] = "reports"


ARTIFACTS: dict[str, ArtifactSpec] = {
    spec.key: spec
    for spec in (
        ArtifactSpec(
            "eval_test",
            "_eval_test.json",
            "json",
            "Held-out gate metrics",
            "satchangegate eval --split test",
            "Confusion matrix and Wilson intervals on the held-out split.",
        ),
        ArtifactSpec(
            "eval_test_md",
            "_eval_test.md",
            "markdown",
            "Held-out gate metrics (rendered)",
            "satchangegate eval --split test",
            "The same numbers as prose.",
        ),
        ArtifactSpec(
            "eval_train",
            "_eval_train.json",
            "json",
            "In-sample gate metrics",
            "satchangegate eval --split train",
            "Training-split metrics. In-sample; not the number to quote.",
        ),
        ArtifactSpec(
            "features_test",
            "_eval_test_features.csv",
            "csv",
            "Held-out feature matrix",
            "satchangegate eval --split test",
            "All 18 gate features per tile, plus label and decision.",
            in_sample=False,
        ),
        ArtifactSpec(
            "features_train",
            "_eval_train_features.csv",
            "csv",
            "Training feature matrix",
            "satchangegate eval --split train",
            "The development population for the threshold playground.",
            in_sample=False,
        ),
        ArtifactSpec(
            "e2e_test",
            "_e2e_test.json",
            "json",
            "Funnel summary and measured cost",
            "satchangegate e2e --split test --vlm --batch",
            "Stage counts, per-city VLM coverage, and the cost ledger.",
        ),
        ArtifactSpec(
            "e2e_ledger",
            "_e2e_test.jsonl",
            "jsonl",
            "Per-tile decision ledger",
            "satchangegate e2e --split test",
            "One row per tile: gate decision, reason, confidence, verdict, cost.",
            in_sample=False,
        ),
        ArtifactSpec(
            "vlm_calls",
            "_e2e_vlm_calls.json",
            "json",
            "Second-tier verdicts",
            "satchangegate vlm-report --split test",
            "The 100 verifications, with what the gate had said about each.",
        ),
        ArtifactSpec(
            "vlm_report_md",
            "_e2e_vlm_report.md",
            "markdown",
            "What the VLM tier added",
            "satchangegate vlm-report --split test",
            "Gate-alone against gate-plus-VLM precision, with intervals.",
        ),
        ArtifactSpec(
            "conformal",
            "_conformal.json",
            "json",
            "Risk-controlled threshold",
            "satchangegate conformal --alpha 0.20 --delta 0.10",
            "Lambda, its bound, and the per-city falsification.",
        ),
        ArtifactSpec(
            "conformal_md",
            "_conformal.md",
            "markdown",
            "Conformal report",
            "satchangegate conformal",
            "The guarantee and where it breaks.",
        ),
        ArtifactSpec(
            "operating_points",
            "_operating_points.json",
            "json",
            "Budget to operating point",
            "satchangegate operating-points --split test",
            "What each review budget buys, and what it leaves unspent.",
        ),
        ArtifactSpec(
            "baselines",
            "_baselines.json",
            "json",
            "Rule gate against learned models",
            "satchangegate baselines",
            "Full precision-recall curves for all three scorers.",
        ),
        ArtifactSpec(
            "pr_curves",
            "pr_curves.png",
            "png",
            "Precision-recall curves",
            "satchangegate baselines",
            "The rendered curve figure.",
        ),
        ArtifactSpec(
            "tuning_report",
            "_gate_tuning_report.json",
            "json",
            "Threshold sweep",
            "satchangegate tune --split train",
            "8,640 combinations, and what the sweep chose.",
        ),
        ArtifactSpec(
            "dev_tests",
            "_dev_tests.json",
            "json",
            "Offline control battery",
            "satchangegate dev-tests",
            "Five checks that fail loudly rather than by inspection.",
        ),
        ArtifactSpec(
            "embedding_coverage",
            "_embedding_coverage.json",
            "json",
            "AlphaEarth coverage",
            "satchangegate embedding-coverage",
            "How many pairs the embedding dataset can actually speak to.",
        ),
        ArtifactSpec(
            "ab_normalize",
            "_ab_normalize.json",
            "json",
            "Radiometric normalization A/B",
            "(no producing command yet)",
            "Committed comparison of normalization on and off. Its producing "
            "command is still missing, which is itself a known gap.",
        ),
        ArtifactSpec(
            "scorer_card",
            "gate_scorer.card.json",
            "json",
            "Learned scorer model card",
            "satchangegate fit-scorer --split train",
            "Feature list, training cities, held-out metrics and limitations.",
            base="models",
        ),
    )
}


class ArtifactMissing(LookupError):
    """The artifact is not on disk. Carries the command that would make it."""

    def __init__(self, spec: ArtifactSpec) -> None:
        super().__init__(
            f"{spec.title} has not been computed here. Produce it with: {spec.produced_by}"
        )
        self.spec = spec


@dataclass(frozen=True)
class Resolved:
    spec: ArtifactSpec
    path: Path
    source: Source


class ArtifactStore:
    """Resolves registry keys to files, local first, committed sample second."""

    def __init__(
        self,
        reports: Path | None = None,
        sample: Path | None = None,
        models: Path | None = None,
    ) -> None:
        self.reports = Path(reports or DEFAULT_REPORTS)
        self.sample = Path(sample or DEFAULT_SAMPLE)
        self.models = Path(models or DEFAULT_MODELS)

    # ---------------------------------------------------------------- resolve

    def _base(self, spec: ArtifactSpec) -> Path:
        return self.models if spec.base == "models" else self.reports

    def resolve(self, key: str) -> Resolved:
        spec = ARTIFACTS.get(key)
        if spec is None:
            raise ArtifactMissing(
                ArtifactSpec(key, "", "json", key, "(unknown artifact)", "Not in the registry.")
            )
        local = self._base(spec) / spec.filename
        if local.is_file():
            return Resolved(spec, local, "local")
        fallback = self.sample / spec.filename
        if spec.in_sample and fallback.is_file():
            return Resolved(spec, fallback, "sample")
        raise ArtifactMissing(spec)

    def status(self, key: str) -> dict[str, Any]:
        spec = ARTIFACTS[key]
        try:
            resolved = self.resolve(key)
        except ArtifactMissing:
            return {
                "key": key,
                "title": spec.title,
                "kind": spec.kind,
                "available": False,
                "source": None,
                "produced_by": spec.produced_by,
                "description": spec.description,
                "in_sample": spec.in_sample,
            }
        return {
            "key": key,
            "title": spec.title,
            "kind": spec.kind,
            "available": True,
            "source": resolved.source,
            "produced_by": spec.produced_by,
            "description": spec.description,
            "in_sample": spec.in_sample,
            "bytes": resolved.path.stat().st_size,
        }

    def catalogue(self) -> list[dict[str, Any]]:
        return [self.status(key) for key in ARTIFACTS]

    # ------------------------------------------------------------------- read

    def read_json(self, key: str) -> dict[str, Any]:
        resolved = self.resolve(key)
        data = json.loads(resolved.path.read_text(encoding="utf-8"))
        return {"source": resolved.source, "produced_by": resolved.spec.produced_by, "data": data}

    def read_text(self, key: str) -> dict[str, Any]:
        resolved = self.resolve(key)
        return {
            "source": resolved.source,
            "produced_by": resolved.spec.produced_by,
            "text": resolved.path.read_text(encoding="utf-8"),
        }

    def read_bytes(self, key: str) -> tuple[bytes, str]:
        resolved = self.resolve(key)
        media = {"png": "image/png"}.get(resolved.spec.kind, "application/octet-stream")
        return resolved.path.read_bytes(), media

    def read_ledger(self, key: str = "e2e_ledger") -> list[dict[str, Any]]:
        """One dict per JSONL row, skipping blanks and unparseable lines."""
        resolved = self.resolve(key)
        rows: list[dict[str, Any]] = []
        for line in resolved.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    # -------------------------------------------------------- evidence packages

    def evidence_root(self) -> Path:
        return self.reports / "e2e_packages"

    def evidence_tiles(self) -> list[str]:
        root = self.evidence_root()
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir() and _TILE_ID.match(p.name))

    def evidence_file(self, tile_id: str, name: str) -> tuple[bytes, str]:
        """One image or the redacted metadata from a package, by validated name.

        ``classical_full.json`` is deliberately unreachable. It is the
        *unredacted* copy that sits beside the metadata the model was shown, and
        the redaction is load-bearing: the VLM tier is the gate's independent
        verifier, so letting the gate's own features reach anything that rendered
        alongside a verdict would confound the agreement statistic the repo
        publishes.
        """
        allowed = {
            "before_rgb.png": "image/png",
            "after_rgb.png": "image/png",
            "change_overlay.png": "image/png",
            "change_heatmap.png": "image/png",
            "metadata.json": "application/json",
        }
        if name not in allowed:
            raise ArtifactMissing(
                ArtifactSpec(
                    name,
                    name,
                    "json",
                    name,
                    "(not served)",
                    "Only the evidence actually shown to the model is served.",
                )
            )
        if not _TILE_ID.match(tile_id):
            raise ArtifactMissing(
                ArtifactSpec(tile_id, "", "json", tile_id, "(invalid tile id)", "Rejected.")
            )
        path = self.evidence_root() / tile_id / name
        # Resolve and re-check: a validated name plus a validated id should make
        # escape impossible, but the check is cheap and the failure is a key leak.
        root = self.evidence_root().resolve()
        if not path.is_file() or root not in path.resolve().parents:
            raise ArtifactMissing(
                ArtifactSpec(
                    f"{tile_id}/{name}",
                    name,
                    "png",
                    f"{tile_id} {name}",
                    "satchangegate e2e --split test",
                    "Evidence packages are written by the funnel run.",
                )
            )
        return path.read_bytes(), allowed[name]
