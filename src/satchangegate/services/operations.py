"""Every operation the project exposes, as a validated request and a runner.

This is the single place both the CLI and the web API go through, which is what
makes the UI's "here is the command that produced this" honest rather than
decorative. Each entry also carries the metadata a caller needs *before* it runs
anything: whether it can spend money, roughly how long it takes, and what it
needs on disk. The capability endpoint and the console form are both generated
from that, so an operation cannot appear runnable in a UI while being unable to
run.

Runners return plain dictionaries. Presentation -- Rich tables, HTML, anything --
stays with the caller, because two renderers of one result is fine and two
computations of one result is the defect this layer exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from satchangegate.data.oscd import default_oscd_root
from satchangegate.services.base import ServiceRequest

Split = Literal["train", "test", "all"]
REPORTS = Path("data/reports")


# --------------------------------------------------------------------- result


@dataclass(frozen=True)
class ServiceResult:
    """What an operation produced, plus the command that reproduces it."""

    command: str
    data: dict[str, Any]
    artifacts: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "data": self.data,
            "artifacts": [str(p) for p in self.artifacts],
            "warnings": list(self.warnings),
        }


# -------------------------------------------------------------------- requests


class VerifyRequest(ServiceRequest):
    command_name: ClassVar[str] = "verify"
    cli_flags: ClassVar[dict[str, str]] = {"root": "--root"}
    root: Path = Field(default_factory=default_oscd_root)


class DownloadRequest(ServiceRequest):
    command_name: ClassVar[str] = "download-oscd"
    cli_flags: ClassVar[dict[str, str]] = {
        "out": "--out",
        "force": "--force",
        "keep_archives": "--keep-archives",
    }
    paired_bools: ClassVar[frozenset[str]] = frozenset({"force", "keep_archives"})
    out: Path = Field(default_factory=default_oscd_root)
    force: bool = False
    keep_archives: bool = False


class TilesRequest(ServiceRequest):
    command_name: ClassVar[str] = "tiles"
    cli_flags: ClassVar[dict[str, str]] = {
        "root": "--root",
        "out": "--out",
        "tile_size": "--tile-size",
        "stride": "--stride",
        "pos_min_fraction": "--pos-min-fraction",
    }
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS / "tile_index.json"
    tile_size: int = Field(64, ge=8, le=1024)
    # Exposed here and on the CLI together, because a UI-only knob would make the
    # displayed command a lie. Overlapping tiles are a different population, not
    # a rendering option -- see the plan's measurement notes.
    stride: int | None = Field(None, ge=1, le=1024)
    pos_min_fraction: float = Field(0.005, ge=0.0, le=1.0)


class RunPairRequest(ServiceRequest):
    command_name: ClassVar[str] = "run"
    cli_flags: ClassVar[dict[str, str]] = {
        "pair": "--pair",
        "root": "--root",
        "out": "--out",
        "vlm": "--vlm",
        "llm": "--llm",
        "model": "--model",
        "negative_mode": "--negative-mode",
    }
    paired_bools: ClassVar[frozenset[str]] = frozenset({"vlm", "llm"})
    pair: str
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    vlm: bool = False
    llm: bool = False
    model: str | None = None
    negative_mode: Literal["identity", "stable", "photometric"] | None = None


class EvalRequest(ServiceRequest):
    command_name: ClassVar[str] = "eval"
    cli_flags: ClassVar[dict[str, str]] = {
        "split": "--split",
        "root": "--root",
        "out": "--out",
        "pixel_metrics": "--pixel-metrics",
    }
    paired_bools: ClassVar[frozenset[str]] = frozenset({"pixel_metrics"})
    split: Split = "test"
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    pixel_metrics: bool = False


class TuneRequest(ServiceRequest):
    command_name: ClassVar[str] = "tune"
    cli_flags: ClassVar[dict[str, str]] = {"split": "--split", "root": "--root", "out": "--out"}
    split: Split = "train"
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS


class E2ERequest(ServiceRequest):
    command_name: ClassVar[str] = "e2e"
    cli_flags: ClassVar[dict[str, str]] = {
        "split": "--split",
        "n": "--n",
        "seed": "--seed",
        "root": "--root",
        "out": "--out",
        "vlm": "--vlm",
        "max_vlm_calls": "--max-vlm-calls",
        "sample": "--sample",
        "batch": "--batch",
        "resume": "--resume",
        "overwrite": "--overwrite",
    }
    paired_bools: ClassVar[frozenset[str]] = frozenset({"vlm", "batch", "resume", "overwrite"})
    split: Split = "test"
    n: int | None = Field(None, ge=1)
    seed: int = 42
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    vlm: bool = False
    max_vlm_calls: int | None = Field(None, ge=0)
    sample: Literal["stratified", "sequential"] = "stratified"
    batch: bool = False
    resume: bool = False
    overwrite: bool = False


class VlmReportRequest(ServiceRequest):
    command_name: ClassVar[str] = "vlm-report"
    cli_flags: ClassVar[dict[str, str]] = {"split": "--split", "out": "--out"}
    split: Split = "test"
    out: Path = REPORTS


class BaselinesRequest(ServiceRequest):
    command_name: ClassVar[str] = "baselines"
    cli_flags: ClassVar[dict[str, str]] = {"root": "--root", "out": "--out"}
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS


class FitScorerRequest(ServiceRequest):
    command_name: ClassVar[str] = "fit-scorer"
    cli_flags: ClassVar[dict[str, str]] = {"split": "--split", "root": "--root", "out": "--out"}
    split: Split = "train"
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = Path("data/models/gate_scorer.pkl")


class ConformalRequest(ServiceRequest):
    command_name: ClassVar[str] = "conformal"
    cli_flags: ClassVar[dict[str, str]] = {
        "root": "--root",
        "out": "--out",
        "alpha": "--alpha",
        "delta": "--delta",
    }
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    alpha: float = Field(0.20, gt=0.0, lt=1.0)
    delta: float = Field(0.10, gt=0.0, lt=1.0)


class OperatingPointsRequest(ServiceRequest):
    command_name: ClassVar[str] = "operating-points"
    cli_flags: ClassVar[dict[str, str]] = {
        "split": "--split",
        "root": "--root",
        "out": "--out",
        "budget_usd": "--budget-usd",
        "cost_per_call": "--cost-per-call",
    }
    split: Split = "test"
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    budget_usd: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00, 5.00)
    cost_per_call: float | None = Field(None, gt=0.0)


class EmbeddingCoverageRequest(ServiceRequest):
    command_name: ClassVar[str] = "embedding-coverage"
    cli_flags: ClassVar[dict[str, str]] = {"root": "--root", "out": "--out"}
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS


class DevTestsRequest(ServiceRequest):
    command_name: ClassVar[str] = "dev-tests"
    cli_flags: ClassVar[dict[str, str]] = {"root": "--root", "out": "--out", "city": "--city"}
    root: Path = Field(default_factory=default_oscd_root)
    out: Path = REPORTS
    city: str = "beirut"


# --------------------------------------------------------------------- runners


def _result(request: ServiceRequest, data: dict[str, Any], *artifacts: Path) -> ServiceResult:
    return ServiceResult(
        command=request.to_command(),
        data=data,
        artifacts=tuple(p for p in artifacts if p is not None),
    )


def run_verify(request: VerifyRequest) -> ServiceResult:
    from satchangegate.config import get_settings
    from satchangegate.data.oscd import verify_layout

    ok, message = verify_layout(request.root)
    settings = get_settings()
    return _result(
        request,
        {
            "ok": ok,
            "message": message,
            "root": str(request.root),
            "bands": list(settings.bands),
            "provenance": _provenance(),
        },
    )


def run_download(request: DownloadRequest) -> ServiceResult:
    from satchangegate.data.download import download_oscd
    from satchangegate.data.oscd import verify_layout

    root = download_oscd(request.out, force=request.force, keep_archives=request.keep_archives)
    ok, message = verify_layout(root)
    return _result(request, {"ok": ok, "message": message, "root": str(root)}, root)


def run_tiles(request: TilesRequest) -> ServiceResult:
    from satchangegate.data.tiles import build_tile_index, save_tile_index, summarise

    tiles = build_tile_index(
        request.root,
        tile_size=request.tile_size,
        stride=request.stride,
        pos_min_fraction=request.pos_min_fraction,
    )
    if not tiles:
        raise ServiceUnavailable("No tiles. Run: satchangegate download-oscd")
    save_tile_index(tiles, request.out)
    stride = request.stride or request.tile_size
    return _result(
        request,
        {
            "n_tiles": len(tiles),
            "tile_size": request.tile_size,
            "stride": stride,
            "overlapping": stride < request.tile_size,
            "by_split": summarise(tiles),
        },
        request.out,
    )


def run_pair_service(request: RunPairRequest) -> ServiceResult:
    from satchangegate.data.oscd import discover_pairs
    from satchangegate.pipeline import run_pair

    matches = [p for p in discover_pairs(request.root) if p.pair_id == request.pair]
    if not matches:
        raise ServiceUnavailable(
            f"Pair {request.pair!r} not found under {request.root}. "
            "Run `satchangegate download-oscd`, or name a city that exists."
        )
    result = run_pair(
        matches[0],
        out_dir=request.out,
        skip_vlm=not request.vlm,
        skip_llm=not request.llm,
        negative_mode=request.negative_mode,
        vlm_model=request.model,
    )
    return _result(
        request,
        {
            "pair": request.pair,
            "gate": result.classical.classical_gate,
            "classical": result.classical.model_dump(),
            "quality": result.quality.model_dump(),
            "vlm_called": result.vlm_called,
            "vlm_verdict": result.vlm_verdict.model_dump() if result.vlm_verdict else None,
            "llm_called": result.llm_called,
            "cost_usd": result.cost_usd,
            "error": result.error,
        },
        result.result_path,
    )


def run_eval_service(request: EvalRequest) -> ServiceResult:
    from satchangegate.evaluate import run_eval

    data = run_eval(
        request.root,
        split=request.split,
        out_dir=request.out,
        pixel_metrics=request.pixel_metrics,
    )
    return _result(
        request,
        data,
        request.out / f"_eval_{request.split}.json",
        request.out / f"_eval_{request.split}_features.csv",
    )


def run_tune_service(request: TuneRequest) -> ServiceResult:
    from satchangegate.tune_gate import sweep

    best = sweep(request.root, split=request.split, out_dir=request.out)
    return _result(
        request,
        {
            "in_sample_balanced_accuracy": round(best.score, 4),
            "thresholds": best.thresholds,
            "metrics": best.metrics,
            "note": "In-sample. Quote `eval --split test` instead.",
        },
        request.out / "_gate_tuning_report.json",
        request.out / "tuned_thresholds.yaml",
    )


def run_e2e_service(request: E2ERequest) -> ServiceResult:
    from satchangegate.e2e import E2EConfig, run_e2e

    data = run_e2e(
        request.root,
        request.out,
        config=E2EConfig(
            split=request.split,
            n=request.n,
            seed=request.seed,
            skip_vlm=not request.vlm,
            max_vlm_calls=request.max_vlm_calls,
            sample=request.sample,
            batch=request.batch,
            overwrite=request.overwrite,
        ),
        resume=request.resume,
    )
    return _result(
        request,
        data,
        request.out / f"_e2e_{request.split}.json",
        request.out / f"_e2e_{request.split}.jsonl",
    )


def run_vlm_report_service(request: VlmReportRequest) -> ServiceResult:
    from satchangegate.vlm_report import run_vlm_report

    data = run_vlm_report(split=request.split, out_dir=request.out)
    return _result(request, data, request.out / "_e2e_vlm_calls.json")


def run_baselines_service(request: BaselinesRequest) -> ServiceResult:
    from satchangegate.baseline import run_baselines

    data = run_baselines(request.root, request.out)
    return _result(request, data, request.out / "_baselines.json", request.out / "pr_curves.png")


def run_fit_scorer_service(request: FitScorerRequest) -> ServiceResult:
    """Fit and persist the learned scorer.

    The tiling, splitting, feature computation, leakage assertion and persistence
    all live here rather than in the CLI handler. They used to live in the
    handler, which meant a web form mapped onto ``scorer.fit_scorer`` would have
    reproduced none of it while still printing a `fit-scorer` command.
    """
    from satchangegate.config import get_settings
    from satchangegate.data.tiles import build_tile_index
    from satchangegate.evaluate import compute_tile_features
    from satchangegate.scorer import fit_scorer, save_scorer
    from satchangegate.tune_gate import assert_disjoint

    tiles = build_tile_index(request.root)
    fit_tiles = [t for t in tiles if t.split == request.split]
    held_tiles = [t for t in tiles if t.split != request.split]
    if not fit_tiles:
        raise ServiceUnavailable(f"No tiles in split {request.split!r}")
    assert_disjoint({t.city for t in fit_tiles}, {t.city for t in held_tiles})

    settings = get_settings()
    fit_rows = compute_tile_features(request.root, fit_tiles, settings)
    held_rows = compute_tile_features(request.root, held_tiles, settings) if held_tiles else None

    artifact = fit_scorer(fit_rows, held_rows)
    path = save_scorer(artifact, request.out)
    return _result(
        request,
        {
            "path": str(path),
            "card": artifact.card() if hasattr(artifact, "card") else None,
            "metrics": artifact.metrics,
            "n_fit_tiles": len(fit_tiles),
            "note": (
                "The rule gate remains the default. Set scorer.kind: learned in "
                "thresholds.yaml to use this, and read the model card first."
            ),
        },
        path,
        path.with_suffix(".card.json"),
    )


def run_conformal_service(request: ConformalRequest) -> ServiceResult:
    from satchangegate.conformal import run_conformal

    data = run_conformal(request.root, request.out, alpha=request.alpha, delta=request.delta)
    return _result(request, data, request.out / "_conformal.json")


def run_operating_points_service(request: OperatingPointsRequest) -> ServiceResult:
    from satchangegate.operating_points import run_operating_points

    data = run_operating_points(
        request.root,
        request.out,
        split=request.split,
        cost_per_call_usd=request.cost_per_call,
        budgets_usd=tuple(request.budget_usd),
    )
    return _result(request, data, request.out / "_operating_points.json")


def run_embedding_coverage_service(request: EmbeddingCoverageRequest) -> ServiceResult:
    import json

    from satchangegate.data.embeddings import coverage_report
    from satchangegate.data.oscd import discover_pairs

    data = coverage_report(discover_pairs(request.root))
    request.out.mkdir(parents=True, exist_ok=True)
    path = request.out / "_embedding_coverage.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return _result(request, data, path)


def run_dev_tests_service(request: DevTestsRequest) -> ServiceResult:
    from satchangegate.dev_controls import run_dev_tests

    data = run_dev_tests(request.root, request.out, city=request.city)
    return _result(request, data, request.out / "_dev_tests.json")


class ServiceUnavailable(RuntimeError):
    """The operation cannot run here, and says why in plain words."""


def _provenance() -> dict[str, Any]:
    """Version and settings fingerprint, for "which config produced this"."""
    from satchangegate.config import get_settings
    from satchangegate.pipeline import provenance

    return provenance(get_settings())


# -------------------------------------------------------------------- registry


@dataclass(frozen=True)
class ServiceSpec:
    """Everything a caller needs to decide whether to offer an operation."""

    name: str
    request_type: type[ServiceRequest]
    runner: Callable[[Any], ServiceResult]
    summary: str
    speed: Literal["instant", "seconds", "minutes", "long"]
    spends_money: bool = False
    needs_network: bool = False
    requires: tuple[str, ...] = field(default_factory=tuple)


SERVICES: dict[str, ServiceSpec] = {
    s.name: s
    for s in (
        ServiceSpec(
            "verify",
            VerifyRequest,
            run_verify,
            "Check the dataset layout and that config resolves.",
            "instant",
        ),
        ServiceSpec(
            "download-oscd",
            DownloadRequest,
            run_download,
            "Fetch and checksum-verify the 13-band OSCD dataset (~513 MB).",
            "long",
            needs_network=True,
        ),
        ServiceSpec(
            "tiles",
            TilesRequest,
            run_tiles,
            "Build the labelled tile index and report class balance.",
            "seconds",
            requires=("dataset",),
        ),
        ServiceSpec(
            "run",
            RunPairRequest,
            run_pair_service,
            "Run one OSCD city pair through the whole funnel.",
            "seconds",
            spends_money=True,
            needs_network=True,
            requires=("dataset",),
        ),
        ServiceSpec(
            "eval",
            EvalRequest,
            run_eval_service,
            "Score the gate on a labelled split, with Wilson intervals.",
            "minutes",
            requires=("dataset",),
        ),
        ServiceSpec(
            "tune",
            TuneRequest,
            run_tune_service,
            "Sweep gate thresholds on one split; the other is held out.",
            "long",
            requires=("dataset",),
        ),
        ServiceSpec(
            "e2e",
            E2ERequest,
            run_e2e_service,
            "Run the funnel over a split and measure its cost.",
            "long",
            spends_money=True,
            needs_network=True,
            requires=("dataset",),
        ),
        ServiceSpec(
            "vlm-report",
            VlmReportRequest,
            run_vlm_report_service,
            "Recompute the second-tier figures from the run ledger. No API calls.",
            "instant",
            requires=("ledger",),
        ),
        ServiceSpec(
            "baselines",
            BaselinesRequest,
            run_baselines_service,
            "Compare the rule gate against learned models on identical features.",
            "long",
            requires=("dataset", "baseline-extra"),
        ),
        ServiceSpec(
            "fit-scorer",
            FitScorerRequest,
            run_fit_scorer_service,
            "Fit and persist the learned scorer, with a model card.",
            "long",
            requires=("dataset", "baseline-extra"),
        ),
        ServiceSpec(
            "conformal",
            ConformalRequest,
            run_conformal_service,
            "Risk-controlled threshold with the calibration cities withheld, then falsify it.",
            "long",
            requires=("dataset",),
        ),
        ServiceSpec(
            "operating-points",
            OperatingPointsRequest,
            run_operating_points_service,
            "What each review budget buys.",
            "minutes",
            requires=("dataset",),
        ),
        ServiceSpec(
            "embedding-coverage",
            EmbeddingCoverageRequest,
            run_embedding_coverage_service,
            "How much of this benchmark AlphaEarth can speak to.",
            "instant",
            requires=("dataset",),
        ),
        ServiceSpec(
            "dev-tests",
            DevTestsRequest,
            run_dev_tests_service,
            "Offline control battery.",
            "seconds",
            requires=("dataset",),
        ),
    )
}


def run_service(name: str, payload: dict[str, Any] | None = None) -> ServiceResult:
    """Validate a payload against an operation and run it."""
    spec = SERVICES.get(name)
    if spec is None:
        raise ServiceUnavailable(f"Unknown operation {name!r}")
    request = spec.request_type(**(payload or {}))
    return spec.runner(request)
