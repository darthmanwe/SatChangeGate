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

from satchangegate.config import S2_REFLECTANCE_SCALE
from satchangegate.data.oscd import default_oscd_root
from satchangegate.services.base import ServiceRequest

Split = Literal["train", "test", "all"]

#: A durable progress sink: ``report(kind, **payload)``. Optional everywhere,
#: so a caller never needs to know which operations can report and which
#: cannot -- an operation that has nothing to say simply does not call it.
ProgressFn = Callable[..., None]
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


class RunImagesRequest(ServiceRequest):
    """Run the funnel over two image files under a declared contract.

    Exists on the CLI as well as the API for a reason the plan is strict about:
    a web-only knob makes the displayed command a lie. An uploaded pair that
    cannot be reproduced from a command line is a result nobody can check.
    """

    command_name: ClassVar[str] = "run-images"
    cli_flags: ClassVar[dict[str, str]] = {
        "t1": "--t1",
        "t2": "--t2",
        "bands": "--bands",
        "reflectance_scale": "--reflectance-scale",
        "reflectance_offset": "--reflectance-offset",
        "date_t1": "--date-t1",
        "date_t2": "--date-t2",
        "alignment": "--alignment",
        "resolution_m": "--resolution-m",
        "name": "--name",
        "out": "--out",
        "vlm": "--vlm",
    }
    paired_bools: ClassVar[frozenset[str]] = frozenset({"vlm"})

    t1: Path
    t2: Path
    #: "B04=1,B03=2,B02=3" -- the band each raster index carries. Named rather
    #: than positional, because band 4 of an arbitrary GeoTIFF is not red.
    bands: str
    reflectance_scale: float = Field(S2_REFLECTANCE_SCALE, gt=0)
    reflectance_offset: float = 0.0
    date_t1: str | None = None
    date_t2: str | None = None
    alignment: Literal["georeferenced", "already_aligned"] = "georeferenced"
    resolution_m: float | None = Field(None, gt=0)
    name: str = "upload"
    out: Path = REPORTS
    vlm: bool = False

    def band_map(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for chunk in str(self.bands).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Bad band mapping {chunk!r}; expected NAME=index.")
            name, _, index = chunk.partition("=")
            out[name.strip()] = int(index)
        if not out:
            raise ValueError("No bands declared.")
        return out


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


def run_verify(request: VerifyRequest, __report: ProgressFn | None = None) -> ServiceResult:
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


def run_download(request: DownloadRequest, __report: ProgressFn | None = None) -> ServiceResult:
    from satchangegate.data.download import download_oscd
    from satchangegate.data.oscd import verify_layout

    root = download_oscd(request.out, force=request.force, keep_archives=request.keep_archives)
    ok, message = verify_layout(root)
    return _result(request, {"ok": ok, "message": message, "root": str(root)}, root)


def run_tiles(request: TilesRequest, __report: ProgressFn | None = None) -> ServiceResult:
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


def run_pair_service(request: RunPairRequest, __report: ProgressFn | None = None) -> ServiceResult:
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


def run_images_service(
    request: RunImagesRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    """Verify a declared image pair, then run whichever lane it supports.

    Two lanes, and the difference is not a quality setting. A source carrying
    near-infrared and short-wave infrared gets the gate. One that does not gets
    structural evidence and an explicit refusal, because every gate threshold in
    this repo is defined over indices that RGB cannot express.
    """
    from satchangegate.config import get_settings
    from satchangegate.features.classical import rgb_only_gate
    from satchangegate.pipeline import run_from_bands
    from satchangegate.preprocess.align import estimate_registration_error
    from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
    from satchangegate.preprocess.quality import compute_quality_score
    from satchangegate.webui.ingest import Declaration, IngestRefused, read_pair

    declaration = Declaration(
        band_map=request.band_map(),
        reflectance_scale=request.reflectance_scale,
        reflectance_offset=request.reflectance_offset,
        date_t1=request.date_t1,
        date_t2=request.date_t2,
        alignment=request.alignment,
        target_resolution_m=request.resolution_m,
        source_name=request.name,
    )
    try:
        pair = read_pair(request.t1, request.t2, declaration)
    except IngestRefused as exc:
        raise ServiceUnavailable(str(exc)) from None

    settings = get_settings()
    date_t1 = request.date_t1 or "t1"
    date_t2 = request.date_t2 or "t2"

    if pair.capability.lane == "full":
        result = run_from_bands(
            request.name,
            pair.bands_t1,
            pair.bands_t2,
            settings=settings,
            out_dir=request.out,
            date_t1=date_t1,
            date_t2=date_t2,
            source="upload",
            skip_vlm=not request.vlm,
        )
        data = {
            "lane": "full",
            "ingest": pair.to_dict(),
            "gate": result.classical.classical_gate,
            "classical": result.classical.model_dump(),
            "quality": result.quality.model_dump(),
            "vlm_called": result.vlm_called,
            "vlm_verdict": result.vlm_verdict.model_dump() if result.vlm_verdict else None,
            "cost_usd": result.cost_usd,
            "error": result.error,
        }
        return _result(request, data, result.result_path)

    # Degraded lane. The masks tier needs bands this source lacks too, so it
    # reports itself unassessed rather than clean.
    masks_t1 = compute_ephemeral_masks(pair.bands_t1, settings.masks)
    masks_t2 = compute_ephemeral_masks(pair.bands_t2, settings.masks)
    combined = combine_pair_masks(masks_t1, masks_t2)
    combined.valid[~pair.valid] = False
    registration = estimate_registration_error(pair.bands_t1, pair.bands_t2)
    quality = compute_quality_score(
        masks_t1,
        masks_t2,
        settings.quality,
        registration_error_px=registration,
        date_t1=date_t1,
        date_t2=date_t2,
        combined=combined,
    )
    artifacts = rgb_only_gate(
        request.name,
        pair.bands_t1,
        pair.bands_t2,
        combined,
        quality,
        settings.gate,
        date_t1=date_t1,
        date_t2=date_t2,
        source="upload",
    )
    return _result(
        request,
        {
            "lane": "rgb_only",
            "ingest": pair.to_dict(),
            "gate": artifacts.result.classical_gate,
            "structural": artifacts.result.model_dump(),
            "quality": quality.model_dump(),
            "vlm_called": False,
            "note": (
                "No gate decision was produced and none could be. The evidence "
                "above is structural only; a verification request from here is "
                "recorded as a manual review, outside the funnel's metrics."
            ),
        },
    )


def run_eval_service(request: EvalRequest, __report: ProgressFn | None = None) -> ServiceResult:
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


def run_tune_service(request: TuneRequest, __report: ProgressFn | None = None) -> ServiceResult:
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


def run_e2e_service(request: E2ERequest, report: ProgressFn | None = None) -> ServiceResult:
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
        report=report,
    )
    return _result(
        request,
        data,
        request.out / f"_e2e_{request.split}.json",
        request.out / f"_e2e_{request.split}.jsonl",
    )


def run_vlm_report_service(
    request: VlmReportRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    from satchangegate.vlm_report import run_vlm_report

    data = run_vlm_report(split=request.split, out_dir=request.out)
    return _result(request, data, request.out / "_e2e_vlm_calls.json")


def run_baselines_service(
    request: BaselinesRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    from satchangegate.baseline import run_baselines

    data = run_baselines(request.root, request.out)
    return _result(request, data, request.out / "_baselines.json", request.out / "pr_curves.png")


def run_fit_scorer_service(
    request: FitScorerRequest, _report: ProgressFn | None = None
) -> ServiceResult:
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


def run_conformal_service(
    request: ConformalRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    from satchangegate.conformal import run_conformal

    data = run_conformal(request.root, request.out, alpha=request.alpha, delta=request.delta)
    return _result(request, data, request.out / "_conformal.json")


def run_operating_points_service(
    request: OperatingPointsRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    from satchangegate.operating_points import run_operating_points

    data = run_operating_points(
        request.root,
        request.out,
        split=request.split,
        cost_per_call_usd=request.cost_per_call,
        budgets_usd=tuple(request.budget_usd),
    )
    return _result(request, data, request.out / "_operating_points.json")


def run_embedding_coverage_service(
    request: EmbeddingCoverageRequest, _report: ProgressFn | None = None
) -> ServiceResult:
    import json

    from satchangegate.data.embeddings import coverage_report
    from satchangegate.data.oscd import discover_pairs

    data = coverage_report(discover_pairs(request.root))
    request.out.mkdir(parents=True, exist_ok=True)
    path = request.out / "_embedding_coverage.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return _result(request, data, path)


def run_dev_tests_service(
    request: DevTestsRequest, _report: ProgressFn | None = None
) -> ServiceResult:
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
    runner: Callable[..., ServiceResult]
    summary: str
    speed: Literal["instant", "seconds", "minutes", "long"]
    spends_money: bool = False
    needs_network: bool = False
    requires: tuple[str, ...] = field(default_factory=tuple)
    #: Whether a UI run may redirect this operation's ``out`` into its own
    #: directory. True for anything that writes a report; false for
    #: ``download-oscd``, whose ``out`` is where the *dataset* lives rather than
    #: where a result goes -- isolating that would re-download 513 MB per run.
    isolate_out: bool = True
    #: Request fields that, when set, are what actually causes spend.
    #: ``spends_money`` says an operation *can* spend; this says whether a
    #: particular request *will*. Without the distinction, `e2e --no-vlm` -- which
    #: costs nothing but CPU and is the most useful thing to run -- gets blocked
    #: alongside the paid path.
    spend_fields: tuple[str, ...] = field(default_factory=tuple)

    def request_spends(self, request: ServiceRequest) -> bool:
        """Whether this specific request would spend money."""
        if not self.spends_money:
            return False
        if not self.spend_fields:
            return True
        return any(bool(getattr(request, name, False)) for name in self.spend_fields)


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
            isolate_out=False,
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
            spend_fields=("vlm", "llm"),
        ),
        ServiceSpec(
            "run-images",
            RunImagesRequest,
            run_images_service,
            "Run the funnel over two declared image files.",
            "seconds",
            spends_money=True,
            needs_network=True,
            spend_fields=("vlm",),
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
            spend_fields=("vlm",),
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


def run_service(
    name: str,
    payload: dict[str, Any] | None = None,
    report: ProgressFn | None = None,
) -> ServiceResult:
    """Validate a payload against an operation and run it."""
    spec = SERVICES.get(name)
    if spec is None:
        raise ServiceUnavailable(f"Unknown operation {name!r}")
    request = spec.request_type(**(payload or {}))
    return spec.runner(request, report)
