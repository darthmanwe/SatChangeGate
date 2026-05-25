"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from satchangegate.config import load_thresholds
from satchangegate.data.goes import GoesContext
from satchangegate.data.oscd import ImagePair, load_bands
from satchangegate.features.classical import classical_gate, save_heatmap_png
from satchangegate.preprocess.align import align_pair_bands
from satchangegate.preprocess.masks import compute_ephemeral_masks
from satchangegate.preprocess.quality import QualityScore, compute_quality_score
from satchangegate.report.llm import generate_analyst_report
from satchangegate.vlm.client import verify_candidate
from satchangegate.vlm.package import package_candidate_for_vlm
from satchangegate.vlm.schemas import ReportInput, VlmVerdict

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]
Layout = Literal["multispectral", "rgb_png"]


@dataclass
class PipelineResult:
    pair_id: str
    quality: QualityScore
    classical_gate: str
    vlm_called: bool
    vlm_verdict: VlmVerdict | None
    report_path: Path | None
    result_path: Path | None
    feature_row: dict[str, Any] = field(default_factory=dict)
    goes_context: dict[str, Any] | None = None
    negative_mode: str | None = None
    ground_truth_label: int | None = None


def _band_paths(pair_dir: Path, bands: list[str]) -> dict[str, str]:
    paths = {}
    for b in bands:
        p = pair_dir / f"{b}.tif"
        if p.is_file():
            paths[b] = str(p)
    return paths


def run_from_bands(
    run_id: str,
    raw_t1: dict[str, np.ndarray],
    raw_t2: dict[str, np.ndarray],
    *,
    cfg: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    layout: Layout = "rgb_png",
    date_t1: str = "t1",
    date_t2: str = "t2",
    split: str = "test",
    skip_vlm: bool = False,
    skip_llm: bool = False,
    api_key: str | None = None,
    goes_context: GoesContext | None = None,
    negative_mode: str | None = None,
    ground_truth_label: int | None = None,
) -> PipelineResult:
    """Run pipeline on pre-loaded band dictionaries."""
    cfg = cfg or load_thresholds()
    out_dir = Path(out_dir or Path("data/reports"))
    processed_dir = Path("data/processed") / run_id
    vlm_dir = Path("data/vlm_packages") / run_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    bands_t1, bands_t2 = align_pair_bands(raw_t1, raw_t2, None, None, cfg)

    masks_t1 = compute_ephemeral_masks(bands_t1, cfg, layout=layout)
    masks_t2 = compute_ephemeral_masks(bands_t2, cfg, layout=layout)
    from satchangegate.preprocess.masks import combine_pair_masks

    combined_masks = combine_pair_masks(masks_t1, masks_t2)
    quality = compute_quality_score(masks_t1, masks_t2, bands_t1, bands_t2, cfg)

    gate_art = classical_gate(
        pair_id=run_id,
        bands_t1=bands_t1,
        bands_t2=bands_t2,
        masks=combined_masks,
        quality=quality,
        cfg=cfg,
        date_t1=date_t1,
        date_t2=date_t2,
    )

    save_heatmap_png(gate_art.heatmap, processed_dir / "heatmap.png")
    (processed_dir / "quality.json").write_text(quality.model_dump_json(indent=2), encoding="utf-8")
    (processed_dir / "classical.json").write_text(
        gate_art.result.model_dump_json(indent=2), encoding="utf-8"
    )

    vlm_verdict: VlmVerdict | None = None
    vlm_called = False

    if gate_art.result.classical_gate == "candidate_change" and not skip_vlm:
        package_candidate_for_vlm(
            vlm_dir, bands_t1, bands_t2, gate_art.heatmap, quality, gate_art.result, cfg
        )
        if api_key or __import__("os").environ.get("ANTHROPIC_API_KEY"):
            vlm_called = True
            vlm_verdict = verify_candidate(vlm_dir, api_key=api_key)
            (vlm_dir / "verdict.json").write_text(
                vlm_verdict.model_dump_json(indent=2), encoding="utf-8"
            )

    report_md = ""
    if not skip_llm:
        report_md = generate_analyst_report(
            ReportInput.from_pipeline(quality, gate_art.result, vlm_verdict),
            api_key=api_key,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_id}.md"
    result_path = out_dir / f"{run_id}.result.json"
    if report_md:
        report_path.write_text(report_md, encoding="utf-8")

    goes_blob = None
    if goes_context is not None:
        goes_blob = {
            "satellite": goes_context.satellite,
            "timestamp": goes_context.timestamp,
            "product": goes_context.product,
            "domain": goes_context.domain,
            "brightness_mean": goes_context.brightness_mean,
            "local_path": str(goes_context.local_path) if goes_context.local_path else None,
        }

    result_blob = {
        "pair_id": run_id,
        "split": split,
        "negative_mode": negative_mode,
        "ground_truth_label": ground_truth_label,
        "quality": quality.model_dump(),
        "classical": gate_art.result.model_dump(),
        "vlm_called": vlm_called,
        "vlm": vlm_verdict.model_dump() if vlm_verdict else None,
        "goes_context": goes_blob,
        "report_path": str(report_path) if report_path.exists() else None,
    }
    result_path.write_text(json.dumps(result_blob, indent=2), encoding="utf-8")

    return PipelineResult(
        pair_id=run_id,
        quality=quality,
        classical_gate=gate_art.result.classical_gate,
        vlm_called=vlm_called,
        vlm_verdict=vlm_verdict,
        report_path=report_path if report_path.exists() else None,
        result_path=result_path,
        feature_row=gate_art.result.model_dump(),
        goes_context=goes_blob,
        negative_mode=negative_mode,
        ground_truth_label=ground_truth_label,
    )


def run_pair(
    pair: ImagePair,
    cfg: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    *,
    skip_vlm: bool = False,
    skip_llm: bool = False,
    api_key: str | None = None,
    negative_mode: str | None = None,
    goes_context: GoesContext | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Run full pipeline for one OSCD pair (optional negative-control overrides)."""
    cfg = cfg or load_thresholds()
    bands_list = cfg.get("bands", BAND_NAMES)
    suffix = ""

    if negative_mode:
        from satchangegate.data.negative_controls import prepare_negative_pair

        raw_t1, raw_t2, suffix = prepare_negative_pair(
            pair, negative_mode, bands_list  # type: ignore[arg-type]
        )
    else:
        raw_t1 = load_bands(pair.img1_dir, bands_list)
        raw_t2 = load_bands(pair.img2_dir, bands_list)

    rid = run_id or f"{pair.pair_id}{suffix}"

    return run_from_bands(
        rid,
        raw_t1,
        raw_t2,
        cfg=cfg,
        out_dir=out_dir,
        layout=pair.layout,
        date_t1=pair.date_t1 or "t1",
        date_t2=pair.date_t2 or "t2",
        split=pair.split,
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
        api_key=api_key,
        goes_context=goes_context,
        negative_mode=negative_mode,
    )
