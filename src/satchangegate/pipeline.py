"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satchangegate.config import load_thresholds
from satchangegate.data.oscd import ImagePair, load_bands, load_label_mask
from satchangegate.features.classical import classical_gate, save_heatmap_png
from satchangegate.preprocess.align import align_pair_bands
from satchangegate.preprocess.masks import compute_ephemeral_masks
from satchangegate.preprocess.quality import QualityScore, compute_quality_score
from satchangegate.report.llm import generate_analyst_report
from satchangegate.vlm.client import verify_candidate
from satchangegate.vlm.package import package_candidate_for_vlm
from satchangegate.vlm.schemas import ReportInput, VlmVerdict

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]


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


def _band_paths(pair_dir: Path, bands: list[str]) -> dict[str, str]:
    paths = {}
    for b in bands:
        p = pair_dir / f"{b}.tif"
        if p.is_file():
            paths[b] = str(p)
    return paths


def run_pair(
    pair: ImagePair,
    cfg: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    *,
    skip_vlm: bool = False,
    skip_llm: bool = False,
    api_key: str | None = None,
) -> PipelineResult:
    """Run full pipeline for one OSCD pair."""
    cfg = cfg or load_thresholds()
    out_dir = Path(out_dir or Path("data/reports"))
    processed_dir = Path("data/processed") / pair.pair_id
    vlm_dir = Path("data/vlm_packages") / pair.pair_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    bands_list = cfg.get("bands", BAND_NAMES)
    raw_t1 = load_bands(pair.img1_dir, bands_list)
    raw_t2 = load_bands(pair.img2_dir, bands_list)

    paths_t1 = _band_paths(pair.img1_dir, bands_list) if pair.img1_dir.is_dir() else {}
    paths_t2 = _band_paths(pair.img2_dir, bands_list) if pair.img2_dir.is_dir() else {}
    bands_t1, bands_t2 = align_pair_bands(
        raw_t1, raw_t2, paths_t1, paths_t2, cfg
    )

    layout = pair.layout
    masks_t1 = compute_ephemeral_masks(bands_t1, cfg, layout=layout)
    masks_t2 = compute_ephemeral_masks(bands_t2, cfg, layout=layout)
    from satchangegate.preprocess.masks import combine_pair_masks

    combined_masks = combine_pair_masks(masks_t1, masks_t2)
    quality = compute_quality_score(masks_t1, masks_t2, bands_t1, bands_t2, cfg)

    gate_art = classical_gate(
        pair_id=pair.pair_id,
        bands_t1=bands_t1,
        bands_t2=bands_t2,
        masks=combined_masks,
        quality=quality,
        cfg=cfg,
        date_t1=pair.date_t1 or "t1",
        date_t2=pair.date_t2 or "t2",
    )

    heatmap_path = processed_dir / "heatmap.png"
    save_heatmap_png(gate_art.heatmap, heatmap_path)

    quality_path = processed_dir / "quality.json"
    quality_path.write_text(quality.model_dump_json(indent=2), encoding="utf-8")

    classical_path = processed_dir / "classical.json"
    classical_path.write_text(
        gate_art.result.model_dump_json(indent=2), encoding="utf-8"
    )

    vlm_verdict: VlmVerdict | None = None
    vlm_called = False

    if gate_art.result.classical_gate == "candidate_change" and not skip_vlm:
        package_candidate_for_vlm(
            vlm_dir,
            bands_t1,
            bands_t2,
            gate_art.heatmap,
            quality,
            gate_art.result,
            cfg,
        )
        if api_key or __import__("os").environ.get("ANTHROPIC_API_KEY"):
            vlm_called = True
            vlm_verdict = verify_candidate(vlm_dir, api_key=api_key)
            (vlm_dir / "verdict.json").write_text(
                vlm_verdict.model_dump_json(indent=2), encoding="utf-8"
            )

    report_input = ReportInput.from_pipeline(quality, gate_art.result, vlm_verdict)
    report_md = generate_analyst_report(report_input, api_key=api_key) if not skip_llm else ""

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{pair.pair_id}.md"
    result_path = out_dir / f"{pair.pair_id}.result.json"

    if report_md:
        report_path.write_text(report_md, encoding="utf-8")

    result_blob = {
        "pair_id": pair.pair_id,
        "split": pair.split,
        "quality": quality.model_dump(),
        "classical": gate_art.result.model_dump(),
        "vlm_called": vlm_called,
        "vlm": vlm_verdict.model_dump() if vlm_verdict else None,
        "report_path": str(report_path) if report_path.exists() else None,
    }
    result_path.write_text(json.dumps(result_blob, indent=2), encoding="utf-8")

    return PipelineResult(
        pair_id=pair.pair_id,
        quality=quality,
        classical_gate=gate_art.result.classical_gate,
        vlm_called=vlm_called,
        vlm_verdict=vlm_verdict,
        report_path=report_path if report_path.exists() else None,
        result_path=result_path,
        feature_row=gate_art.result.model_dump(),
    )
