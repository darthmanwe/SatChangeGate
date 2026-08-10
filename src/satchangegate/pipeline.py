"""End-to-end pipeline: preprocess -> Tier 0 quality -> classical gate -> VLM -> report.

All artifacts for a run live under ``out_dir/<run_id>/``. Previously the
processed rasters and VLM packages were written to CWD-relative
``data/processed`` and ``data/vlm_packages`` regardless of ``out_dir``, so
parallel runs collided and the test suite wrote into the repository root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from satchangegate.config import Settings, get_settings
from satchangegate.data.goes import GoesContext
from satchangegate.data.oscd import ImagePair, load_bands
from satchangegate.features.classical import ClassicalResult, classical_gate, save_heatmap_png
from satchangegate.preprocess.align import (
    estimate_registration_error,
    resample_to_common_grid,
)
from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
from satchangegate.preprocess.quality import QualityScore, compute_quality_score
from satchangegate.report.llm import generate_analyst_report
from satchangegate.vlm.client import UsageRecord, verify_candidate
from satchangegate.vlm.package import package_candidate_for_vlm
from satchangegate.vlm.schemas import ReportInput, VlmVerdict


@dataclass
class PipelineResult:
    pair_id: str
    quality: QualityScore
    classical: ClassicalResult
    classical_gate: str
    vlm_called: bool
    vlm_verdict: VlmVerdict | None
    report_path: Path | None
    result_path: Path
    feature_row: dict[str, Any]
    goes_context: dict[str, Any] | None = None
    negative_mode: str | None = None
    ground_truth_label: int | None = None
    llm_called: bool = False
    usage: list[UsageRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usage), 6)


def run_from_bands(
    run_id: str,
    raw_t1: dict[str, np.ndarray],
    raw_t2: dict[str, np.ndarray],
    *,
    settings: Settings | None = None,
    out_dir: Path | None = None,
    date_t1: str = "t1",
    date_t2: str = "t2",
    split: str | None = None,
    source: str = "oscd",
    rgb_only: bool = False,
    skip_vlm: bool = True,
    skip_llm: bool = True,
    api_key: str | None = None,
    goes_context: GoesContext | None = None,
    negative_mode: str | None = None,
    ground_truth_label: int | None = None,
    vlm_model: str | None = None,
    llm_model: str | None = None,
) -> PipelineResult:
    """Run the funnel over one already-loaded bitemporal pair."""
    settings = settings or get_settings()
    out_dir = Path(out_dir or Path("data/reports"))
    run_dir = out_dir / run_id
    processed_dir = run_dir / "processed"
    vlm_dir = run_dir / "vlm_package"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bands_t1, bands_t2 = resample_to_common_grid(raw_t1, raw_t2)
    registration_px = estimate_registration_error(bands_t1, bands_t2)

    masks_t1 = compute_ephemeral_masks(bands_t1, settings.masks)
    masks_t2 = compute_ephemeral_masks(bands_t2, settings.masks)
    combined = combine_pair_masks(masks_t1, masks_t2)

    quality = compute_quality_score(
        masks_t1,
        masks_t2,
        settings.quality,
        registration_error_px=registration_px,
        date_t1=date_t1,
        date_t2=date_t2,
        combined=combined,
    )

    gate_art = classical_gate(
        run_id,
        bands_t1,
        bands_t2,
        combined,
        quality,
        settings.gate,
        date_t1=date_t1,
        date_t2=date_t2,
        source=source,
        rgb_only=rgb_only,
    )

    save_heatmap_png(gate_art.heatmap, processed_dir / "heatmap.png")
    (processed_dir / "quality.json").write_text(quality.model_dump_json(indent=2), encoding="utf-8")
    (processed_dir / "classical.json").write_text(
        gate_art.result.model_dump_json(indent=2), encoding="utf-8"
    )

    usage: list[UsageRecord] = []
    vlm_verdict: VlmVerdict | None = None
    vlm_called = False
    error: str | None = None

    if gate_art.result.classical_gate == "candidate_change" and not skip_vlm:
        package_candidate_for_vlm(
            vlm_dir,
            bands_t1,
            bands_t2,
            gate_art.heatmap,
            quality,
            gate_art.result,
            rgb_bands=settings.rgb_bands,
        )
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            vlm_called = True
            try:
                vlm_verdict, record = verify_candidate(vlm_dir, api_key=api_key, model=vlm_model)
                usage.append(record)
                (vlm_dir / "verdict.json").write_text(
                    vlm_verdict.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception as exc:  # degrade, never kill a batch
                error = f"vlm: {type(exc).__name__}: {exc}"

    report_md = ""
    llm_called = False
    if not skip_llm:
        # Degrade like the VLM path above. An unguarded call here would throw
        # away the quality score, the classical result, and an already-paid-for
        # VLM verdict before result.json is written.
        try:
            report_md, llm_called, llm_record = generate_analyst_report(
                ReportInput.from_pipeline(quality, gate_art.result, vlm_verdict),
                api_key=api_key,
                model=llm_model,
            )
            if llm_record is not None:
                usage.append(llm_record)
        except Exception as exc:
            error = "; ".join(filter(None, [error, f"llm: {type(exc).__name__}: {exc}"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.md"
    result_path = run_dir / "result.json"
    # Only claim a report when this run actually produced one; the previous
    # implementation reported a stale file left by an earlier run.
    wrote_report = bool(report_md)
    if wrote_report:
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
        "source": source,
        "negative_mode": negative_mode,
        "ground_truth_label": ground_truth_label,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(settings),
        "quality": quality.model_dump(),
        "classical": gate_art.result.model_dump(),
        "vlm_called": vlm_called,
        "llm_called": llm_called,
        "vlm": vlm_verdict.model_dump() if vlm_verdict else None,
        "usage": [u.model_dump() for u in usage],
        "cost_usd": round(sum(u.cost_usd for u in usage), 6),
        "goes_context": goes_blob,
        "report_path": str(report_path) if wrote_report else None,
        "error": error,
    }
    result_path.write_text(json.dumps(result_blob, indent=2), encoding="utf-8")

    return PipelineResult(
        pair_id=run_id,
        quality=quality,
        classical=gate_art.result,
        classical_gate=gate_art.result.classical_gate,
        vlm_called=vlm_called,
        vlm_verdict=vlm_verdict,
        report_path=report_path if wrote_report else None,
        result_path=result_path,
        feature_row=gate_art.result.model_dump(),
        goes_context=goes_blob,
        negative_mode=negative_mode,
        ground_truth_label=ground_truth_label,
        llm_called=llm_called,
        usage=usage,
        error=error,
    )


def provenance(settings: Settings) -> dict[str, Any]:
    """Version and config fingerprint recorded with every result."""
    import hashlib
    import platform

    import satchangegate

    blob = settings.model_dump_json().encode("utf-8")
    return {
        "satchangegate_version": getattr(satchangegate, "__version__", "unknown"),
        "python": platform.python_version(),
        "config_sha256": hashlib.sha256(blob).hexdigest()[:16],
    }


def run_pair(
    pair: ImagePair,
    settings: Settings | None = None,
    out_dir: Path | None = None,
    *,
    skip_vlm: bool = True,
    skip_llm: bool = True,
    api_key: str | None = None,
    negative_mode: str | None = None,
    goes_context: GoesContext | None = None,
    run_id: str | None = None,
    ground_truth_label: int | None = None,
    vlm_model: str | None = None,
    llm_model: str | None = None,
) -> PipelineResult:
    """Run the full pipeline for one OSCD pair."""
    settings = settings or get_settings()
    bands_list = list(settings.bands)
    suffix = ""

    if negative_mode:
        from satchangegate.data.negative_controls import prepare_negative_pair

        raw_t1, raw_t2, suffix = prepare_negative_pair(
            pair,
            negative_mode,  # type: ignore[arg-type]
            bands_list,
        )
    else:
        raw_t1 = load_bands(pair.img1_dir, bands_list)
        raw_t2 = load_bands(pair.img2_dir, bands_list)

    return run_from_bands(
        run_id or f"{pair.pair_id}{suffix}",
        raw_t1,
        raw_t2,
        settings=settings,
        out_dir=out_dir,
        date_t1=pair.date_t1 or "t1",
        date_t2=pair.date_t2 or "t2",
        split=pair.split,
        source="oscd",
        skip_vlm=skip_vlm,
        skip_llm=skip_llm,
        api_key=api_key,
        goes_context=goes_context,
        negative_mode=negative_mode,
        ground_truth_label=ground_truth_label,
        vlm_model=vlm_model,
        llm_model=llm_model,
    )
