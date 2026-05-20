"""Anthropic text-only LLM analyst report."""

from __future__ import annotations

import json
import os

from anthropic import Anthropic

from satchangegate.vlm.schemas import ReportInput

DEFAULT_LLM_MODEL = os.environ.get("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-20250514")

REPORT_SYSTEM = """You are a geospatial analyst writing a concise change-detection report.
Use only the structured JSON evidence provided — you have not seen the images.
Write markdown with these sections:
## AOI
## Date range
## Overall verdict
## Change type
## Confidence
## Evidence from indices
## Evidence from model
## Evidence from VLM
## Artifact and weather risk
## Recommended next step
Be factual; cite numbers from the evidence."""


def generate_analyst_report(
    report_input: ReportInput,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Generate markdown analyst report from structured evidence."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_report(report_input)

    client = Anthropic(api_key=api_key)
    model = model or DEFAULT_LLM_MODEL
    evidence = report_input.model_dump(mode="json")

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=REPORT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Generate the analyst report from this evidence:\n{json.dumps(evidence, indent=2)}",
            }
        ],
    )
    return response.content[0].text


def _fallback_report(inp: ReportInput) -> str:
    """Template report when no API key (tests / offline)."""
    c = inp.classical
    vlm = inp.vlm
    vlm_section = "VLM not invoked (gate filtered or no API key)."
    if vlm:
        vlm_section = (
            f"VLM verdict: {vlm.vlm_verdict}, type: {vlm.change_type}, "
            f"confidence: {vlm.confidence}"
        )
    return f"""## AOI
{inp.aoi_id}

## Date range
{inp.date_t1} to {inp.date_t2}

## Overall verdict
Classical gate: **{c.classical_gate}**. Changed area: {c.changed_area_percent}%.

## Change type
{vlm.change_type if vlm else "pending / not verified"}

## Confidence
Quality score: {inp.quality.quality_score}. {vlm_section}

## Evidence from indices
NDVI delta mean: {c.ndvi_delta_mean}, NDBI delta mean: {c.ndbi_delta_mean}, NDWI delta mean: {c.ndwi_delta_mean}.

## Evidence from model
SSIM: {c.ssim}, pHash distance: {c.phash_distance}, CVA mean: {c.cva_magnitude_mean}.

## Evidence from VLM
{vlm_section}

## Artifact and weather risk
Cloud max: {c.cloud_fraction_max}, snow max: {c.snow_fraction_max}, registration error: {c.registration_error_px} px.

## Recommended next step
{"Human review recommended." if (vlm and vlm.requires_human_review) else "Monitor next acquisition or expand AOI review."}
"""
