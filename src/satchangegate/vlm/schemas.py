"""Pydantic schemas for VLM verdict and LLM report input."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from satchangegate.features.classical import ClassicalResult
from satchangegate.preprocess.quality import QualityScore

ArtifactLevel = Literal["low", "medium", "high"]
VlmVerdictType = Literal["real_change", "likely_artifact", "uncertain"]
ChangeType = Literal[
    "construction",
    "demolition",
    "vegetation_clearing",
    "flooding",
    "no_meaningful_change",
    "weather_artifact",
    "alignment_artifact",
    "uncertain",
]


class ArtifactRisk(BaseModel):
    cloud_shadow: ArtifactLevel
    snow: ArtifactLevel
    seasonality: ArtifactLevel
    registration: ArtifactLevel


class VlmVerdict(BaseModel):
    vlm_verdict: VlmVerdictType
    change_type: ChangeType
    visual_evidence: list[str] = Field(min_length=1)
    artifact_risk: ArtifactRisk
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool


class ReportInput(BaseModel):
    """Structured evidence bundle for LLM report (no images)."""

    aoi_id: str
    date_t1: str
    date_t2: str
    quality: QualityScore
    classical: ClassicalResult
    vlm: VlmVerdict | None = None
    change_score: float | None = None

    @classmethod
    def from_pipeline(
        cls,
        quality: QualityScore,
        classical: ClassicalResult,
        vlm: VlmVerdict | None = None,
    ) -> ReportInput:
        score = classical.changed_area_percent / 100.0
        return cls(
            aoi_id=classical.aoi_id,
            date_t1=classical.date_t1,
            date_t2=classical.date_t2,
            quality=quality,
            classical=classical,
            vlm=vlm,
            change_score=round(score, 4),
        )
