"""Pydantic schemas for the VLM verdict and the LLM report input.

Field descriptions are load-bearing: the verdict schema is submitted to the API
as a JSON schema via structured outputs, so these strings are what tell the
model what each field means. They replaced a hand-written schema pasted into the
system prompt plus a regex that scraped JSON back out of prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from satchangegate.features.classical import ClassicalResult
from satchangegate.preprocess.quality import QualityScore

ArtifactLevel = Literal["low", "medium", "high"]
VlmVerdictType = Literal["real_change", "likely_artifact", "uncertain"]
ChangeType = Literal[
    "construction",
    "demolition",
    "vegetation_clearing",
    "vegetation_regrowth",
    "flooding",
    "no_meaningful_change",
    "weather_artifact",
    "alignment_artifact",
    "uncertain",
]


class ArtifactRisk(BaseModel):
    """Per-factor risk that the apparent change is an artifact, not ground truth."""

    model_config = ConfigDict(extra="forbid")

    cloud_shadow: ArtifactLevel = Field(
        description="Risk that cloud or cloud shadow explains the apparent change."
    )
    snow: ArtifactLevel = Field(
        description="Risk that snow or ice cover explains the apparent change."
    )
    seasonality: ArtifactLevel = Field(
        description="Risk that seasonal phenology (leaf-on/leaf-off, crop cycle) "
        "explains the apparent change rather than a persistent physical change."
    )
    registration: ArtifactLevel = Field(
        description="Risk that image misalignment between the two dates explains "
        "the apparent change."
    )


class VlmVerdict(BaseModel):
    """The vision model's independent verification of a gate candidate."""

    model_config = ConfigDict(extra="forbid")

    vlm_verdict: VlmVerdictType = Field(
        description="real_change when a persistent physical change on the ground is "
        "visible; likely_artifact when the difference is explained by weather, "
        "season, illumination, or misalignment; uncertain when the evidence "
        "does not support either conclusion."
    )
    change_type: ChangeType = Field(
        description="The dominant kind of change observed, or the artifact category."
    )
    visual_evidence: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Short concrete observations grounded in what is visible in the "
        "images, e.g. 'new rectangular structures in the north-east quadrant'. "
        "Do not restate the metadata.",
    )
    artifact_risk: ArtifactRisk
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the stated verdict, 0 to 1."
    )
    requires_human_review: bool = Field(
        description="True when an analyst should inspect this pair before it is acted on."
    )

    @model_validator(mode="after")
    def _low_confidence_needs_review(self) -> VlmVerdict:
        # Business rule that previously lived nowhere: an uncertain or
        # low-confidence verdict must not be auto-accepted.
        if (
            self.vlm_verdict == "uncertain" or self.confidence < 0.5
        ) and not self.requires_human_review:
            object.__setattr__(self, "requires_human_review", True)
        return self

    @classmethod
    def uncertain_fallback(cls, reason: str) -> VlmVerdict:
        """Degraded verdict used when the model response cannot be validated."""
        return cls(
            vlm_verdict="uncertain",
            change_type="uncertain",
            visual_evidence=[f"No usable model response: {reason}"],
            artifact_risk=ArtifactRisk(
                cloud_shadow="medium", snow="medium", seasonality="medium", registration="medium"
            ),
            confidence=0.0,
            requires_human_review=True,
        )


class ReportInput(BaseModel):
    """Structured evidence bundle for the LLM analyst report (no images)."""

    model_config = ConfigDict(extra="forbid")

    aoi_id: str
    date_t1: str
    date_t2: str
    quality: QualityScore
    classical: ClassicalResult
    vlm: VlmVerdict | None = None
    change_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_pipeline(
        cls,
        quality: QualityScore,
        classical: ClassicalResult,
        vlm: VlmVerdict | None = None,
    ) -> ReportInput:
        return cls(
            aoi_id=classical.aoi_id,
            date_t1=classical.date_t1,
            date_t2=classical.date_t2,
            quality=quality,
            classical=classical,
            vlm=vlm,
            # The gate's calibrated confidence, not changed_area/100. The latter
            # is a noise-dominated area fraction and was being handed to the LLM
            # as though it were a confidence.
            change_score=classical.gate_confidence,
        )
