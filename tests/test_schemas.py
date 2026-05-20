"""Tests for pydantic schemas."""

import json

from satchangegate.vlm.schemas import ArtifactRisk, ReportInput, VlmVerdict


def test_vlm_verdict_roundtrip():
    data = {
        "vlm_verdict": "real_change",
        "change_type": "construction",
        "visual_evidence": ["Coherent cleared region in after image."],
        "artifact_risk": {
            "cloud_shadow": "low",
            "snow": "low",
            "seasonality": "medium",
            "registration": "low",
        },
        "confidence": 0.78,
        "requires_human_review": False,
    }
    v = VlmVerdict.model_validate(data)
    assert v.vlm_verdict == "real_change"
    dumped = json.loads(v.model_dump_json())
    assert dumped["change_type"] == "construction"


def test_artifact_risk_levels():
    r = ArtifactRisk(
        cloud_shadow="high",
        snow="low",
        seasonality="medium",
        registration="low",
    )
    assert r.cloud_shadow == "high"
