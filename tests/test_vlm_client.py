"""Tests for VLM client with mocked Anthropic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from satchangegate.vlm.client import verify_candidate
from satchangegate.vlm.schemas import VlmVerdict


@pytest.fixture
def minimal_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "before_rgb.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (pkg / "after_rgb.png").write_bytes((pkg / "before_rgb.png").read_bytes())
    (pkg / "change_heatmap.png").write_bytes((pkg / "before_rgb.png").read_bytes())
    (pkg / "metadata.json").write_text(
        '{"aoi_id": "test", "classical": {"classical_gate": "candidate_change"}}',
        encoding="utf-8",
    )
    return pkg


def test_verify_candidate_mocked(minimal_package):
    verdict_json = {
        "vlm_verdict": "real_change",
        "change_type": "construction",
        "visual_evidence": ["Built surface visible."],
        "artifact_risk": {
            "cloud_shadow": "low",
            "snow": "low",
            "seasonality": "low",
            "registration": "low",
        },
        "confidence": 0.85,
        "requires_human_review": False,
    }

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(verdict_json))]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("satchangegate.vlm.client.Anthropic", return_value=mock_client):
        result = verify_candidate(minimal_package, api_key="test-key")

    assert isinstance(result, VlmVerdict)
    assert result.vlm_verdict == "real_change"
