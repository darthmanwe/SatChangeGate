"""VLM client, schemas, redaction, cost accounting, and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from satchangegate.features.classical import ClassicalResult
from satchangegate.metrics import (
    FunnelCost,
    confusion_from_pairs,
    wilson_interval,
)
from satchangegate.preprocess.quality import QualityScore
from satchangegate.vlm.client import (
    DEFAULT_VLM_MODEL,
    PRICING_USD_PER_MTOK,
    UsageRecord,
    build_vlm_messages,
    resolve_model,
    verify_candidate,
)
from satchangegate.vlm.package import redacted_metadata
from satchangegate.vlm.schemas import ReportInput, VlmVerdict

GOOD_VERDICT = {
    "vlm_verdict": "real_change",
    "change_type": "construction",
    "visual_evidence": ["New rectangular structures in the north-east quadrant."],
    "artifact_risk": {
        "cloud_shadow": "low",
        "snow": "low",
        "seasonality": "low",
        "registration": "low",
    },
    "confidence": 0.9,
    "requires_human_review": False,
}


def _classical(**kw: object) -> ClassicalResult:
    base = {
        "tile_id": "t",
        "aoi_id": "oscd_t",
        "date_t1": "2020-01-01",
        "date_t2": "2022-01-01",
        "ssim": 0.5,
        "phash_distance": 4,
        "ndvi_delta_mean": -0.1,
        "ndbi_delta_mean": 0.07,
        "ndwi_delta_mean": 0.0,
        "changed_area_percent": 12.0,
        "classical_gate": "candidate_change",
        "gate_confidence": 0.8,
    }
    base.update(kw)
    return ClassicalResult(**base)  # type: ignore[arg-type]


class TestModelSelection:
    def test_default_model_is_current(self) -> None:
        """The previous default was past its retirement date and returned 404."""
        assert DEFAULT_VLM_MODEL == "claude-sonnet-5"
        assert "4-20250514" not in DEFAULT_VLM_MODEL

    def test_env_override_is_read_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading at import time meant .env was loaded too late to ever apply."""
        monkeypatch.setenv("ANTHROPIC_VLM_MODEL", "claude-opus-5")
        assert resolve_model(None, "ANTHROPIC_VLM_MODEL", DEFAULT_VLM_MODEL) == "claude-opus-5"

    def test_explicit_argument_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_VLM_MODEL", "claude-opus-5")
        assert resolve_model("claude-haiku-4-5", "ANTHROPIC_VLM_MODEL", DEFAULT_VLM_MODEL) == (
            "claude-haiku-4-5"
        )

    def test_every_default_model_has_pricing(self) -> None:
        assert DEFAULT_VLM_MODEL in PRICING_USD_PER_MTOK


class TestSchemas:
    def test_valid_verdict_round_trips(self) -> None:
        v = VlmVerdict.model_validate(GOOD_VERDICT)
        assert v.vlm_verdict == "real_change"

    @pytest.mark.parametrize(
        "mutation",
        [
            {"confidence": 1.5},
            {"confidence": -0.1},
            {"vlm_verdict": "definitely_changed"},
            {"visual_evidence": []},
            {"change_type": "alien_invasion"},
        ],
    )
    def test_malformed_verdicts_are_rejected(self, mutation: dict) -> None:
        with pytest.raises(ValidationError):
            VlmVerdict.model_validate({**GOOD_VERDICT, **mutation})

    def test_extra_keys_are_rejected_not_silently_dropped(self) -> None:
        with pytest.raises(ValidationError):
            VlmVerdict.model_validate({**GOOD_VERDICT, "hallucinated": "field"})

    def test_uncertain_verdict_forces_human_review(self) -> None:
        v = VlmVerdict.model_validate(
            {**GOOD_VERDICT, "vlm_verdict": "uncertain", "requires_human_review": False}
        )
        assert v.requires_human_review is True

    def test_low_confidence_forces_human_review(self) -> None:
        v = VlmVerdict.model_validate(
            {**GOOD_VERDICT, "confidence": 0.2, "requires_human_review": False}
        )
        assert v.requires_human_review is True

    def test_report_change_score_uses_gate_confidence(self) -> None:
        """Not changed_area/100, which is a noise-dominated area fraction."""
        c = _classical(gate_confidence=0.42, changed_area_percent=99.0)
        r = ReportInput.from_pipeline(QualityScore(masks_assessed=True, valid_observation=True), c)
        assert r.change_score == pytest.approx(0.42)


class TestRedaction:
    def test_gate_verdict_is_withheld_from_the_model(self) -> None:
        """The VLM verifies the gate; telling it the answer confounds agreement."""
        meta = redacted_metadata(
            QualityScore(masks_assessed=True, valid_observation=True), _classical()
        )
        blob = json.dumps(meta)
        for leak in (
            "classical_gate",
            "candidate_change",
            "ndvi_delta",
            "changed_area",
            "gate_confidence",
            "ssim",
            "phash",
        ):
            assert leak not in blob, f"{leak} leaked into VLM metadata"

    def test_acquisition_context_is_kept(self) -> None:
        meta = redacted_metadata(
            QualityScore(masks_assessed=True, valid_observation=True), _classical()
        )
        assert meta["date_t1"] == "2020-01-01"
        assert "observation_quality" in meta


class TestVlmClient:
    def _package(self, tmp_path: Path, *, images: int = 3) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
        import cv2

        for name in ("before_rgb.png", "after_rgb.png", "change_overlay.png")[:images]:
            cv2.imwrite(str(tmp_path / name), np.zeros((8, 8, 3), np.uint8))
        return tmp_path

    def test_request_carries_metadata_and_three_images(self, tmp_path: Path) -> None:
        content = build_vlm_messages(self._package(tmp_path / "pkg"))
        assert sum(1 for c in content if c["type"] == "image") == 3
        assert content[0]["type"] == "text"
        assert all(
            c["source"]["media_type"] == "image/png" for c in content if c["type"] == "image"
        )

    def test_package_without_images_refuses(self, tmp_path: Path) -> None:
        """Never ask for a visual verdict with no imagery attached."""
        pkg = self._package(tmp_path / "pkg", images=0)
        with pytest.raises(FileNotFoundError, match="no images"):
            build_vlm_messages(pkg)

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            build_vlm_messages(tmp_path / "empty")

    def _mock_client(self, *, parsed=None, stop_reason="end_turn"):
        client = MagicMock()
        response = MagicMock()
        response.parsed_output = parsed
        response.stop_reason = stop_reason
        response.usage = MagicMock(input_tokens=1500, output_tokens=200, cache_read_input_tokens=0)
        client.messages.parse.return_value = response
        return client

    def test_sends_the_expected_model_and_records_usage(self, tmp_path: Path) -> None:
        client = self._mock_client(parsed=VlmVerdict.model_validate(GOOD_VERDICT))
        verdict, usage = verify_candidate(self._package(tmp_path / "pkg"), client=client)
        kwargs = client.messages.parse.call_args.kwargs
        assert kwargs["model"] == DEFAULT_VLM_MODEL
        assert kwargs["output_format"] is VlmVerdict
        assert kwargs["max_tokens"] >= 4096
        assert verdict.vlm_verdict == "real_change"
        assert usage.input_tokens == 1500 and usage.output_tokens == 200
        assert usage.cost_usd > 0

    def test_refusal_degrades_to_uncertain(self, tmp_path: Path) -> None:
        client = self._mock_client(parsed=None, stop_reason="refusal")
        verdict, _ = verify_candidate(self._package(tmp_path / "pkg"), client=client)
        assert verdict.vlm_verdict == "uncertain"
        assert verdict.requires_human_review is True

    def test_truncation_degrades_to_uncertain(self, tmp_path: Path) -> None:
        client = self._mock_client(parsed=None, stop_reason="max_tokens")
        verdict, _ = verify_candidate(self._package(tmp_path / "pkg"), client=client)
        assert verdict.vlm_verdict == "uncertain"

    def test_unparseable_response_does_not_kill_the_batch(self, tmp_path: Path) -> None:
        client = self._mock_client(parsed={"garbage": True})
        verdict, _ = verify_candidate(self._package(tmp_path / "pkg"), client=client)
        assert verdict.vlm_verdict == "uncertain"


class TestUsageAndCost:
    def test_cost_matches_published_rates(self) -> None:
        rec = UsageRecord(
            model="claude-sonnet-5", kind="vlm", input_tokens=1_000_000, output_tokens=0
        )
        assert rec.cost_usd == pytest.approx(PRICING_USD_PER_MTOK["claude-sonnet-5"][0])

    def test_unknown_model_costs_zero_rather_than_guessing(self) -> None:
        rec = UsageRecord(model="mystery", kind="vlm", input_tokens=10_000, output_tokens=10_000)
        assert rec.cost_usd == 0.0

    def test_funnel_reports_savings_against_review_everything(self) -> None:
        cost = FunnelCost(n_pairs=100, n_gate_filtered=60, n_candidates=40, n_vlm_calls=40)
        cost.vlm_cost_usd = 0.40
        cost.per_call_costs = [0.01] * 40
        out = cost.to_dict()
        assert out["gate_filtered_pct"] == 60.0
        assert out["counterfactual_review_everything_usd"] == pytest.approx(1.0)
        assert out["savings_pct"] == pytest.approx(60.0)
        assert out["vlm_budget_capped"] is False

    def test_budget_cap_is_not_credited_to_the_gate(self) -> None:
        """A truncated run must not report the cap as a gate saving."""
        cost = FunnelCost(n_pairs=100, n_gate_filtered=60, n_candidates=40, n_vlm_calls=1)
        cost.vlm_cost_usd = 0.01
        cost.per_call_costs = [0.01]
        out = cost.to_dict()
        assert out["vlm_budget_capped"] is True
        # 40 candidates x $0.01 = $0.40 projected, against $1.00 for all 100.
        assert out["projected_cost_at_gate_rate_usd"] == pytest.approx(0.40)
        assert out["savings_pct"] == pytest.approx(60.0)


class TestMetrics:
    def test_no_negatives_makes_precision_undefined(self) -> None:
        """Precision pinned at 1.0 by construction must be reported as undefined."""
        cm = confusion_from_pairs([1, 1, 1], [1, 1, 0])
        assert cm.has_negatives is False
        assert cm.precision is None and cm.f1 is None
        d = cm.to_dict()
        assert d["precision"] is None
        assert "metrics_note" in d

    def test_confusion_counts(self) -> None:
        cm = confusion_from_pairs([1, 0, 1, 0], [1, 1, 0, 0])
        assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
        assert cm.precision == pytest.approx(0.5)
        assert cm.recall == pytest.approx(0.5)
        assert cm.balanced_accuracy == pytest.approx(0.5)

    def test_wilson_interval_brackets_the_estimate(self) -> None:
        lo, hi = wilson_interval(7, 11)
        assert lo < 7 / 11 < hi
        assert hi - lo > 0.3  # small n must show wide uncertainty

    def test_wilson_interval_narrows_with_n(self) -> None:
        small = wilson_interval(70, 110)
        large = wilson_interval(700, 1100)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_length_mismatch_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            confusion_from_pairs([1, 0], [1])

    def test_empty_interval_is_maximally_uncertain(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)
