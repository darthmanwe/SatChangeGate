"""E2E random eval tests (mocked + optional live VLM)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from satchangegate.e2e_random_eval import (
    build_pair_pool,
    run_e2e_random_eval,
    sample_random_pairs,
)
from satchangegate.vlm.schemas import VlmVerdict


@pytest.fixture
def mock_vlm_verdict():
    return VlmVerdict(
        vlm_verdict="real_change",
        change_type="construction",
        visual_evidence=["Visible change in candidate region."],
        artifact_risk={
            "cloud_shadow": "low",
            "snow": "low",
            "seasonality": "medium",
            "registration": "low",
        },
        confidence=0.82,
        requires_human_review=False,
    )


def test_build_pair_pool_non_empty():
    pool = build_pair_pool()
    if not pool:
        pytest.skip("No OSCD/OPTIMUS data on disk")
    sources = {p.source for p in pool}
    assert "oscd" in sources or "optimus" in sources


def test_sample_random_pairs_deterministic():
    pool = build_pair_pool()
    if len(pool) < 2:
        pytest.skip("Need >=2 pairs in pool")
    a = sample_random_pairs(5, pool, seed=99)
    b = sample_random_pairs(5, pool, seed=99)
    assert [x.pair_id for x in a] == [x.pair_id for x in b]


def test_e2e_random_smoke(tmp_path, mock_vlm_verdict):
    pool = build_pair_pool()
    if not pool:
        pytest.skip("No OSCD/OPTIMUS data on disk")

    with patch("satchangegate.pipeline.verify_candidate", return_value=mock_vlm_verdict):
        metrics = run_e2e_random_eval(
            n=min(3, len(pool)),
            seed=7,
            out_dir=tmp_path,
            skip_vlm=False,
            skip_llm=True,
            api_key="fake-key",
        )

    assert metrics["n_completed"] >= 1
    assert metrics["skip_vlm"] is False
    assert (tmp_path / "_e2e_random_summary.md").is_file()


@pytest.mark.e2e
@pytest.mark.vlm
def test_e2e_random_live_vlm_one():
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    pool = build_pair_pool()
    if not pool:
        pytest.skip("No OSCD/OPTIMUS data on disk")

    metrics = run_e2e_random_eval(
        n=1,
        seed=0,
        out_dir=Path("data/reports/e2e_random_test"),
        skip_vlm=False,
        skip_llm=True,
    )
    assert metrics["n_completed"] == 1
