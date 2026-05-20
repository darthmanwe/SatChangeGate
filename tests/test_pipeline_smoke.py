"""Full pipeline smoke test with synthetic data and mocked APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from satchangegate.data.oscd import ImagePair
from satchangegate.pipeline import run_pair
from satchangegate.vlm.schemas import VlmVerdict


@dataclass
class FakePair:
    pair_id: str = "synthetic"
    split: str = "test"
    img1_dir: Path = Path(".")
    img2_dir: Path = Path(".")
    label_path: None = None


def _write_fake_tifs(directory: Path, bands: dict[str, np.ndarray]) -> None:
    import rasterio
    from rasterio.transform import from_origin

    directory.mkdir(parents=True, exist_ok=True)
    h, w = next(iter(bands.values())).shape
    transform = from_origin(0, h, 1, 1)
    for name, arr in bands.items():
        with rasterio.open(
            directory / f"{name}.tif",
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(arr.astype(np.float32), 1)


@pytest.fixture
def synthetic_oscd_pair(tmp_path, synthetic_bands_vegetation, synthetic_bands_bare):
    t1 = tmp_path / "t1"
    t2 = tmp_path / "t2"
    _write_fake_tifs(t1, synthetic_bands_vegetation)
    _write_fake_tifs(t2, synthetic_bands_bare)
    return ImagePair(
        pair_id="synthetic",
        split="test",
        img1_dir=t1,
        img2_dir=t2,
        label_path=None,
    )


def test_pipeline_smoke(synthetic_oscd_pair, tmp_path):
    verdict = VlmVerdict(
        vlm_verdict="real_change",
        change_type="construction",
        visual_evidence=["Change visible."],
        artifact_risk={
            "cloud_shadow": "low",
            "snow": "low",
            "seasonality": "low",
            "registration": "low",
        },
        confidence=0.8,
        requires_human_review=False,
    )

    with patch("satchangegate.pipeline.verify_candidate", return_value=verdict):
        with patch(
            "satchangegate.pipeline.generate_analyst_report",
            return_value="## AOI\nsynthetic\n## Overall verdict\nchange detected",
        ):
            result = run_pair(
                synthetic_oscd_pair,
                out_dir=tmp_path / "reports",
                skip_vlm=False,
                api_key="fake",
            )

    assert result.classical_gate in ("candidate_change", "no_change", "low_quality")
    assert result.report_path is not None
    assert result.report_path.exists()
    text = result.report_path.read_text(encoding="utf-8")
    assert "AOI" in text or "verdict" in text.lower()
    assert result.result_path.exists()
