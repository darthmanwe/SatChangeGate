"""Spectral indices, masks, and quality scoring."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from satchangegate.config import MaskThresholds, QualityThresholds
from satchangegate.features.indices import MIN_DENOMINATOR, ndbi, ndsi, ndvi, ndwi
from satchangegate.preprocess.masks import (
    can_assess,
    combine_pair_masks,
    compute_ephemeral_masks,
    unassessed_masks,
)
from satchangegate.preprocess.quality import compute_quality_score, season_delta_days


class TestIndices:
    def test_ndvi_high_over_vegetation(self) -> None:
        assert ndvi(np.array([0.32]), np.array([0.035]))[0] > 0.6

    def test_ndvi_near_zero_over_bare_soil(self) -> None:
        assert abs(ndvi(np.array([0.21]), np.array([0.20]))[0]) < 0.1

    def test_ndbi_positive_over_built_surfaces(self) -> None:
        assert ndbi(np.array([0.24]), np.array([0.18]))[0] > 0.0

    def test_ndwi_positive_over_water(self) -> None:
        assert ndwi(np.array([0.04]), np.array([0.015]))[0] > 0.3

    def test_ndsi_positive_over_snow(self) -> None:
        assert ndsi(np.array([0.75]), np.array([0.10]))[0] > 0.4

    def test_low_denominator_returns_zero_not_garbage(self) -> None:
        """Dark water must not produce a huge spurious index.

        Deep-water SWIR is ~0.008; with a naive epsilon guard, noise of a few
        thousandths swings the index across most of its range and a static lake
        reads as violent change.
        """
        tiny = MIN_DENOMINATOR / 4
        assert ndbi(np.array([tiny]), np.array([tiny / 2]))[0] == 0.0

    def test_nan_and_inf_do_not_propagate(self) -> None:
        out = ndvi(np.array([np.nan, np.inf, 0.3]), np.array([0.1, 0.1, 0.1]))
        assert np.isfinite(out).all()
        assert out[0] == 0.0 and out[1] == 0.0

    def test_output_is_float32_and_bounded(self) -> None:
        out = ndvi(
            np.random.rand(32, 32).astype(np.float32), np.random.rand(32, 32).astype(np.float32)
        )
        assert out.dtype == np.float32
        assert np.all(np.abs(out) <= 1.0)


class TestMasks:
    def test_cloud_detected_on_bright_unvegetated_pixels(self, synthetic_bands: Callable) -> None:
        bands = synthetic_bands(blue=0.5, green=0.5, red=0.5, nir=0.5, swir1=0.4)
        masks = compute_ephemeral_masks(bands, MaskThresholds())
        assert masks.assessed
        assert masks.cloud_fraction is not None and masks.cloud_fraction > 0.5

    def test_vegetation_is_not_flagged_as_cloud(self, synthetic_bands: Callable) -> None:
        masks = compute_ephemeral_masks(synthetic_bands(), MaskThresholds())
        assert masks.cloud_fraction == 0.0

    def test_snow_is_not_double_counted_as_cloud(self, synthetic_bands: Callable) -> None:
        """A pixel must not be both snow and cloud; that double-penalised quality."""
        bands = synthetic_bands(green=0.75, swir1=0.05, nir=0.55, blue=0.7, red=0.7)
        masks = compute_ephemeral_masks(bands, MaskThresholds())
        assert masks.snow_fraction is not None and masks.snow_fraction > 0.5
        assert not (masks.snow & masks.cloud).any()

    def test_water_is_reported_but_stays_valid(self, synthetic_bands: Callable) -> None:
        """Water must not be subtracted from validity, or flooding is undetectable."""
        bands = synthetic_bands(green=0.04, nir=0.015, red=0.028, blue=0.045, swir1=0.008)
        masks = compute_ephemeral_masks(bands, MaskThresholds())
        assert masks.water_fraction is not None and masks.water_fraction > 0.5
        assert masks.valid.all()

    def test_missing_bands_report_unassessed_not_clean(self) -> None:
        """The critical honesty property: unavailable != zero contamination."""
        rgb_only = {k: np.zeros((8, 8), np.float32) for k in ("B02", "B03", "B04")}
        assert not can_assess(rgb_only)
        masks = compute_ephemeral_masks(rgb_only, MaskThresholds())
        assert masks.assessed is False
        assert masks.cloud_fraction is None

    def test_combine_is_union_of_contamination(self, synthetic_bands: Callable) -> None:
        a = compute_ephemeral_masks(
            synthetic_bands(blue=0.5, green=0.5, red=0.5, nir=0.5), MaskThresholds()
        )
        b = compute_ephemeral_masks(synthetic_bands(), MaskThresholds())
        combined = combine_pair_masks(a, b)
        assert combined.cloud_fraction == a.cloud_fraction
        assert combined.valid_fraction is not None and combined.valid_fraction < 1.0


class TestQuality:
    def test_unassessed_masks_yield_unknown_not_perfect(self) -> None:
        u = unassessed_masks((8, 8))
        q = compute_quality_score(u, u, QualityThresholds())
        assert q.masks_assessed is False
        assert q.cloud_fraction_max is None
        assert q.quality_score is None

    def test_contamination_never_exceeds_one(self, synthetic_bands: Callable) -> None:
        """Overlapping masks previously summed past 1.0 and clipped the score to 0."""
        bands = synthetic_bands(blue=0.6, green=0.6, red=0.6, nir=0.5, swir1=0.05)
        m = compute_ephemeral_masks(bands, MaskThresholds())
        q = compute_quality_score(m, m, QualityThresholds())
        assert q.quality_score is not None and 0.0 <= q.quality_score <= 1.0

    def test_registration_error_gates_the_observation(self, synthetic_bands: Callable) -> None:
        m = compute_ephemeral_masks(synthetic_bands(), MaskThresholds())
        ok = compute_quality_score(m, m, QualityThresholds(), registration_error_px=0.4)
        bad = compute_quality_score(m, m, QualityThresholds(), registration_error_px=9.0)
        assert ok.valid_observation is True
        assert bad.valid_observation is False

    @pytest.mark.parametrize(
        ("d1", "d2", "expected"),
        [
            ("2020-01-01", "2020-01-01", 0),
            ("2020-01-01", "2021-01-01", 0),  # same season a year apart
            ("2020-01-01", "2020-07-01", 182),
            (None, "2020-07-01", None),
        ],
    )
    def test_season_delta(self, d1: str | None, d2: str, expected: int | None) -> None:
        result = season_delta_days(d1, d2)
        if expected is None:
            assert result is None
        else:
            assert result is not None and abs(result - expected) <= 1
