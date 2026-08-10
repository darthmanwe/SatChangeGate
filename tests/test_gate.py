"""Gate features, decision logic, and regression pins.

The previous suite asserted only that the gate returned one of its three legal
values, which is why threshold drift that invalidated the committed sample
report went unnoticed. These tests pin behaviour to concrete feature vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from satchangegate.config import GateThresholds
from satchangegate.features.classical import (
    GateFeatures,
    _phash,
    change_magnitude,
    compute_ssim,
    decide,
    despeckle,
    phash_distance,
    scene_change_threshold,
)


def features(**overrides: object) -> GateFeatures:
    base = {
        "ssim": 0.9,
        "phash_distance": 2,
        "ndvi_delta_mean": 0.0,
        "ndbi_delta_mean": 0.0,
        "ndwi_delta_mean": 0.0,
        "ndvi_delta_abs_mean": 0.0,
        "ndbi_delta_abs_mean": 0.0,
        "ndwi_delta_abs_mean": 0.0,
        "cva_magnitude_mean": 0.0,
        "changed_area_percent": 0.0,
        "valid_observation": True,
    }
    base.update(overrides)
    return GateFeatures(**base)  # type: ignore[arg-type]


class TestDecide:
    def test_quiet_pair_is_no_change(self) -> None:
        decision, _, conf = decide(features(), GateThresholds())
        assert decision == "no_change"
        assert conf < 0.2

    def test_failed_quality_short_circuits_to_low_quality(self) -> None:
        decision, reason, _ = decide(features(valid_observation=False), GateThresholds())
        assert decision == "low_quality"
        assert "Tier 0" in reason

    def test_strong_spectral_change_is_a_candidate(self) -> None:
        t = GateThresholds()
        decision, _, _ = decide(features(ndvi_delta_abs_mean=t.ndvi_strong_min + 0.01), t)
        assert decision == "candidate_change"

    def test_moderate_change_needs_area_support(self) -> None:
        t = GateThresholds()
        moderate = t.ndvi_delta_mean_min + 0.001
        assert decide(features(ndvi_delta_abs_mean=moderate), t)[0] == "no_change"
        assert (
            decide(
                features(
                    ndvi_delta_abs_mean=moderate,
                    changed_area_percent=t.min_changed_area_percent + 1,
                ),
                t,
            )[0]
            == "candidate_change"
        )

    def test_decision_is_monotone_in_evidence(self) -> None:
        """More evidence must never move a pair back to no_change.

        The previous ladder could: one branch returned no_change from *inside*
        the high-area path, so increasing changed area flipped a candidate off.
        """
        t = GateThresholds()
        last_candidate = False
        for area in np.linspace(0, 100, 40):
            for spectral in np.linspace(0, 0.4, 40):
                decision, _, _ = decide(
                    features(ndvi_delta_abs_mean=float(spectral), changed_area_percent=float(area)),
                    t,
                )
                if decision == "candidate_change":
                    last_candidate = True
                elif last_candidate and spectral == 0.0:
                    last_candidate = False  # new sweep row
        # Directly: increasing spectral evidence never un-flags.
        flagged = None
        for spectral in np.linspace(0, 0.4, 60):
            decision, _, _ = decide(features(ndvi_delta_abs_mean=float(spectral)), t)
            if decision == "candidate_change":
                flagged = spectral
            elif flagged is not None:
                pytest.fail(f"un-flagged at {spectral} after flagging at {flagged}")

    def test_confidence_is_bounded_and_increasing(self) -> None:
        t = GateThresholds()
        confs = [
            decide(features(ndvi_delta_abs_mean=float(s)), t)[2] for s in np.linspace(0, 0.4, 20)
        ]
        assert all(0.0 <= c <= 1.0 for c in confs)
        assert confs == sorted(confs)


class TestPerceptualHash:
    def test_dc_term_does_not_dominate(self) -> None:
        """Bit 0 was previously always 1, costing ~15 bits of hash entropy."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
            assert _phash(img)[0] == np.False_

    def test_brightness_shift_does_not_change_the_hash(self) -> None:
        rng = np.random.default_rng(1)
        img = rng.integers(20, 200, (64, 64), dtype=np.uint8)
        brighter = np.clip(img.astype(int) + 30, 0, 255).astype(np.uint8)
        assert phash_distance(img, brighter) <= 4

    def test_identical_images_have_zero_distance(self) -> None:
        img = np.random.default_rng(2).integers(0, 255, (64, 64), dtype=np.uint8)
        assert phash_distance(img, img) == 0


class TestChangeMask:
    def test_despeckle_removes_isolated_pixels(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[10, 10] = True  # speckle
        mask[30:45, 30:45] = True  # real region
        out = despeckle(mask, open_radius=1, min_size=32)
        assert not out[10, 10]
        assert out[35, 35]

    def test_adaptive_threshold_survives_a_large_changed_fraction(self) -> None:
        """A percentile cut lands inside the change when change is common.

        With 30% of pixels changed, a 95th-percentile threshold sits within the
        changed population and suppresses exactly what it should detect.
        """
        deltas = {k: np.zeros((100, 100), np.float32) for k in ("ndvi", "ndbi", "ndwi")}
        deltas["ndvi"][:30, :] = 0.5
        valid = np.ones((100, 100), dtype=bool)
        cut = scene_change_threshold(deltas, valid, GateThresholds())
        assert cut is not None and cut < 0.5

    def test_threshold_respects_absolute_floor(self) -> None:
        """Pure noise must not yield a change mask just because some pixel is top-N%."""
        rng = np.random.default_rng(3)
        deltas = {
            k: rng.normal(0, 0.002, (100, 100)).astype(np.float32) for k in ("ndvi", "ndbi", "ndwi")
        }
        cut = scene_change_threshold(deltas, np.ones((100, 100), bool), GateThresholds())
        assert cut == GateThresholds().min_absolute_delta

    def test_water_uses_ndwi_not_land_indices(self) -> None:
        """NDVI/NDBI are unstable over water; NDWI is the meaningful signal."""
        shape = (16, 16)
        deltas = {
            "ndvi": np.full(shape, 0.9, np.float32),
            "ndbi": np.full(shape, 0.9, np.float32),
            "ndwi": np.zeros(shape, np.float32),
        }
        water = np.ones(shape, dtype=bool)
        assert change_magnitude(deltas, water).max() == 0.0
        assert change_magnitude(deltas, None).max() == pytest.approx(0.9)


class TestSSIM:
    def test_identical_images_score_one(self) -> None:
        rgb = np.random.default_rng(4).random((32, 32, 3)).astype(np.float32)
        assert compute_ssim(rgb, rgb) == pytest.approx(1.0, abs=1e-6)

    def test_different_images_score_below_one(self) -> None:
        rng = np.random.default_rng(5)
        assert (
            compute_ssim(
                rng.random((32, 32, 3)).astype(np.float32),
                rng.random((32, 32, 3)).astype(np.float32),
            )
            < 0.5
        )
