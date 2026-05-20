"""Tests for classical change gate."""

import numpy as np

from satchangegate.features.classical import classical_gate
from satchangegate.preprocess.masks import EphemeralMasks
from satchangegate.preprocess.quality import QualityScore


def _quality(cloud=0.02, valid=True):
    return QualityScore(
        valid_observation=valid,
        cloud_fraction=cloud,
        shadow_fraction=0.01,
        snow_fraction=0.0,
        water_fraction=0.01,
        registration_error_px=0.0,
        illumination_delta=0.05,
        season_delta_days=None,
        quality_score=0.9 if valid else 0.1,
    )


def test_identical_images_no_change(cfg, synthetic_bands_vegetation, all_valid_mask):
    q = _quality()
    art = classical_gate(
        "test",
        synthetic_bands_vegetation,
        synthetic_bands_vegetation,
        all_valid_mask,
        q,
        cfg,
    )
    assert art.result.classical_gate == "no_change"


def test_vegetation_to_bare_candidate_change(
    cfg, synthetic_bands_vegetation, synthetic_bands_bare, all_valid_mask
):
    q = _quality()
    art = classical_gate(
        "test",
        synthetic_bands_vegetation,
        synthetic_bands_bare,
        all_valid_mask,
        q,
        cfg,
    )
    assert art.result.classical_gate == "candidate_change"
    assert art.result.changed_area_percent > 0


def test_high_cloud_low_quality(cfg, synthetic_bands_cloudy, all_valid_mask):
    q = _quality(cloud=0.9, valid=False)
    art = classical_gate(
        "test",
        synthetic_bands_cloudy,
        synthetic_bands_cloudy,
        all_valid_mask,
        q,
        cfg,
    )
    assert art.result.classical_gate == "low_quality"
