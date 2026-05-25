"""Tests for OSCD negative controls (identity, stable, photometric)."""

import numpy as np
import pytest

from satchangegate.data.negative_controls import (
    apply_photometric_perturbation,
    find_stable_crop_box,
    prepare_negative_pair,
)
from satchangegate.data.oscd import discover_pairs


@pytest.fixture
def beirut_pair():
    pairs = discover_pairs()
    match = [p for p in pairs if p.pair_id == "beirut"]
    if not match:
        pytest.skip("OSCD beirut not on disk")
    return match[0]


def test_identity_mode_no_change(beirut_pair):
    t1, t2, suffix = prepare_negative_pair(beirut_pair, "identity", ["B02", "B04"])
    assert suffix == "_identity"
    assert np.allclose(t1["B02"], t2["B02"])


def test_stable_crop_fits_label(beirut_pair):
    from satchangegate.data.oscd import load_label_mask

    label = load_label_mask(beirut_pair)
    assert label is not None
    box = find_stable_crop_box(label, target_size=128)
    assert box is not None


def test_photometric_changes_values():
    bands = {"B02": np.full((8, 8), 0.4, dtype=np.float32), "B04": np.full((8, 8), 0.3, dtype=np.float32)}
    out = apply_photometric_perturbation(bands)
    assert not np.allclose(out["B02"], bands["B02"])
