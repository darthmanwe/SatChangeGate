"""OSCD loader tests (skipped if dataset absent)."""

from pathlib import Path

import pytest

from satchangegate.data.oscd import (
    default_oscd_root,
    discover_pairs,
    list_pairs,
    load_bands,
    load_label_mask,
)

OSCD_ROOT = default_oscd_root()


@pytest.mark.oscd
@pytest.mark.skipif(not OSCD_ROOT.exists(), reason="OSCD dataset not on disk")
def test_list_pairs_finds_at_least_one():
    pairs = discover_pairs(OSCD_ROOT) or list_pairs(OSCD_ROOT, "all")
    assert len(pairs) >= 1


@pytest.mark.oscd
@pytest.mark.skipif(not OSCD_ROOT.exists(), reason="OSCD dataset not on disk")
def test_load_bands_shapes():
    pairs = discover_pairs(OSCD_ROOT) or list_pairs(OSCD_ROOT, "all")
    pair = pairs[0]
    bands = load_bands(pair.img1_dir, ["B02", "B03", "B04", "B08", "B11", "B12"])
    for name, arr in bands.items():
        assert arr.ndim == 2
        assert arr.shape[0] > 0 and arr.shape[1] > 0


@pytest.mark.oscd
@pytest.mark.skipif(not OSCD_ROOT.exists(), reason="OSCD dataset not on disk")
def test_load_label_mask_if_present():
    pairs = [p for p in (discover_pairs(OSCD_ROOT) or list_pairs(OSCD_ROOT, "test")) if p.label_path]
    if not pairs:
        pytest.skip("No labeled pairs")
    label = load_label_mask(pairs[0])
    assert label is not None
    assert label.ndim == 2
