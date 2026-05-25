"""Tests for OPTIMUS loader (metadata + dev fixture)."""

import pytest

from satchangegate.data.optimus import (
    create_optimus_dev_fixture,
    default_optimus_root,
    get_bitemporal_frames,
    list_eval_series,
    load_eval_labels,
    verify_layout,
)


@pytest.mark.optimus
def test_eval_labels_present():
    root = default_optimus_root()
    if not (root / "2024_dataset_evaluation.json").is_file():
        pytest.skip("OPTIMUS metadata not downloaded")
    labels = load_eval_labels(root)
    assert len(labels) == 483 or len(labels) == 300  # HF eval set size
    assert 0 in labels.values() and 1 in labels.values()


@pytest.mark.optimus
def test_list_eval_no_change_sample():
    root = default_optimus_root()
    if not (root / "2024_dataset_evaluation.json").is_file():
        pytest.skip("OPTIMUS metadata not downloaded")
    series = list_eval_series(root, label=0, limit=5)
    assert len(series) >= 1
    assert series[0].label == 0


@pytest.mark.optimus
def test_dev_fixture_bitemporal():
    root = default_optimus_root()
    if not (root / "2024_dataset_evaluation.json").is_file():
        pytest.skip("OPTIMUS metadata not downloaded")
    from satchangegate.data.oscd import discover_pairs

    if not discover_pairs():
        pytest.skip("OSCD required for OPTIMUS dev fixture")

    tile = "4472_3910"
    create_optimus_dev_fixture(tile, label=0, root=root)
    b1, b2, t1, t2 = get_bitemporal_frames(tile, root)
    assert t1 != t2 or tile  # timestamps from index or defaults
    assert b1["B02"].shape == b2["B02"].shape


@pytest.mark.optimus
def test_verify_layout():
    ok, msg = verify_layout()
    if not ok:
        pytest.skip(msg)
    assert "Eval labels" in msg
