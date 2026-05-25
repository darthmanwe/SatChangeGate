"""Tests for GOES ABI fetch (optional, network)."""

from pathlib import Path

import pytest

from satchangegate.data.goes import default_goes_root, verify_layout


@pytest.mark.goes
def test_goes_verify_or_skip():
    ok, msg = verify_layout(default_goes_root())
    if not ok:
        pytest.skip(msg)
    assert Path(default_goes_root()).is_dir()
