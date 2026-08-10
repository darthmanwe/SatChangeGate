"""Loaders, alignment, packaging, and the end-to-end pipeline on fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from satchangegate.config import ALL_BANDS, Settings, get_settings, load_settings
from satchangegate.data.download import OSCD_FILES, _safe_extract
from satchangegate.data.oscd import discover_pairs, list_pairs, load_bands, load_label_mask
from satchangegate.data.tiles import build_tile_index, summarise, tiles_for_pair
from satchangegate.preprocess.align import (
    estimate_registration_error,
    resample_band_to_ref,
    resample_to_common_grid,
    rgb_to_uint8,
)
from satchangegate.tune_gate import assert_disjoint


class TestConfig:
    def test_packaged_config_loads_without_a_source_checkout(self) -> None:
        """The old loader used parents[2] and could not find its own config in a wheel."""
        s = get_settings()
        assert s.gate.background_sigma > 0
        assert len(s.bands) == 6

    def test_unknown_keys_are_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings.model_validate({"gate": {"a_key_that_does_not_exist": 1}})

    def test_settings_are_immutable(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            get_settings().gate.min_changed_area_percent = 1.0  # type: ignore[misc]

    def test_missing_file_falls_back_to_model_defaults(self, tmp_path: Path) -> None:
        s = load_settings(tmp_path / "nope.yaml")
        assert s.gate.min_changed_area_percent > 0

    def test_every_gate_threshold_is_consumed(self) -> None:
        """Dead config was a real defect: eleven keys had no reader."""
        import inspect

        from satchangegate.features import classical

        src = inspect.getsource(classical)
        unused = [
            k
            for k in get_settings().gate.model_dump()
            if f"t.{k}" not in src and f"thresholds.{k}" not in src
        ]
        assert unused == []


class TestAlign:
    def test_resample_preserves_content_not_tiles_it(self) -> None:
        """np.resize flattens and tiles; it scrambled any mismatched band."""
        src = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
        out = resample_band_to_ref(src, (64, 64))
        assert out.shape == (64, 64)
        # A true resample keeps the monotonic gradient; tiling would not.
        assert out[0, 0] < out[-1, -1]
        assert np.all(np.diff(out.mean(axis=1)) > -1e-3)

    def test_mismatched_band_sets_raise(self) -> None:
        a = {"B02": np.zeros((4, 4), np.float32), "B03": np.zeros((4, 4), np.float32)}
        b = {"B02": np.zeros((4, 4), np.float32)}
        with pytest.raises(ValueError, match="unmatched"):
            resample_to_common_grid(a, b)

    def test_registration_recovers_a_known_shift(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.random((64, 64)).astype(np.float32)
        shifted = np.roll(base, 4, axis=0)
        err = estimate_registration_error({"B04": base}, {"B04": shifted})
        assert err == pytest.approx(4.0, abs=0.2)

    def test_identical_scenes_have_zero_registration_error(self) -> None:
        base = np.random.default_rng(1).random((64, 64)).astype(np.float32)
        assert estimate_registration_error({"B04": base}, {"B04": base}) == pytest.approx(0.0)

    def test_rgb_to_uint8_rounds(self) -> None:
        assert rgb_to_uint8(np.array([[[0.5]]], dtype=np.float32))[0, 0, 0] == 128


class TestDownloadSafety:
    def test_checksums_are_pinned_for_every_archive(self) -> None:
        assert len(OSCD_FILES) == 3
        assert all(len(v) == 64 for v in OSCD_FILES.values())

    def test_zip_traversal_is_refused(self, tmp_path: Path) -> None:
        import zipfile

        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../escaped.txt", "pwned")
        with pytest.raises(ValueError, match="Unsafe path"):
            _safe_extract(evil, tmp_path / "dest")


class TestFixtures:
    def test_pairs_discovered(self, fixture_root: Path) -> None:
        pairs = discover_pairs(fixture_root)
        assert {p.pair_id for p in pairs} == {"fixtureville", "stableton"}

    def test_splits_are_disjoint(self, fixture_root: Path) -> None:
        train = {p.pair_id for p in list_pairs(fixture_root, "train")}
        test = {p.pair_id for p in list_pairs(fixture_root, "test")}
        assert train and test and not (train & test)

    def test_bands_load_as_reflectance(self, fixture_root: Path) -> None:
        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "fixtureville")
        bands = load_bands(pair.img1_dir)
        assert set(bands) == {"B02", "B03", "B04", "B08", "B11", "B12"}
        for arr in bands.values():
            assert arr.dtype == np.float32
            assert float(arr.min()) >= 0.0 and float(arr.max()) < 1.5

    def test_bands_are_not_linear_combinations_of_rgb(self, fixture_root: Path) -> None:
        """Regression against the deleted pseudo-band synthesis.

        The old loader built NIR as 0.6*green + 0.4*red, which forced R^2 == 1
        and made NDVI and NDWI 0.995-correlated aliases of each other.
        """
        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "fixtureville")
        b = load_bands(pair.img1_dir)
        A = np.stack(
            [b["B04"].ravel(), b["B03"].ravel(), b["B02"].ravel(), np.ones(b["B04"].size)], axis=1
        )
        coef, *_ = np.linalg.lstsq(A, b["B08"].ravel(), rcond=None)
        resid = ((A @ coef - b["B08"].ravel()) ** 2).sum()
        total = ((b["B08"].ravel() - b["B08"].mean()) ** 2).sum()
        assert 1 - resid / max(total, 1e-12) < 0.999

    def test_unknown_band_rejected(self, fixture_root: Path) -> None:
        pair = next(iter(discover_pairs(fixture_root)))
        with pytest.raises(ValueError, match="Unknown"):
            load_bands(pair.img1_dir, ["B99"])
        assert "B8A" in ALL_BANDS

    def test_labels_load_as_binary(self, fixture_root: Path) -> None:
        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "fixtureville")
        label = load_label_mask(pair)
        assert label is not None
        assert set(np.unique(label)).issubset({0, 1})
        assert label.sum() > 0


class TestTiles:
    def test_tiles_carry_both_classes(self, fixture_root: Path) -> None:
        tiles = build_tile_index(fixture_root, tile_size=32)
        labels = {t.label for t in tiles}
        assert labels == {0, 1}, "tile set must contain a real negative class"

    def test_ambiguous_tiles_are_discarded(self, fixture_root: Path) -> None:
        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "fixtureville")
        label = load_label_mask(pair)
        assert label is not None
        tiles = tiles_for_pair(pair, label, tile_size=32, pos_min_fraction=0.5)
        for t in tiles:
            assert t.change_fraction == 0.0 or t.change_fraction >= 0.5

    def test_index_is_deterministic(self, fixture_root: Path) -> None:
        a = [t.tile_id for t in build_tile_index(fixture_root, tile_size=32)]
        b = [t.tile_id for t in build_tile_index(fixture_root, tile_size=32)]
        assert a == b == sorted(a)

    def test_summary_counts_match(self, fixture_root: Path) -> None:
        tiles = build_tile_index(fixture_root, tile_size=32)
        s = summarise(tiles)
        assert sum(v["positive"] + v["negative"] for v in s.values()) == len(tiles)


class TestLeakageGuard:
    def test_overlapping_splits_raise(self) -> None:
        with pytest.raises(RuntimeError, match="leakage"):
            assert_disjoint({"paris", "beirut"}, {"beirut"})

    def test_disjoint_splits_pass(self) -> None:
        assert_disjoint({"paris"}, {"beirut"})  # must not raise


class TestPipeline:
    def test_end_to_end_on_fixtures_without_api_key(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        from satchangegate.pipeline import run_pair

        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "fixtureville")
        result = run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=False)

        assert result.classical_gate == "candidate_change"
        assert result.vlm_called is False
        assert result.llm_called is False
        assert result.cost_usd == 0.0

    def test_no_change_fixture_is_filtered(self, fixture_root: Path, tmp_path: Path) -> None:
        from satchangegate.pipeline import run_pair

        pair = next(p for p in discover_pairs(fixture_root) if p.pair_id == "stableton")
        assert run_pair(pair, out_dir=tmp_path).classical_gate == "no_change"

    def test_artifacts_stay_under_out_dir(self, fixture_root: Path, tmp_path: Path) -> None:
        """Outputs previously went to CWD-relative data/processed regardless of out_dir."""
        from satchangegate.pipeline import run_pair

        pair = next(iter(discover_pairs(fixture_root)))
        result = run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=True)
        assert tmp_path in result.result_path.parents
        assert (tmp_path / pair.pair_id / "processed" / "classical.json").is_file()
        assert not Path("data/processed").exists()

    def test_result_records_provenance_and_llm_flag(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        from satchangegate.pipeline import run_pair

        pair = next(iter(discover_pairs(fixture_root)))
        result = run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=False)
        blob = json.loads(result.result_path.read_text(encoding="utf-8"))
        assert blob["provenance"]["config_sha256"]
        assert blob["llm_called"] is False
        assert blob["report_path"] is not None

    def test_skipped_report_is_not_claimed(self, fixture_root: Path, tmp_path: Path) -> None:
        """A stale report from an earlier run was previously reported as this run's."""
        from satchangegate.pipeline import run_pair

        pair = next(iter(discover_pairs(fixture_root)))
        run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=False)
        second = run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=True)
        assert second.report_path is None
        assert json.loads(second.result_path.read_text(encoding="utf-8"))["report_path"] is None

    def test_offline_report_is_labelled_as_a_template(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        """A template and a real model report must not be indistinguishable."""
        from satchangegate.pipeline import run_pair

        pair = next(iter(discover_pairs(fixture_root)))
        result = run_pair(pair, out_dir=tmp_path, skip_vlm=True, skip_llm=False)
        assert result.report_path is not None
        assert "TEMPLATE REPORT" in result.report_path.read_text(encoding="utf-8")


class TestControls:
    def test_control_battery_passes_on_fixtures(self, fixture_root: Path, tmp_path: Path) -> None:
        from satchangegate.dev_controls import run_dev_tests

        summary = run_dev_tests(fixture_root, tmp_path, city="fixtureville")
        failed = [c["name"] for c in summary["checks"] if not c["passed"]]
        assert failed == [], f"controls failed: {failed}"


class TestE2E:
    def test_sampling_is_deterministic(self, fixture_root: Path) -> None:
        from satchangegate.e2e import sample_tiles

        tiles = build_tile_index(fixture_root, tile_size=32)
        a = [t.tile_id for t in sample_tiles(tiles, 5, seed=42)]
        b = [t.tile_id for t in sample_tiles(tiles, 5, seed=42)]
        c = [t.tile_id for t in sample_tiles(tiles, 5, seed=7)]
        assert a == b
        assert a != c or len(tiles) <= 5

    def test_funnel_runs_gate_only_and_resumes(self, fixture_root: Path, tmp_path: Path) -> None:
        from satchangegate.e2e import E2EConfig, run_e2e

        cfg = E2EConfig(split="all", n=8, seed=1, skip_vlm=True)
        first = run_e2e(fixture_root, tmp_path, config=cfg)
        assert first["n"] > 0
        assert first["funnel_cost"]["n_vlm_calls"] == 0
        assert (tmp_path / "_e2e_all.jsonl").is_file()

        # Resuming must not duplicate completed work.
        second = run_e2e(fixture_root, tmp_path, config=cfg, resume=True)
        assert second["n"] == first["n"]
