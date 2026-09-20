"""The upload contract, tested on the ways an upload can lie.

``run_from_bands`` asks only for two dictionaries of arrays, which makes it a
clean seam and a dangerous one. Every test here is a way to hand it something
that satisfies its shape checks and means nothing: two rasters of different
continents, a missing CRS, nodata filled with zero, a band index that points at
the wrong band.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satchangegate.webui.ingest import Declaration, IngestRefused, Limits, read_pair
from satchangegate.webui.uploads import Manifest, UploadStore


def raster(path, *, bands=6, origin=(400000, 5000000), crs="EPSG:32631", size=32, fill=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="uint16",
        crs=crs,
        transform=from_origin(*origin, 10, 10),
        nodata=0,
    ) as dst:
        for band in range(1, bands + 1):
            if fill is None:
                data = (np.random.RandomState(band).rand(size, size) * 3000 + 500).astype("uint16")
            else:
                data = np.full((size, size), fill, dtype="uint16")
            dst.write(data, band)
    return path


def declaration(**overrides):
    base = {
        "band_map": {"B02": 1, "B03": 2, "B04": 3, "B08": 4, "B11": 5, "B12": 6},
        "reflectance_scale": 10000.0,
        "target_resolution_m": 10.0,
    }
    base.update(overrides)
    return Declaration(**base)


class TestFootprintsMustActuallyOverlap:
    def test_an_overlapping_pair_is_cropped_to_the_shared_ground(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        b = raster(tmp_path / "b.tif", origin=(400100, 5000000))
        pair = read_pair(a, b, declaration())
        assert pair.width < 32, "cropped to the intersection, not stretched onto one shape"
        assert pair.capability.lane == "full"
        assert any("shared footprint" in note for note in pair.warnings)

    def test_images_of_different_places_are_refused(self, tmp_path) -> None:
        """Equal pixel dimensions do not make two rasters comparable."""
        a = raster(tmp_path / "a.tif")
        b = raster(tmp_path / "far.tif", origin=(900000, 4000000))
        with pytest.raises(IngestRefused, match="do not overlap"):
            read_pair(a, b, declaration())

    def test_mismatched_coordinate_systems_are_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif", crs="EPSG:32631")
        b = raster(tmp_path / "b.tif", crs="EPSG:32632")
        with pytest.raises(IngestRefused, match="different coordinate systems"):
            read_pair(a, b, declaration())


class TestAlignmentIsDeclaredNotAssumed:
    def test_a_missing_crs_is_refused_by_default(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        b = raster(tmp_path / "b.tif", crs=None)
        with pytest.raises(IngestRefused, match="no CRS"):
            read_pair(a, b, declaration())

    def test_already_aligned_claims_no_map_position(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        b = raster(tmp_path / "b.tif", crs=None)
        pair = read_pair(a, b, declaration(alignment="already_aligned"))
        assert pair.crs is None and pair.bbox is None

    def test_already_aligned_requires_identical_shapes(self, tmp_path) -> None:
        """Resizing one to match invents a correspondence rather than verifying one."""
        a = raster(tmp_path / "a.tif", size=32)
        b = raster(tmp_path / "b.tif", size=48)
        with pytest.raises(IngestRefused, match="pixel-for-pixel"):
            read_pair(a, b, declaration(alignment="already_aligned"))


class TestNodataIsNotZero:
    def test_nodata_becomes_invalid(self, tmp_path) -> None:
        """Filling nodata with zero fabricates a clean observation."""
        for name in ("a.tif", "b.tif"):
            with rasterio.open(
                tmp_path / name,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=6,
                dtype="uint16",
                crs="EPSG:32631",
                transform=from_origin(400000, 5000000, 10, 10),
                nodata=0,
            ) as dst:
                for band in range(1, 7):
                    data = np.full((32, 32), 1500, dtype="uint16")
                    data[:8, :] = 0
                    dst.write(data, band)
        pair = read_pair(tmp_path / "a.tif", tmp_path / "b.tif", declaration())
        assert pair.valid[:8, :].sum() == 0
        assert pair.valid.mean() == pytest.approx(0.75)

    def test_a_pair_with_no_shared_valid_pixel_is_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif", fill=0)
        with pytest.raises(IngestRefused, match="nothing here to compare"):
            read_pair(a, a, declaration())


class TestBandsAreNamedNotGuessed:
    def test_unknown_band_names_are_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        with pytest.raises(IngestRefused, match="Unknown band names"):
            read_pair(a, a, declaration(band_map={"NIR": 1}))

    def test_a_band_index_beyond_the_raster_is_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif", bands=3)
        with pytest.raises(IngestRefused, match="raster index"):
            read_pair(a, a, declaration())

    def test_three_bands_route_to_the_degraded_lane(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif", bands=3)
        pair = read_pair(a, a, declaration(band_map={"B04": 1, "B03": 2, "B02": 3}))
        assert pair.capability.lane == "rgb_only"
        assert pair.capability.spectral_indices is False
        assert set(pair.capability.missing_for_gate) >= {"B08", "B11"}


class TestReflectanceIsDeclared:
    def test_values_above_one_are_not_clipped(self, tmp_path) -> None:
        """Bright targets legitimately exceed 1.0; clipping hides cloud tops."""
        a = raster(tmp_path / "a.tif")
        pair = read_pair(a, a, declaration(reflectance_scale=100.0))
        assert max(float(v.max()) for v in pair.bands_t1.values()) > 1.0

    def test_a_non_positive_scale_is_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        with pytest.raises(IngestRefused, match="reflectance_scale"):
            read_pair(a, a, declaration(reflectance_scale=0.0))


class TestResolutionChangesMeaning:
    def test_a_resolution_other_than_ten_metres_is_flagged(self, tmp_path) -> None:
        """Despeckling, component sizes and the registration tolerance are in pixels."""
        a = raster(tmp_path / "a.tif")
        pair = read_pair(a, a, declaration(target_resolution_m=30.0))
        assert any("30" in note for note in pair.warnings)

    def test_an_undeclared_resolution_is_read_from_the_transform(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif")
        pair = read_pair(a, a, declaration(target_resolution_m=None))
        assert pair.resolution_m == pytest.approx(10.0)


class TestLimitsAreCheckedFromTheHeader:
    def test_too_many_bands_is_refused(self, tmp_path) -> None:
        a = raster(tmp_path / "a.tif", bands=8)
        with pytest.raises(IngestRefused, match="bands, over the limit"):
            read_pair(a, a, declaration(band_map={"B02": 1}), limits=Limits(max_bands=4))

    def test_too_many_pixels_is_refused_before_decoding(self, tmp_path) -> None:
        """Compressed size is not a memory bound."""
        a = raster(tmp_path / "a.tif", size=64)
        with pytest.raises(IngestRefused, match="Mpx limit"):
            read_pair(a, a, declaration(), limits=Limits(max_pixels=1000))


class TestStagingIsNotPathHandling:
    def test_a_client_supplied_path_never_reaches_the_filesystem(self, tmp_path) -> None:
        store = UploadStore(tmp_path)
        upload_id = store.new()
        stored = store.store(upload_id, "../../evil.tif", b"x" * 10, Limits())
        assert stored.startswith("file_")
        assert (store.directory(upload_id) / stored).is_file()
        assert store.client_names(upload_id)[stored] == "../../evil.tif"

    def test_formats_that_can_reference_other_files_are_refused(self, tmp_path) -> None:
        store = UploadStore(tmp_path)
        upload_id = store.new()
        with pytest.raises(IngestRefused, match="only GeoTIFF"):
            store.store(upload_id, "sneaky.vrt", b"x", Limits())

    @pytest.mark.parametrize("bad", ["../secret", "file_000/../../x", "secret.tif", ""])
    def test_a_name_outside_the_staging_area_is_refused(self, tmp_path, bad) -> None:
        store = UploadStore(tmp_path)
        upload_id = store.new()
        with pytest.raises(IngestRefused):
            store.path(upload_id, bad)

    def test_an_unknown_upload_id_is_refused(self, tmp_path) -> None:
        store = UploadStore(tmp_path)
        with pytest.raises(IngestRefused):
            store.directory("../../etc")

    def test_the_file_count_is_bounded(self, tmp_path) -> None:
        store = UploadStore(tmp_path)
        upload_id = store.new()
        limits = Limits(max_files=2)
        store.store(upload_id, "a.tif", b"x", limits)
        store.store(upload_id, "b.tif", b"x", limits)
        with pytest.raises(IngestRefused, match="At most 2 files"):
            store.store(upload_id, "c.tif", b"x", limits)


class TestPairingIsExplicit:
    def test_a_manifest_naming_an_unknown_file_is_refused(self) -> None:
        with pytest.raises(IngestRefused, match="was not uploaded"):
            Manifest.parse(
                {
                    "declaration": {"band_map": {"B04": 1}},
                    "pairs": [{"t1": "file_000.tif", "t2": "nope.tif"}],
                },
                {"file_000.tif"},
            )

    def test_pairing_cannot_be_omitted(self) -> None:
        """Sorting filenames and pairing neighbours mis-pairs silently."""
        with pytest.raises(IngestRefused, match="No pairs declared"):
            Manifest.parse({"declaration": {"band_map": {"B04": 1}}}, {"file_000.tif"})

    def test_duplicate_pair_keys_are_refused(self) -> None:
        pair = {"key": "a", "t1": "file_000.tif", "t2": "file_000.tif"}
        with pytest.raises(IngestRefused, match="Duplicate pair key"):
            Manifest.parse(
                {"declaration": {"band_map": {"B04": 1}}, "pairs": [pair, dict(pair)]},
                {"file_000.tif"},
            )

    def test_a_manifest_round_trips_a_valid_declaration(self) -> None:
        manifest = Manifest.parse(
            {
                "declaration": {"band_map": {"B04": 1, "B03": 2}, "reflectance_scale": 255.0},
                "pairs": [{"key": "site", "t1": "file_000.tif", "t2": "file_001.tif"}],
            },
            {"file_000.tif", "file_001.tif"},
        )
        assert manifest.declaration.reflectance_scale == 255.0
        assert manifest.pairs[0].key == "site"


class TestTheDegradedLaneRefusesRatherThanScoring:
    def test_missing_index_bands_raise_a_named_error(self) -> None:
        from satchangegate.features.classical import MissingBandsError, _index_deltas

        rgb = {b: np.zeros((8, 8), dtype=np.float32) for b in ("B02", "B03", "B04")}
        with pytest.raises(MissingBandsError) as caught:
            _index_deltas(rgb, rgb)
        assert "B08" in caught.value.missing and "B11" in caught.value.missing

    def test_the_rgb_gate_refuses_and_names_what_is_unavailable(self) -> None:
        from satchangegate.config import get_settings
        from satchangegate.features.classical import RGB_ONLY_REASON, rgb_only_gate
        from satchangegate.preprocess.masks import combine_pair_masks, compute_ephemeral_masks
        from satchangegate.preprocess.quality import compute_quality_score

        settings = get_settings()
        rgb = {b: np.full((16, 16), 0.2, dtype=np.float32) for b in ("B02", "B03", "B04")}
        m1 = compute_ephemeral_masks(rgb, settings.masks)
        m2 = compute_ephemeral_masks(rgb, settings.masks)
        combined = combine_pair_masks(m1, m2)
        quality = compute_quality_score(
            m1,
            m2,
            settings.quality,
            registration_error_px=0.1,
            date_t1="2020-01-01",
            date_t2="2021-01-01",
            combined=combined,
        )
        result = rgb_only_gate("x", rgb, rgb, combined, quality, settings.gate).result
        assert result.classical_gate == "low_quality"
        assert result.gate_reason == RGB_ONLY_REASON
        assert result.spectral_indices_available is False
        assert len(result.unavailable) == 9

    def test_the_degraded_result_is_a_separate_type(self) -> None:
        """Nulls poured into ClassicalResult would invite averaging the two."""
        from satchangegate.features.classical import ClassicalResult, RgbOnlyResult

        assert RgbOnlyResult is not ClassicalResult
        assert "ndvi_delta_mean" not in RgbOnlyResult.model_fields
        assert "spectral_indices_available" not in ClassicalResult.model_fields

    def test_the_refusal_is_distinguishable_from_a_tier_zero_refusal(self) -> None:
        from satchangegate.features.classical import RGB_ONLY_REASON

        assert "spectral bands unavailable" in RGB_ONLY_REASON
        assert RGB_ONLY_REASON != "failed Tier 0 quality checks"

    def test_the_learned_scorer_cannot_see_a_degraded_row(self) -> None:
        """The degraded lane produces no GateFeatures at all, so there is nothing
        for a learned scorer to score -- the refusal cannot be bypassed."""
        from satchangegate.baseline import FEATURE_NAMES
        from satchangegate.features.classical import RgbOnlyResult

        fields = set(RgbOnlyResult.model_fields)
        assert not set(FEATURE_NAMES) <= fields
