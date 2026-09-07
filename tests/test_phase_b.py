"""Tests for the scorer, the conformal threshold, budgets, PIF, and embeddings.

Each of these ships a number the repo intends to publish, so each gets a test
that could falsify it rather than only one that exercises it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from satchangegate.conformal import (
    calibrate,
    evaluate_guarantee,
    false_negative_rate,
    hoeffding_bentkus_ucb,
    split_calibration,
)
from satchangegate.data.embeddings import (
    EMBEDDING_FEATURE_NAMES,
    FIRST_EMBEDDING_YEAR,
    PROBE_FEATURE_NAMES,
    aoi_bbox,
    clamp_to_coverage,
    coverage_report,
    embedding_change_probe,
    embedding_features,
    scene_centroid,
)
from satchangegate.operating_points import (
    curve_for_budgets,
    threshold_for_call_budget,
)
from satchangegate.preprocess.radiometric import normalize_to_reference

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


class TestConformal:
    """A guarantee that cannot fail is not a guarantee."""

    @staticmethod
    def _separable(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
        rng = np.random.RandomState(seed)
        labels = rng.binomial(1, 0.5, n)
        scores = np.clip(
            rng.normal(0.68, 0.16, n) * labels + rng.normal(0.32, 0.16, n) * (1 - labels),
            0.0,
            1.0,
        )
        return scores, labels, [f"c{i % 6}" for i in range(n)]

    def test_the_bound_is_respected_on_calibration_data(self) -> None:
        scores, labels, cities = self._separable()
        r = calibrate(scores, labels, cities, alpha=0.2, delta=0.1)
        assert r.controlled
        assert r.risk_ucb <= 0.2
        assert r.risk_hat <= r.risk_ucb, "the bound must sit above the point estimate"

    def test_a_tighter_alpha_buys_a_lower_threshold(self) -> None:
        """Lower tolerated miss rate means flagging more, which costs more."""
        scores, labels, cities = self._separable()
        loose = calibrate(scores, labels, cities, alpha=0.30, delta=0.1)
        tight = calibrate(scores, labels, cities, alpha=0.10, delta=0.1)
        assert tight.lam is not None and loose.lam is not None
        assert tight.lam <= loose.lam
        assert tight.flagged_fraction >= loose.flagged_fraction

    def test_an_uncertifiable_alpha_is_reported_not_faked(self) -> None:
        """With too few positives no threshold can be certified; say so."""
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        r = calibrate(scores, labels, ["a"] * 4, alpha=0.01, delta=0.01)
        assert not r.controlled
        assert r.lam is None

    def test_ucb_widens_as_the_sample_shrinks(self) -> None:
        assert hoeffding_bentkus_ucb(0.1, 10, 0.1) > hoeffding_bentkus_ucb(0.1, 1000, 0.1)

    def test_ucb_is_never_below_the_point_estimate(self) -> None:
        for n in (1, 10, 100, 10_000):
            assert hoeffding_bentkus_ucb(0.25, n, 0.1) >= 0.25

    def test_false_negative_rate_is_over_positives_only(self) -> None:
        scores = np.array([0.9, 0.1, 0.9, 0.1])
        labels = np.array([1, 1, 0, 0])
        assert false_negative_rate(scores, labels, 0.5) == pytest.approx(0.5)

    def test_no_positives_means_no_false_negatives(self) -> None:
        assert false_negative_rate(np.array([0.1, 0.2]), np.array([0, 0]), 0.5) == 0.0

    def test_guarantee_is_evaluated_per_city(self) -> None:
        """Held-out cities are not exchangeable with calibration cities; the
        per-city breakdown is the finding, not a diagnostic."""
        scores, labels, cities = self._separable()
        r = calibrate(scores, labels, cities, alpha=0.2, delta=0.1)
        v = evaluate_guarantee(r, scores, labels, cities)
        assert v["cities_tested"] == len(set(cities))
        assert set(v["per_city"]) == set(cities)
        assert 0 <= v["cities_where_it_held"] <= v["cities_tested"]

    def test_calibration_splits_by_city_not_by_tile(self) -> None:
        """A tile-level split would leak scene identity across the boundary."""
        rows = [{"city": f"c{i % 5}", "label": i % 2} for i in range(100)]
        cal, fit = split_calibration(rows, fraction=0.4, seed=1)
        assert cal and fit
        assert not ({r["city"] for r in cal} & {r["city"] for r in fit})

    def test_splitting_needs_more_than_one_city(self) -> None:
        with pytest.raises(ValueError, match="at least two cities"):
            split_calibration([{"city": "only", "label": 1}])


class TestOperatingPoints:
    @staticmethod
    def _scores() -> tuple[np.ndarray, np.ndarray]:
        scores = np.linspace(0.0, 1.0, 100)
        labels = (scores > 0.5).astype(int)
        return scores, labels

    def test_a_budget_never_buys_more_calls_than_it_can_pay_for(self) -> None:
        scores, labels = self._scores()
        for point in curve_for_budgets(scores, labels, cost_per_call_usd=0.01):
            assert point.n_flagged <= point.affordable_calls

    def test_a_bigger_budget_never_reduces_recall(self) -> None:
        scores, labels = self._scores()
        points = curve_for_budgets(scores, labels, 0.01, (0.10, 0.25, 0.50, 1.00))
        recalls = [p.recall for p in points]
        assert recalls == sorted(recalls)

    def test_a_zero_budget_flags_nothing(self) -> None:
        scores, labels = self._scores()
        point = curve_for_budgets(scores, labels, 1000.0, (0.0,))[0]
        assert point.affordable_calls == 0
        assert point.n_flagged == 0

    def test_threshold_for_budget_respects_ties(self) -> None:
        """Several tiles can share a score; admitting all of them would overshoot."""
        scores = np.array([0.5] * 10)
        assert threshold_for_call_budget(scores, 3) == pytest.approx(0.5)

    def test_spend_is_derived_from_what_is_actually_flagged(self) -> None:
        scores, labels = self._scores()
        point = curve_for_budgets(scores, labels, 0.02, (0.50,))[0]
        assert point.spend_usd == pytest.approx(point.n_flagged * 0.02)


class TestRadiometricNormalization:
    @staticmethod
    def _pair(gain: float, offset: float, seed: int = 0):
        rng = np.random.RandomState(seed)
        b1 = {b: np.clip(rng.rand(96, 96).astype(np.float32) * 0.4 + 0.05, 0, 1) for b in BANDS}
        b2 = {b: (v * gain + offset).astype(np.float32) for b, v in b1.items()}
        return b1, b2

    def test_it_recovers_a_known_gain_and_offset(self) -> None:
        b1, b2 = self._pair(gain=1.30, offset=0.04)
        out, report = normalize_to_reference(b1, b2)
        assert report.applied
        # Inverting y = 1.3x + 0.04 gives gain 1/1.3 and offset -0.04/1.3.
        assert report.gains["B04"] == pytest.approx(1 / 1.30, abs=0.03)
        assert report.offsets["B04"] == pytest.approx(-0.04 / 1.30, abs=0.01)
        assert np.abs(out["B04"] - b1["B04"]).mean() < 1e-3

    def test_it_preserves_genuine_change_while_removing_the_shift(self) -> None:
        """The point of PIF selection: correct the scene, keep the change."""
        b1, b2 = self._pair(gain=1.25, offset=0.03)
        for band in BANDS:
            b2[band] = b2[band].copy()
            b2[band][10:30, 10:30] = 0.9
        out, report = normalize_to_reference(b1, b2)
        assert report.applied

        changed = np.zeros((96, 96), dtype=bool)
        changed[10:30, 10:30] = True
        residual_unchanged = np.abs(out["B04"] - b1["B04"])[~changed].mean()
        residual_changed = np.abs(out["B04"] - b1["B04"])[changed].mean()
        assert residual_unchanged < 0.01
        assert residual_changed > 10 * residual_unchanged

    def test_an_already_matched_pair_is_left_alone(self) -> None:
        b1, b2 = self._pair(gain=1.0, offset=0.0)
        _, report = normalize_to_reference(b1, b2)
        assert report.gains["B08"] == pytest.approx(1.0, abs=0.02)
        assert report.offsets["B08"] == pytest.approx(0.0, abs=0.005)

    def test_too_few_valid_pixels_declines_rather_than_guessing(self) -> None:
        b1, b2 = self._pair(gain=1.2, offset=0.02)
        valid = np.zeros((96, 96), dtype=bool)
        valid[:2, :2] = True
        out, report = normalize_to_reference(b1, b2, valid)
        assert not report.applied
        assert "too few" in report.reason
        assert np.array_equal(out["B04"], b2["B04"])

    def test_the_reference_acquisition_is_never_modified(self) -> None:
        b1, b2 = self._pair(gain=1.4, offset=0.05)
        before = b1["B04"].copy()
        normalize_to_reference(b1, b2)
        assert np.array_equal(b1["B04"], before)


class TestEmbeddings:
    @staticmethod
    def _block(value: float, noise: float = 0.0, seed: int = 0, size: int = 8) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return (np.full((size, size, 64), value) + rng.randn(size, size, 64) * noise).astype(
            np.float32
        )

    def test_missing_embeddings_read_as_unknown_not_zero(self) -> None:
        """Unknown is not clean -- the rule this repo applies to masks applies here."""
        feats = embedding_features(None, None)
        assert set(feats) == set(EMBEDDING_FEATURE_NAMES)
        assert all(v is None for v in feats.values())

    def test_a_tile_unlike_its_scene_scores_a_greater_centroid_distance(self) -> None:
        centroid = scene_centroid(self._block(0.1))
        near = embedding_features(self._block(0.1), centroid)
        far = embedding_features(self._block(0.9), centroid)
        assert far["emb_centroid_distance"] > near["emb_centroid_distance"]

    def test_dispersion_tracks_internal_heterogeneity(self) -> None:
        centroid = scene_centroid(self._block(0.4))
        uniform = embedding_features(self._block(0.4, noise=0.0), centroid)
        mixed = embedding_features(self._block(0.4, noise=0.5), centroid)
        assert mixed["emb_dispersion"] > uniform["emb_dispersion"]

    def test_no_raw_dimension_is_exposed_as_a_feature(self) -> None:
        """Raw coordinates encode absolute location, which lets a model key on
        city identity instead of on change."""
        assert not any(name.startswith("emb_dim") for name in EMBEDDING_FEATURE_NAMES)
        assert len(EMBEDDING_FEATURE_NAMES) == 3

    def test_the_probe_is_named_and_kept_separate_from_results(self) -> None:
        assert PROBE_FEATURE_NAMES == ("emb_change_probe",)
        assert not set(PROBE_FEATURE_NAMES) & set(EMBEDDING_FEATURE_NAMES)

    def test_probe_reports_dissimilarity_between_two_years(self) -> None:
        same = embedding_change_probe(self._block(0.5), self._block(0.5))
        differ = embedding_change_probe(self._block(0.5), self._block(-0.5))
        assert same["emb_change_probe"] == pytest.approx(0.0, abs=1e-5)
        assert differ["emb_change_probe"] > 1.0

    def test_clamping_is_reported_because_it_is_what_contaminates_the_probe(self) -> None:
        assert clamp_to_coverage(2018) == (2018, False)
        assert clamp_to_coverage(2015) == (FIRST_EMBEDDING_YEAR, True)

    def test_coverage_report_counts_what_is_measurable(self) -> None:
        class _Pair:
            def __init__(self, pid: str, t1: str, t2: str) -> None:
                self.pair_id, self.date_t1, self.date_t2 = pid, t1, t2

        report = coverage_report(
            [
                _Pair("old", "2015-08-20", "2018-02-05"),
                _Pair("new", "2017-04-14", "2018-04-02"),
            ]
        )
        assert report["n_t2_covered"] == 2
        assert report["n_t1_covered"] == 1
        assert report["n_bitemporal_measurable"] == 1


class TestScorerArtifact:
    """A stale artifact must fail loudly, not score quietly."""

    @staticmethod
    def _rows(n: int = 80) -> list[dict]:
        from satchangegate.baseline import FEATURE_NAMES

        rng = np.random.RandomState(0)
        rows = []
        for i in range(n):
            label = i % 2
            row = {name: float(rng.rand() + label) for name in FEATURE_NAMES}
            row.update({"label": label, "city": f"c{i % 4}"})
            rows.append(row)
        return rows

    def test_round_trip_preserves_predictions(self, tmp_path: Path) -> None:
        from satchangegate.baseline import feature_matrix
        from satchangegate.scorer import fit_scorer, load_scorer, save_scorer

        rows = self._rows()
        artifact = fit_scorer(rows)
        path = save_scorer(artifact, tmp_path / "scorer.pkl")
        reloaded = load_scorer(path)
        x, _ = feature_matrix(rows)
        assert np.allclose(artifact.predict_proba(x), reloaded.predict_proba(x))

    def test_a_model_card_is_written_alongside(self, tmp_path: Path) -> None:
        import json

        from satchangegate.scorer import fit_scorer, save_scorer

        path = save_scorer(fit_scorer(self._rows()), tmp_path / "scorer.pkl")
        card = json.loads(path.with_suffix(".card.json").read_text(encoding="utf-8"))
        assert card["feature_names"]
        assert card["limitations"]
        assert card["train_cities"]

    def test_a_changed_feature_vector_refuses_to_load(self, tmp_path: Path) -> None:
        from satchangegate.scorer import StaleScorerError, load_scorer, save_scorer

        artifact = self._fitted()
        object.__setattr__(artifact, "feature_hash", "0000000000000000")
        path = save_scorer(artifact, tmp_path / "scorer.pkl")
        with pytest.raises(StaleScorerError, match="different feature vector"):
            load_scorer(path)

    def test_a_future_artifact_version_refuses_to_load(self, tmp_path: Path) -> None:
        from satchangegate.scorer import StaleScorerError, load_scorer, save_scorer

        artifact = self._fitted()
        object.__setattr__(artifact, "artifact_version", 999)
        path = save_scorer(artifact, tmp_path / "scorer.pkl")
        with pytest.raises(StaleScorerError, match="artifact version"):
            load_scorer(path)

    def test_a_missing_artifact_says_how_to_make_one(self, tmp_path: Path) -> None:
        from satchangegate.scorer import load_scorer

        with pytest.raises(FileNotFoundError, match="fit-scorer"):
            load_scorer(tmp_path / "absent.pkl")

    def test_tier0_refusals_are_never_scored_by_the_model(self) -> None:
        """Tier 0 keeps the first and final word under the learned scorer too."""
        from satchangegate.scorer import score_rows_learned

        rows = self._rows(20)
        rows[0]["valid_observation"] = False
        scored = score_rows_learned(rows, self._fitted(), threshold=0.5)
        assert scored[0]["gate"] == "low_quality"
        assert scored[0]["gate_confidence"] == 0.0

    def test_threshold_controls_the_decision(self) -> None:
        from satchangegate.scorer import score_rows_learned

        rows = self._rows(40)
        artifact = self._fitted()
        permissive = score_rows_learned(rows, artifact, threshold=0.0)
        strict = score_rows_learned(rows, artifact, threshold=1.01)
        n_perm = sum(1 for r in permissive if r["gate"] == "candidate_change")
        n_strict = sum(1 for r in strict if r["gate"] == "candidate_change")
        assert n_perm > n_strict == 0

    def test_every_decision_still_carries_a_reason(self) -> None:
        from satchangegate.scorer import score_rows_learned

        scored = score_rows_learned(self._rows(10), self._fitted(), threshold=0.5)
        assert all(r["gate_reason"] for r in scored)

    def _fitted(self):
        from satchangegate.scorer import fit_scorer

        return fit_scorer(self._rows())


@pytest.mark.oscd
def test_aoi_bbox_reads_the_only_georeference_oscd_ships(oscd_root: Path) -> None:
    """The rect rasters carry no CRS, so the AOI polygon is all there is."""
    bbox = aoi_bbox(oscd_root / "beirut")
    assert bbox is not None
    min_lon, min_lat, max_lon, max_lat = bbox
    assert min_lon < max_lon and min_lat < max_lat
    assert 35.0 < min_lon < 36.0
    assert 33.0 < min_lat < 34.0


class TestScorerDispatch:
    """Opting into the learned scorer must not be able to break a run."""

    def test_rules_are_the_default(self) -> None:
        from satchangegate.config import get_settings

        assert get_settings().scorer.kind == "rules"

    def test_a_missing_artifact_warns_and_falls_back_to_the_rules(self, tmp_path: Path) -> None:
        """A missing model is an operations problem; it must not read as a
        modelling result."""
        from satchangegate.config import get_settings
        from satchangegate.evaluate import score_rows

        settings = get_settings().model_copy(
            update={
                "scorer": get_settings().scorer.model_copy(
                    update={"kind": "learned", "path": str(tmp_path / "absent.pkl")}
                )
            }
        )
        rows = [
            {
                **dict.fromkeys(
                    __import__("satchangegate.baseline", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES,
                    0.0,
                ),
                "label": 1,
                "city": "c",
                "valid_observation": True,
            }
        ]
        with pytest.warns(RuntimeWarning, match="fit-scorer"):
            _cm, scored = score_rows(rows, settings)
        assert scored[0]["gate"] in {"no_change", "candidate_change"}
        assert "learned scorer" not in scored[0]["gate_reason"]
