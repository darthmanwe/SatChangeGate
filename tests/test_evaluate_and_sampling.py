"""Tests for the modules that produce the published numbers.

``evaluate.py`` computes every headline metric this repo quotes and had no direct
test of its own; the ``low_quality`` exclusion rule in ``score_rows`` -- arguably
the most consequential methodological decision in the project -- was entirely
unexercised. The sampling and second-tier reporting below are new, and each test
here pins a specific way the previous version misreported something.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from satchangegate.config import Settings, get_settings
from satchangegate.e2e import SAMPLE_STRATEGIES, select_candidates
from satchangegate.evaluate import run_eval, score_rows
from satchangegate.metrics import ConfusionMatrix, pixel_confusion, pixel_metrics
from satchangegate.vlm_report import analyse


def _row(label: int, *, valid: bool = True, strong: float = 0.0) -> dict:
    """A feature row good enough for ``score_rows``."""
    return {
        "tile_id": f"t{label}{strong}{valid}",
        "city": "somewhere",
        "split": "test",
        "label": label,
        "ssim": 1.0,
        "phash_distance": 0,
        "ndvi_delta_mean": 0.0,
        "ndbi_delta_mean": 0.0,
        "ndwi_delta_mean": 0.0,
        "ndvi_delta_abs_mean": strong,
        "ndbi_delta_abs_mean": 0.0,
        "ndwi_delta_abs_mean": 0.0,
        "cva_magnitude_mean": 0.0,
        "changed_area_percent": 0.0,
        "valid_observation": valid,
    }


class TestScoreRows:
    """The low_quality exclusion is a refusal to judge, not a negative prediction."""

    def test_low_quality_is_excluded_from_the_confusion_matrix(self, settings: Settings) -> None:
        rows = [
            _row(1, strong=0.99),  # candidate
            _row(0),  # no_change
            _row(1, valid=False),  # refused by Tier 0
            _row(0, valid=False),  # refused by Tier 0
        ]
        cm, scored = score_rows(rows, settings)
        assert cm.n == 2, "Tier-0 refusals must not enter the metrics"
        assert sum(1 for r in scored if r["gate"] == "low_quality") == 2

    def test_a_tier0_refusal_is_not_counted_as_a_negative_prediction(
        self, settings: Settings
    ) -> None:
        """Counting refusals as negatives would inflate recall's denominator.

        A positive tile the pipeline declined to look at is not a miss by the
        gate; it is a pair no decision was made about.
        """
        cm_refused, _ = score_rows([_row(1, valid=False)], settings)
        assert cm_refused.n_positive == 0
        assert cm_refused.fn == 0

    def test_every_scored_row_carries_a_reason_and_confidence(self, settings: Settings) -> None:
        _, scored = score_rows([_row(1, strong=0.99), _row(0)], settings)
        assert all(r["gate_reason"] for r in scored)
        assert all(0.0 <= r["gate_confidence"] <= 1.0 for r in scored)


class TestRunEval:
    def test_reports_three_separate_reductions(self, fixture_root: Path, tmp_path: Path) -> None:
        """Tier 0 refuses; the gate decides. One blended number credited the gate
        with co-registration failures it had no part in."""
        summary = run_eval(fixture_root, "all", tmp_path)
        for key in ("tier0_refused_pct", "gate_filtered_pct", "total_reduction_pct"):
            assert key in summary
        assert summary["n_assessable"] == summary["n_tiles"] - summary["n_low_quality"]

    def test_gate_filter_rate_is_over_assessable_tiles_only(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        s = run_eval(fixture_root, "all", tmp_path)
        n_gate_filtered = s["n_assessable"] - s["n_candidate_change"]
        expected = 100.0 * n_gate_filtered / max(1, s["n_assessable"])
        assert s["gate_filtered_pct"] == pytest.approx(expected, abs=0.01)

    def test_cities_scored_can_be_fewer_than_cities_present(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        """`valid_observation` is scene-level, so one bad city drops all its tiles.

        This is how "measured on 10 held-out cities" was true of the tile index
        but false of the metrics computed from it.
        """
        s = run_eval(fixture_root, "all", tmp_path)
        assert set(s["cities_scored"]) <= set(s["cities"])

    def test_pixel_metrics_are_opt_in_and_reproducible(
        self, fixture_root: Path, tmp_path: Path
    ) -> None:
        assert "pixel_metrics" not in run_eval(fixture_root, "all", tmp_path)
        s = run_eval(fixture_root, "all", tmp_path, pixel_metrics=True)
        pm = s["pixel_metrics"]
        assert pm["n"] > 0
        assert 0.0 <= pm["f1"] <= 1.0
        assert 0.0 <= pm["iou"] <= 1.0


class TestPixelMetrics:
    def test_perfect_prediction_scores_one(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        assert pixel_metrics(mask, mask)["f1"] == pytest.approx(1.0)
        assert pixel_metrics(mask, mask)["iou"] == pytest.approx(1.0)

    def test_valid_mask_excludes_unobserved_pixels(self) -> None:
        """Masked-out cloud must not accumulate free true negatives."""
        pred = np.zeros((4, 4), dtype=bool)
        gt = np.zeros((4, 4), dtype=bool)
        valid = np.zeros((4, 4), dtype=bool)
        valid[0, :] = True
        assert pixel_confusion(pred, gt, valid).n == 4

    def test_shape_mismatch_raises_rather_than_comparing_garbage(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            pixel_confusion(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


class TestCandidateSelection:
    """Regression for the sample that was actually an alphabetical truncation."""

    @staticmethod
    def _pool() -> list[tuple[str, str]]:
        # Deliberately lopsided, like the real split: one city dominates.
        plan = {"alpha": 11, "beta": 49, "gamma": 33, "delta": 70, "epsilon": 9}
        return [(c, f"{c}_t{i}") for c, n in plan.items() for i in range(n)]

    def test_stratified_reaches_every_city(self) -> None:
        """The failure this exists to prevent: five of ten cities never sampled."""
        chosen = select_candidates(self._pool(), 100, strategy="stratified", seed=42)
        cities = {t.split("_t")[0] for t in chosen}
        assert cities == {"alpha", "beta", "gamma", "delta", "epsilon"}

    def test_stratified_spends_exactly_the_budget(self) -> None:
        for budget in (5, 17, 100, 171):
            chosen = select_candidates(self._pool(), budget, strategy="stratified", seed=1)
            assert len(chosen) == min(budget, len(self._pool()))

    def test_stratified_is_roughly_proportional(self) -> None:
        chosen = select_candidates(self._pool(), 100, strategy="stratified", seed=42)
        counts: dict[str, int] = {}
        for t in chosen:
            counts[t.split("_t")[0]] = counts.get(t.split("_t")[0], 0) + 1
        # delta holds 70/172 of the pool, so it should get roughly 40 of 100.
        assert 33 <= counts["delta"] <= 47

    def test_stratified_is_deterministic_under_a_seed(self) -> None:
        a = select_candidates(self._pool(), 40, strategy="stratified", seed=7)
        b = select_candidates(self._pool(), 40, strategy="stratified", seed=7)
        c = select_candidates(self._pool(), 40, strategy="stratified", seed=8)
        assert a == b
        assert a != c

    def test_sequential_reproduces_the_alphabetical_truncation(self) -> None:
        """Kept only so the earlier, unrepresentative run can be reproduced."""
        chosen = select_candidates(self._pool(), 60, strategy="sequential", seed=42)
        cities = {t.split("_t")[0] for t in chosen}
        assert cities < {"alpha", "beta", "gamma", "delta", "epsilon"}
        assert "epsilon" not in cities

    def test_no_budget_returns_everything(self) -> None:
        assert len(select_candidates(self._pool(), None)) == len(self._pool())

    def test_zero_budget_spends_nothing(self) -> None:
        assert select_candidates(self._pool(), 0) == []

    def test_unknown_strategy_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ValueError, match="unknown sample strategy"):
            select_candidates(self._pool(), 10, strategy="whatever")

    @pytest.mark.parametrize("strategy", SAMPLE_STRATEGIES)
    def test_never_exceeds_the_budget(self, strategy: str) -> None:
        """The cap is a spend limit; overshooting it costs real money."""
        assert len(select_candidates(self._pool(), 13, strategy=strategy)) <= 13


class TestVlmReport:
    """The gate+VLM headline must come from a command, not from a spreadsheet."""

    @staticmethod
    def _calls() -> list[dict]:
        # 10 verified: 8 truly changed, 2 gate errors. The VLM forwards 6 of the
        # 8 and rejects both errors.
        rows = []
        for i in range(8):
            rows.append(
                {
                    "tile_id": f"c_t{i}",
                    "city": "c",
                    "label": 1,
                    "vlm_called": True,
                    "vlm_verdict": "real_change" if i < 6 else "uncertain",
                    "vlm_change_type": "construction",
                    "cost_usd": 0.004,
                }
            )
        for i in range(2):
            rows.append(
                {
                    "tile_id": f"d_t{i}",
                    "city": "d",
                    "label": 0,
                    "vlm_called": True,
                    "vlm_verdict": "likely_artifact",
                    "vlm_change_type": "weather_artifact",
                    "cost_usd": 0.004,
                }
            )
        return rows

    def test_precision_improves_and_carries_intervals(self) -> None:
        s = analyse(self._calls())
        assert s["precision"]["gate_alone"]["value"] == pytest.approx(0.8)
        assert s["precision"]["gate_plus_vlm"]["value"] == pytest.approx(1.0)
        for key in ("gate_alone", "gate_plus_vlm"):
            lo, hi = s["precision"][key]["ci95"]
            assert 0.0 <= lo <= hi <= 1.0

    def test_uncertain_counts_as_a_rejection_not_a_dropped_row(self) -> None:
        """An unconfirmed candidate still needs an analyst; dropping it would
        flatter the tier."""
        s = analyse(self._calls())
        assert s["vlm_retained_true_changes"]["successes"] == 6
        assert s["vlm_retained_true_changes"]["total"] == 8

    def test_rejection_rate_is_measured_against_gate_errors_only(self) -> None:
        s = analyse(self._calls())
        assert s["vlm_rejected_gate_errors"]["total"] == 2
        assert s["vlm_rejected_gate_errors"]["successes"] == 2

    def test_coverage_by_city_is_reported(self) -> None:
        """The previous run verified 3 of 10 cities and said nothing about it."""
        assert analyse(self._calls())["cities_verified"] == {"c": 8, "d": 2}


@pytest.mark.oscd
class TestAgainstRealDataset:
    """Exercised only by `make test-all`, which needs `make data` first.

    The `oscd` marker was declared in pyproject and deselected by default, but no
    test carried it -- so `make test-all` ran exactly the same suite as `make
    test` and nothing ever touched the real 13-band imagery.
    """

    def test_split_is_disjoint_and_covers_every_city(self, oscd_root: Path) -> None:
        splits = json.loads((oscd_root / "splits.json").read_text(encoding="utf-8"))
        train, test = set(splits["train"]), set(splits["test"])
        assert not (train & test)
        assert len(train) + len(test) == 24

    def test_a_real_city_gates_end_to_end(self, oscd_root: Path) -> None:
        from satchangegate.data.oscd import discover_pairs, load_label_mask
        from satchangegate.data.tiles import tiles_for_pair
        from satchangegate.evaluate import build_scene_cache, mask_and_features_for_tile

        pairs = [p for p in discover_pairs(oscd_root) if p.pair_id == "beirut"]
        if not pairs:
            pytest.skip("beirut not present")
        pair = pairs[0]
        label = load_label_mask(pair)
        assert label is not None
        cache = build_scene_cache(pair, get_settings())
        tiles = tiles_for_pair(pair, label)[:5]
        assert tiles
        for tile in tiles:
            mask, feats = mask_and_features_for_tile(cache, tile, get_settings())
            assert mask.shape == (tile.y1 - tile.y0, tile.x1 - tile.x0)
            assert np.isfinite(feats.ssim)

    def test_registration_error_is_measured_not_hardcoded(self, oscd_root: Path) -> None:
        """saclay_w exceeds the 1.5 px tolerance; that is the whole reason Tier 0
        fires on this benchmark at all."""
        from satchangegate.data.oscd import discover_pairs
        from satchangegate.evaluate import build_scene_cache

        pairs = {p.pair_id: p for p in discover_pairs(oscd_root)}
        if "saclay_w" not in pairs:
            pytest.skip("saclay_w not present")
        cache = build_scene_cache(pairs["saclay_w"], get_settings())
        assert cache.registration_px > get_settings().quality.registration_error_px_max


def test_confusion_matrix_accumulates_across_tiles() -> None:
    """The pixel accumulator sums tile-level confusions into one scene total."""
    acc = ConfusionMatrix()
    for _ in range(3):
        cm = pixel_confusion(np.ones((2, 2), bool), np.ones((2, 2), bool))
        acc.tp += cm.tp
        acc.tn += cm.tn
    assert acc.tp == 12


class TestBatchReattachment:
    """A submitted batch is already paid for; a rerun must not buy it twice."""

    @staticmethod
    def _manifest(tmp_path: Path, tile_ids: list[str], model: str = "claude-sonnet-5") -> Path:
        path = tmp_path / "_e2e_test_batch.json"
        path.write_text(
            json.dumps({"batch_id": "msgbatch_x", "model": model, "tile_ids": tile_ids}),
            encoding="utf-8",
        )
        return path

    def test_reattaches_when_the_manifest_covers_the_same_work(self, tmp_path: Path) -> None:
        from satchangegate.e2e import _live_batch

        path = self._manifest(tmp_path, ["a", "b", "c"])
        assert _live_batch(path, ["c", "b", "a"], "claude-sonnet-5") == "msgbatch_x"

    def test_refuses_a_manifest_for_a_different_selection(self, tmp_path: Path) -> None:
        """Collecting it would attach verdicts to tiles they were not computed for."""
        from satchangegate.e2e import _live_batch

        path = self._manifest(tmp_path, ["a", "b"])
        assert _live_batch(path, ["a", "b", "c"], "claude-sonnet-5") is None

    def test_refuses_a_manifest_from_a_different_model(self, tmp_path: Path) -> None:
        from satchangegate.e2e import _live_batch

        path = self._manifest(tmp_path, ["a"], model="claude-haiku-4-5")
        assert _live_batch(path, ["a"], "claude-sonnet-5") is None

    def test_absent_or_corrupt_manifest_is_not_an_error(self, tmp_path: Path) -> None:
        from satchangegate.e2e import _live_batch

        assert _live_batch(tmp_path / "nope.json", ["a"], "m") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert _live_batch(bad, ["a"], "m") is None
