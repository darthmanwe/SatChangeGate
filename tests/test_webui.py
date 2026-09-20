"""The web layer, tested where it can leak, mislead, or spend.

Route smoke tests are the least of it. What matters here is that the server
refuses the things it must refuse -- an arbitrary path, an unredacted artifact, a
state change without a token, a Host it does not recognise -- and that it tells
the truth when something has not been computed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="the web UI needs the [ui] extra")

from fastapi.testclient import TestClient

from satchangegate.services import (
    SERVICES,
    E2ERequest,
    EvalRequest,
    OperatingPointsRequest,
)
from satchangegate.webui.app import AppConfig, create_app
from satchangegate.webui.artifacts import ArtifactMissing, ArtifactStore


@pytest.fixture
def config(tmp_path):
    """An app that never reads a real .env.

    The session fixture unsets ANTHROPIC_API_KEY so no test can authenticate by
    accident. An app factory that re-read .env at startup would undo exactly
    that, which is why load_env is injectable at all.
    """
    return AppConfig(
        reports=tmp_path / "reports",
        sample=tmp_path / "sample",
        run_root=tmp_path / "runs",
        load_env=False,
    )


@pytest.fixture
def client(config):
    # base_url sets the Host header. TestClient defaults to "testserver", which
    # the origin guard correctly refuses -- so pointing it at loopback is part of
    # the test, not a workaround for it.
    return TestClient(create_app(config), base_url="http://127.0.0.1")


class TestTrustBoundary:
    def test_a_hostile_host_header_is_refused(self, client) -> None:
        """Loopback binding does not stop a rebinding attack; checking Host does."""
        res = client.get("/api/health", headers={"Host": "evil.example.com"})
        assert res.status_code == 421

    def test_loopback_hosts_are_accepted(self, client) -> None:
        for host in ("127.0.0.1:8000", "localhost:8000"):
            assert client.get("/api/health", headers={"Host": host}).status_code == 200

    def test_state_changing_routes_need_the_token(self, client) -> None:
        res = client.post("/api/commands/preview", json={"operation": "eval"})
        assert res.status_code == 403

    def test_a_wrong_token_is_refused(self, client) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval"},
            headers={"X-SCG-Token": "not-the-token"},
        )
        assert res.status_code == 403

    def test_the_right_token_is_accepted(self, client, config) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval", "params": {"split": "test"}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 200

    def test_cross_origin_state_change_is_refused(self, client, config) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval"},
            headers={"X-SCG-Token": config.token, "Origin": "https://evil.example.com"},
        )
        assert res.status_code == 403

    def test_get_routes_never_need_a_token(self, client) -> None:
        """A side-effect-free route that demanded a token would be theatre."""
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/artifacts").status_code == 200

    def test_the_page_carries_the_token_and_the_api_does_not(self, client, config) -> None:
        page = client.get("/").text
        assert config.token in page
        assert config.token not in client.get("/api/capabilities").text


class TestNoGenericFileServing:
    def test_an_unknown_artifact_key_is_refused(self, client) -> None:
        assert client.get("/api/artifacts/../../.env").status_code in (404, 400)
        assert client.get("/api/artifacts/nope").status_code == 404

    def test_the_unredacted_gate_features_are_not_served(self, tmp_path) -> None:
        """`classical_full.json` sits beside the metadata the model saw.

        Serving it anywhere the verdict is displayed would confound the one
        statistic the second tier exists to produce: agreement between the gate
        and something that could not see the gate's answer.
        """
        store = ArtifactStore(reports=tmp_path)
        pkg = tmp_path / "e2e_packages" / "city_r0c0"
        pkg.mkdir(parents=True)
        (pkg / "classical_full.json").write_text("{}", encoding="utf-8")
        (pkg / "metadata.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ArtifactMissing):
            store.evidence_file("city_r0c0", "classical_full.json")
        assert store.evidence_file("city_r0c0", "metadata.json")[0] == b"{}"

    @pytest.mark.parametrize(
        "tile_id", ["../..", "..", "a/../../b", "C:\\Windows", "tile-with-dash", ""]
    )
    def test_traversal_and_odd_tile_ids_are_refused(self, tmp_path, tile_id) -> None:
        store = ArtifactStore(reports=tmp_path)
        with pytest.raises(ArtifactMissing):
            store.evidence_file(tile_id, "metadata.json")


class TestMissingIsAState:
    def test_an_absent_artifact_names_its_producing_command(self, client) -> None:
        res = client.get("/api/artifacts/e2e_ledger")
        assert res.status_code == 404
        detail = res.json()["detail"]
        assert detail["available"] is False
        assert "satchangegate e2e" in detail["produced_by"]

    def test_the_catalogue_lists_everything_with_availability(self, client) -> None:
        rows = client.get("/api/artifacts").json()["artifacts"]
        assert rows
        for row in rows:
            assert "available" in row and "produced_by" in row

    def test_the_committed_sample_answers_when_local_is_absent(self, tmp_path) -> None:
        sample = tmp_path / "sample"
        sample.mkdir()
        (sample / "_eval_test.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        store = ArtifactStore(reports=tmp_path / "nothing", sample=sample)
        assert store.resolve("eval_test").source == "sample"

    def test_local_wins_over_the_sample(self, tmp_path) -> None:
        """Half a local run stitched to half a sample is a number nobody can reproduce."""
        reports, sample = tmp_path / "r", tmp_path / "s"
        reports.mkdir()
        sample.mkdir()
        (reports / "_eval_test.json").write_text('{"metrics":{"f1":1}}', encoding="utf-8")
        (sample / "_eval_test.json").write_text('{"metrics":{"f1":0}}', encoding="utf-8")
        store = ArtifactStore(reports=reports, sample=sample)
        assert store.resolve("eval_test").source == "local"
        assert store.read_json("eval_test")["data"]["metrics"]["f1"] == 1

    def test_artifacts_absent_from_the_sample_say_so(self) -> None:
        from satchangegate.webui.artifacts import ARTIFACTS

        assert ARTIFACTS["e2e_ledger"].in_sample is False
        assert ARTIFACTS["features_test"].in_sample is False


class TestCapabilities:
    def test_capabilities_never_offers_what_it_cannot_run(self, client) -> None:
        caps = client.get("/api/capabilities").json()
        by_name = {o["name"]: o for o in caps["operations"]}
        assert set(by_name) == set(SERVICES)
        for op in caps["operations"]:
            if not op["available"]:
                assert op["blocked_by"], "blocked with no reason given"

    def test_unconditionally_paid_operations_are_blocked_while_spend_is_off(self, client) -> None:
        caps = client.get("/api/capabilities").json()
        for op in caps["operations"]:
            if op["spends_money"] and not op["spend_fields"]:
                assert "spend-disabled" in op["blocked_by"]

    def test_conditionally_paid_operations_are_judged_per_request(self, client) -> None:
        """`e2e --no-vlm` costs nothing but CPU and is the most useful thing to run.

        Blocking it alongside the paid path would make the spend guard an
        obstacle rather than a control, so the decision moves to the request.
        """
        caps = client.get("/api/capabilities").json()
        by_name = {o["name"]: o for o in caps["operations"]}
        assert by_name["e2e"]["spend_fields"] == ["vlm"]
        assert "spend-disabled" not in by_name["e2e"]["blocked_by"]

    def test_a_request_that_would_spend_is_refused(self, client, config) -> None:
        res = client.post(
            "/api/runs",
            json={"operation": "e2e", "params": {"split": "test", "vlm": True}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 403
        assert "--allow-spend" in res.json()["detail"]

    def test_the_same_operation_without_the_paid_flag_is_allowed(self, client, config) -> None:
        res = client.post(
            "/api/runs",
            json={"operation": "e2e", "params": {"split": "test", "vlm": False, "n": 1}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 200, res.text
        assert res.json()["run_id"]

    def test_a_key_being_present_is_not_permission_to_spend(self, client) -> None:
        caps = client.get("/api/capabilities").json()
        assert caps["spend"]["allowed"] is False

    def test_provenance_is_reported(self, client) -> None:
        caps = client.get("/api/capabilities").json()
        assert caps["provenance"]["config_sha256"]


class TestCommandRendering:
    def test_the_command_comes_from_the_validated_request(self, client, config) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval", "params": {"split": "test", "pixel_metrics": True}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.json()["command"] == "satchangegate eval --pixel-metrics"

    def test_an_invalid_request_is_rejected_not_rendered(self, client, config) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval", "params": {"split": "nonsense"}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 422

    def test_unknown_fields_are_refused(self, client, config) -> None:
        res = client.post(
            "/api/commands/preview",
            json={"operation": "eval", "params": {"not_a_field": 1}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 422

    def test_defaults_are_omitted_from_the_command(self) -> None:
        assert EvalRequest().to_command() == "satchangegate eval"

    def test_paired_booleans_render_their_negative_form(self) -> None:
        cmd = E2ERequest(split="test", vlm=True, batch=True, max_vlm_calls=100).to_command()
        assert "--vlm" in cmd and "--batch" in cmd and "--max-vlm-calls 100" in cmd

    def test_repeatable_options_repeat(self) -> None:
        cmd = OperatingPointsRequest(budget_usd=(0.5, 1.0)).to_command()
        assert cmd.count("--budget-usd") == 2

    def test_every_operation_renders_a_command_from_its_defaults(self) -> None:
        """A field with no flag is a field no command can reproduce."""
        required = {
            "pair": "beirut",
            "t1": "before.tif",
            "t2": "after.tif",
            "bands": "B04=1,B03=2,B02=3",
        }
        for name, spec in SERVICES.items():
            fields = spec.request_type.model_fields
            kwargs = {k: v for k, v in required.items() if k in fields}
            request = spec.request_type(**kwargs)
            assert request.to_command().startswith(
                f"satchangegate {spec.request_type.command_name}"
            )
            assert name


class TestPlayground:
    def test_thresholds_are_split_by_what_the_cache_can_answer(self, client) -> None:
        body = client.get("/api/playground/thresholds").json()
        names = {t["name"]: t["kind"] for t in body["thresholds"]}
        assert names["ndvi_strong_min"] == "decision"
        assert names["background_sigma"] == "mask"
        assert names["min_absolute_delta"] == "mask"

    def test_every_threshold_is_classified(self, client) -> None:
        body = client.get("/api/playground/thresholds").json()
        assert body["thresholds"]
        unclassified = [t["name"] for t in body["thresholds"] if t["kind"] == "other"]
        # 'other' is allowed, but it must not silently contain a decision knob.
        from satchangegate.webui.features import DECISION_ONLY, FEATURE_CHANGING

        assert not (set(unclassified) & (DECISION_ONLY | FEATURE_CHANGING))

    def test_a_mask_threshold_is_reported_stale_not_applied(self, client, config) -> None:
        """Re-scoring a cached matrix after moving a mask knob looks live and is wrong."""
        from satchangegate.webui.features import classify_overrides

        decision, changing = classify_overrides({"ndvi_strong_min": 0.1, "background_sigma": 3.0})
        assert decision == ["ndvi_strong_min"]
        assert changing == ["background_sigma"]

    def test_scoring_without_a_matrix_names_the_command(self, client, config) -> None:
        res = client.post(
            "/api/playground/score",
            json={"split": "train", "thresholds": {}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 404
        assert "eval --split train" in res.json()["detail"]["produced_by"]

    def test_an_unknown_threshold_is_refused(self, client, config) -> None:
        res = client.post(
            "/api/playground/score",
            json={"split": "train", "thresholds": {"made_up": 1}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 422


class TestScoringUsesTheRealGate:
    def test_the_playground_calls_decide_rather_than_a_copy(self) -> None:
        """A JavaScript or Python copy of the ladder is the drift tune_gate warns about."""
        import inspect

        from satchangegate.webui import features

        source = inspect.getsource(features)
        assert "from satchangegate.features.classical import GateFeatures, decide" in source
        assert "decide(" in source

    def test_metrics_match_a_hand_computed_confusion(self, tmp_path) -> None:
        from satchangegate.config import get_settings
        from satchangegate.webui.features import FeatureMatrix, score

        # A wholly unremarkable tile: no spectral delta, no changed area,
        # structurally identical. Whatever the shipped thresholds happen to be,
        # this has to come out as no_change -- which makes the confusion matrix
        # predictable without pinning a threshold value into the test.
        quiet = {
            "ssim": 1.0,
            "phash_distance": 0,
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
        matrix = FeatureMatrix(
            split="train",
            rows=(
                {"tile_id": "a", "city": "x", "label": 0, "features": {**quiet}},
                {"tile_id": "b", "city": "x", "label": 1, "features": {**quiet}},
            ),
            source="local",
            produced_by="test",
        )
        result = score(matrix, get_settings().gate)
        assert result["n_scored"] == 2
        assert result["confusion"]["tn"] == 1
        assert result["confusion"]["fn"] == 1


class TestLayerRendering:
    """Rendering rules that exist because breaking them would mislead."""

    def test_signed_deltas_use_a_fixed_scale_not_a_per_scene_stretch(self) -> None:
        """A self-normalised diverging map makes every scene look dramatic."""
        import numpy as np

        from satchangegate.webui.render import SIGNED_FULL_SCALE, _diverging

        valid = np.ones((4, 4), dtype=bool)
        quiet = _diverging(np.full((4, 4), 0.01), valid, SIGNED_FULL_SCALE)
        loud = _diverging(np.full((4, 4), 0.40), valid, SIGNED_FULL_SCALE)
        # A quiet scene must stay near white; a loud one must not.
        assert quiet.min() > loud.min()
        assert quiet[..., 0].min() > 200

    def test_the_scale_does_not_move_between_scenes(self) -> None:
        import numpy as np

        from satchangegate.webui.render import SIGNED_FULL_SCALE, _diverging

        valid = np.ones((2, 2), dtype=bool)
        a = _diverging(np.full((2, 2), 0.25), valid, SIGNED_FULL_SCALE)
        b = _diverging(np.array([[0.25, 0.9], [0.25, 0.9]]), valid, SIGNED_FULL_SCALE)
        # The 0.25 pixels render identically despite b containing larger values.
        assert a[0, 0].tolist() == b[0, 0].tolist()

    def test_unobserved_pixels_are_hatched_not_filled(self) -> None:
        """Unknown is not zero, and a flat fill reads as a measurement."""
        import numpy as np

        from satchangegate.webui.render import _hatch_invalid

        rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
        valid = np.ones((8, 8), dtype=bool)
        valid[0, :] = False
        out = _hatch_invalid(rgb, valid)
        row = {tuple(px) for px in out[0]}
        assert len(row) == 2, "an invalid row should be hatched, not one flat colour"
        assert (out[1] == 200).all(), "valid pixels must be untouched"

    def test_every_advertised_layer_has_a_renderer(self) -> None:
        from satchangegate.webui.render import describe_layers

        names = {spec["name"] for spec in describe_layers()}
        assert "change_mask" in names and "heatmap" in names
        assert {"mask_cloud_t1", "mask_cloud_t2"} <= names, "masks must name their timestep"

    def test_mask_layers_declare_their_timestep(self) -> None:
        """Cloud at t1, cloud at t2 and their union are three different pictures."""
        from satchangegate.webui.render import describe_layers

        for spec in describe_layers():
            if spec["name"].startswith("mask_"):
                assert spec["name"].endswith(("_t1", "_t2", "_union"))

    def test_an_unknown_layer_is_refused(self, client) -> None:
        res = client.get("/api/scenes/nowhere/layers/rgb_t1")
        assert res.status_code == 404


class TestRoutesSmoke:
    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/api/health",
            "/api/capabilities",
            "/api/operations",
            "/api/artifacts",
            "/api/scenes",
            "/api/playground/thresholds",
        ],
    )
    def test_route_responds(self, client, path) -> None:
        assert client.get(path).status_code == 200

    def test_security_headers_are_set(self, client) -> None:
        res = client.get("/api/health")
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["Referrer-Policy"] == "no-referrer"


class TestPaidWorkNeedsAQuote:
    """Approval is for one request, not for a session."""

    @pytest.fixture
    def paying(self, tmp_path):
        return AppConfig(
            reports=tmp_path / "reports",
            sample=tmp_path / "sample",
            run_root=tmp_path / "runs",
            allow_spend=True,
            spend_cap_usd=1.00,
            load_env=False,
        )

    @pytest.fixture
    def paying_client(self, paying):
        return TestClient(create_app(paying), base_url="http://127.0.0.1")

    def test_a_paid_request_without_a_quote_is_refused(self, paying_client, paying) -> None:
        res = paying_client.post(
            "/api/runs",
            json={"operation": "e2e", "params": {"split": "test", "vlm": True, "n": 1}},
            headers={"X-SCG-Token": paying.token},
        )
        assert res.status_code == 402
        assert "quote" in res.json()["detail"]

    def test_quoting_is_refused_entirely_when_spend_is_off(self, client, config) -> None:
        res = client.post(
            "/api/spend/quote",
            json={"operation": "e2e", "params": {"vlm": True}},
            headers={"X-SCG-Token": config.token},
        )
        assert res.status_code == 403

    def test_a_free_request_needs_no_quote(self, paying_client, paying) -> None:
        res = paying_client.post(
            "/api/spend/quote",
            json={"operation": "e2e", "params": {"split": "test", "vlm": False}},
            headers={"X-SCG-Token": paying.token},
        )
        assert res.json()["quote"] is None

    def test_a_quote_is_an_upper_bound_not_an_estimate(self, paying_client, paying) -> None:
        res = paying_client.post(
            "/api/spend/quote",
            json={
                "operation": "e2e",
                "params": {"split": "test", "vlm": True, "max_vlm_calls": 1, "batch": True},
                "n_calls": 1,
            },
            headers={"X-SCG-Token": paying.token},
        )
        quote = res.json()["quote"]
        # The recorded run averaged $0.004689 per batched call; a bound that sat
        # near the average would be no bound at all.
        assert quote["worst_case_usd"] > 0.004689 * 10
        assert "upper bound" in quote["note"]

    def test_a_quote_cannot_be_spent_on_a_different_request(self, paying_client, paying) -> None:
        quoted = paying_client.post(
            "/api/spend/quote",
            json={
                "operation": "e2e",
                "params": {"split": "test", "vlm": True, "max_vlm_calls": 1},
                "n_calls": 1,
            },
            headers={"X-SCG-Token": paying.token},
        ).json()["quote"]
        res = paying_client.post(
            "/api/runs",
            json={
                "operation": "e2e",
                "params": {"split": "test", "vlm": True, "max_vlm_calls": 2},
                "quote_id": quoted["quote_id"],
            },
            headers={"X-SCG-Token": paying.token},
        )
        assert res.status_code == 402
        assert "not the one that was quoted" in res.json()["detail"]

    def test_the_ledger_reports_what_it_is(self, paying_client) -> None:
        body = paying_client.get("/api/spend").json()
        assert body["enabled"] is True
        assert body["cap_usd"] == 1.0
        assert "not an invoice" in body["note"]
        assert "reserved" in body["policy"] or "reserved" in body["note"]

    def test_an_unknown_reservation_cannot_be_released(self, paying_client, paying) -> None:
        res = paying_client.post(
            "/api/spend/made-up/release",
            json={"note": "nope"},
            headers={"X-SCG-Token": paying.token},
        )
        assert res.status_code == 404
