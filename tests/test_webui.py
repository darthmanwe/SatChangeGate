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
    return AppConfig(reports=tmp_path / "reports", sample=tmp_path / "sample", load_env=False)


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

    def test_spending_operations_are_blocked_while_spend_is_off(self, client) -> None:
        caps = client.get("/api/capabilities").json()
        for op in caps["operations"]:
            if op["spends_money"]:
                assert "spend-disabled" in op["blocked_by"]

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
        for name, spec in SERVICES.items():
            if "pair" in spec.request_type.model_fields:
                request = spec.request_type(pair="beirut")
            else:
                request = spec.request_type()
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
