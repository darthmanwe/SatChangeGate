"""The shared service layer: the thing that makes a displayed command honest.

The UI promises that every panel shows the command producing it. That promise
is worth nothing unless the command and the panel run the same code, so the
tests here are mostly about the contract between a request, the command it
renders, and the runner it reaches -- not about any individual operation's
arithmetic, which its own module already covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from satchangegate.services import SERVICES, EvalRequest, ServiceUnavailable, run_service
from satchangegate.services.base import ServiceRequest


def _example(spec) -> ServiceRequest:
    """A minimally valid request for any operation."""
    required = {
        "pair": "beirut",
        "t1": "before.tif",
        "t2": "after.tif",
        "bands": "B04=1,B03=2,B02=3",
    }
    fields = spec.request_type.model_fields
    return spec.request_type(**{k: v for k, v in required.items() if k in fields})


class TestEveryOperationIsReproducible:
    def test_each_renders_a_command_naming_its_subcommand(self) -> None:
        for name, spec in SERVICES.items():
            command = _example(spec).to_command()
            assert command.startswith(f"satchangegate {spec.request_type.command_name}"), name

    def test_a_field_without_a_cli_flag_is_a_hard_error(self) -> None:
        """A web-only knob would make the displayed command a lie."""

        class Leaky(ServiceRequest):
            command_name = "leaky"
            cli_flags = {}
            knob: int = 1

        with pytest.raises(NotImplementedError, match="no CLI flag"):
            Leaky(knob=5).to_argv()

    def test_locations_are_omitted_unless_asked_for(self) -> None:
        request = EvalRequest(split="train")
        assert "--out" not in request.to_command()
        assert "--out" in request.to_command(include_locations=True)

    def test_defaults_never_appear_in_the_command(self) -> None:
        assert EvalRequest().to_command() == "satchangegate eval"

    def test_every_spec_declares_what_it_needs(self) -> None:
        for name, spec in SERVICES.items():
            assert spec.summary, name
            assert spec.speed in ("instant", "seconds", "minutes", "long"), name
            if spec.spends_money:
                assert spec.needs_network, f"{name} spends but claims no network"


class TestSpendIsJudgedPerRequest:
    def test_an_operation_that_cannot_spend_never_does(self) -> None:
        for name in ("eval", "conformal", "verify", "vlm-report", "ab-normalize"):
            spec = SERVICES[name]
            assert spec.request_spends(_example(spec)) is False, name

    def test_a_conditional_operation_depends_on_its_flags(self) -> None:
        spec = SERVICES["e2e"]
        assert spec.request_spends(spec.request_type(vlm=False)) is False
        assert spec.request_spends(spec.request_type(vlm=True)) is True

    def test_run_spends_on_either_paid_tier(self) -> None:
        spec = SERVICES["run"]
        assert spec.request_spends(spec.request_type(pair="x")) is False
        assert spec.request_spends(spec.request_type(pair="x", vlm=True)) is True
        assert spec.request_spends(spec.request_type(pair="x", llm=True)) is True


class TestCustomThresholds:
    def test_scoring_operations_accept_a_thresholds_file(self) -> None:
        """Without a flag to apply it, the playground's export would be a file
        nothing could consume."""
        for name in ("eval", "e2e", "conformal", "operating-points", "baselines", "tune"):
            assert "thresholds" in SERVICES[name].request_type.model_fields, name

    def test_the_thresholds_path_reaches_the_command(self) -> None:
        command = EvalRequest(thresholds=Path("tuned.yaml")).to_command()
        assert "--thresholds tuned.yaml" in command

    def test_a_declared_thresholds_file_changes_the_settings_used(self, tmp_path) -> None:
        from satchangegate.services.operations import _settings_for

        path = tmp_path / "t.yaml"
        path.write_text("gate:\n  ndvi_strong_min: 0.99\n", encoding="utf-8")
        assert _settings_for(EvalRequest(thresholds=path)).gate.ndvi_strong_min == 0.99
        assert _settings_for(EvalRequest()).gate.ndvi_strong_min != 0.99


class TestDispatch:
    def test_an_unknown_operation_is_refused(self) -> None:
        with pytest.raises(ServiceUnavailable, match="Unknown operation"):
            run_service("not-a-thing")

    def test_an_invalid_payload_is_refused_before_anything_runs(self) -> None:
        with pytest.raises(ValueError):
            run_service("eval", {"split": "nonsense"})

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValueError):
            run_service("eval", {"not_a_field": 1})

    def test_verify_runs_and_carries_its_command(self) -> None:
        result = run_service("verify")
        assert result.command == "satchangegate verify"
        assert "provenance" in result.data


class TestAbNormalizeHasAProducingCommand:
    def test_the_operation_exists(self) -> None:
        """Its artifact was committed from 0.3.0 with nothing that could
        regenerate it, which is the defect the audit was about. A negative
        result is not exempt from needing a command."""
        assert "ab-normalize" in SERVICES
        assert SERVICES["ab-normalize"].request_type.command_name == "ab-normalize"

    def test_it_declares_the_extra_it_needs(self) -> None:
        assert "baseline-extra" in SERVICES["ab-normalize"].requires

    def test_the_renderer_shows_both_sides_and_the_delta(self) -> None:
        from satchangegate.ab_normalize import _render

        summary = {
            "split": "test",
            "off": {"gate": {"f1": 0.6, "precision": 0.8}, "models": {"m": 0.85}},
            "on": {"gate": {"f1": 0.5, "precision": 0.7}, "models": {"m": 0.75}},
            "delta_on_minus_off": {"gate_f1": -0.1, "gate_precision": -0.1, "m_ap": -0.1},
            "reading": "normalization hurt",
            "provenance": "committed without a command",
        }
        text = _render(summary)
        assert "-0.1000" in text and "normalization hurt" in text
