"""The reservation ledger, tested at the boundary rather than at a safe value.

The 0.3.0 review found a cap violation a green suite had missed, because the
test asserted at a comfortable budget rather than at the edge. Every test here
sits on the edge: exactly at the cap, one micro-dollar over, two claims racing
for the last of it, an outcome nobody knows.
"""

from __future__ import annotations

import time

import pytest

from satchangegate.webui.spend import (
    MICRO,
    SpendLedger,
    SpendRefused,
    fingerprint,
    worst_case_micro,
)

REQUEST = {"operation": "e2e", "tiles": ["a", "b"], "model": "claude-sonnet-5"}


@pytest.fixture
def ledger(tmp_path):
    return SpendLedger(tmp_path / "spend.db", cap_usd=1.00)


def _quote(ledger, **overrides):
    params = {"request": REQUEST, "model": "claude-sonnet-5", "n_calls": 1}
    params.update(overrides)
    return ledger.quote(**params)


class TestPricingRefusesWhatItCannotBound:
    def test_an_unpriced_model_is_refused_not_charged_zero(self) -> None:
        """`UsageRecord.cost_usd` returns 0.0 for an unknown model.

        That is right for a report, where "unknown" can be shown, and
        catastrophic for an authorisation, where a zero is a blank cheque.
        """
        with pytest.raises(SpendRefused, match="No published rate"):
            worst_case_micro(
                model="claude-imaginary-9",
                n_calls=1,
                batch=False,
                max_output_tokens=4096,
                attempts=1,
            )

    def test_the_bound_covers_max_output_and_every_permitted_retry(self) -> None:
        one = worst_case_micro(
            model="claude-sonnet-5", n_calls=1, batch=False, max_output_tokens=4096, attempts=1
        )
        five = worst_case_micro(
            model="claude-sonnet-5", n_calls=1, batch=False, max_output_tokens=4096, attempts=5
        )
        assert five == 5 * one

    def test_the_bound_is_far_above_the_observed_mean(self) -> None:
        """A reservation that is too small is the only kind that matters.

        The recorded run averaged $0.004689 per batched call.
        """
        bound = worst_case_micro(
            model="claude-sonnet-5", n_calls=1, batch=True, max_output_tokens=4096, attempts=5
        )
        assert bound / MICRO > 0.004689 * 10

    def test_batching_prices_lower_than_synchronous(self) -> None:
        kwargs = {
            "model": "claude-sonnet-5",
            "n_calls": 3,
            "max_output_tokens": 4096,
            "attempts": 1,
        }
        assert worst_case_micro(batch=True, **kwargs) < worst_case_micro(batch=False, **kwargs)


class TestTheCapIsAHardCap:
    def test_a_request_exactly_at_the_cap_is_allowed(self, tmp_path) -> None:
        ledger = SpendLedger(tmp_path / "s.db", cap_usd=1.0)
        quote = _quote(ledger)
        # Size the cap to exactly this quote.
        ledger.cap_micro = quote.worst_case_micro
        assert ledger.reserve(quote.quote_id, request=REQUEST)

    def test_one_micro_dollar_over_the_cap_is_refused(self, tmp_path) -> None:
        """A cap compared against rounded display floats leaks at the fourth decimal."""
        ledger = SpendLedger(tmp_path / "s.db", cap_usd=1.0)
        quote = _quote(ledger)
        ledger.cap_micro = quote.worst_case_micro - 1
        with pytest.raises(SpendRefused, match="exceed"):
            ledger.reserve(quote.quote_id, request=REQUEST)

    def test_outstanding_holds_count_against_the_cap(self, ledger) -> None:
        first = _quote(ledger, n_calls=2)
        ledger.reserve(first.quote_id, request=REQUEST)
        second = _quote(ledger, n_calls=2)
        with pytest.raises(SpendRefused, match="still held"):
            ledger.reserve(second.quote_id, request=REQUEST)

    def test_settling_below_the_hold_frees_the_difference(self, ledger) -> None:
        first = _quote(ledger, n_calls=2)
        reservation = ledger.reserve(first.quote_id, request=REQUEST)
        ledger.settle(reservation, 0.01)
        second = _quote(ledger, n_calls=2)
        assert ledger.reserve(second.quote_id, request=REQUEST)

    def test_two_claims_on_the_last_of_the_budget_do_not_both_succeed(self, ledger) -> None:
        a = _quote(ledger, n_calls=2)
        b = _quote(ledger, n_calls=2)
        ledger.reserve(a.quote_id, request=REQUEST)
        with pytest.raises(SpendRefused):
            ledger.reserve(b.quote_id, request=REQUEST)


class TestConfirmationIsBoundToTheRequest:
    def test_a_changed_request_invalidates_its_quote(self, ledger) -> None:
        """Approval is for one request, not for a session."""
        quote = _quote(ledger)
        changed = {**REQUEST, "tiles": ["a", "b", "c"]}
        with pytest.raises(SpendRefused, match="not the one that was quoted"):
            ledger.reserve(quote.quote_id, request=changed)

    def test_a_quote_cannot_be_spent_twice(self, ledger) -> None:
        quote = _quote(ledger)
        ledger.reserve(quote.quote_id, request=REQUEST)
        with pytest.raises(SpendRefused, match="already been used"):
            ledger.reserve(quote.quote_id, request=REQUEST)

    def test_an_expired_quote_is_refused(self, ledger, monkeypatch) -> None:
        quote = _quote(ledger)
        monkeypatch.setattr(time, "time", lambda: quote.expires_at + 1)
        with pytest.raises(SpendRefused, match="expired"):
            ledger.reserve(quote.quote_id, request=REQUEST)

    def test_an_unknown_quote_is_refused(self, ledger) -> None:
        with pytest.raises(SpendRefused, match="No such quote"):
            ledger.reserve("made-up", request=REQUEST)

    def test_the_fingerprint_ignores_key_order(self) -> None:
        assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


class TestUncertainOutcomesStayHeld:
    def test_an_uncertain_request_keeps_its_reservation(self, ledger) -> None:
        """A request that timed out may still have been served and billed.

        Releasing it would let the next request spend the same money again.
        """
        quote = _quote(ledger, n_calls=2)
        reservation = ledger.reserve(quote.quote_id, request=REQUEST)
        ledger.mark_uncertain(reservation, "timeout; provider outcome unknown")
        summary = ledger.summary()
        assert summary["outstanding_usd"] > 0
        assert summary["reservations"]["uncertain"] == 1

    def test_uncertain_money_is_not_available_to_the_next_request(self, ledger) -> None:
        first = _quote(ledger, n_calls=2)
        reservation = ledger.reserve(first.quote_id, request=REQUEST)
        ledger.mark_uncertain(reservation, "timeout")
        second = _quote(ledger, n_calls=2)
        with pytest.raises(SpendRefused):
            ledger.reserve(second.quote_id, request=REQUEST)

    def test_releasing_is_possible_but_explicit(self, ledger) -> None:
        quote = _quote(ledger, n_calls=2)
        reservation = ledger.reserve(quote.quote_id, request=REQUEST)
        ledger.release(reservation, "refused before dispatch")
        assert ledger.summary()["outstanding_usd"] == 0


class TestDurability:
    def test_holds_survive_a_restart(self, tmp_path) -> None:
        """A cap that forgets across restarts is not a cap."""
        path = tmp_path / "s.db"
        first = SpendLedger(path, cap_usd=1.0)
        quote = _quote(first, n_calls=2)
        first.reserve(quote.quote_id, request=REQUEST)

        reopened = SpendLedger(path, cap_usd=1.0)
        assert reopened.summary()["outstanding_usd"] > 0
        again = _quote(reopened, n_calls=2)
        with pytest.raises(SpendRefused):
            reopened.reserve(again.quote_id, request=REQUEST)


class TestReportingIsNotAnInvoice:
    def test_the_summary_says_what_it_is(self, ledger) -> None:
        note = ledger.summary()["note"]
        assert "not an invoice" in note

    def test_a_quote_says_it_is_an_upper_bound(self, ledger) -> None:
        assert "upper bound" in _quote(ledger).to_dict()["note"]

    def test_available_never_goes_negative(self, tmp_path) -> None:
        ledger = SpendLedger(tmp_path / "s.db", cap_usd=1.0)
        quote = _quote(ledger)
        reservation = ledger.reserve(quote.quote_id, request=REQUEST)
        ledger.settle(reservation, 99.0)  # a bill far beyond the cap
        assert ledger.summary()["available_usd"] == 0
