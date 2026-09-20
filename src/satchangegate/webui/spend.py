"""A reservation ledger, so a dollar cap is actually a dollar cap.

Counting calls and multiplying by an average is not a spend control. The mean
per-call cost of the recorded run is $0.004689, but a call is bounded at 4,096
output tokens with four SDK retries, and the analyst report is a separate paid
call the count never saw. A cap enforced against an average is a cap that a
worse-than-average run walks straight through.

So spend is **reserved before dispatch** at a conservative upper bound, and
settled afterwards at the price the usage actually implies. Three consequences,
each deliberate:

- **A model with no published rate is refused, not priced at zero.**
  ``UsageRecord.cost_usd`` returns 0.0 for an unknown model, which is correct
  for reporting and catastrophic for authorising. The env override the README
  invites is exactly how a caller reaches that path.
- **Money is integer micro-dollars.** A cap compared against rounded display
  floats is a cap that leaks at the fourth decimal place, forever.
- **An uncertain outcome keeps its reservation.** A request that timed out may
  still have been served and billed. Releasing that money would let the next
  request spend it a second time, so it stays held until someone reconciles.

What this can never be is an invoice. It prices from a versioned rate card and
the tokens the API reports; the bill is the provider's.
"""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: One US dollar in the integer unit this ledger counts in.
MICRO = 1_000_000

#: A generous upper bound on the input a verification package can carry: three
#: images at a 512 px long edge, plus redacted metadata. The recorded run
#: averaged 2,893 input tokens per call; this is roughly triple that, because a
#: reservation that is too small is the only kind that matters.
WORST_CASE_INPUT_TOKENS = 9_000

#: How long a quote is honoured. Long enough to read and confirm, short enough
#: that a stale price cannot authorise a run.
QUOTE_TTL_S = 300.0

State = Literal["reserved", "settled", "released", "uncertain"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    quote_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    worst_case_micro INTEGER NOT NULL,
    model TEXT NOT NULL,
    n_calls INTEGER NOT NULL,
    issued_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    quote_id TEXT NOT NULL,
    run_id TEXT,
    fingerprint TEXT NOT NULL,
    reserved_micro INTEGER NOT NULL,
    settled_micro INTEGER,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS reservations_state ON reservations(state);
"""


class SpendRefused(RuntimeError):
    """The cap, or the rate card, will not permit this."""


@dataclass(frozen=True)
class Quote:
    """A server-issued price for one specific request."""

    quote_id: str
    fingerprint: str
    worst_case_micro: int
    model: str
    n_calls: int
    issued_at: float
    expires_at: float
    detail: dict[str, Any]

    @property
    def worst_case_usd(self) -> float:
        return self.worst_case_micro / MICRO

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote_id": self.quote_id,
            "fingerprint": self.fingerprint,
            "worst_case_usd": round(self.worst_case_usd, 6),
            "model": self.model,
            "n_calls": self.n_calls,
            "expires_in_s": round(max(0.0, self.expires_at - time.time()), 1),
            "detail": self.detail,
            "note": (
                "A conservative upper bound, not an estimate of the likely cost. "
                "It assumes every call returns the maximum output and every "
                "permitted retry is used. The settled figure will usually be lower."
            ),
        }


def worst_case_micro(
    *,
    model: str,
    n_calls: int,
    batch: bool,
    max_output_tokens: int,
    attempts: int,
    input_tokens: int = WORST_CASE_INPUT_TOKENS,
) -> int:
    """The most this request could cost, in micro-dollars.

    Refuses a model with no published rate rather than pricing it at zero. That
    is the difference between a cost report, where "unknown" can be shown, and
    an authorisation, where it cannot.
    """
    from satchangegate.vlm.client import PRICING_USD_PER_MTOK

    rate = PRICING_USD_PER_MTOK.get(model)
    if rate is None:
        raise SpendRefused(
            f"No published rate for {model!r}, so no upper bound can be computed. "
            f"Cost reporting can say 'unknown'; authorising a spend cannot. Known "
            f"models: {', '.join(sorted(PRICING_USD_PER_MTOK))}."
        )
    if n_calls < 0 or attempts < 1 or max_output_tokens < 0:
        raise SpendRefused("A quote needs a non-negative call count and at least one attempt.")

    rate_in = rate.batch_input_usd if batch else rate.input_usd
    rate_out = rate.batch_output_usd if batch else rate.output_usd
    # tokens / 1e6 * usd_per_mtok, in micro-dollars, is tokens * usd_per_mtok.
    per_call = input_tokens * rate_in + max_output_tokens * rate_out
    return math.ceil(per_call * attempts * n_calls)


def fingerprint(payload: dict[str, Any]) -> str:
    """A stable identity for exactly this request.

    Confirmation is bound to this, so changing the images, the model, the output
    cap or the number of calls invalidates the quote rather than carrying an old
    approval onto new work.
    """
    import hashlib

    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


class SpendLedger:
    """Reservations against a hard cap, durable across restarts."""

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_micro = round(cap_usd * MICRO)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ----------------------------------------------------------------- quote

    def quote(
        self,
        *,
        request: dict[str, Any],
        model: str,
        n_calls: int,
        batch: bool = False,
        max_output_tokens: int = 4096,
        attempts: int = 5,
    ) -> Quote:
        bound = worst_case_micro(
            model=model,
            n_calls=n_calls,
            batch=batch,
            max_output_tokens=max_output_tokens,
            attempts=attempts,
        )
        quote = Quote(
            quote_id=secrets.token_urlsafe(18),
            fingerprint=fingerprint(request),
            worst_case_micro=bound,
            model=model,
            n_calls=n_calls,
            issued_at=time.time(),
            expires_at=time.time() + QUOTE_TTL_S,
            detail={
                "batch": batch,
                "max_output_tokens": max_output_tokens,
                "attempts_priced": attempts,
                "input_tokens_assumed": WORST_CASE_INPUT_TOKENS,
            },
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO quotes (quote_id, fingerprint, worst_case_micro, model, "
                "n_calls, issued_at, expires_at, detail) VALUES (?,?,?,?,?,?,?,?)",
                (
                    quote.quote_id,
                    quote.fingerprint,
                    quote.worst_case_micro,
                    quote.model,
                    quote.n_calls,
                    quote.issued_at,
                    quote.expires_at,
                    json.dumps(quote.detail),
                ),
            )
        return quote

    # ------------------------------------------------------------- reserve

    def reserve(self, quote_id: str, *, request: dict[str, Any], run_id: str | None = None) -> str:
        """Hold the quoted amount against the cap, or refuse.

        The check and the insert happen inside one immediate transaction, so two
        requests racing for the last of the budget cannot both be told yes.
        """
        now = time.time()
        want = fingerprint(request)
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM quotes WHERE quote_id = ?", (quote_id,)
                ).fetchone()
                if row is None:
                    raise SpendRefused("No such quote. Ask for a price before spending.")
                if row["consumed"]:
                    raise SpendRefused("That quote has already been used.")
                if now > row["expires_at"]:
                    raise SpendRefused(
                        "That quote has expired. Prices and request contents can both "
                        "move; confirm against a fresh one."
                    )
                if not secrets.compare_digest(str(row["fingerprint"]), want):
                    raise SpendRefused(
                        "This request is not the one that was quoted. Changing the "
                        "images, the model, the output cap or the call count "
                        "invalidates a quote rather than carrying its approval over."
                    )

                settled, outstanding = self._totals(conn)
                new = int(row["worst_case_micro"])
                if settled + outstanding + new > self.cap_micro:
                    raise SpendRefused(
                        f"Refused: ${settled / MICRO:.4f} already settled and "
                        f"${outstanding / MICRO:.4f} still held, plus ${new / MICRO:.4f} "
                        f"for this request, would exceed the ${self.cap_micro / MICRO:.2f} "
                        f"session cap."
                    )
                reservation_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO reservations (reservation_id, quote_id, run_id, "
                    "fingerprint, reserved_micro, state, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (reservation_id, quote_id, run_id, want, new, "reserved", now, now),
                )
                conn.execute("UPDATE quotes SET consumed = 1 WHERE quote_id = ?", (quote_id,))
                conn.execute("COMMIT")
                return reservation_id
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def settle(self, reservation_id: str, actual_usd: float) -> None:
        """Replace a hold with what the usage actually implies."""
        micro = math.ceil(max(actual_usd, 0.0) * MICRO)
        self._set_state(reservation_id, "settled", settled_micro=micro)

    def mark_uncertain(self, reservation_id: str, note: str) -> None:
        """Keep the hold: the request may have been served and billed.

        Releasing it would let the next request spend the same money again,
        which is exactly the wrong direction to fail in.
        """
        self._set_state(reservation_id, "uncertain", note=note)

    def release(self, reservation_id: str, note: str = "not dispatched") -> None:
        """Give the money back. Only for work that demonstrably never left."""
        self._set_state(reservation_id, "released", settled_micro=0, note=note)

    def _set_state(
        self,
        reservation_id: str,
        state: State,
        *,
        settled_micro: int | None = None,
        note: str | None = None,
    ) -> None:
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE reservations SET state = ?, settled_micro = COALESCE(?, settled_micro), "
                "note = COALESCE(?, note), updated_at = ? WHERE reservation_id = ?",
                (state, settled_micro, note, time.time(), reservation_id),
            )
            if cur.rowcount == 0:
                raise SpendRefused(f"No reservation {reservation_id!r}.")

    # ------------------------------------------------------------- reporting

    def _totals(self, conn: sqlite3.Connection) -> tuple[int, int]:
        """(settled, outstanding) in micro-dollars.

        ``uncertain`` counts as outstanding, not as free budget: a request whose
        outcome is unknown may well have been billed.
        """
        settled = conn.execute(
            "SELECT COALESCE(SUM(settled_micro), 0) AS s FROM reservations WHERE state = 'settled'"
        ).fetchone()["s"]
        outstanding = conn.execute(
            "SELECT COALESCE(SUM(reserved_micro), 0) AS s FROM reservations "
            "WHERE state IN ('reserved', 'uncertain')"
        ).fetchone()["s"]
        return int(settled), int(outstanding)

    def summary(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            settled, outstanding = self._totals(conn)
            counts = {
                row["state"]: row["n"]
                for row in conn.execute(
                    "SELECT state, COUNT(*) AS n FROM reservations GROUP BY state"
                )
            }
        return {
            "cap_usd": round(self.cap_micro / MICRO, 6),
            "settled_usd": round(settled / MICRO, 6),
            "outstanding_usd": round(outstanding / MICRO, 6),
            "available_usd": round(max(0, self.cap_micro - settled - outstanding) / MICRO, 6),
            "reservations": counts,
            "note": (
                "Settled figures are derived from the tokens the API reported and a "
                "versioned rate card. They are this project's arithmetic, not an "
                "invoice. Anything left 'uncertain' stays held, because a request "
                "that timed out may still have been served."
            ),
        }
