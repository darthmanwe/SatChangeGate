"""Immutable runs: a server-generated identity, its own directory, one writer.

The UI never writes into ``data/reports``. That directory holds the benchmark --
a 621-row ledger containing 100 verifications that cost real money -- and most of
the library's entry points write fixed filenames into it. ``run_e2e`` used to
delete its ledger outright on any run without ``resume``. A browser that can
trigger a run is a browser that can destroy a measurement, so runs made here go
to ``data/ui_runs/<run_id>/`` and the benchmark is mounted read-only.

Identity is server-generated and separate from the display name, because a name
a person typed is not a key. Each run records the effective settings it ran
under, so "which configuration produced this" is answerable later rather than
inferred from whatever the config happens to say now.

One writer, enforced by a lock file. A second server on the same store is
refused rather than allowed to interleave. That is the light version of run
isolation and it is the right one here: this is a single-user local tool, not a
cluster.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]

DEFAULT_RUN_ROOT = Path("data/ui_runs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    display_name TEXT NOT NULL,
    request TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    fingerprint TEXT,
    error TEXT,
    result TEXT,
    artifacts TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
"""


class StoreLocked(RuntimeError):
    """Another process already owns this run store."""


@dataclass
class Run:
    run_id: str
    operation: str
    display_name: str
    request: dict[str, Any]
    command: str
    status: Status
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    fingerprint: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = (
            round((self.finished_at or time.time()) - self.started_at, 2)
            if self.started_at
            else None
        )
        return data


class RunStore:
    """SQLite index plus one directory per run. One writer, by design."""

    def __init__(self, root: Path | None = None, *, take_lock: bool = True) -> None:
        self.root = Path(root or DEFAULT_RUN_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / ".writer.lock"
        self._lock_fd: int | None = None
        if take_lock:
            self._acquire()
        self._db_path = self.root / "index.db"
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
        if take_lock:
            self.recover_interrupted()

    def recover_interrupted(self) -> list[str]:
        """Settle runs left mid-flight by a process that died.

        Only safe because the lock was just acquired: anything still marked
        running belongs to an owner that is demonstrably gone, since a live one
        would still hold the lock. Left alone these sit at "running" forever and
        a reader cannot tell them from work in progress.

        They are marked failed rather than cancelled, because nobody chose to
        stop them, and their bundles stay incomplete -- which is what they are.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE status IN ('running', 'queued')"
            ).fetchall()
            stranded = [r["run_id"] for r in rows]
            if stranded:
                conn.execute(
                    "UPDATE runs SET status = 'failed', finished_at = ?, error = ? "
                    "WHERE status IN ('running', 'queued')",
                    (
                        time.time(),
                        "Interrupted: the server stopped before this run finished. "
                        "Any work it had already dispatched to a provider was still "
                        "dispatched; check the run directory before re-running.",
                    ),
                )
                conn.commit()
        for run_id in stranded:
            self.append_event(run_id, "interrupted", {"recovered_at_startup": True})
        return stranded

    # ------------------------------------------------------------------ lock

    def _acquire(self) -> None:
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            stale = self._stale_owner()
            if stale is None:
                raise StoreLocked(
                    f"{self.root} is already owned by another process. Two writers "
                    f"would interleave runs into one index; stop the other server, "
                    f"or delete {self._lock_path} if you are sure it is stale."
                ) from None
            self._lock_path.unlink(missing_ok=True)
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        self._lock_fd = fd

    def _stale_owner(self) -> int | None:
        """The pid in the lock file if that process is demonstrably gone, else None.

        A lock held by *this* process is the least stale lock there is, so it is
        reported live. Getting that backwards makes a second store in the same
        process delete a lock whose descriptor is still open -- which on Windows
        fails outright, and on POSIX would silently hand out a second writer.
        """
        try:
            pid = int(self._lock_path.read_text(encoding="utf-8").strip() or "0")
        except (ValueError, OSError):
            return 0  # unreadable lock: treat as debris, not as an owner
        if pid == os.getpid():
            return None
        if pid <= 0:
            return 0
        try:
            os.kill(pid, 0)
        except OSError:
            return pid
        except Exception:  # pragma: no cover - platform differences
            return None
        return None

    def release(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        self._lock_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------- db

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------------------------------------------- write

    def create(
        self,
        operation: str,
        request: dict[str, Any],
        command: str,
        *,
        display_name: str | None = None,
        fingerprint: str | None = None,
    ) -> Run:
        run = Run(
            run_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
            operation=operation,
            display_name=display_name or operation,
            request=request,
            command=command,
            status="queued",
            created_at=time.time(),
            fingerprint=fingerprint,
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO runs (run_id, operation, display_name, request, command, "
                "status, created_at, fingerprint, artifacts) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.operation,
                    run.display_name,
                    json.dumps(run.request),
                    run.command,
                    run.status,
                    run.created_at,
                    run.fingerprint,
                    "[]",
                ),
            )
            conn.commit()
        # The request and its effective settings land on disk before anything
        # runs, so an interrupted run is still identifiable afterwards.
        (self.run_dir(run.run_id) / "request.json").write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "operation": operation,
                    "request": request,
                    "command": command,
                    "fingerprint": fingerprint,
                    "created_at": run.created_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return run

    def record_request(self, run_id: str, request: dict[str, Any], command: str) -> None:
        """Re-persist the request after a caller has finalised it.

        A run's recorded request has to be the one that *ran*. Redirecting output
        into the run directory happens after the run has an identity, so without
        this the index would keep claiming the run wrote to the default reports
        directory -- which is exactly the provenance claim this UI is built on.
        """
        self._update(run_id, request=json.dumps(request), command=command)
        directory = self.run_dir(run_id)
        path = directory / "request.json"
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - written moments ago
            blob = {"run_id": run_id}
        blob["request"] = request
        blob["command"] = command
        path.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    def _update(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with closing(self._connect()) as conn:
            conn.execute(f"UPDATE runs SET {sets} WHERE run_id = ?", (*fields.values(), run_id))
            conn.commit()

    def mark_running(self, run_id: str) -> None:
        self._update(run_id, status="running", started_at=time.time())

    def mark_succeeded(self, run_id: str, result: dict[str, Any], artifacts: list[str]) -> None:
        self._update(
            run_id,
            status="succeeded",
            finished_at=time.time(),
            result=json.dumps(result, default=str),
            artifacts=json.dumps(artifacts),
        )
        self._publish(run_id, result)

    def mark_failed(self, run_id: str, error: str) -> None:
        self._update(run_id, status="failed", finished_at=time.time(), error=error)

    def mark_cancelled(self, run_id: str) -> None:
        self._update(run_id, status="cancelled", finished_at=time.time())

    def _publish(self, run_id: str, result: dict[str, Any]) -> None:
        """Write the result, then mark the bundle complete.

        Two steps, in that order, so a reader can never pick up a half-written
        bundle and treat it as finished.
        """
        directory = self.run_dir(run_id)
        tmp = directory / "result.json.partial"
        tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        tmp.replace(directory / "result.json")
        (directory / "COMPLETE").write_text(str(time.time()), encoding="utf-8")

    def is_complete(self, run_id: str) -> bool:
        return (self.root / run_id / "COMPLETE").is_file()

    # ------------------------------------------------------------------ read

    def get(self, run_id: str) -> Run | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    def recent(self, limit: int = 50) -> list[Run]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    # ---------------------------------------------------------------- events

    def append_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> int:
        """Record a progress event durably, and return its sequence number.

        Events carry a sequence so a reconnecting client can resume from where
        it left off instead of replaying a run from the beginning or, worse,
        silently missing the middle of one.
        """
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS s FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()
            seq = int(row["s"]) + 1
            conn.execute(
                "INSERT INTO events (run_id, seq, at, kind, payload) VALUES (?,?,?,?,?)",
                (run_id, seq, time.time(), kind, json.dumps(payload, default=str)),
            )
            conn.commit()
        return seq

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT seq, at, kind, payload FROM events WHERE run_id = ? AND seq > ? "
                "ORDER BY seq",
                (run_id, after),
            ).fetchall()
        return [
            {"seq": r["seq"], "at": r["at"], "kind": r["kind"], **json.loads(r["payload"])}
            for r in rows
        ]


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"],
        operation=row["operation"],
        display_name=row["display_name"],
        request=json.loads(row["request"]),
        command=row["command"],
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        fingerprint=row["fingerprint"],
        error=row["error"],
        result=json.loads(row["result"]) if row["result"] else None,
        artifacts=json.loads(row["artifacts"] or "[]"),
    )
