"""One bounded worker, and progress that survives a disconnect.

Everything expensive in this repo -- a scene cache, a feature matrix, a
threshold sweep -- runs for minutes and holds real memory. One worker is the
right number for a single-user local tool: two would contend for the same
scene cache and double the peak footprint for no gain, so a second submission
queues rather than racing.

Progress is *persisted* and then streamed, not streamed and hoped for. The
queue is bounded and says so when full, rather than accepting work it will not
get to. Closing a tab does not cancel anything: a browser going away is not a
decision, and if the work has already been paid for, treating it as one would
be expensive.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from satchangegate.webui.runs import RunStore

#: Deeper than this and a queue stops being a queue and starts being a promise
#: nobody is going to keep.
DEFAULT_MAX_QUEUE = 8


@dataclass
class Submission:
    run_id: str
    operation: str
    payload: dict[str, Any]


class QueueFull(RuntimeError):
    """The worker is busy and the queue is at its limit."""


class JobRunner:
    """A single worker thread over a durable run store."""

    def __init__(
        self,
        store: RunStore,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        runner: Callable[[str, dict[str, Any]], Any] | None = None,
        on_settled: Callable[[str, dict[str, Any] | None, str | None], None] | None = None,
    ) -> None:
        self.store = store
        #: Called once a run reaches a terminal state, with its result or its
        #: error. The spend ledger uses it to turn a hold into a settled figure,
        #: which has to happen from the worker: the request that queued the run
        #: returned long before it finished.
        self._on_settled = on_settled
        self._queue: queue.Queue[Submission | None] = queue.Queue(maxsize=max_queue)
        self._runner = runner
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._work, name="scg-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        with contextlib.suppress(queue.Full):  # a full queue still drains
            self._queue.put_nowait(None)
        if self._thread:
            self._thread.join(timeout=timeout)

    # --------------------------------------------------------------- submit

    def submit(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        command: str,
        display_name: str | None = None,
        fingerprint: str | None = None,
        prepare: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        """Queue an operation and return its run id.

        ``prepare`` runs after the run has an identity but *before* the work is
        enqueued. That ordering matters: it is how a caller points the run's
        output at a directory named after it. Rewriting the payload after
        enqueueing would race the worker, which may already have started.
        """
        run = self.store.create(
            operation, payload, command, display_name=display_name, fingerprint=fingerprint
        )
        if prepare is not None:
            prepare(run.run_id, payload)
            # The stored request must be the one that runs, not the one that was
            # asked for before the run had a directory to write into.
            self.store.record_request(run.run_id, payload, command)
        try:
            self._queue.put_nowait(Submission(run.run_id, operation, payload))
        except queue.Full:
            self.store.mark_failed(
                run.run_id,
                "The queue is full. This is one worker by design; wait for the "
                "current run to finish rather than piling up work behind it.",
            )
            raise QueueFull("Worker queue is full") from None
        self.store.append_event(run.run_id, "queued", {"operation": operation})
        self.start()
        return run.run_id

    def cancel(self, run_id: str) -> bool:
        """Ask a run to stop at its next checkpoint.

        Cooperative, not forced: a run that has already dispatched paid work is
        not made cheaper by killing the thread that was waiting for it.
        """
        with self._lock:
            self._cancelled.add(run_id)
        run = self.store.get(run_id)
        if run and run.status == "queued":
            self.store.mark_cancelled(run_id)
            self.store.append_event(run_id, "cancelled", {"where": "before it started"})
            return True
        return bool(run and run.status == "running")

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancelled

    @property
    def current(self) -> str | None:
        return self._current

    def status(self) -> dict[str, Any]:
        return {
            "running": self._current,
            "queued": self._queue.qsize(),
            "capacity": self._queue.maxsize,
            "worker_alive": bool(self._thread and self._thread.is_alive()),
        }

    # --------------------------------------------------------------- worker

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            if self.is_cancelled(item.run_id):
                self.store.mark_cancelled(item.run_id)
                self._queue.task_done()
                continue
            self._execute(item)
            self._queue.task_done()

    def _execute(self, item: Submission) -> None:
        self._current = item.run_id
        started = time.time()
        self.store.mark_running(item.run_id)
        self.store.append_event(item.run_id, "started", {"operation": item.operation})
        try:
            result = self._dispatch(item)
            artifacts = [str(p) for p in getattr(result, "artifacts", ())]
            payload = result.to_dict() if hasattr(result, "to_dict") else {"data": result}
            self.store.mark_succeeded(item.run_id, payload, artifacts)
            self._settle(item.run_id, payload.get("data"), None)
            self.store.append_event(
                item.run_id,
                "finished",
                {"status": "succeeded", "duration_s": round(time.time() - started, 2)},
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.store.mark_failed(item.run_id, detail)
            self._settle(item.run_id, None, detail)
            self.store.append_event(
                item.run_id,
                "finished",
                {
                    "status": "failed",
                    "error": detail,
                    "traceback": traceback.format_exc(limit=4),
                    "duration_s": round(time.time() - started, 2),
                },
            )
        finally:
            self._current = None

    def _settle(self, run_id: str, result: dict[str, Any] | None, error: str | None) -> None:
        """Tell the ledger what happened, and never fail a run for trying.

        A run that has already spent money must not be reported as failed
        because bookkeeping raised afterwards.
        """
        if self._on_settled is None:
            return
        with contextlib.suppress(Exception):
            self._on_settled(run_id, result, error)

    def _dispatch(self, item: Submission) -> Any:
        if self._runner is not None:
            return self._runner(item.operation, item.payload)
        from satchangegate.services import run_service

        # The reporter is the whole point of running through a worker rather than
        # inline: without it the caller gets "started" and then silence for
        # minutes, because the ledger a naive tailer would watch is not written
        # until the expensive pass has already finished.
        return run_service(item.operation, item.payload, self.reporter(item.run_id))

    # -------------------------------------------------------------- reporter

    def reporter(self, run_id: str) -> Callable[..., None]:
        """A progress callback that writes durably before anything streams it."""

        def emit(kind: str, **payload: Any) -> None:
            self.store.append_event(run_id, kind, payload)

        return emit
