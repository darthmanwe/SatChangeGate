"""The run store and the worker, tested where they can lose or corrupt work.

Three properties matter here and none of them are about throughput: a run cannot
overwrite the benchmark, a half-written bundle cannot read as finished, and
progress already reported cannot be lost by a client that went away.
"""

from __future__ import annotations

import time

import pytest

from satchangegate.webui.jobs import JobRunner, QueueFull
from satchangegate.webui.runs import RunStore, StoreLocked


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "runs")
    yield s
    s.release()


def _ok(_operation, _payload):
    """A runner that succeeds immediately, for testing the machinery around it."""
    return {"ok": True}


def _wait(store, run_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = store.get(run_id)
        if run and run.status in ("succeeded", "failed", "cancelled"):
            return run
        time.sleep(0.02)
    raise AssertionError(f"{run_id} never settled")


class TestOneWriter:
    def test_a_second_writer_is_refused(self, store, tmp_path) -> None:
        """Two writers would interleave runs into one index."""
        with pytest.raises(StoreLocked):
            RunStore(tmp_path / "runs")

    def test_a_lock_from_a_dead_process_is_reclaimed(self, tmp_path) -> None:
        root = tmp_path / "runs"
        root.mkdir()
        # A pid that cannot be running: the lock is debris, not an owner.
        (root / ".writer.lock").write_text("999999999", encoding="utf-8")
        reclaimed = RunStore(root)
        reclaimed.release()

    def test_an_unreadable_lock_is_treated_as_debris(self, tmp_path) -> None:
        root = tmp_path / "runs"
        root.mkdir()
        (root / ".writer.lock").write_text("not a pid", encoding="utf-8")
        reclaimed = RunStore(root)
        reclaimed.release()


class TestBundlesArePublishedAtomically:
    def test_a_finished_run_is_marked_complete(self, store) -> None:
        runner = JobRunner(store, runner=_ok)
        run_id = runner.submit("x", {}, command="satchangegate x")
        _wait(store, run_id)
        assert store.is_complete(run_id)
        assert (store.run_dir(run_id) / "result.json").is_file()
        runner.stop()

    def test_a_failed_run_is_never_marked_complete(self, store) -> None:
        """An interrupted bundle must not read as a finished one."""

        def boom(_operation, _payload):
            raise ValueError("deliberate")

        runner = JobRunner(store, runner=boom)
        run_id = runner.submit("x", {}, command="satchangegate x")
        run = _wait(store, run_id)
        assert run.status == "failed"
        assert "deliberate" in run.error
        assert not store.is_complete(run_id)
        runner.stop()

    def test_no_partial_file_survives_a_publish(self, store) -> None:
        runner = JobRunner(store, runner=_ok)
        run_id = runner.submit("x", {}, command="x")
        _wait(store, run_id)
        leftovers = list(store.run_dir(run_id).glob("*.partial"))
        assert leftovers == []
        runner.stop()


class TestProgressSurvivesTheClient:
    def test_events_are_durable_and_resumable(self, store) -> None:
        """A client that reconnects resumes; one that never connects loses nothing."""
        run = store.create("x", {}, "x")
        for i in range(5):
            store.append_event(run.run_id, "tick", {"i": i})
        assert len(store.events(run.run_id)) == 5
        resumed = store.events(run.run_id, after=3)
        assert [e["i"] for e in resumed] == [3, 4]

    def test_sequence_numbers_are_monotonic(self, store) -> None:
        run = store.create("x", {}, "x")
        seqs = [store.append_event(run.run_id, "tick", {}) for _ in range(4)]
        assert seqs == sorted(seqs) and len(set(seqs)) == 4

    def test_a_reporter_writes_before_anything_streams(self, store) -> None:
        runner = JobRunner(store, runner=_ok)
        run = store.create("x", {}, "x")
        report = runner.reporter(run.run_id)
        report("city_finished", city="bercy", done=1, total=3)
        events = store.events(run.run_id)
        assert events[0]["kind"] == "city_finished"
        assert events[0]["city"] == "bercy"
        runner.stop()

    def test_a_broken_reporter_never_breaks_a_run(self) -> None:
        """A run that has already spent money must not fail on a status line."""
        from satchangegate.e2e import _emit

        def hostile(kind, **payload):
            raise RuntimeError("the sink is down")

        _emit(hostile, "city_finished", city="x")  # must not raise
        _emit(None, "city_finished", city="x")


class TestQueueIsBounded:
    def test_a_full_queue_is_refused_not_silently_dropped(self, store) -> None:
        gate = {"open": False}

        def slow(_operation, _payload):
            while not gate["open"]:
                time.sleep(0.01)
            return {"ok": True}

        runner = JobRunner(store, max_queue=2, runner=slow)
        runner.submit("x", {}, command="x")
        time.sleep(0.2)  # let the worker pick the first one up
        accepted = 0
        try:
            for _ in range(6):
                runner.submit("x", {}, command="x")
                accepted += 1
        except QueueFull:
            pass
        else:  # pragma: no cover - the bound must bite
            raise AssertionError("the queue accepted more than its capacity")
        assert accepted <= 2
        gate["open"] = True
        runner.stop()

    def test_the_refused_run_is_recorded_as_failed_with_a_reason(self, store) -> None:
        gate = {"open": False}

        def slow(_operation, _payload):
            while not gate["open"]:
                time.sleep(0.01)
            return {}

        runner = JobRunner(store, max_queue=1, runner=slow)
        runner.submit("x", {}, command="x")
        time.sleep(0.2)
        with pytest.raises(QueueFull):
            for _ in range(5):
                runner.submit("x", {}, command="x")
        failed = [r for r in store.recent() if r.status == "failed"]
        assert failed and "queue is full" in failed[0].error
        gate["open"] = True
        runner.stop()


class TestRequestProvenance:
    def test_the_recorded_request_is_the_one_that_ran(self, store) -> None:
        """Output is redirected after the run has an identity.

        Without re-recording, the index would keep claiming the run wrote to the
        default reports directory -- which is the provenance claim this whole UI
        rests on.
        """
        seen = {}

        def capture(_operation, payload):
            seen.update(payload)
            return {"ok": True}

        runner = JobRunner(store, runner=capture)

        def prepare(run_id, queued):
            queued["out"] = str(store.run_dir(run_id))

        run_id = runner.submit("x", {"out": "data/reports"}, command="x", prepare=prepare)
        _wait(store, run_id)
        recorded = store.get(run_id).request
        assert recorded["out"] != "data/reports"
        assert recorded["out"] == seen["out"], "recorded request must match what ran"
        runner.stop()

    def test_prepare_runs_before_the_worker_can_see_the_payload(self, store) -> None:
        order = []

        def capture(_operation, payload):
            order.append(("ran", payload.get("out")))
            return {}

        runner = JobRunner(store, runner=capture)

        def prepare(run_id, queued):
            order.append(("prepared", run_id))
            queued["out"] = "isolated"

        run_id = runner.submit("x", {"out": "shared"}, command="x", prepare=prepare)
        _wait(store, run_id)
        assert order[0][0] == "prepared"
        assert order[1] == ("ran", "isolated")
        runner.stop()


class TestCancellation:
    def test_a_queued_run_can_be_cancelled_before_it_starts(self, store) -> None:
        gate = {"open": False}

        def slow(_operation, _payload):
            while not gate["open"]:
                time.sleep(0.01)
            return {}

        runner = JobRunner(store, max_queue=4, runner=slow)
        first = runner.submit("x", {}, command="x")
        second = runner.submit("x", {}, command="x")
        time.sleep(0.2)
        assert runner.cancel(second) is True
        assert store.get(second).status == "cancelled"
        gate["open"] = True
        _wait(store, first)
        runner.stop()


class TestInterruptedRunsAreSettled:
    def test_a_run_left_running_by_a_dead_server_is_marked_failed(self, tmp_path) -> None:
        """Left alone these sit at "running" forever, indistinguishable from live work."""
        root = tmp_path / "runs"
        first = RunStore(root)
        run = first.create("e2e", {}, "satchangegate e2e")
        first.mark_running(run.run_id)
        first.release()

        second = RunStore(root)
        reopened = second.get(run.run_id)
        assert reopened.status == "failed"
        assert "Interrupted" in reopened.error
        assert not second.is_complete(run.run_id)
        assert "interrupted" in [e["kind"] for e in second.events(run.run_id)]
        second.release()

    def test_a_finished_run_is_left_alone(self, tmp_path) -> None:
        root = tmp_path / "runs"
        first = RunStore(root)
        run = first.create("x", {}, "x")
        first.mark_succeeded(run.run_id, {"ok": True}, [])
        first.release()

        second = RunStore(root)
        assert second.get(run.run_id).status == "succeeded"
        second.release()
