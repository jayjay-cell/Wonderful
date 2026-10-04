"""API and persistence tests (FR-G4, FR-H1..H4).

The store tests matter more than they look: these tables are what make
later scoring purely additive (it reads history rather than re-simulating),
and what make a debrief possible at all. A transcript that records intended
rather than delivered text would make a debrief a lie.
"""

from __future__ import annotations

import pytest

from api.store import SqliteSessionStore


@pytest.fixture
def store() -> SqliteSessionStore:
    return SqliteSessionStore(":memory:")


@pytest.fixture
def session_id(store) -> str:
    store.create_session(
        session_id="s1", mission_id="m1", mission_version=1,
        realism_seed=42, started_at="2026-10-04T10:00:00Z",
        mission_title="Test mission",
    )
    return "s1"


class TestSessionLifecycle:
    def test_session_is_recorded_with_its_seed(self, store, session_id) -> None:
        """FR-H4: the seed is what makes 'run that again and watch what you
        missed' possible."""
        record = store.session(session_id)
        assert record["realism_seed"] == 42
        assert record["status"] == "running"

    def test_ending_sets_status_and_timestamp(self, store, session_id) -> None:
        store.end_session(session_id, "completed")
        record = store.session(session_id)
        assert record["status"] == "completed"
        assert record["ended_at"]

    def test_unknown_session_returns_none(self, store) -> None:
        assert store.session("nope") is None

    def test_sessions_list_newest_first(self, store) -> None:
        store.create_session(session_id="a", mission_id="m", mission_version=1,
                             realism_seed=1, started_at="2026-10-01T00:00:00Z")
        store.create_session(session_id="b", mission_id="m", mission_version=1,
                             realism_seed=1, started_at="2026-10-03T00:00:00Z")
        assert [s.session_id for s in store.list_sessions()][0] == "b"


class TestTranscript:
    def test_delivered_text_is_what_is_stored(self, store, session_id) -> None:
        """FR-H3. Storing the intended text would show the trainee words
        they never heard."""
        store.add_utterance(
            session_id, text="מפקדה, נחשון 3", origin="reactive",
            mission_seconds=10.0, status="interrupted",
            planned_text="מפקדה, נחשון 3, דלק 180 ליברות",
            delivered_segments=1, total_segments=3,
        )
        entry = store.transcript(session_id)[0]
        assert entry["text"] == "מפקדה, נחשון 3"
        assert entry["planned_text"] != entry["text"]
        assert entry["status"] == "interrupted"

    def test_transcript_is_interleaved_by_mission_time(self, store, session_id) -> None:
        """Ordered here rather than in the UI, so every consumer -- review
        page, future scoring, an export -- sees the same conversation."""
        store.add_utterance(session_id, text="second", origin="reactive",
                            mission_seconds=20.0)
        store.add_trainee_message(session_id, "first", 10.0)
        store.add_utterance(session_id, text="third", origin="initiated",
                            mission_seconds=30.0, trigger_id="t1")

        entries = store.transcript(session_id)
        assert [e["text"] for e in entries] == ["first", "second", "third"]
        assert [e["speaker"] for e in entries] == ["trainee", "counterpart", "counterpart"]

    def test_initiated_utterances_name_their_trigger(self, store, session_id) -> None:
        """FR-H2: a debrief must distinguish what he was asked from what he
        raised himself."""
        store.add_utterance(session_id, text="בינגו", origin="initiated",
                            mission_seconds=5.0, trigger_id="fuel_below_bingo")
        entry = store.transcript(session_id)[0]
        assert entry["origin"] == "initiated"
        assert entry["trigger_id"] == "fuel_below_bingo"


class TestSuppressionRecord:
    def test_suppressed_firings_are_recorded_with_a_reason(self, store, session_id) -> None:
        """FR-C8. Without this, 'why didn't he warn me about the fuel?' is
        unanswerable: the absence of an event leaves no trace."""
        store.add_trigger_firing(session_id, "fuel_below_bingo", 100.0,
                                 suppressed=True, reason="trainee_composing")
        store.add_trigger_firing(session_id, "fuel_below_bingo", 105.0,
                                 suppressed=False)

        firings = store.firings(session_id)
        assert len(firings) == 2
        assert firings[0]["suppressed"] and firings[0]["reason"] == "trainee_composing"
        assert not firings[1]["suppressed"]


class TestSnapshots:
    def test_snapshot_records_state_and_cause(self, store, session_id) -> None:
        """FR-D7: answers 'what did he know at 03:12?'"""
        store.add_snapshot(session_id, 192.0, {"fuel_lb": 176.3, "mode": "idle"},
                           cause="trigger:movement_spotted")
        snapshot = store.snapshots(session_id)[0]
        assert snapshot["state"]["fuel_lb"] == 176.3
        assert snapshot["cause"] == "trigger:movement_spotted"

    def test_snapshots_are_append_only_in_order(self, store, session_id) -> None:
        for second in (10.0, 20.0, 30.0):
            store.add_snapshot(session_id, second, {"fuel_lb": 100 - second}, "tick")
        assert [s["mission_seconds"] for s in store.snapshots(session_id)] == [10.0, 20.0, 30.0]

    def test_hebrew_survives_the_round_trip(self, store, session_id) -> None:
        """ensure_ascii=False in the store, so a debrief is readable rather
        than full of escape sequences."""
        store.add_snapshot(session_id, 1.0, {"sensor": "עוקב"}, "tick")
        assert store.snapshots(session_id)[0]["state"]["sensor"] == "עוקב"


class TestApiSurface:
    """Import-level checks, so a broken route is caught without a server."""

    def test_app_imports_and_exposes_expected_routes(self) -> None:
        from api.main import app
        paths = {route.path for route in app.routes if hasattr(route, "path")}
        for expected in ("/health", "/missions", "/sessions",
                         "/sessions/{session_id}/stream",
                         "/sessions/{session_id}/messages",
                         "/sessions/{session_id}/interrupt",
                         "/sessions/{session_id}/composing",
                         "/sessions/{session_id}/review"):
            assert expected in paths, f"missing route {expected}"

    def test_message_length_is_bounded(self) -> None:
        """An unbounded transmission field is an easy way to blow the
        context window."""
        import pydantic
        from api.schemas import MessageRequest
        with pytest.raises(pydantic.ValidationError):
            MessageRequest(text="x" * 5000)

    def test_empty_message_rejected(self) -> None:
        import pydantic
        from api.schemas import MessageRequest
        with pytest.raises(pydantic.ValidationError):
            MessageRequest(text="")

    def test_exception_handler_hides_internals(self) -> None:
        """FR-G4: a generic message plus a request id; the real exception
        is logged server-side only."""
        import asyncio
        from api.main import unhandled_exception_handler

        response = asyncio.run(unhandled_exception_handler(
            None, RuntimeError("secret internal detail: /path/to/file.py")
        ))
        body = response.body.decode()
        assert "secret internal detail" not in body
        assert "file.py" not in body
        assert "request_id" in body
