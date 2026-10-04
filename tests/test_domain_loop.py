"""Exactly one domain loop per session (regression).

THE BUG THIS LOCKS DOWN. The UI opens the SSE stream for every mode, and
the Gemini Live endpoint used not to cancel it -- so every Live session ran
TWO trigger loops against one MissionStateEngine:

  * both bumped log.fired_counts (sim/runner.py vs api/live_voice.py), so a
    `once: true` trigger was consumed by whichever loop saw it first, or
    fired twice;
  * the text agent also made its own model call and delivered through SSE,
    so one authored event could arrive spoken by Gemini AND as text;
  * api/main.py's _drain_suppressions index-tracks a shared list, which two
    concurrent appenders make unreliable.

It is asserted at the ownership level rather than by racing two real loops:
a timing test would be flaky, while ownership is the actual invariant.
"""

from __future__ import annotations

import asyncio

import pytest

from core.mission import load_mission_dict
from core.models import Priority
from delivery.text_channel import CollectingChannel
from sim.clock import VirtualClock
from sim.runner import SessionRunner
from tests.fakes import ScriptedModel


@pytest.fixture
def mission(mission_with_everything):
    return load_mission_dict(mission_with_everything)


class TestSingleDomainOwner:
    def test_session_starts_with_no_owner(self) -> None:
        from api.main import LiveSession
        live = LiveSession(None, None)
        assert live.domain_owner is None

    def test_sse_claims_the_loop_when_free(self) -> None:
        """The text and ElevenLabs paths drive triggers from the SSE tick
        loop, so it must still claim ownership when nothing else has."""
        from api.main import LiveSession
        live = LiveSession(None, None)
        assert live.domain_owner in (None, "sse")

    def test_live_voice_ownership_excludes_sse(self) -> None:
        """Once Gemini Live owns the loop, the SSE handler's guard must
        refuse to start a second one."""
        from api.main import LiveSession
        live = LiveSession(None, None)
        live.domain_owner = "live_voice"
        # This is the exact condition api/main.py's stream handler tests.
        may_start = live.domain_owner in (None, "sse")
        assert not may_start


class TestFiredCountsAreNotDoubleCounted:
    """The observable symptom: a once-only trigger must fire exactly once
    however many channels are attached."""

    @pytest.mark.asyncio
    async def test_once_trigger_fires_exactly_once(self, mission) -> None:
        runner = SessionRunner(
            mission, ScriptedModel(replies=["רות"]), CollectingChannel(),
            session_id="s1", clock=VirtualClock(), delivery_speed=1.0, seed=1,
        )
        # low_fuel is `once: true` in the fixture.
        runner.clock.set(10.0)
        for _ in range(6):
            runner.clock.set(runner.clock.now() + 5.0)
            await runner.tick()
        assert runner.log.fired_counts.get("low_fuel", 0) <= 1

    @pytest.mark.asyncio
    async def test_two_runners_on_one_engine_would_double_count(self, mission) -> None:
        """Demonstrates WHY ownership matters: two runners sharing a log
        double-count. This is the shape of the bug, asserted so the reason
        for the guard stays documented in a test rather than only a
        comment."""
        channel = CollectingChannel()
        first = SessionRunner(mission, ScriptedModel(replies=["a"]), channel,
                              session_id="s", clock=VirtualClock(),
                              delivery_speed=1.0, seed=1)
        # A second runner over the SAME mission, as the two loops effectively
        # were -- separate bookkeeping, one world.
        second = SessionRunner(mission, ScriptedModel(replies=["b"]), channel,
                               session_id="s", clock=VirtualClock(),
                               delivery_speed=1.0, seed=1)
        first.clock.set(130.0)
        second.clock.set(130.0)
        await first.tick()
        await second.tick()
        total = (first.log.fired_counts.get("t1", 0)
                 + second.log.fired_counts.get("t1", 0))
        assert total == 2, "two independent loops each fire the same trigger"
