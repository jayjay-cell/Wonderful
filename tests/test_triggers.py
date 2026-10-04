"""Trigger and initiative tests (FR-C1..C9, FR-B5).

The asymmetry worth stating: the hard part is not making him talk, it is
making him SHUT UP. A counterpart who speaks whenever a condition fires
talks over the trainee and buries critical reports in chatter. Most of
these tests are therefore about suppression, deferral and priority rather
than about firing.

All run against a virtual clock, so a 45-second idle trigger and a
30-minute session both assert in milliseconds with exact ordering.
"""

from __future__ import annotations

import pytest

from core.mission import load_mission_dict
from core.models import Priority
from core.triggers import (
    IdleTriggerRule,
    ThresholdTriggerRule,
    TimelineTriggerRule,
    TriggerContext,
    build_triggers,
    select_firing,
)
from core.state import MissionStateEngine
from delivery.text_channel import CollectingChannel
from sim.clock import VirtualClock
from sim.runner import SessionRunner
from tests.fakes import ScriptedModel


@pytest.fixture
def mission(mission_with_everything):
    return load_mission_dict(mission_with_everything)


def context(**overrides) -> TriggerContext:
    defaults = dict(state={"fuel": 100.0, "mode": "idle", "comms": "good"},
                    mission_seconds=0.0)
    return TriggerContext(**{**defaults, **overrides})


class TestTimelineTriggers:
    def test_fires_at_its_time(self, mission) -> None:
        rule = TimelineTriggerRule(mission.triggers.timeline[0])   # t1 at 120s
        assert rule.evaluate(context(mission_seconds=119.0)) is None
        assert rule.evaluate(context(mission_seconds=120.0)) is not None

    def test_fires_only_once(self, mission) -> None:
        rule = TimelineTriggerRule(mission.triggers.timeline[0])
        ctx = context(mission_seconds=200.0, fired_counts={"t1": 1})
        assert rule.evaluate(ctx) is None

    def test_condition_gates_firing(self, mission_with_everything) -> None:
        """FR-C2 applied to a timeline: a checkpoint that only makes sense
        if the trainee missed something must not fire when they did not.
        Otherwise it teaches the opposite of the intended lesson."""
        mission_with_everything["triggers"]["timeline"].append({
            "id": "conditional", "at_mission_seconds": 100.0,
            "when": "mode != 'active'", "say_intent": "report miss",
        })
        mission = load_mission_dict(mission_with_everything)
        rule = next(TimelineTriggerRule(t) for t in mission.triggers.timeline
                    if t.id == "conditional")

        missed = context(mission_seconds=150.0, state={"mode": "idle"})
        tracked = context(mission_seconds=150.0, state={"mode": "active"})
        assert rule.evaluate(missed) is not None
        assert rule.evaluate(tracked) is None

    def test_after_waits_for_its_predecessor(self, mission_with_everything) -> None:
        """A chain stays ordered even when the earlier step was delayed by
        its own condition."""
        mission_with_everything["triggers"]["timeline"].append({
            "id": "second", "at_mission_seconds": 50.0, "after": "t1",
            "say_intent": "follow up",
        })
        mission = load_mission_dict(mission_with_everything)
        rule = next(TimelineTriggerRule(t) for t in mission.triggers.timeline
                    if t.id == "second")

        assert rule.evaluate(context(mission_seconds=200.0)) is None
        assert rule.evaluate(context(mission_seconds=200.0,
                                     fired_counts={"t1": 1})) is not None


class TestThresholdTriggers:
    def test_fires_when_condition_becomes_true(self, mission) -> None:
        rule = ThresholdTriggerRule(mission.triggers.thresholds[0])  # fuel <= 20
        assert rule.evaluate(context(state={"fuel": 50.0})) is None
        assert rule.evaluate(context(state={"fuel": 15.0})) is not None

    def test_once_prevents_refiring(self, mission) -> None:
        rule = ThresholdTriggerRule(mission.triggers.thresholds[0])
        ctx = context(state={"fuel": 15.0}, fired_counts={"low_fuel": 1})
        assert rule.evaluate(ctx) is None

    def test_unevaluatable_condition_does_not_crash(self, mission) -> None:
        """A derived value may be briefly unavailable (division by a
        drained-to-zero quantity). Ending the session over it would be
        worse than not firing."""
        rule = ThresholdTriggerRule(mission.triggers.thresholds[0])
        assert rule.evaluate(context(state={})) is None


class TestIdleTriggers:
    def test_fires_after_configured_silence(self, mission) -> None:
        rule = IdleTriggerRule(mission.triggers.idle[0])            # 45s
        assert rule.evaluate(context(mission_seconds=30.0,
                                     last_trainee_message_at=0.0)) is None
        assert rule.evaluate(context(mission_seconds=50.0,
                                     last_trainee_message_at=0.0)) is not None

    def test_max_fires_is_honoured(self, mission) -> None:
        """A prompt that repeats indefinitely stops reading as checking in
        and starts reading as nagging, training the trainee to ignore him."""
        rule = IdleTriggerRule(mission.triggers.idle[0])
        ctx = context(mission_seconds=500.0, last_trainee_message_at=0.0,
                      fired_counts={"quiet": 3})
        assert rule.evaluate(ctx) is None

    def test_fires_when_trainee_never_transmitted(self, mission) -> None:
        """A trainee who never checks in is a realistic failure worth
        training, so silence is measured from session start."""
        rule = IdleTriggerRule(mission.triggers.idle[0])
        assert rule.evaluate(context(mission_seconds=60.0,
                                     last_trainee_message_at=None)) is not None

    def test_does_not_prompt_straight_after_speaking(self, mission) -> None:
        rule = IdleTriggerRule(mission.triggers.idle[0])
        ctx = context(mission_seconds=60.0, last_trainee_message_at=0.0,
                      last_counterpart_message_at=55.0)
        assert rule.evaluate(ctx) is None


class TestSuppressionGuards:
    """The four guards (docs/specs/agent-initiative.md §4)."""

    def test_composing_suppresses_non_critical(self, mission) -> None:
        """Guard 3: he waits while the trainee is still talking."""
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 50.0, "mode": "idle", "comms": "good"},
                      mission_seconds=200.0, trainee_composing=True)
        firing, suppressions = select_firing(triggers, ctx)
        assert firing is None
        assert any(s.reason == "trainee_composing" for s in suppressions)

    def test_critical_overrides_composing(self, mission) -> None:
        """FR-B5: a bingo call does not wait for a gap in the conversation."""
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 10.0, "mode": "idle", "comms": "good"},
                      mission_seconds=200.0, trainee_composing=True)
        firing, _ = select_firing(triggers, ctx)
        assert firing is not None
        assert firing.priority is Priority.CRITICAL

    def test_delivery_active_defers_equal_priority(self, mission) -> None:
        """Guard 2 -- and note deferred, not dropped."""
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 50.0, "mode": "idle", "comms": "good"},
                      mission_seconds=200.0, delivery_active=True,
                      active_priority=Priority.NORMAL)
        firing, suppressions = select_firing(triggers, ctx)
        assert firing is None
        assert all(s.deferred for s in suppressions)

    def test_higher_priority_interrupts_lower_delivery(self, mission) -> None:
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 10.0, "mode": "idle", "comms": "good"},
                      mission_seconds=200.0, delivery_active=True,
                      active_priority=Priority.LOW)
        firing, _ = select_firing(triggers, ctx)
        assert firing is not None and firing.priority is Priority.CRITICAL

    def test_highest_priority_wins_when_several_fire(self, mission) -> None:
        """Guard 4: a bingo call is never buried behind a weather remark."""
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 10.0, "mode": "idle", "comms": "bad"},
                      mission_seconds=200.0)
        firing, suppressions = select_firing(triggers, ctx)
        assert firing.priority is Priority.CRITICAL
        assert any(s.reason == "lower_priority_than_chosen" for s in suppressions)

    def test_suppressions_are_always_reported(self, mission) -> None:
        """FR-C8: the absence of an event leaves no trace unless recorded,
        so 'why didn't he warn me?' is otherwise undebuggable."""
        triggers = build_triggers(mission)
        ctx = context(state={"fuel": 10.0, "mode": "idle", "comms": "good"},
                      mission_seconds=200.0, trainee_composing=True,
                      delivery_active=True, active_priority=Priority.CRITICAL)
        _, suppressions = select_firing(triggers, ctx)
        assert suppressions
        assert all(s.trigger_id and s.reason for s in suppressions)


class TestExtensibility:
    def test_a_new_trigger_kind_needs_no_other_changes(self, mission) -> None:
        """FR-C9. Any object with id, priority and evaluate() works."""

        class TraineeErrorTrigger:
            id = "too_many_errors"
            priority = Priority.HIGH

            def evaluate(self, ctx):
                from core.triggers import TriggerFiring
                if ctx.state.get("errors", 0) >= 3:
                    return TriggerFiring(
                        trigger_id=self.id, priority=self.priority,
                        say_intent="he is getting impatient",
                    )
                return None

        triggers = [*build_triggers(mission), TraineeErrorTrigger()]
        firing, _ = select_firing(triggers, context(state={"errors": 3}))
        assert firing is not None and firing.trigger_id == "too_many_errors"


class TestRunnerIntegration:
    """The full loop against a virtual clock."""

    @pytest.fixture
    def runner(self, mission):
        return SessionRunner(
            mission, ScriptedModel(replies=["מדווח."]), CollectingChannel(),
            session_id="test", clock=VirtualClock(), delivery_speed=0, seed=1,
        )

    @pytest.mark.asyncio
    async def test_timeline_checkpoint_fires_on_time(self, runner) -> None:
        # T+30s: before the checkpoint at 120s, and before the idle
        # trigger's 45s of silence, so nothing should fire at all.
        runner.clock.set(30.0)
        assert await runner.tick() is None

        runner.clock.set(130.0)
        record = await runner.tick()
        assert record is not None and record.trigger_id == "t1"

    @pytest.mark.asyncio
    async def test_effects_apply_when_a_checkpoint_fires(self, runner) -> None:
        runner.clock.set(130.0)
        await runner.tick()
        assert runner.session.engine.snapshot()["mode"] == "active"

    @pytest.mark.asyncio
    async def test_silent_checkpoint_applies_without_speaking(self, mission_with_everything) -> None:
        """A world-only checkpoint must not lose its effect by competing
        with a talking trigger for the single-flight slot."""
        mission_with_everything["triggers"]["timeline"].append({
            "id": "silent_change", "at_mission_seconds": 100.0,
            "effects": [{"parameter": "comms", "value": "bad"}],
        })
        mission = load_mission_dict(mission_with_everything)
        runner = SessionRunner(mission, ScriptedModel(replies=["x"]),
                               CollectingChannel(), clock=VirtualClock(),
                               delivery_speed=0, seed=1)
        runner.clock.set(110.0)
        await runner.tick()
        assert runner.session.engine.snapshot()["comms"] == "bad"
        assert runner.log.fired_counts["silent_change"] == 1

    @pytest.mark.asyncio
    async def test_deferred_firing_is_retried_not_dropped(self, runner) -> None:
        """A bingo call withheld because he was mid-sentence must still
        happen. Dropping it would be the simulation lying."""
        runner.set_composing(True)
        runner.clock.set(130.0)
        assert await runner.tick() is None
        assert runner.log.suppressions

        runner.set_composing(False)
        record = await runner.tick()
        assert record is not None and record.trigger_id == "t1"

    @pytest.mark.asyncio
    async def test_idle_capped_across_many_ticks(self, runner) -> None:
        for second in range(50, 600, 50):
            runner.clock.set(float(second))
            await runner.tick()
        assert runner.log.fired_counts.get("quiet", 0) <= 3

    @pytest.mark.asyncio
    async def test_trainee_message_runs_a_reactive_turn(self, runner) -> None:
        record = await runner.handle_trainee_message("דווח מצב")
        assert record is not None and record.origin == "reactive"
        assert record.text

    @pytest.mark.asyncio
    async def test_tone_shifts_once_its_trigger_fires(self, mission_with_everything) -> None:
        """Tone follows the situation, not the model's mood."""
        mission = load_mission_dict(mission_with_everything)
        runner = SessionRunner(mission, ScriptedModel(replies=["x"]),
                               CollectingChannel(), clock=VirtualClock(),
                               delivery_speed=0, seed=1)
        assert runner._current_tone() == mission.tone.baseline
        runner.log.fired_counts["low_fuel"] = 1
        assert runner._current_tone() == "urgent"


class TestInterruptionTiming:
    """Regression: an interrupt arriving while the model is still thinking
    must survive until delivery begins.

    OBSERVED LIVE: the runner cleared the barge-in flag immediately before
    execute(), so an interrupt sent during the model's thinking time was
    discarded -- and that is precisely when a trainee interrupts, because
    the counterpart has gone quiet.
    """

    @pytest.mark.asyncio
    async def test_interrupt_before_delivery_still_lands(self, mission) -> None:
        runner = SessionRunner(
            mission, ScriptedModel(replies=["מפקדה, נחשון 3, דלק 180, ראות 8"]),
            CollectingChannel(), clock=VirtualClock(), delivery_speed=0, seed=1,
        )
        # Interrupt first, then start the turn: the flag must persist
        # through the agent call and take effect at delivery.
        runner.interrupt()
        record = await runner._deliver(
            marked_text="מפקדה, נחשון 3, דלק 180",
            origin="initiated", trigger_id="t1", priority=Priority.NORMAL,
        )
        assert record.status in {"abandoned", "interrupted"}
        assert record.text == ""
        # What was intended is preserved for the debrief even though
        # nothing was heard.
        assert record.planned_text

    @pytest.mark.asyncio
    async def test_new_trainee_turn_clears_a_stale_interrupt(self, mission) -> None:
        """The flag is cleared when a trainee turn starts -- the one moment
        a pending interruption is genuinely stale. Otherwise every later
        reply would be cut off by an old signal."""
        runner = SessionRunner(
            mission, ScriptedModel(replies=["רות"]), CollectingChannel(),
            clock=VirtualClock(), delivery_speed=0, seed=1,
        )
        runner.interrupt()
        record = await runner.handle_trainee_message("דווח מצב")
        assert record is not None
        assert record.status == "completed"
        assert record.text == "רות"

    @pytest.mark.asyncio
    async def test_critical_initiative_survives_a_pending_interrupt(self, mission) -> None:
        """FR-B5 through the runner: a bingo call is delivered in full even
        with barge-in already pending."""
        runner = SessionRunner(
            mission, ScriptedModel(replies=["בינגו דלק"]), CollectingChannel(),
            clock=VirtualClock(), delivery_speed=0, seed=1,
        )
        runner.interrupt()
        record = await runner._deliver(
            marked_text="בינגו דלק", origin="initiated",
            trigger_id="low_fuel", priority=Priority.CRITICAL,
        )
        assert record.status == "completed"
        assert record.text == "בינגו דלק"


class TestDisplayRounding:
    """Regression: the trainer panel showed 179.84942222222202.

    No instrument reads like that, and a counterpart handed the long form
    will read the whole thing aloud."""

    def test_display_rounds_but_engine_keeps_precision(self, mission) -> None:
        engine = MissionStateEngine(mission)
        engine.advance_to(24.3)
        displayed = next(r["value"] for r in engine.describe_for_prompt(for_persona=False)
                         if r["id"] == "fuel")
        assert displayed == round(displayed, 1)
        # Full precision survives where it matters: arithmetic.
        assert engine.snapshot()["fuel"] != displayed
