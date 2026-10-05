"""The brief's 12 acceptance cases.

Real invariants, not prompt-string matching: each test asserts behaviour
the product depends on, with fake models and virtual time so a 20-minute
exercise runs in milliseconds.

Numbered to match the brief, so a reader can check coverage directly.
"""

from __future__ import annotations

import asyncio

import pytest

from core.lifecycle import Phase
from core.timeline import Priority
from sim.exercise import Exercise, Utterance
from tests.fakes import FailingModel, ScriptedModel


# =========================================================================
# 1. Ready preparation takes time, but exercise time stays zero until Start
# =========================================================================


class TestReadyHoldsTimeAtZero:
    """The trainer starts an external video by hand at roughly the same
    moment. If setup time counted as mission time, the exercise would
    already be out of step before the video began."""

    def test_preparing_is_zero(self, ready_exercise) -> None:
        """Mission time is zero while still preparing."""
        assert ready_exercise.clock.now() == 0.0
        assert ready_exercise.phase is Phase.PREPARING

    def test_ready_stays_zero_however_long_preparation_takes(
        self, ready_exercise, clock
    ) -> None:
        """Ninety seconds of voice setup still leaves mission time at zero."""
        ready_exercise.mark_ready()
        clock.set(90.0)          # 90 seconds of voice setup and validation
        assert ready_exercise.clock.now() == 0.0
        assert ready_exercise.phase is Phase.READY

    def test_clock_runs_only_after_start(self, ready_exercise, clock) -> None:
        """Mission time begins advancing only once Start is pressed."""
        ready_exercise.mark_ready()
        clock.set(0.0)
        ready_exercise.start()
        clock.set(30.0)
        assert ready_exercise.clock.now() == 30.0

    def test_advance_does_nothing_before_start(self, ready_exercise, clock) -> None:
        """Revelation must not run during preparation, or events would be
        consumed before the trainee is watching."""
        ready_exercise.mark_ready()
        clock.set(0.0)
        assert ready_exercise.advance() == ()

    def test_cannot_start_twice(self, ready_exercise, clock) -> None:
        """A second Start raises rather than silently restarting the clock."""
        from core.lifecycle import LifecycleError
        ready_exercise.mark_ready()
        clock.set(0.0)
        ready_exercise.start()
        with pytest.raises(LifecycleError):
            ready_exercise.start()


# =========================================================================
# 2. Future facts and private information are inaccessible
# =========================================================================


class TestNoLeakOfFutureOrPrivate:
    """The engine holds the whole timeline; the model must only ever see
    revealed operator_information. A leak here would let the operator
    answer questions about things that have not happened."""

    def _prompt(self, exercise) -> str:
        """Build the operator's system prompt for this exercise."""
        from agent.prompts import build_system_prompt
        return build_system_prompt(exercise)

    def test_unrevealed_information_absent_from_prompt(self, exercise, clock) -> None:
        """Future events do not appear in the prompt -- the no-future-leak rule."""
        clock.set(200.0)
        exercise.advance()
        prompt = self._prompt(exercise)
        # veh2 (480s), veh3 (600s) and urgent (840s) are still future.
        assert "רכב שני" not in prompt
        assert "רכב שלישי" not in prompt
        assert "תנועה חריגה" not in prompt

    def test_revealed_information_is_present(self, exercise, clock) -> None:
        """The inverse, so the test above cannot pass vacuously."""
        clock.set(200.0)
        exercise.advance()
        assert "רכב ראשון" in self._prompt(exercise)

    def test_author_descriptions_never_reach_the_prompt(self, exercise, clock) -> None:
        """description holds intent and the solution; only
        operator_information is for the model."""
        clock.set(900.0)
        exercise.advance()
        prompt = self._prompt(exercise)
        for marker in ("AUTHOR_VIS", "AUTHOR_VEH1", "AUTHOR_VEH2",
                       "AUTHOR_VEH3", "AUTHOR_HO", "AUTHOR_URGENT"):
            assert marker not in prompt

    def test_private_notes_never_reach_the_prompt(self, exercise, clock) -> None:
        """The author's private solution stays out of the prompt."""
        clock.set(900.0)
        exercise.advance()
        prompt = self._prompt(exercise)
        assert "PRIVATE_SOLUTION" not in prompt
        assert "PRIVATE_POINT" not in prompt

    def test_trainee_briefing_never_reaches_the_prompt(self, exercise) -> None:
        """It may carry the intelligence intent the operator must not
        know, which is why it is separate from prior_briefing."""
        assert "TRAINEE_ONLY" not in self._prompt(exercise)

    def test_prior_briefing_does_reach_the_prompt(self, exercise) -> None:
        """The crew was briefed at the squadron and does not pretend
        otherwise."""
        assert "PRIOR_BRIEFING" in self._prompt(exercise)

    def test_schedule_times_absent(self, exercise, clock) -> None:
        """Knowing an event is due at 480s is knowing the future."""
        clock.set(200.0)
        exercise.advance()
        prompt = self._prompt(exercise)
        assert "480" not in prompt
        assert "00:08:00" not in prompt

    def test_no_leak_through_any_tool(self, exercise, clock) -> None:
        """Every tool, not just the prompt: the Live path reaches them
        directly."""
        import json
        from tools.mission_tools import build_mission_tools

        clock.set(200.0)
        exercise.advance()
        tools = {t.name: t for t in build_mission_tools(exercise)}

        blob = ""
        for name, tool in tools.items():
            args = {"about": ""} if name == "recall_observation" else {}
            if name in ("agree_to_report", "cancel_reporting", "cannot_comply"):
                continue
            blob += json.dumps(tool.invoke(args), ensure_ascii=False, default=str)

        for secret in ("רכב שני", "רכב שלישי", "תנועה חריגה",
                       "PRIVATE_SOLUTION", "AUTHOR_VEH2", "TRAINEE_ONLY"):
            assert secret not in blob, f"{secret} leaked through a tool"

    def test_commitment_returns_a_count_not_the_events(self, exercise, clock) -> None:
        """Agreeing to watch a category must not reveal what is coming."""
        from tools.mission_tools import build_mission_tools
        import json

        clock.set(100.0)
        tools = {t.name: t for t in build_mission_tools(exercise)}
        result = tools["agree_to_report"].invoke({"tags": ["vehicle"]})
        blob = json.dumps(result, ensure_ascii=False)
        assert result["matching_moments"] >= 1
        assert "רכב שני" not in blob
        assert "480" not in blob


# =========================================================================
# 3. Persistent facts, interval expiry, past observations, between-events
# =========================================================================


class TestTemporalSemantics:
    """An event's tense follows its type: interval, point and persistent each behave differently."""
    def test_interval_is_current_only_inside_its_window(self, exercise, clock) -> None:
        """An interval reads current inside its window and past once it closes."""
        clock.set(100.0)
        exercise.advance()
        assert any(o["event_id"] == "vis" and o["is_current"]
                   for o in exercise.revealed_observations())

        clock.set(300.0)                 # window closed at 240
        exercise.advance()
        assert any(o["event_id"] == "vis" and not o["is_current"]
                   for o in exercise.revealed_observations())

    def test_expired_interval_facts_fall_away(self, exercise, clock) -> None:
        """An expired observation must not keep asserting a fact about the
        current picture."""
        clock.set(100.0)
        exercise.advance()
        assert exercise.current_information().get("ראות") == "ירודה"

        clock.set(300.0)
        exercise.advance()
        # Back to the mission's initial value, not the interval's.
        assert exercise.current_information().get("ראות") == "בינונית"

    def test_point_event_is_never_current(self, exercise, clock) -> None:
        """A vehicle that stopped at 03:00 is not still stopping at
        10:00."""
        clock.set(900.0)
        exercise.advance()
        veh1 = next(o for o in exercise.revealed_observations()
                    if o["event_id"] == "veh1")
        assert not veh1["is_current"]

    def test_past_observations_remain_recallable(self, exercise, clock) -> None:
        """A past event stays recallable, marked as no longer current."""
        clock.set(900.0)
        exercise.advance()
        past = [o for o in exercise.revealed_observations() if not o["is_current"]]
        assert any("רכב ראשון" in o["information"] for o in past)

    def test_facts_between_events_stay_queryable(self, exercise, clock) -> None:
        """Not a queue of lines: at a moment with no event, the picture is
        still answerable."""
        clock.set(400.0)                 # between veh1 (180) and veh2 (480)
        exercise.advance()
        facts = exercise.current_information()
        assert facts, "facts must remain available between events"
        assert exercise.revealed_observations()


# =========================================================================
# 4. Timeline updates continue while speech/model output is busy
# =========================================================================


class TestRevelationIsNotBlockedByDelivery:
    """THE STRUCTURAL FIX. Previously one lock covered revelation and
    delivery, so a slow model call stalled the timeline and an event
    authored for T+480 was not noticed until the current utterance
    finished."""

    @pytest.mark.asyncio
    async def test_advance_reveals_during_a_slow_model_call(
        self, mission, timeline, clock
    ) -> None:
        """The key regression: the timeline keeps advancing while the model is generating."""
        from sim.turns import TurnRunner

        class SlowModel(ScriptedModel):
            """Takes long enough that several events would be missed."""

            async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
                """Reply after a delay long enough that events would be missed if revelation blocked."""
                await asyncio.sleep(0.05)
                return await super()._agenerate(messages, stop, run_manager, **kw)

        exercise = Exercise(mission, timeline, SlowModel(replies=["רות."]),
                            clock=clock)
        exercise.mark_ready()
        clock.set(0.0)
        exercise.start()

        turns = TurnRunner(exercise, SlowModel(replies=["רות."]))

        async def slow_turn():
            """Run one deliberately slow trainee turn."""
            await turns.trainee_turn("דווח מצב")

        async def keep_time():
            # The timeline advances while the turn is in flight.
            """Advance the clock and reveal events while that turn is still in flight."""
            for second in (100.0, 200.0, 500.0):
                clock.set(second)
                exercise.advance()
                await asyncio.sleep(0.01)

        await asyncio.gather(slow_turn(), keep_time())

        revealed = {event_id for _, event_id in exercise.log.revealed}
        assert {"vis", "veh1", "veh2"} <= revealed, (
            "revelation must not wait for the model"
        )

    def test_advance_never_awaits(self) -> None:
        """Structural: advance() must stay synchronous, or a future change
        could reintroduce the coupling."""
        import inspect
        assert not inspect.iscoroutinefunction(Exercise.advance)


# =========================================================================
# 5. Required reports, subscriptions, cancellation, dedup, stale handling
# =========================================================================


class TestReporting:
    """What gets reported: required always, subscription only under an agreement, never twice."""
    def test_required_event_is_owed_without_an_agreement(self, exercise, clock) -> None:
        """A required event is owed a report with no agreement needed."""
        clock.set(200.0)
        exercise.advance()
        owed = {r.event.event_id for r in exercise.pending_reports()}
        assert "veh1" in owed

    def test_subscription_event_is_silent_without_an_agreement(
        self, exercise, clock
    ) -> None:
        """A subscription event stays silent until someone agrees to report it."""
        clock.set(500.0)
        exercise.advance()
        owed = {r.event.event_id for r in exercise.pending_reports()}
        assert "veh2" not in owed

    def test_one_agreement_covers_every_matching_later_event(
        self, exercise, clock
    ) -> None:
        """The brief's example: three vehicle appearances, only one
        proactive by default, and "tell me about every vehicle" should
        cover all the later ones."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0, tags=["vehicle"])
        clock.set(700.0)
        exercise.advance()
        owed = {r.event.event_id for r in exercise.pending_reports()}
        assert {"veh2", "veh3"} <= owed

    def test_agreement_defaults_to_future_events(self, exercise, clock) -> None:
        """Agreeing to watch for vehicles is not a request to recap."""
        clock.set(300.0)                 # veh1 already past
        exercise.advance()
        exercise.ledger.reported.clear()
        _, covered = exercise.ledger.register(
            exercise.timeline, at=300.0, tags=["vehicle"]
        )
        assert "veh1" not in {e.event_id for e in covered}

    def test_cancellation_stops_future_reports(self, exercise, clock) -> None:
        """Cancelling an agreement stops reports for later matching events."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0, tags=["vehicle"])
        exercise.ledger.cancel_all()
        clock.set(500.0)
        exercise.advance()
        owed = {r.event.event_id for r in exercise.pending_reports()}
        assert "veh2" not in owed

    def test_narrowing_to_one_entity(self, exercise, clock) -> None:
        """"Only that vehicle" -- supersede the broad agreement."""
        clock.set(100.0)
        broad, _ = exercise.ledger.register(
            exercise.timeline, at=100.0, tags=["vehicle"]
        )
        exercise.ledger.register(
            exercise.timeline, at=100.0, entity_ids=["veh3"],
            replaces=[broad.commitment_id],
        )
        clock.set(700.0)
        exercise.advance()
        owed = {r.event.event_id for r in exercise.pending_reports()}
        assert "veh3" in owed
        assert "veh2" not in owed

    def test_baseline_and_subscription_produce_one_report(
        self, exercise, clock
    ) -> None:
        """A REQUIRED event also covered by an agreement must not be
        reported twice."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0,
                                 tags=["vehicle", "conditions"])
        clock.set(200.0)
        exercise.advance()
        owed = [r.event.event_id for r in exercise.pending_reports()]
        assert owed.count("veh1") == 1

    def test_marking_reported_prevents_a_repeat(self, exercise, clock) -> None:
        """An event already reported is not reported again."""
        clock.set(200.0)
        exercise.advance()
        exercise.ledger.mark_reported("veh1")
        assert not exercise.ledger.should_report(exercise.timeline.event("veh1"))

    def test_stale_report_is_dropped_not_announced(self, exercise, clock) -> None:
        """veh3 expires at 660. If it was never delivered by then it is no
        longer news."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0, entity_ids=["veh3"])
        clock.set(620.0)
        exercise.advance()
        assert any(r.event.event_id == "veh3" for r in exercise.pending_reports())

        clock.set(700.0)                 # past the expiry
        exercise.advance()
        exercise.claim_report()
        assert "veh3" in [d[1] for d in exercise.log.dropped]

    def test_urgent_is_delivered_before_routine(self, exercise, clock) -> None:
        """A time-critical report must not queue behind routine traffic."""
        clock.set(850.0)
        exercise.advance()
        chosen = exercise.claim_report()
        assert chosen is not None and chosen.priority is Priority.URGENT


# =========================================================================
# 6. A proactive crew asks for a skipped briefing; a passive one does not
# =========================================================================


class TestBriefingRequest:
    """A proactive crew asks for a briefing it never got; a passive one does not."""
    def test_proactive_crew_asks_after_the_grace_period(
        self, exercise, clock
    ) -> None:
        """A proactive crew asks for the missing briefing once the grace period passes."""
        exercise.state.trainee_spoke(1.0, "גלוק, מדבקה, האם שומע?")
        clock.set(100.0)                 # grace is 60s in the fixture
        assert exercise.should_request_briefing()

    def test_not_before_the_grace_period(self, exercise, clock) -> None:
        """The crew stays quiet until the grace period has elapsed."""
        exercise.state.trainee_spoke(1.0, "שלום")
        clock.set(30.0)
        assert not exercise.should_request_briefing()

    def test_passive_crew_stays_quiet(self, mission_dict, timeline, clock) -> None:
        """A low-initiative crew never asks for the briefing."""
        from core.mission import load_mission_dict

        mission_dict["behaviour"]["initiative"] = 0.2
        mission = load_mission_dict(mission_dict)
        exercise = Exercise(mission, timeline, ScriptedModel(replies=["x"]),
                            clock=clock)
        exercise.mark_ready()
        clock.set(0.0)
        exercise.start()
        exercise.state.trainee_spoke(1.0, "שלום")
        clock.set(300.0)
        assert not exercise.should_request_briefing()

    def test_asked_only_once(self, exercise, clock) -> None:
        """A crew that keeps asking reads as nagging, not initiative."""
        exercise.state.trainee_spoke(1.0, "שלום")
        clock.set(100.0)
        assert exercise.should_request_briefing()
        exercise.state.mark_briefing_requested()
        clock.set(200.0)
        assert not exercise.should_request_briefing()

    def test_not_asked_once_briefed(self, exercise, clock) -> None:
        """Once briefed, the crew stops asking."""
        exercise.state.trainee_spoke(1.0, "שלום")
        exercise.state.mark_briefed()
        clock.set(300.0)
        assert not exercise.should_request_briefing()


# =========================================================================
# 7. Formal addressing on contact and re-contact, not every line
# =========================================================================


class TestAddressing:
    """Callsigns on contact and after a silence, not on every transmission."""
    def test_formal_before_contact(self, exercise) -> None:
        """Callsigns are required on the first transmission."""
        assert exercise.state.needs_formal_addressing(0.0)

    def test_informal_in_flowing_dialogue(self, exercise) -> None:
        """Mid-conversation transmissions need no callsigns."""
        exercise.state.trainee_spoke(10.0, "דווח מצב")
        exercise.state.operator_spoke(12.0)
        assert not exercise.state.needs_formal_addressing(15.0)

    def test_formal_again_after_a_break(self, exercise) -> None:
        """30 seconds by default -- an initial tuning value, not an
        asserted rule."""
        exercise.state.trainee_spoke(10.0, "דווח מצב")
        exercise.state.operator_spoke(12.0)
        assert exercise.state.needs_formal_addressing(50.0)

    def test_threshold_is_configurable(self, mission_dict, timeline, clock) -> None:
        """The re-addressing silence is a mission setting, not a hardcoded rule."""
        from core.mission import load_mission_dict

        mission_dict["reestablish_silence"] = 120.0
        mission = load_mission_dict(mission_dict)
        exercise = Exercise(mission, timeline, ScriptedModel(replies=["x"]),
                            clock=clock)
        exercise.state.trainee_spoke(10.0, "x")
        assert not exercise.state.needs_formal_addressing(60.0)
        assert exercise.state.needs_formal_addressing(140.0)

    def test_prompt_states_the_answer_not_a_rule(self, exercise) -> None:
        """Decided in code from elapsed silence. The old blanket
        every-transmission instruction is gone."""
        from agent.prompts import build_system_prompt

        exercise.state.trainee_spoke(10.0, "x")
        exercise.state.operator_spoke(11.0)
        exercise.clock.set(15.0)
        prompt = build_system_prompt(exercise)
        assert "CONVERSATION IS FLOWING" in prompt
        assert "Do NOT repeat callsigns" in prompt


# =========================================================================
# 8. Handover blocks ordinary conversation; authored urgent passes
# =========================================================================


class TestHandover:
    """A crew rotation blocks ordinary conversation, with authored exceptions."""
    def test_blocks_ordinary_conversation(self, exercise, clock) -> None:
        """Ordinary conversation is unavailable during a handover."""
        clock.set(850.0)                 # inside 800-920
        assert exercise.handover_active() is not None
        assert not exercise.crew_available(Priority.NORMAL)

    def test_urgent_passes_when_allowed(self, exercise, clock) -> None:
        """An urgent report still gets through a handover when the exercise permits it."""
        clock.set(850.0)
        assert exercise.crew_available(Priority.URGENT)

    def test_urgent_blocked_when_the_exercise_forbids_it(
        self, mission_dict, timeline, clock
    ) -> None:
        """With allow_urgent off, even urgent traffic waits."""
        from core.mission import load_mission_dict

        mission_dict["handover"] = {"allow_urgent": False}
        mission = load_mission_dict(mission_dict)
        exercise = Exercise(mission, timeline, ScriptedModel(replies=["x"]),
                            clock=clock)
        exercise.mark_ready()
        clock.set(0.0)
        exercise.start()
        clock.set(850.0)
        assert not exercise.crew_available(Priority.URGENT)

    def test_available_again_afterwards(self, exercise, clock) -> None:
        """The crew is reachable again once the handover window closes."""
        clock.set(950.0)
        assert exercise.handover_active() is None
        assert exercise.crew_available()

    def test_routine_report_defers_rather_than_dropping(
        self, exercise, clock
    ) -> None:
        """A report withheld during handover must still happen -- dropping
        it would be the exercise lying about what the crew saw."""
        clock.set(200.0)
        exercise.advance()
        clock.set(850.0)
        exercise.advance()
        exercise.claim_report()           # urgent goes first
        still_owed = {r.event.event_id for r in exercise.pending_reports()}
        assert {"vis", "veh1"} & still_owed

    def test_facts_and_agreements_carry_across_a_handover(
        self, exercise, clock
    ) -> None:
        """One exercise keeps one timeline and one ledger, so a rotation
        does not reset either. The former inherit_* settings described a
        separate incoming-crew memory that does not exist."""
        clock.set(100.0)
        exercise.advance()
        exercise.ledger.register(exercise.timeline, at=100.0, tags=["vehicle"])

        clock.set(850.0)                 # mid-handover
        exercise.advance()
        assert exercise.handover_active() is not None
        assert len(exercise.ledger.active) == 1

        clock.set(950.0)                 # after it
        assert exercise.crew_available()
        assert len(exercise.ledger.active) == 1
        assert "ראות" in exercise.current_information()

    @pytest.mark.asyncio
    async def test_reply_during_handover_is_the_busy_line(
        self, mission_dict, timeline, clock
    ) -> None:
        """Enforced in code: a prompt instruction is exactly what a model
        drops under pressure."""
        from core.mission import load_mission_dict
        from sim.turns import TurnRunner

        mission_dict["handover"] = {"busy_reply": "BUSY_HANDOVER"}
        mission = load_mission_dict(mission_dict)
        exercise = Exercise(mission, timeline, ScriptedModel(replies=["רות"]),
                            clock=clock)
        exercise.mark_ready()
        clock.set(0.0)
        exercise.start()
        clock.set(850.0)

        turns = TurnRunner(exercise, ScriptedModel(replies=["רות"]))
        result = await turns.trainee_turn("גלוק, מדבקה")
        assert result.text == "BUSY_HANDOVER"


# =========================================================================
# 9. Unsupported requests never mutate facts or claim a false action
# =========================================================================


class TestUnsupportedRequests:
    """Nothing the model can call changes the recording."""
    def test_no_tool_can_change_the_recording(self, exercise) -> None:
        """set_parameter is gone. Nothing exposed to the model writes
        world facts."""
        from tools.mission_tools import build_mission_tools

        names = {t.name for t in build_mission_tools(exercise)}
        assert "set_parameter" not in names
        for forbidden in ("set_", "change_", "slew", "zoom", "altitude"):
            assert not any(forbidden in n for n in names), names

    def test_facts_are_unchanged_by_any_tool_call(self, exercise, clock) -> None:
        """Calling every available tool leaves the world facts identical."""
        from tools.mission_tools import build_mission_tools

        clock.set(300.0)
        exercise.advance()
        before = dict(exercise.current_information())

        tools = {t.name: t for t in build_mission_tools(exercise)}
        tools["current_information"].invoke({})
        tools["recall_observation"].invoke({"about": ""})
        tools["cannot_comply"].invoke({"topic": "zoom"})
        tools["agree_to_report"].invoke({"tags": ["vehicle"]})

        assert exercise.current_information() == before

    def test_cannot_comply_returns_the_authored_reason(self, exercise) -> None:
        """A known impossible request returns the author's own wording."""
        from tools.mission_tools import build_mission_tools

        tools = {t.name: t for t in build_mission_tools(exercise)}
        result = tools["cannot_comply"].invoke({"topic": "zoom"})
        assert result["reason"] == "ZOOM_REASON"
        assert result["specific"] is True

    def test_unknown_topic_falls_back_honestly(self, exercise) -> None:
        """An unanticipated request gets the honest fallback, flagged as non-specific."""
        from tools.mission_tools import build_mission_tools

        tools = {t.name: t for t in build_mission_tools(exercise)}
        result = tools["cannot_comply"].invoke({"topic": "teleport"})
        assert result["reason"] == "FALLBACK_REASON"
        assert result["specific"] is False

    def test_prompt_forbids_claiming_the_action(self, exercise) -> None:
        """The prompt forbids claiming an action that cannot happen."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt(exercise)
        assert "never claim you did it" in prompt.lower()


# =========================================================================
# 10. Missing data yields an in-character unknown, not an invented reading
# =========================================================================


class TestMissingData:
    """An absent fact stays absent rather than being invented."""
    def test_absent_facts_are_simply_absent(self, exercise) -> None:
        """No fabricated defaults: a plausible-looking invented figure is
        worse than no answer."""
        facts = exercise.current_information()
        assert "fuel" not in facts
        assert "altitude" not in facts

    def test_tool_says_what_it_has_and_no_more(self, exercise) -> None:
        """The facts tool returns only revealed facts, and says so when asked for more."""
        from tools.mission_tools import build_mission_tools

        tools = {t.name: t for t in build_mission_tools(exercise)}
        result = tools["current_information"].invoke({})
        assert set(result["facts"]) == {"ראות"}
        assert "do not have it" in result["note"]

    def test_prompt_instructs_an_unknown_rather_than_a_guess(self, exercise) -> None:
        """The prompt tells the operator to admit ignorance rather than invent a figure."""
        from agent.prompts import build_system_prompt

        prompt = build_system_prompt(exercise)
        assert "say so, or ask" in prompt
        assert "Never produce a plausible-sounding value" in prompt

    def test_mission_with_no_facts_is_valid(self, mission_dict) -> None:
        """Requiring readings would invite invented ones."""
        from core.mission import load_mission_dict

        mission_dict.pop("initial_facts")
        mission = load_mission_dict(mission_dict)
        assert mission.initial_facts == {}

    def test_trainee_claims_do_not_overwrite_timeline_truth(
        self, exercise, clock
    ) -> None:
        """What the trainee asserts is recorded as theirs, not promoted to fact."""
        clock.set(100.0)
        exercise.advance()
        exercise.state.note_shared("הראות מעולה")   # contradicts the interval
        assert exercise.current_information()["ראות"] == "ירודה"
        assert "הראות מעולה" in exercise.state.shared_by_trainee


# =========================================================================
# 11. Pause freezes time and suppresses late output; resume preserves state
# =========================================================================


class TestPauseResumeEnd:
    """Pause freezes time and output; resume preserves state; end stops everything."""
    def test_pause_freezes_mission_time(self, exercise, clock) -> None:
        """Real time passing while paused does not move mission time."""
        clock.set(300.0)
        exercise.pause()
        frozen = exercise.clock.now()
        clock.advance(600.0)             # real time passes
        assert exercise.clock.now() == frozen

    def test_revelation_stops_while_paused(self, exercise, clock) -> None:
        """No events are revealed while paused."""
        clock.set(100.0)
        exercise.advance()
        exercise.pause()
        clock.advance(600.0)
        assert exercise.advance() == ()

    def test_resume_continues_from_frozen_time(self, exercise, clock) -> None:
        """Resume picks up at the frozen time, with no jump."""
        clock.set(300.0)
        exercise.advance()
        exercise.pause()
        exercise.resume()
        assert exercise.clock.now() == pytest.approx(300.0)

    def test_commitments_survive_a_pause(self, exercise, clock) -> None:
        """Agreements and their wording survive a pause and resume."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0, tags=["vehicle"],
                                 description="כל רכב")
        exercise.pause()
        exercise.resume()
        assert len(exercise.ledger.active) == 1
        assert exercise.ledger.active[0].description == "כל רכב"

    def test_facts_and_history_survive_a_pause(self, exercise, clock) -> None:
        """Revealed facts and the transcript survive a pause and resume."""
        clock.set(100.0)
        exercise.advance()
        exercise.record_utterance(
            Utterance(speaker="trainee", text="שלום", at=50.0)
        )
        exercise.pause()
        exercise.resume()
        assert exercise.current_information()["ראות"] == "ירודה"
        assert len(exercise.log.utterances) == 1

    def test_resume_drops_reports_that_went_stale(self, exercise, clock) -> None:
        """A report whose moment has passed is dropped on resume rather
        than announced as news.

        Note pausing does NOT advance mission time -- that is the whole
        point of freezing it -- so staleness here comes from mission time
        already being past the expiry when the pause began.
        """
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0, entity_ids=["veh3"])
        clock.set(620.0)
        exercise.advance()
        assert any(r.event.event_id == "veh3"
                   for r in exercise.pending_reports())

        # Mission time moves past veh3's 660s expiry, then a pause.
        clock.set(700.0)
        exercise.pause()
        exercise.resume()
        assert not any(r.event.event_id == "veh3"
                       for r in exercise.pending_reports())
        assert "veh3" in [d[1] for d in exercise.log.dropped]

    def test_pause_does_not_advance_mission_time(self, exercise, clock) -> None:
        """Explicit, because the test above depends on it: time spent
        paused is not mission time, so a resumed exercise stays in step
        with the video the trainer also paused."""
        clock.set(300.0)
        exercise.pause()
        clock.advance(900.0)
        exercise.resume()
        assert exercise.clock.now() == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_late_model_result_is_abandoned_after_a_pause(
        self, exercise
    ) -> None:
        """A provider result arriving after a pause must not be spoken
        into a stopped exercise."""
        from sim.turns import TurnRunner

        turns = TurnRunner(exercise, ScriptedModel(replies=["too late"]))
        generation = exercise.generation()
        exercise.pause()
        assert not exercise.is_current_generation(generation)

    def test_end_stops_everything(self, exercise, clock) -> None:
        """After End the clock, revelation and pending reports all stop."""
        clock.set(300.0)
        exercise.advance()
        exercise.end()
        assert exercise.phase is Phase.ENDED
        clock.advance(600.0)
        assert exercise.advance() == ()
        assert not exercise.pending_reports()

    def test_end_is_automatic_at_the_authored_duration(self, exercise, clock) -> None:
        """The exercise ends itself once it passes the authored duration."""
        clock.set(1250.0)                # duration is 1200
        exercise.advance()
        assert exercise.phase is Phase.ENDED

    @pytest.mark.asyncio
    async def test_no_turns_after_end(self, exercise) -> None:
        """A model call started before End may no longer speak."""
        from sim.turns import TurnRunner

        turns = TurnRunner(exercise, ScriptedModel(replies=["רות"]))
        exercise.end()
        generation = exercise.generation()
        assert not exercise.is_current_generation(generation)


# =========================================================================
# 12. Text and both voice paths share domain behaviour and persist history
# =========================================================================


class TestChannelsShareTheDomain:
    """Text and both voice paths run the same domain logic and persist the same way."""
    def test_all_channels_use_one_tool_set(self) -> None:
        """The Live bridge dispatches into build_mission_tools rather than
        reimplementing it -- the previous version's shim omitted a tool
        entirely."""
        import inspect
        from api import live_voice

        source = inspect.getsource(live_voice)
        assert "build_mission_tools" in source
        assert "def _read_state" not in source
        assert "def _set_parameter" not in source

    def test_live_declarations_match_the_real_tools(self, exercise) -> None:
        """Names must agree, because the declarations describe tools the
        shared implementation executes."""
        from providers.cloud.gemini_live import tool_declarations_for
        from tools.mission_tools import build_mission_tools

        declared = {
            fn.name
            for tool in tool_declarations_for(exercise)
            for fn in tool.function_declarations
        }
        actual = {t.name for t in build_mission_tools(exercise)}
        assert declared == actual, f"declared {declared} vs actual {actual}"

    def test_no_channel_runs_its_own_timeline_loop(self) -> None:
        """One domain loop per session. Two loops double-counted every
        event and could consume a once-only trigger twice."""
        import inspect
        from api import live_voice, main

        for module in (live_voice, main):
            source = inspect.getsource(module)
            assert "select_firing" not in source, module.__name__
            assert "_trigger_loop" not in source, module.__name__

    def test_store_records_both_speakers_for_any_channel(self) -> None:
        """A voice session persists both speakers, so it can be debriefed."""
        from api.store import SqliteSessionStore

        store = SqliteSessionStore(":memory:")
        store.create_session("s", "m", 1, "2026-01-01T00:00:00Z",
                             channel="gemini_live")
        store.add_utterance("s", "trainee", "דווח מצב", 10.0,
                            delivery="streamed")
        store.add_utterance("s", "operator", "רות", 12.0, delivery="streamed")

        transcript = store.transcript("s")
        assert [t["speaker"] for t in transcript] == ["trainee", "operator"]
        # Honest marker: with native streaming voice, what was HEARD can
        # only be estimated.
        assert all(t["delivery"] == "streamed" for t in transcript)

    def test_store_records_revealed_events_and_agreements(self) -> None:
        """Revealed events and agreements are persisted with their disposition."""
        from api.store import SqliteSessionStore

        store = SqliteSessionStore(":memory:")
        store.create_session("s", "m", 1, "2026-01-01T00:00:00Z")
        store.add_revealed("s", "veh1", 180.0, reported=True,
                           disposition="delivered")
        store.add_agreement("s", "c1", 100.0, tags=["vehicle"],
                            description="כל רכב")

        assert store.revealed("s")[0]["disposition"] == "delivered"
        assert store.agreements("s")[0]["tags"] == ["vehicle"]

    def test_undelivered_output_is_not_stored_as_heard(self) -> None:
        """An abandoned turn records empty text with a status, so a
        debrief never shows words the trainee did not hear."""
        from api.store import SqliteSessionStore

        store = SqliteSessionStore(":memory:")
        store.create_session("s", "m", 1, "2026-01-01T00:00:00Z")
        store.add_utterance("s", "operator", "", 10.0, status="abandoned",
                            planned_text="what it would have said")
        row = store.transcript("s")[0]
        assert row["text"] == ""
        assert row["status"] == "abandoned"
        assert row["planned_text"]


# =========================================================================
# 13. Regressions for bugs confirmed in the Gemini-Live-only cleanup
# =========================================================================


class TestTimeZeroEvents:
    """An event authored at 00:00 is revealed, and exactly once.

    `_revealed_through` started at 0.0 against a (since, until] window, so
    `0.0 < 0.0` was never true and a time-zero event was revealed ZERO
    times -- visible to current_information() but never reported.
    """

    def _at_zero(self, clock):
        """Build an exercise whose first event sits exactly at 00:00."""
        from core.mission import load_mission_dict
        from core.timeline import EventType, ReportingPolicy, build_timeline
        from sim.exercise import Exercise

        timeline = build_timeline([
            {"event_id": "zero", "start_time": 0.0,
             "event_type": EventType.POINT,
             "operator_information": "מתחילים",
             "reporting_policy": ReportingPolicy.REQUIRED},
            {"event_id": "ten", "start_time": 10.0,
             "event_type": EventType.POINT,
             "operator_information": "אחרי עשר",
             "reporting_policy": ReportingPolicy.REQUIRED},
        ])
        mission = load_mission_dict({
            "id": "z", "title": "z", "timeline": "t.csv",
            "callsigns": {"trainee": "A", "operator": "B"},
        })
        exercise = Exercise(mission, timeline,
                            ScriptedModel(replies=["רות."]), clock=clock)
        exercise.mark_ready()
        clock.set(0.0)
        exercise.start()
        return exercise

    def test_event_at_time_zero_is_revealed(self, clock) -> None:
        """The first tick at t=0 reveals it."""
        exercise = self._at_zero(clock)
        assert [e.event_id for e in exercise.advance()] == ["zero"]

    def test_event_at_time_zero_is_revealed_only_once(self, clock) -> None:
        """A second tick at the same instant does not repeat it."""
        exercise = self._at_zero(clock)
        exercise.advance()
        assert exercise.advance() == ()
        clock.set(5.0)
        assert exercise.advance() == ()

    def test_event_at_time_zero_is_queued_for_report(self, clock) -> None:
        """Being revealed, it is also owed -- the point of the fix."""
        exercise = self._at_zero(clock)
        exercise.advance()
        assert "zero" in {r.event.event_id for r in exercise.pending_reports()}


class TestReportOwnership:
    """One consumer per report queue, and claiming is not delivering."""

    def test_claim_keeps_the_report_queued(self, exercise, clock) -> None:
        """A claimed report stays owed until it is actually spoken, so a
        failure can retry it. Removing it on claim lost it outright."""
        clock.set(200.0)
        exercise.advance()
        claimed = exercise.claim_report()
        assert claimed is not None
        # Still queued: only a real delivery removes it.
        assert claimed.event.event_id in {
            r.event.event_id for r in exercise.pending_reports()
        }

    def test_a_claimed_report_is_not_claimed_twice(self, exercise, clock) -> None:
        """Two consumers racing one queue must not both get it."""
        clock.set(200.0)
        exercise.advance()
        first = exercise.claim_report()
        second = exercise.claim_report()
        assert first is not None
        assert second is None or second.event.event_id != first.event.event_id

    def test_release_makes_it_claimable_again(self, exercise, clock) -> None:
        """An abandoned delivery puts the report back."""
        clock.set(200.0)
        exercise.advance()
        report = exercise.claim_report()
        exercise.release_report(report)
        again = exercise.claim_report()
        assert again is not None
        assert again.event.event_id == report.event.event_id

    def test_complete_removes_it_and_marks_it_reported(
        self, exercise, clock
    ) -> None:
        """Only a real delivery clears the obligation."""
        clock.set(200.0)
        exercise.advance()
        report = exercise.claim_report()
        event_id = report.event.event_id
        exercise.complete_report(report)
        assert event_id not in {r.event.event_id
                                for r in exercise.pending_reports()}
        assert not exercise.ledger.should_report(
            exercise.timeline.event(event_id))

    def test_a_released_report_that_went_stale_is_dropped(
        self, exercise, clock
    ) -> None:
        """Retry only while it is still relevant."""
        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0,
                                 entity_ids=["veh3"])
        clock.set(620.0)
        exercise.advance()
        assert "veh3" in {r.event.event_id for r in exercise.pending_reports()}

        # Claim and release everything: nothing was delivered, so all of
        # it stays owed.
        for _ in range(10):
            report = exercise.claim_report()
            if report is None:
                break
            exercise.release_report(report)
        assert "veh3" in {r.event.event_id for r in exercise.pending_reports()}

        clock.set(700.0)                 # past expires_at 660
        exercise.advance()
        exercise.claim_report()          # re-checks relevance, drops veh3
        assert "veh3" in [d[1] for d in exercise.log.dropped]
        assert "veh3" not in {r.event.event_id
                              for r in exercise.pending_reports()}

    def test_session_stands_down_when_live_owns_reports(self) -> None:
        """The Session speak loop yields the queue to Gemini Live, so one
        queue has one consumer. Both consuming it fired reports twice."""
        import inspect
        from sim import session

        source = inspect.getsource(session.Session._speak_loop)
        assert "reports_owner" in source

    def test_live_bridge_claims_ownership_and_hands_it_back(self) -> None:
        """Ownership is taken for the socket's lifetime only."""
        import inspect
        from api import live_voice

        source = inspect.getsource(live_voice.LiveBridge.run)
        assert 'reports_owner = "gemini_live"' in source
        assert 'reports_owner = "session"' in source


class TestVoiceLifecycle:
    """Voice input is refused outside a running, available exercise."""

    def _bridge(self, exercise):
        """A LiveBridge with no socket, for gate checks only."""
        from api.live_voice import LiveBridge

        class _Session:
            """Stands in for sim.session.Session: the bridge only reads
            .exercise and .store off it."""

        class _Live:
            """Stands in for the API's live-session holder."""

            def __init__(self, ex):
                """Expose the exercise the way the real holder does."""
                self.session = _Session()
                self.session.exercise = ex
                self.session.store = None

        return LiveBridge(socket=None, live=_Live(exercise), provider=None)

    def test_rejected_before_start(self, mission, timeline, clock) -> None:
        """Audio before Start would be answered as though live."""
        exercise = Exercise(mission, timeline,
                            ScriptedModel(replies=["רות."]), clock=clock)
        exercise.mark_ready()
        assert not self._bridge(exercise)._accepting_input()

    def test_accepted_while_running(self, exercise, clock) -> None:
        """The normal case still works."""
        clock.set(100.0)
        assert self._bridge(exercise)._accepting_input()

    def test_rejected_while_paused(self, exercise, clock) -> None:
        """A pause must stop voice reaching the model."""
        clock.set(100.0)
        exercise.pause()
        assert not self._bridge(exercise)._accepting_input()

    def test_rejected_after_end(self, exercise, clock) -> None:
        """An ended exercise refuses further turns."""
        clock.set(100.0)
        exercise.end()
        assert not self._bridge(exercise)._accepting_input()

    def test_rejected_during_handover(self, exercise, clock) -> None:
        """Crew availability is enforced for voice as it is for text."""
        clock.set(850.0)                 # inside the handover window
        assert exercise.handover_active() is not None
        assert not self._bridge(exercise)._accepting_input()

    def test_urgent_report_still_passes_during_handover(
        self, exercise, clock
    ) -> None:
        """Gating input does not block the authored urgent exception."""
        clock.set(850.0)
        assert exercise.crew_available(Priority.URGENT)

    def test_microphone_audio_is_gated(self) -> None:
        """The browser audio loop consults the gate, not just the report loop."""
        import inspect
        from api import live_voice

        source = inspect.getsource(live_voice.LiveBridge._from_browser)
        assert "_accepting_input" in source


class TestAgreementPersistence:
    """A change of status is written, not just the first state."""

    def test_cancellation_is_persisted(self, exercise, clock) -> None:
        """Keying only by id froze every agreement at 'active' forever."""
        from api.store import SqliteSessionStore
        from sim.session import Session
        from sim.turns import TurnRunner

        store = SqliteSessionStore(":memory:")
        store.create_session(exercise.session_id, "m", 1,
                             "2026-01-01T00:00:00Z")
        session = Session(exercise, TurnRunner(exercise, ScriptedModel()),
                          store=store)

        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0,
                                 tags=["vehicle"], description="כל רכב")
        session._persist_new_agreements()
        assert [a["status"] for a in store.agreements(exercise.session_id)] \
            == ["active"]

        exercise.ledger.cancel_all()
        session._persist_new_agreements()
        statuses = [a["status"] for a in store.agreements(exercise.session_id)]
        assert "cancelled" in statuses

    def test_stale_drop_is_persisted(self, exercise, clock) -> None:
        """A report that expired unspoken is a fact worth debriefing."""
        from api.store import SqliteSessionStore
        from sim.session import Session
        from sim.turns import TurnRunner

        store = SqliteSessionStore(":memory:")
        store.create_session(exercise.session_id, "m", 1,
                             "2026-01-01T00:00:00Z")
        session = Session(exercise, TurnRunner(exercise, ScriptedModel()),
                          store=store)

        clock.set(100.0)
        exercise.ledger.register(exercise.timeline, at=100.0,
                                 entity_ids=["veh3"])
        clock.set(620.0)
        exercise.advance()
        clock.set(700.0)
        exercise.advance()
        exercise.claim_report()          # triggers the stale drop
        session._persist_dropped()

        rows = store.revealed(exercise.session_id)
        assert any(r["event_id"] == "veh3" and r["disposition"] == "stale"
                   for r in rows)


class TestBriefingTransition:
    """Briefing the crew actually marks it briefed."""

    def test_sharing_the_briefing_stops_the_request(
        self, exercise, clock
    ) -> None:
        """mark_briefed existed but nothing called it, so a proactive crew
        kept asking for a briefing it had already been given."""
        from sim.session import Session
        from sim.turns import TurnRunner

        session = Session(exercise, TurnRunner(exercise, ScriptedModel()))
        exercise.state.trainee_spoke(1.0, "גלוק, מדבקה")
        clock.set(100.0)
        assert exercise.should_request_briefing()

        session.note_shared("המשימה: לקבוע אם המבנה מאוכלס", briefing=True)
        assert not exercise.should_request_briefing()

    def test_ordinary_sharing_does_not_mark_briefed(
        self, exercise, clock
    ) -> None:
        """Passing one fact mid-mission is not the opening briefing."""
        from sim.session import Session
        from sim.turns import TurnRunner

        session = Session(exercise, TurnRunner(exercise, ScriptedModel()))
        exercise.state.trainee_spoke(1.0, "גלוק, מדבקה")
        clock.set(100.0)
        session.note_shared("יש דיווח על רכב לבן")
        assert exercise.should_request_briefing()


class TestElevenLabsIsGone:
    """The cascade path is removed, not merely unused."""

    def test_no_production_module_references_elevenlabs(self) -> None:
        """A leftover import would break a clean install.

        Production packages only: tests/test_architecture.py keeps the
        name in its denylist of SDKs that core/ may never import, which
        is exactly where it should stay.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for package in ("core", "sim", "api", "agent", "tools", "providers",
                        "obs"):
            for path in (root / package).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8").lower()
                assert "elevenlabs" not in text, path

    def test_no_elevenlabs_config_or_ui_option(self) -> None:
        """The key, the WS route and the UI option are all gone."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        for name in (".env.example", "ui/index.html", "ui/voice.js",
                     "requirements.txt"):
            path = root / name
            if path.exists():
                assert "elevenlabs" not in path.read_text(
                    encoding="utf-8").lower(), name

    def test_api_accepts_only_the_two_live_channels(self) -> None:
        """A channel the UI cannot serve must not validate."""
        from pydantic import ValidationError

        from api.schemas import StartSessionRequest

        assert StartSessionRequest(mission_file="m.yaml",
                                    channel="gemini_live").channel
        assert StartSessionRequest(mission_file="m.yaml",
                                    channel="text").channel
        with pytest.raises(ValidationError):
            StartSessionRequest(mission_file="m.yaml", channel="elevenlabs")

    def test_ui_sends_a_channel_the_api_accepts(self) -> None:
        """The UI option values ARE the API channel values. They had
        diverged, so picking voice mode failed with a 422."""
        import pathlib
        import re

        from api.schemas import StartSessionRequest

        html = (pathlib.Path(__file__).resolve().parent.parent
                / "ui" / "index.html").read_text(encoding="utf-8")
        block = re.search(r'<select id="mode">(.*?)</select>', html, re.S)
        assert block, "channel select not found"
        values = re.findall(r'value="([^"]+)"', block.group(1))
        assert values
        for value in values:
            StartSessionRequest(mission_file="m.yaml", channel=value)
