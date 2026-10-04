"""Realism layer tests: marker compilation and delivery (FR-A1..A9, FR-B1..B5).

The model is non-deterministic; THIS LAYER IS NOT. That is the whole point
of keeping timing outside the model, and it is what makes the tests below
possible at all: same markers plus same seed gives a byte-identical plan,
so pacing is assertable rather than merely observable.

Distribution properties are asserted over many samples rather than per
plan, because a single draw proves nothing about a random process.
"""

from __future__ import annotations

import asyncio

import pytest

from core.mission import load_mission_dict
from core.realism import (
    FORCE_GARBLE,
    FORCE_NO_REALISM,
    FORCE_STALL,
    compile_plan,
    strip_markers,
)
from delivery.scheduler import (
    LeadInStarted,
    PlanCompleted,
    PlanInterrupted,
    PlanStarted,
    SegmentStarted,
    execute,
)
from delivery.text_channel import CollectingChannel, TextChannel


@pytest.fixture
def mission(mission_with_everything):
    mission_with_everything["realism"].update({
        "stall_probability": 0.2,
        "filler_probability": 0.2,
        "filler_sounds": ["אה", "רגע"],
        "response_delay_ms": {"min": 600, "max": 2400, "distribution": "lognormal"},
        "stall_duration_ms": {"min": 400, "max": 1600},
    })
    return load_mission_dict(mission_with_everything)


def plan_for(mission, text, **kwargs):
    defaults = dict(session_id="s1", turn_id="t1", plan_id="p1", seed=42)
    return compile_plan(text, mission=mission, **{**defaults, **kwargs})


class TestMarkerStripping:
    """FR-A9: markers must never reach the transcript or replayed history.

    If they persist, the model starts imitating its own marker density and
    drifts within a few turns."""

    def test_all_markers_removed(self) -> None:
        text = "a «hesitate» b «filler» c «breath» d «urgent» e «calm» f «correct» g «garble» h"
        cleaned = strip_markers(text)
        assert "«" not in cleaned and "»" not in cleaned
        for word in "abcdefgh":
            assert word in cleaned

    def test_test_markers_removed(self) -> None:
        assert strip_markers(f"status {FORCE_GARBLE} report") == "status report"

    def test_whitespace_collapsed(self) -> None:
        assert strip_markers("a «hesitate»  «breath»  b") == "a b"

    def test_plain_text_untouched(self) -> None:
        assert strip_markers("מפקדה, נחשון 3, רות.") == "מפקדה, נחשון 3, רות."


class TestDeterminism:
    """FR-A7 -- the property that makes every other timing test possible."""

    def test_same_seed_gives_identical_plan(self, mission) -> None:
        text = "מפקדה, «hesitate» דלק 180, «breath» סוף."
        a = plan_for(mission, text, seed=12345)
        b = plan_for(mission, text, seed=12345)
        assert a.segments == b.segments
        assert a.lead_in_ms == b.lead_in_ms
        assert a.total_ms == b.total_ms

    def test_different_seed_gives_different_timing(self, mission) -> None:
        text = "מפקדה, «hesitate» דלק 180."
        leads = {plan_for(mission, text, seed=s).lead_in_ms for s in range(20)}
        assert len(leads) > 1, "timing must actually vary with the seed"

    def test_mission_seed_is_used_when_none_given(self, mission_with_everything) -> None:
        mission_with_everything["realism"]["seed"] = 777
        mission = load_mission_dict(mission_with_everything)
        a = compile_plan("test", mission=mission, session_id="s", turn_id="t", plan_id="p")
        b = compile_plan("test", mission=mission, session_id="s", turn_id="t", plan_id="p")
        assert a.lead_in_ms == b.lead_in_ms


class TestMarkerPlacement:
    """Markers tie pacing to MEANING -- the reason they exist instead of a
    generic scheduler. A pause before an uncertain figure reads as thought;
    the same pause at a random word reads as lag."""

    def test_hesitate_becomes_a_pause_before_the_figure(self, mission) -> None:
        plan = plan_for(mission, "דלק «hesitate» 180 ליברות")
        kinds = [s.kind for s in plan.segments]
        assert "pause" in kinds
        pause_index = kinds.index("pause")
        following = [s for s in plan.segments[pause_index + 1:] if s.kind == "speech"]
        assert following and "180" in following[0].text

    def test_filler_uses_configured_sounds(self, mission) -> None:
        plan = plan_for(mission, "«filler» רות")
        fillers = [s for s in plan.segments if s.kind == "filler"]
        assert fillers and fillers[0].text in mission.realism.filler_sounds

    def test_breath_is_shorter_than_hesitate(self, mission) -> None:
        """A clause break is not a hesitation; equal durations would make
        «breath» pointless."""
        hesitate = [s for s in plan_for(mission, "a «hesitate» b").segments if s.kind == "pause"]
        breath = [s for s in plan_for(mission, "a «breath» b").segments if s.kind == "pause"]
        assert breath[0].duration_ms < hesitate[0].duration_ms

    def test_no_adjacent_pauses(self, mission) -> None:
        """Two pauses back to back read as a dropped connection rather than
        as thinking -- observed as 620ms + 1490ms of unbroken dead air."""
        plan = plan_for(mission, "a, «hesitate» «breath» b, «breath» «hesitate» c")
        kinds = [s.kind for s in plan.segments]
        for first, second in zip(kinds, kinds[1:]):
            assert not (first == "pause" and second == "pause"), kinds

    def test_tone_shift_has_zero_duration(self, mission) -> None:
        """A manner change takes no time; giving it duration would insert
        a phantom pause."""
        plan = plan_for(mission, "a «urgent» b")
        shifts = [s for s in plan.segments if s.kind == "tone_shift"]
        assert shifts and all(s.duration_ms == 0 for s in shifts)

    def test_urgent_reduces_filler_multiplier(self, mission) -> None:
        """Stressed people use FEWER fillers. Getting this backwards reads
        as confusion rather than urgency."""
        plan = plan_for(mission, "a «urgent» b")
        after = [s for s in plan.segments if s.kind == "speech" and s.index > 0]
        assert after[-1].tone.filler_multiplier < 1.0
        assert after[-1].tone.pace > 1.0


class TestTimingProperties:
    def test_lead_in_within_configured_range(self, mission) -> None:
        delays = [plan_for(mission, "רות", seed=s).lead_in_ms for s in range(200)]
        low = mission.realism.response_delay_ms.min
        high = mission.realism.response_delay_ms.max
        assert all(low <= d <= high for d in delays)

    def test_lognormal_clusters_short(self, mission) -> None:
        """FR-A1. Uniform delay is the single biggest reason naive
        simulators read as network lag: it produces as many 2.4s pauses as
        0.7s ones, which no human conversation does."""
        delays = [plan_for(mission, "רות", seed=s).lead_in_ms for s in range(400)]
        low = mission.realism.response_delay_ms.min
        high = mission.realism.response_delay_ms.max
        midpoint = (low + high) / 2
        below = sum(1 for d in delays if d < midpoint)
        assert below / len(delays) > 0.6, "most replies should be prompt"

    def test_segments_are_contiguous(self, mission) -> None:
        """A gap or overlap would make total_ms wrong and desynchronize the
        voice channel's audio from the plan."""
        plan = plan_for(mission, "a, «hesitate» b, «filler» c")
        cursor = 0
        for segment in plan.segments:
            assert segment.start_ms == cursor
            cursor = segment.end_ms

    def test_priority_scales_lead_in(self, mission) -> None:
        """Urgency is audible in the timing before a word is understood."""
        critical = plan_for(mission, "בינגו", origin="initiated",
                            trigger_id="low_fuel")      # critical in fixture
        idle = plan_for(mission, "בודק", origin="initiated",
                        trigger_id="quiet")             # low in fixture
        assert critical.lead_in_ms < idle.lead_in_ms

    def test_empty_text_yields_no_segments(self, mission) -> None:
        assert plan_for(mission, "").segments == ()


class TestGarbling:
    """FR-A6: degradation is governed by mission STATE, never by the model."""

    def test_good_comms_never_garbles(self, mission) -> None:
        plans = [plan_for(mission, "דיווח מצב", seed=s, state={"comms": "good"})
                 for s in range(50)]
        assert not any(s.garbled for p in plans for s in p.segments)

    def test_bad_comms_sometimes_garbles(self, mission) -> None:
        plans = [plan_for(mission, "דיווח מצב", seed=s, state={"comms": "bad"})
                 for s in range(100)]
        garbled = sum(1 for p in plans for s in p.segments if s.garbled)
        assert garbled > 0

    def test_force_marker_guarantees_garbling(self, mission) -> None:
        """NFR-4: without this, garbling could only be exercised by
        retrying until a 40% chance fired -- a flaky test."""
        plan = plan_for(mission, f"{FORCE_GARBLE} דיווח מצב", state={"comms": "good"})
        assert any(s.garbled for s in plan.segments)

    def test_missing_state_does_not_crash(self, mission) -> None:
        assert plan_for(mission, "דיווח", state={}).segments


class TestInterruptibility:
    def test_number_spans_are_not_interruptible(self, mission) -> None:
        """Half a figure is worse than all of it or none: the trainee may
        act on the fragment."""
        plan = plan_for(mission, "דלק 180 ליברות")
        numeric = [s for s in plan.segments if s.kind == "speech" and any(c.isdigit() for c in s.text)]
        assert numeric and not any(s.interruptible for s in numeric)

    def test_pauses_are_interruptible(self, mission) -> None:
        plan = plan_for(mission, "a «hesitate» b")
        assert all(s.interruptible for s in plan.segments if s.kind == "pause")


class TestDeliveryExecution:
    @pytest.mark.asyncio
    async def test_full_delivery_event_sequence(self, mission) -> None:
        plan = plan_for(mission, "מפקדה, «hesitate» רות")
        channel = CollectingChannel()
        outcome = await execute(plan, channel, speed=0)

        assert outcome.status == "completed"
        assert not outcome.was_cut_short
        kinds = channel.kinds()
        assert kinds[0] == "PlanStarted"
        assert kinds[1] == "LeadInStarted"      # typing indicator before any text
        assert kinds[-1] == "PlanCompleted"

    @pytest.mark.asyncio
    async def test_barge_in_during_lead_in_delivers_nothing(self, mission) -> None:
        """Cut off before he started speaking: the transcript must show
        nothing, not the intended text."""
        plan = plan_for(mission, "מפקדה, רות")
        barge_in = asyncio.Event()
        barge_in.set()
        outcome = await execute(plan, CollectingChannel(), barge_in=barge_in, speed=0)
        assert outcome.status == "abandoned"
        assert outcome.delivered_text == ""

    @pytest.mark.asyncio
    async def test_interruption_records_only_what_was_heard(self, mission) -> None:
        """FR-B3. Recording the full intended text would make a debrief a
        lie -- the trainee would be shown words never spoken."""
        plan = plan_for(mission, "מפקדה, נחשון 3, «breath» דלק 180, «breath» סוף")
        barge_in = asyncio.Event()

        async def cut_in():
            await asyncio.sleep(0.05)
            barge_in.set()

        outcome, _ = await asyncio.gather(
            execute(plan, CollectingChannel(), barge_in=barge_in, speed=40.0),
            cut_in(),
        )
        assert outcome.status == "interrupted"
        assert outcome.was_cut_short
        assert len(outcome.delivered_text) < len(plan.clean_text)

    @pytest.mark.asyncio
    async def test_critical_initiative_rides_through_barge_in(self, mission) -> None:
        """FR-B5. A real operator shouting BINGO FUEL does not stop because
        you started talking, and that asymmetry is pedagogically correct."""
        plan = plan_for(mission, "בינגו דלק", origin="initiated",
                        trigger_id="low_fuel", yield_on_trainee_speech=False)
        barge_in = asyncio.Event()
        barge_in.set()
        outcome = await execute(plan, CollectingChannel(), barge_in=barge_in, speed=0)
        assert outcome.status == "completed"
        assert outcome.delivered_text == plan.clean_text

    @pytest.mark.asyncio
    async def test_no_realism_marker_disables_timing(self, mission) -> None:
        """So a test can assert on content without timing noise."""
        plan = plan_for(mission, f"{FORCE_NO_REALISM} מפקדה, רות")
        assert plan.lead_in_ms == 0
        assert all(s.kind == "speech" for s in plan.segments)


class TestTextChannel:
    @pytest.mark.asyncio
    async def test_emits_typing_then_chunks(self, mission) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        plan = plan_for(mission, "מפקדה, «hesitate» רות")
        await execute(plan, TextChannel(emit), speed=0)

        types = [e["type"] for e in emitted]
        assert types[0] == "utterance_start"
        assert "typing_start" in types
        assert "pause" in types          # UI keeps the indicator alive
        assert "chunk" in types
        assert types[-1] == "utterance_end"

    @pytest.mark.asyncio
    async def test_garbled_chunks_are_flagged(self, mission) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        plan = plan_for(mission, f"{FORCE_GARBLE} דיווח מצב")
        await execute(plan, TextChannel(emit), speed=0)
        chunks = [e for e in emitted if e["type"] == "chunk"]
        assert any(c["garbled"] for c in chunks)

    @pytest.mark.asyncio
    async def test_tone_shift_emitted_for_styling(self, mission) -> None:
        emitted: list[dict] = []

        async def emit(event: dict) -> None:
            emitted.append(event)

        plan = plan_for(mission, "רגוע «urgent» בינגו")
        await execute(plan, TextChannel(emit), speed=0)
        shifts = [e for e in emitted if e["type"] == "tone_shift"]
        assert shifts and shifts[0]["tone"] == "urgent"


class TestDeliveryRate:
    """Regression: text was paced at SPEAKING rate.

    The original constant was ~14 chars/sec, so a 59-character Hebrew
    transmission took 4.2 seconds to appear. In voice that is correct --
    the sentence genuinely takes that long to say. In text the trainee
    reads it at a glance, so it was pure waiting stacked on top of the
    model's own latency, and it is what made replies feel slow even on a
    fast model.
    """

    def test_text_is_much_faster_than_speech(self, mission) -> None:
        text = "מפקדה, נחשון 3, גובה 12,000, דלק 180, חיישן ללא מיקוד."
        as_text = plan_for(mission, text, for_voice=False)
        as_voice = plan_for(mission, text, for_voice=True)
        assert as_text.total_ms < as_voice.total_ms / 2

    def test_mission_can_override_the_rate(self, mission_with_everything) -> None:
        mission_with_everything["realism"]["chars_per_second"] = 10.0
        slow = load_mission_dict(mission_with_everything)
        mission_with_everything["realism"]["chars_per_second"] = 100.0
        fast = load_mission_dict(mission_with_everything)

        text = "מפקדה, נחשון 3, דלק 180 ליברות."
        slow_plan = compile_plan(text, mission=slow, session_id="s", turn_id="t",
                                 plan_id="p", seed=1)
        fast_plan = compile_plan(text, mission=fast, session_id="s", turn_id="t",
                                 plan_id="p", seed=1)
        assert slow_plan.total_ms > fast_plan.total_ms

    def test_voice_rate_matches_real_speech(self, mission) -> None:
        """A sanity bound: roughly 10-20 chars/sec is human speaking rate,
        so a 60-character sentence should take about 3-6 seconds aloud."""
        text = "א" * 60
        plan = plan_for(mission, text, for_voice=True)
        speech_ms = sum(s.duration_ms for s in plan.speech_segments)
        assert 3000 <= speech_ms <= 6500, speech_ms
