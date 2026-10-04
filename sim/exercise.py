"""The exercise runner: one shared domain loop for every channel.

Replaces sim/runner.py's simulator loop. Text, the ElevenLabs cascade and
Gemini Live all drive THIS -- only audio transport differs. The previous
version had a second, silently divergent trigger loop inside
api/live_voice.py, which meant the Live path could not fire callsign or
readback rules at all and double-counted every trigger.

THE STRUCTURAL FIX. Revelation and delivery are separate:

    advance()   pure bookkeeping. Reveals timeline events, queues reports.
                Never awaits a model or a speaker. Cheap enough to call
                on every tick.
    pump()      takes one queued report and speaks it. May block for
                seconds on the model and the channel.

Previously one lock covered both, so a slow model call stalled the
timeline: a fuel event authored for T+300 was not noticed until whatever
was being said finished. Now an event becomes known at its authored time
even while Glok is mid-transmission, the trainee is talking, or the crew
is handing over. Only the REPORT waits.

A queued report is re-checked for relevance before it is spoken, and is
marked delivered only once it actually has been -- not when it was
queued, because a pause or an interruption can cancel it.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from core.commitments import CommitmentLedger
from core.conversation_state import BriefingStage, ConversationState
from core.lifecycle import ExerciseClock, Phase
from core.mission import Mission
from core.timeline import Priority, TimelineEvent, Timeline
from obs.logging import get_logger

logger = get_logger("sim.exercise")

# How often the domain loop reveals facts. Fine enough that a report is
# never visibly late, cheap because advance() does no I/O.
TICK_SECONDS = 0.25


@dataclass
class PendingReport:
    """A report owed but not yet spoken."""

    event: TimelineEvent
    queued_at: float
    reason: Literal["required", "commitment"] = "required"

    @property
    def priority(self) -> Priority:
        return self.event.priority


@dataclass
class Utterance:
    """Something that was actually said, by either side."""

    speaker: Literal["trainee", "operator"]
    text: str
    at: float
    origin: Literal["reactive", "report", "briefing_request"] = "reactive"
    event_id: str | None = None
    status: str = "completed"
    planned_text: str = ""


@dataclass
class ExerciseLog:
    """Everything a debrief needs, including what was withheld."""

    utterances: list[Utterance] = field(default_factory=list)
    revealed: list[tuple[float, str]] = field(default_factory=list)
    dropped: list[tuple[float, str, str]] = field(default_factory=list)
    handover_blocks: int = 0


class Exercise:
    """One running exercise."""

    def __init__(
        self,
        mission: Mission,
        timeline: Timeline,
        model: Any,
        context: str = "",
        session_id: str | None = None,
        clock: Any | None = None,
    ) -> None:
        self.mission = mission
        self.timeline = timeline
        self.context = context
        self.session_id = session_id or secrets.token_urlsafe(12)

        # Injectable for tests (VirtualClock); production always gets the
        # real one, because the exercise runs alongside a video player and
        # any scaling would desynchronise them.
        self.clock = clock or ExerciseClock()

        self.state = ConversationState(
            reestablish_silence=mission.reestablish_silence,
        )
        self.ledger = CommitmentLedger()
        self.log = ExerciseLog()

        self._model = model
        self._pending: list[PendingReport] = []
        self._revealed_through: float = 0.0

        # One utterance at a time, so two reports cannot overlap.
        self._speaking = asyncio.Lock()
        self._channel: Any = None
        self._agent: Any = None

        # Guards against a late provider result arriving after a pause or
        # end and being spoken into a stopped exercise.
        self._generation = 0

    # -- lifecycle --------------------------------------------------------

    def mark_ready(self) -> None:
        duration = self.mission.duration_seconds or self.timeline.duration or None
        self.clock.mark_ready(duration)
        logger.info("exercise.ready", session_id=self.session_id,
                    mission_id=self.mission.id)

    def start(self) -> None:
        self.clock.start()
        logger.info("exercise.started", session_id=self.session_id)

    def pause(self) -> None:
        """Freeze mission time and abandon anything in flight.

        Bumping the generation is what stops a model call that is already
        running from speaking into a paused exercise when it returns.
        """
        self.clock.pause()
        self._generation += 1
        logger.info("exercise.paused", session_id=self.session_id,
                    mission_seconds=int(self.clock.now()))

    def resume(self) -> None:
        """Continue from frozen time.

        Stale pending reports are dropped here rather than delivered in a
        burst: a pause can last minutes, and announcing six old
        observations at once is worse than losing them.
        """
        self.clock.resume()
        now = self.clock.now()
        kept: list[PendingReport] = []
        for report in self._pending:
            if report.event.is_stale_at(now):
                self.log.dropped.append((now, report.event.event_id, "stale_on_resume"))
            else:
                kept.append(report)
        self._pending = kept
        logger.info("exercise.resumed", session_id=self.session_id,
                    mission_seconds=int(now))

    def end(self) -> None:
        self.clock.end()
        self._generation += 1
        self._pending.clear()
        logger.info("exercise.ended", session_id=self.session_id,
                    mission_seconds=int(self.clock.now()))

    @property
    def phase(self) -> Phase:
        return self.clock.phase

    def set_channel(self, channel: Any) -> None:
        self._channel = channel

    # -- revelation (never blocks) ----------------------------------------

    def advance(self) -> tuple[TimelineEvent, ...]:
        """Reveal whatever the clock has reached, and queue owed reports.

        Pure bookkeeping: no model, no speaker, no awaits. This is the half
        that must never be delayed by a slow turn, because an event's
        authored time is a fact about the recording playing beside us.
        """
        if not self.clock.is_running:
            return ()

        now = self.clock.now()
        if now <= self._revealed_through:
            return ()

        newly = self.timeline.newly_revealed(self._revealed_through, now)
        self._revealed_through = now

        for event in newly:
            if event.operator_information:
                self.log.revealed.append((event.start_time, event.event_id))
            if self.ledger.should_report(event):
                reason = ("required"
                          if event.reporting_policy.value == "required"
                          else "commitment")
                self._pending.append(PendingReport(event, now, reason))

        if self.clock.is_past_duration():
            self.end()
        return newly

    # -- queries (operator-visible only) ----------------------------------

    def current_information(self) -> dict[str, Any]:
        """What Glok can see right now.

        Initial facts, overlaid by revealed timeline updates, plus the
        facts that are currently true. Nothing unrevealed, nothing
        author-facing, nothing private.
        """
        now = self.clock.now()
        facts = dict(self.mission.operator_facts())
        facts.update(self.timeline.current_facts(now))
        hidden = set(self.mission.hidden_facts)
        return {k: v for k, v in facts.items() if k not in hidden}

    def revealed_observations(self, current_only: bool = False) -> list[dict[str, Any]]:
        """Revealed facts, flagged current or past.

        The caller phrases past observations in past tense; an expired
        interval must never be presented as the current picture.
        """
        now = self.clock.now()
        out = []
        for fact in self.timeline.revealed(now):
            if current_only and not fact.is_current:
                continue
            out.append({
                "event_id": fact.event_id,
                "information": fact.text,
                "at_seconds": fact.at,
                "is_current": fact.is_current,
            })
        return out

    def pending_reports(self) -> tuple[PendingReport, ...]:
        """Reports owed but not yet spoken.

        A public accessor so the trainer panel does not reach into a
        private field -- the previous API read `runner._pending` directly.
        """
        return tuple(self._pending)

    def handover_active(self) -> TimelineEvent | None:
        return self.timeline.handover_at(self.clock.now())

    def crew_available(self, priority: Priority = Priority.NORMAL) -> bool:
        """Whether ordinary conversation is possible.

        Enforced here, in code, from the timeline -- not suggested in a
        prompt, because 'do not reply normally' is exactly the instruction
        a model drops under pressure.
        """
        handover = self.handover_active()
        if handover is None:
            return True
        if priority is Priority.URGENT and self.mission.handover.allow_urgent:
            return True
        return False

    # -- delivery (may block) ---------------------------------------------

    def next_report(self) -> PendingReport | None:
        """The next report that should be spoken, or None.

        Re-checked against the clock at the point of delivery rather than
        when queued: an interval that has since closed, or an expiry that
        has passed, is dropped instead of announced as news.
        """
        if not self.clock.is_running or not self._pending:
            return None

        now = self.clock.now()
        kept: list[PendingReport] = []
        ready: list[PendingReport] = []
        for report in self._pending:
            if report.event.is_stale_at(now):
                self.log.dropped.append((now, report.event.event_id, "stale"))
                continue
            if not self.crew_available(report.priority):
                self.log.handover_blocks += 1
                kept.append(report)          # deferred, never dropped
                continue
            ready.append(report)

        if not ready:
            self._pending = kept
            return None

        # Urgent first, then oldest, so a time-critical report is not
        # buried behind routine traffic queued earlier.
        order = {Priority.URGENT: 0, Priority.HIGH: 1, Priority.NORMAL: 2}
        ready.sort(key=lambda r: (order[r.priority], r.queued_at))
        chosen = ready[0]
        self._pending = kept + [r for r in ready if r is not chosen]
        return chosen

    def may_interrupt(self, priority: Priority) -> bool:
        """Whether a report may cut across trainee speech.

        Routine traffic waits. Only an urgent event interrupts, and only
        if the exercise allows it -- a crew that cuts in routinely is a
        nuisance rather than a realistic one.
        """
        if priority is not Priority.URGENT:
            return False
        return self.mission.reporting.allow_urgent_interruption

    def record_utterance(self, utterance: Utterance) -> None:
        """Record something actually said.

        `mark_reported` is called from HERE, after delivery, so a report
        cancelled by a pause or an interruption is retried rather than
        silently counted as done.
        """
        self.log.utterances.append(utterance)
        if utterance.speaker == "operator":
            self.state.operator_spoke(utterance.at)
            if utterance.event_id and utterance.status == "completed":
                self.ledger.mark_reported(utterance.event_id)
        else:
            self.state.trainee_spoke(utterance.at, utterance.text)

    # -- briefing ---------------------------------------------------------

    def should_request_briefing(self) -> bool:
        """Whether a proactive crew should ask for a skipped briefing."""
        if not self.clock.is_running or not self.crew_available():
            return False
        return self.state.should_request_briefing(
            self.clock.now(),
            self.mission.behaviour.initiative,
            self.mission.behaviour.briefing_request_after,
        )

    # -- generation guard -------------------------------------------------

    def generation(self) -> int:
        return self._generation

    def is_current_generation(self, generation: int) -> bool:
        """Whether a result from a model call started earlier may still be
        spoken. False after a pause or an end, so a late provider response
        cannot leak across the boundary."""
        return generation == self._generation and self.clock.is_running
