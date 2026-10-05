"""The shared session layer: lifecycle, the domain loop, persistence.

EVERY CHANNEL DRIVES THIS. Text and Gemini Live
differ only in how audio moves; timeline, facts, commitments, handover,
lifecycle and persistence all live here. The previous version let
api/live_voice.py reimplement the domain layer, which silently lost two
procedure rules and a whole tool, and ran a second trigger loop that
double-counted every event.

The loop is split, which is the point of the refactor:

    _reveal_loop   runs advance() every TICK_SECONDS. No awaits on the
                   model or the speaker, so timeline revelation cannot be
                   delayed by a slow turn.
    _speak_loop    takes one owed report at a time and says it. Blocks
                   for seconds; that is fine, because it is not what
                   keeps time.

Persistence hangs off record(), so a voice session is as debriefable as a
text one -- which it previously was not at all.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

from core.lifecycle import Phase
from core.timeline import Priority
from obs.logging import get_logger
from sim.exercise import TICK_SECONDS, Exercise, Utterance
from sim.turns import TurnRunner

logger = get_logger("sim.session")


class Session:
    """One live exercise, with its channel and its store."""

    def __init__(
        self,
        exercise: Exercise,
        turns: TurnRunner,
        store: Any = None,
        channel: str = "text",
        on_utterance: Callable[[Utterance], Any] | None = None,
    ) -> None:
        """Hold the exercise, turn runner and store, and create the loops' task slots and the speaking lock."""
        self.exercise = exercise
        self.turns = turns
        self.store = store
        self.channel = channel

        # Called after every recorded utterance so a channel can render
        # it. Returning an awaitable is fine.
        self.on_utterance = on_utterance

        self._reveal_task: asyncio.Task | None = None
        self._speak_task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

        # One utterance at a time. A reactive reply and a report must not
        # overlap on the net.
        self._speaking = asyncio.Lock()

        self._persisted_reveals: set[str] = set()
        # commitment_id -> last status written, so a change is noticed.
        self._persisted_agreements: dict[str, str] = {}

    # -- lifecycle --------------------------------------------------------

    async def prepare(self) -> None:
        """Finish preparation. Mission time is still zero."""
        self.exercise.mark_ready()
        if self.store is not None:
            self.store.set_status(self.exercise.session_id, "ready")
            self.store.set_channel(self.exercise.session_id, self.channel)

    async def start(self) -> None:
        """Begin, alongside the trainer's manual video start."""
        self.exercise.start()
        if self.store is not None:
            self.store.set_status(self.exercise.session_id, "running")
        self._reveal_task = asyncio.create_task(self._reveal_loop())
        self._speak_task = asyncio.create_task(self._speak_loop())

    async def pause(self) -> None:
        """Freeze time and abandon anything in flight.

        The loops keep running but do nothing, because the clock is frozen
        and `is_running` is false. Cheaper and less error-prone than
        cancelling and recreating them.
        """
        self.exercise.pause()
        if self.store is not None:
            self.store.set_status(self.exercise.session_id, "paused")

    async def resume(self) -> None:
        """Continue from the frozen mission time and mark the session running again."""
        self.exercise.resume()
        if self.store is not None:
            self.store.set_status(self.exercise.session_id, "running")

    async def end(self) -> None:
        """Stop the loops, the clock and the model, and persist the final status."""
        self.exercise.end()
        self._stopped.set()
        for task in (self._reveal_task, self._speak_task):
            if task is not None and not task.done():
                task.cancel()
        if self.store is not None:
            self.store.set_status(
                self.exercise.session_id, "ended",
                datetime.now(timezone.utc).isoformat(),
            )

    @property
    def phase(self) -> Phase:
        """The current lifecycle phase, from the exercise."""
        return self.exercise.phase

    # -- the two loops ----------------------------------------------------

    async def _reveal_loop(self) -> None:
        """Advance the timeline. Never awaits the model or the speaker.

        This is the half that must stay on time: an authored event's
        moment is a fact about the recording playing beside us, and a slow
        turn must not move it.
        """
        try:
            while not self._stopped.is_set():
                newly = self.exercise.advance()
                for event in newly:
                    self._persist_reveal(event.event_id)
                self._persist_dropped()
                self._persist_new_agreements()

                if self.exercise.phase is Phase.ENDED:
                    await self.end()
                    return
                await asyncio.sleep(TICK_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            # A failure here would silently stop all revelation, so it is
            # logged loudly rather than swallowed.
            logger.exception("reveal_loop.failed",
                             session_id=self.exercise.session_id,
                             error_code=type(err).__name__, status="error")

    async def _speak_loop(self) -> None:
        """Deliver owed reports, and briefing requests, one at a time."""
        try:
            while not self._stopped.is_set():
                await asyncio.sleep(TICK_SECONDS)
                if not self.exercise.clock.is_running:
                    continue

                # Gemini Live delivers its own reports, in its own session.
                # Standing down here is what keeps one consumer per queue.
                if self.exercise.reports_owner != "session":
                    continue

                report = self.exercise.claim_report()
                if report is not None:
                    await self._speak_report(report)
                    continue

                if self.exercise.should_request_briefing():
                    await self._speak_briefing_request()
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.exception("speak_loop.failed",
                             session_id=self.exercise.session_id,
                             error_code=type(err).__name__, status="error")

    async def _speak_report(self, report: Any) -> None:
        """Deliver one owed report, recording it only if it was actually said."""
        async with self._speaking:
            result = await self.turns.report_turn(
                report.event.operator_information,
                report.event.event_id,
                report.event.instructions,
                report.event.priority,
            )
            if result.abandoned or result.failed:
                # Paused, ended, or the provider failed mid-generation.
                # Nothing was said, so nothing is recorded and the report
                # goes back on the queue to be retried while it stays
                # relevant.
                self.exercise.release_report(report)
                self._persist_reveal(report.event.event_id,
                                     disposition="pending")
                return
            self._record("operator", result)
            self.exercise.complete_report(report)
            self._persist_reveal(report.event.event_id, reported=True,
                                 disposition="delivered")

    async def _speak_briefing_request(self) -> None:
        """Have the crew ask for the briefing it never received."""
        async with self._speaking:
            result = await self.turns.briefing_request_turn()
            if not result.abandoned:
                self._record("operator", result)

    # -- trainee input ----------------------------------------------------

    async def trainee_says(self, text: str) -> Any:
        """Handle a trainee transmission.

        Takes the speaking lock, so a report in flight finishes first --
        except that an urgent report may already have interrupted, which
        the delivery layer handles.
        """
        if not self.exercise.clock.is_running:
            return None
        async with self._speaking:
            result = await self.turns.trainee_turn(text)
            # The trainee's own transmission is recorded by the turn
            # runner; this records the reply.
            if self.store is not None:
                self.store.add_utterance(
                    self.exercise.session_id, speaker="trainee", text=text,
                    mission_seconds=self.exercise.clock.now(), origin="reactive",
                    delivery=self._delivery_kind(),
                )
            if not result.abandoned:
                self._record("operator", result)
            return result

    def note_shared(self, fact: str, briefing: bool = False) -> None:
        """Record context the trainee supplied.

        Glok may reason with it, attributed -- it never overwrites what the
        recording shows.

        `briefing=True` marks the opening briefing as given, which is the
        transition that stops a proactive crew asking for it. Nothing
        reached BRIEFED before, so the crew asked even after being told.
        """
        self.exercise.state.note_shared(fact)
        if briefing:
            self.exercise.state.mark_briefed()

    # -- recording --------------------------------------------------------

    def _record(self, speaker: str, result: Any) -> None:
        """Record an utterance that was actually delivered."""
        utterance = Utterance(
            speaker=speaker,          # type: ignore[arg-type]
            text=result.text,
            at=self.exercise.clock.now(),
            origin=result.origin,
            event_id=result.event_id,
            status="failed" if result.failed else "completed",
        )
        self.exercise.record_utterance(utterance)

        if self.store is not None:
            self.store.add_utterance(
                self.exercise.session_id, speaker=speaker, text=result.text,
                mission_seconds=utterance.at, origin=result.origin,
                event_id=result.event_id, status=utterance.status,
                delivery=self._delivery_kind(),
            )

        if self.on_utterance is not None:
            maybe = self.on_utterance(utterance)
            if asyncio.iscoroutine(maybe):
                asyncio.create_task(maybe)

    def _delivery_kind(self) -> str:
        """How faithfully the stored text reflects what was heard.

        'streamed' for native speech-to-speech: the model generates audio
        and transcribes its own speech, so if the trainee talks over it
        what they actually heard can only be estimated. Marked rather than
        implying a precision we do not have.
        """
        return "streamed" if self.channel == "gemini_live" else "paced"

    def _persist_reveal(self, event_id: str, reported: bool = False,
                        disposition: str = "pending") -> None:
        """Record that an event became known, and what became of its report."""
        if self.store is None:
            return
        key = f"{event_id}:{disposition}"
        if key in self._persisted_reveals:
            return
        self._persisted_reveals.add(key)
        self.store.add_revealed(
            self.exercise.session_id, event_id,
            self.exercise.clock.now(), reported=reported,
            disposition=disposition,
        )

    def _persist_dropped(self) -> None:
        """Record reports dropped as stale, so a debrief sees them.

        A report that expired before it could be spoken is a fact about
        the exercise; leaving it only in the in-memory log meant it
        vanished with the process.
        """
        if self.store is None:
            return
        for _at, event_id, reason in self.exercise.log.dropped:
            self._persist_reveal(event_id, disposition=reason)

    def _persist_new_agreements(self) -> None:
        """Record agreements the model made via its tools.

        Polled from the ledger rather than hooked into the tool, so the
        tool stays pure and every channel persists identically.
        """
        if self.store is None:
            return
        for commitment in self.exercise.ledger.commitments:
            # Keyed by id AND status, so a later cancel or supersede is
            # written too. Keying by id alone froze every agreement at
            # "active" in the store however it actually ended.
            seen = self._persisted_agreements.get(commitment.commitment_id)
            if seen == commitment.status.value:
                continue
            self._persisted_agreements[commitment.commitment_id] = \
                commitment.status.value
            self.store.add_agreement(
                self.exercise.session_id, commitment.commitment_id,
                self.exercise.clock.now(), list(commitment.tags),
                list(commitment.entity_ids), commitment.description,
                commitment.status.value,
            )
