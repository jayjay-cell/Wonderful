"""The session runner: ties triggers, delivery and the agent together.

This is where step 4 and step 5 meet. It owns:

  * the tick loop that advances mission time and evaluates triggers
  * single-flight enforcement -- one utterance at a time (guard 1)
  * deferral retry -- a withheld firing is retried, never dropped
  * barge-in plumbing -- the trainee's transmission cuts in
  * recording what was ACTUALLY delivered, not what was planned

Deliberately separate from Session (sim/session.py), which handles one
turn. A turn is a request/response; this is a live session with its own
clock, and keeping them apart means the one-turn path stays testable
without starting a background task.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agent.prompts import build_initiative_cue
from core.models import Mission, Priority, StateCommand
from core.realism import compile_plan, new_seed
from core.triggers import (
    Suppression,
    TriggerContext,
    TriggerFiring,
    build_triggers,
    select_firing,
)
from delivery.plan import DeliveryOutcome, DeliveryPlan
from delivery.scheduler import DeliveryChannel, execute
from obs.logging import get_logger
from sim.clock import Clock, RealClock
from sim.session import Session

logger = get_logger("sim.runner")

# How often triggers are evaluated. 250ms is comfortably finer than any
# realistic trigger interval while staying cheap -- trigger evaluation is
# pure predicate work with no I/O.
TICK_SECONDS = 0.25

# The shortest lead-in the trainee ever sees. Exists so the typing
# indicator always appears, even when the model's own latency has already
# covered the persona's delay -- otherwise a reply materialises with no
# prior sign that anyone was composing it.
_MIN_LEAD_IN_MS = 180


@dataclass
class UtteranceRecord:
    """One delivered utterance, as it actually happened."""

    plan_id: str
    turn_id: str
    origin: str
    text: str
    mission_seconds: float
    trigger_id: str | None = None
    status: str = "completed"
    planned_text: str = ""
    delivered_segments: int = 0
    total_segments: int = 0


@dataclass
class SessionLog:
    """Everything a debrief needs, including what was withheld.

    `suppressions` exists because "why didn't he warn me about the fuel?"
    is undebuggable otherwise: the absence of an event leaves no trace
    unless it is recorded (FR-C8).
    """

    utterances: list[UtteranceRecord] = field(default_factory=list)
    suppressions: list[tuple[float, Suppression]] = field(default_factory=list)
    fired_counts: dict[str, int] = field(default_factory=dict)


class SessionRunner:
    """Drives one live training session."""

    def __init__(
        self,
        mission: Mission,
        model: Any,
        channel: DeliveryChannel,
        session_id: str | None = None,
        clock: Clock | None = None,
        delivery_speed: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.mission = mission
        self.session_id = session_id or secrets.token_urlsafe(8)
        self.clock = clock or RealClock()
        self.channel = channel
        self.delivery_speed = delivery_speed
        self.seed = seed if seed is not None else (mission.realism.seed or new_seed())

        self.session = Session(mission, model, self.session_id, clock=self.clock)
        self.triggers = build_triggers(mission)
        self.log = SessionLog()

        # Guard 1: single flight. One utterance at a time, server-side,
        # regardless of what any client does.
        self._lock = asyncio.Lock()
        self._delivery_active = False
        self._active_priority: Priority | None = None

        # True while a voice socket is attached: makes compile_plan use
        # real speaking rate instead of the fast text rate.
        self.for_voice = False

        self._barge_in = asyncio.Event()
        self._trainee_composing = False
        self._deferred: list[TriggerFiring] = []
        self._turn_counter = 0
        self._stopped = asyncio.Event()

    # -- trainee input ----------------------------------------------------

    @property
    def is_delivering(self) -> bool:
        """Whether the counterpart currently holds the net.

        Read by voice turn detection: barge-in only has meaning while he is
        actually speaking.
        """
        return self._delivery_active

    def set_composing(self, composing: bool) -> None:
        """Called when the trainee starts or stops composing.

        Feeds guard 3 (the composition window). In text this is the typing
        indicator; in voice it will be VAD -- the same predicate from two
        sources, so the rule is written once.
        """
        self._trainee_composing = composing

    def interrupt(self) -> None:
        """The trainee has started transmitting: cut in.

        Set unconditionally, even with nothing in flight: the scheduler
        checks it at the next segment boundary, so a race between a send
        and a reply starting cannot produce a talk-over.
        """
        self._barge_in.set()

    async def handle_trainee_message(self, text: str) -> UtteranceRecord | None:
        """Full reactive turn: interrupt, run the agent, deliver the reply."""
        self.interrupt()

        async with self._lock:
            self._barge_in.clear()
            self._trainee_composing = False

            # Time the model call, so the realism lead-in can ABSORB it
            # rather than stack on top. See _deliver's thinking_ms note.
            import time as _time
            started = _time.monotonic()
            result = await self.session.run_trainee_turn(text)
            thinking_ms = int((_time.monotonic() - started) * 1000)

            return await self._deliver(
                marked_text=result.text,
                origin="reactive",
                trigger_id=None,
                priority=Priority.NORMAL,
                thinking_ms=thinking_ms,
            )

    # -- the tick loop ----------------------------------------------------

    async def run_until_stopped(self) -> None:
        """Evaluate triggers on a tick until stopped or the mission ends."""
        limit_seconds = self.mission.limits.session_max_minutes * 60
        try:
            while not self._stopped.is_set():
                await self.tick()
                if self.clock.now() >= limit_seconds:
                    logger.info("session.time_limit",
                                session_id=self.session_id,
                                mission_seconds=int(self.clock.now()))
                    return
                await asyncio.sleep(TICK_SECONDS)
        except asyncio.CancelledError:
            raise

    def stop(self) -> None:
        self._stopped.set()

    async def tick(self) -> UtteranceRecord | None:
        """One evaluation pass. Separated from the loop so tests can step
        it against a virtual clock instead of waiting."""
        now = self.clock.now()
        self.session.engine.advance_to(now)

        context = self._context(now)

        # A previously deferred firing is retried FIRST, so a withheld
        # bingo call is not starved by newer, lower-priority traffic.
        pending = self._take_deferred(context)
        if pending is not None:
            return await self._fire(pending, now)

        firing, suppressions = select_firing(self.triggers, context)

        # Silent checkpoints are applied IMMEDIATELY rather than queued.
        # They only change the world, so there is nothing to say and no
        # reason to compete for the single-flight slot -- and without this
        # a silent checkpoint loses to any talking trigger on the same tick
        # and its world change is lost (observed: visibility never dropped
        # because comms_degraded outranked it).
        applied_silently = self._apply_silent_firings(context, now)

        for suppression in suppressions:
            self.log.suppressions.append((now, suppression))
            logger.info("trigger.suppressed",
                        session_id=self.session_id,
                        trigger_id=suppression.trigger_id,
                        reason=suppression.reason,
                        status="suppressed")
            if suppression.deferred:
                self._defer(suppression.trigger_id)

        if firing is None or firing.trigger_id in applied_silently:
            return None
        return await self._fire(firing, now)

    def _apply_silent_firings(self, ctx: TriggerContext, now: float) -> set[str]:
        """Apply world-only checkpoints at once, outside the speaking path.

        A silent checkpoint has nothing to say, so making it queue behind a
        talking trigger means its world change may never land. Returns the
        ids applied, so the speaking path skips them.
        """
        applied: set[str] = set()
        for trigger in self.triggers:
            firing = trigger.evaluate(ctx)
            if firing is None or not firing.is_silent:
                continue
            self.log.fired_counts[firing.trigger_id] = (
                self.log.fired_counts.get(firing.trigger_id, 0) + 1
            )
            for effect in firing.effects:
                result = self.session.engine.apply(StateCommand(
                    parameter_id=effect.parameter, value=effect.value, source="trigger",
                ))
                if not result.accepted:
                    logger.warning("trigger.effect_rejected",
                                   session_id=self.session_id,
                                   trigger_id=firing.trigger_id,
                                   parameter_id=effect.parameter,
                                   error_code=result.reason_code,
                                   status="rejected")
            logger.info("trigger.fired_silent",
                        session_id=self.session_id,
                        trigger_id=firing.trigger_id,
                        mission_seconds=int(now))
            applied.add(firing.trigger_id)
        return applied

    async def _fire(self, firing: TriggerFiring, now: float) -> UtteranceRecord | None:
        """Apply a firing's world effects, then have him speak if needed."""
        if self._delivery_active:
            self._defer(firing.trigger_id)
            return None

        self.log.fired_counts[firing.trigger_id] = (
            self.log.fired_counts.get(firing.trigger_id, 0) + 1
        )

        # Effects apply even for a silent checkpoint -- the world changes
        # whether or not he mentions it.
        for effect in firing.effects:
            result = self.session.engine.apply(StateCommand(
                parameter_id=effect.parameter, value=effect.value, source="trigger",
            ))
            if not result.accepted:
                logger.warning("trigger.effect_rejected",
                               session_id=self.session_id,
                               trigger_id=firing.trigger_id,
                               parameter_id=effect.parameter,
                               error_code=result.reason_code,
                               status="rejected")

        if firing.is_silent:
            logger.info("trigger.fired_silent",
                        session_id=self.session_id,
                        trigger_id=firing.trigger_id,
                        mission_seconds=int(now))
            return None

        async with self._lock:
            result = await self.session.run_initiated_turn(
                firing.say_intent or "", firing.trigger_id, firing.instructions,
            )
            return await self._deliver(
                marked_text=result.text,
                origin="initiated",
                trigger_id=firing.trigger_id,
                priority=firing.priority,
            )

    # -- delivery ---------------------------------------------------------

    async def _deliver(
        self,
        marked_text: str,
        origin: str,
        trigger_id: str | None,
        priority: Priority,
        thinking_ms: int = 0,
    ) -> UtteranceRecord:
        """Compile the model's marked output into a plan and execute it.

        `thinking_ms` is how long the model actually took. The realism
        lead-in ABSORBS it instead of adding to it: a 1,200ms model call
        and a 1,400ms persona delay become a 1,400ms wait, not 2,600ms.

        This is the one place where real latency is a feature rather than a
        cost. An operator does not reply instantly, so the time the model
        spends thinking is time the counterpart would plausibly have spent
        thinking anyway -- and hiding the latency inside the modelled pause
        is free realism. Only the EXCESS beyond the persona's own delay is
        perceptible as lag, which is why the measurement matters.
        """
        self._turn_counter += 1
        turn_id = f"turn-{self._turn_counter}"

        plan = compile_plan(
            marked_text,
            mission=self.mission,
            session_id=self.session_id,
            turn_id=turn_id,
            plan_id=f"plan-{self._turn_counter}",
            origin=origin,
            # Derived from the session seed plus the turn number, so a
            # session replays identically while each utterance still
            # differs from the last.
            seed=self.seed + self._turn_counter,
            state=self.session.engine.snapshot(),
            tone_name=self._current_tone(),
            # Critical initiative does not yield: a real operator shouting
            # BINGO FUEL does not stop because you started talking.
            yield_on_trainee_speech=priority is not Priority.CRITICAL,
            trigger_id=trigger_id,
            for_voice=self.for_voice,
        )

        # Absorb the model's thinking time into the lead-in. Without this
        # the trainee waits for the model AND then for the persona's pause,
        # which is what made replies feel slow even on a fast model.
        if thinking_ms > 0 and plan.lead_in_ms > 0:
            from dataclasses import replace
            # A floor, not zero. The typing indicator is emitted by the
            # LeadInStarted event, so absorbing the lead-in entirely means
            # the trainee gets no feedback at all until the first word --
            # and then sees a reply appear with no sign anyone was there.
            # A short remainder keeps the indicator visible without adding
            # perceptible delay.
            plan = replace(plan, lead_in_ms=max(_MIN_LEAD_IN_MS,
                                                plan.lead_in_ms - thinking_ms))

        # NOTE: the barge-in flag is deliberately NOT cleared here.
        #
        # It is cleared when a trainee turn STARTS (handle_trainee_message),
        # which is the only moment a pending interruption is genuinely
        # stale. Clearing it here instead discarded any interrupt that
        # arrived while the model was still thinking -- and that is exactly
        # when a trainee interrupts, because the counterpart has gone quiet.
        # Observed live: an interrupt sent 3s into a turn had no effect
        # because delivery had not begun yet.
        self._delivery_active = True
        self._active_priority = priority
        try:
            outcome = await execute(
                plan, self.channel, barge_in=self._barge_in, speed=self.delivery_speed,
            )
        finally:
            self._delivery_active = False
            self._active_priority = None

        return self._record(plan, outcome, origin, trigger_id, turn_id)

    def _record(
        self,
        plan: DeliveryPlan,
        outcome: DeliveryOutcome,
        origin: str,
        trigger_id: str | None,
        turn_id: str,
    ) -> UtteranceRecord:
        """Persist what was ACTUALLY said.

        An interrupted utterance is recorded as what the trainee heard, not
        what was intended -- showing them words never spoken would make a
        debrief a lie.
        """
        record = UtteranceRecord(
            plan_id=plan.plan_id,
            turn_id=turn_id,
            origin=origin,
            text=outcome.delivered_text,
            planned_text=plan.clean_text,
            mission_seconds=self.clock.now(),
            trigger_id=trigger_id,
            status=outcome.status,
            delivered_segments=outcome.delivered_segments,
            total_segments=outcome.total_segments,
        )
        self.log.utterances.append(record)

        # The persisted conversation must match what was heard, or the
        # model's own history diverges from the trainee's experience and it
        # will refer back to things it never actually said.
        if outcome.was_cut_short:
            self.session.replace_last_counterpart_text(outcome.delivered_text)

        logger.info("utterance.delivered",
                    session_id=self.session_id,
                    plan_id=plan.plan_id,
                    origin=origin,
                    trigger_id=trigger_id,
                    outcome=outcome.status,
                    segment_count=outcome.delivered_segments,
                    mission_seconds=int(record.mission_seconds))
        return record

    # -- internals --------------------------------------------------------

    def _context(self, now: float) -> TriggerContext:
        return TriggerContext(
            state=self.session.engine.snapshot(),
            mission_seconds=now,
            last_trainee_message_at=self.session.last_trainee_at,
            last_counterpart_message_at=self.session.last_counterpart_at,
            fired_counts=self.log.fired_counts,
            delivery_active=self._delivery_active,
            active_priority=self._active_priority,
            trainee_composing=self._trainee_composing,
        )

    def _defer(self, trigger_id: str) -> None:
        """Hold a withheld firing for retry.

        Deduplicated: a threshold condition stays true across many ticks,
        and queueing one copy per tick would produce a burst of identical
        reports the moment the blocker clears.
        """
        if any(f.trigger_id == trigger_id for f in self._deferred):
            return
        trigger = next((t for t in self.triggers if t.id == trigger_id), None)
        if trigger is None:
            return
        spec = getattr(trigger, "spec", None)
        if spec is None:
            return
        self._deferred.append(TriggerFiring(
            trigger_id=spec.id,
            priority=spec.priority,
            say_intent=spec.say_intent,
            instructions=spec.instructions,
            effects=tuple(spec.effects),
            label=spec.label,
        ))

    def _take_deferred(self, ctx: TriggerContext) -> TriggerFiring | None:
        """Pop the highest-priority deferred firing that may now proceed."""
        if not self._deferred or ctx.delivery_active:
            return None

        ready = [
            f for f in self._deferred
            if f.priority is Priority.CRITICAL or not ctx.trainee_composing
        ]
        if not ready:
            return None

        ready.sort(key=lambda f: f.priority is Priority.CRITICAL, reverse=True)
        chosen = ready[0]
        self._deferred.remove(chosen)
        return chosen

    def _current_tone(self) -> str | None:
        """The tone in force, from the mission's shift rules.

        Driven by which triggers have fired rather than by the model's
        mood, so an urgent manner is a consequence of the situation.
        """
        for shift in self.mission.tone.shifts:
            if self.log.fired_counts.get(shift.when, 0) > 0:
                return shift.to
        return self.mission.tone.baseline
