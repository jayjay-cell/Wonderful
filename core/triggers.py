"""Trigger evaluation: when the counterpart speaks without being asked.

Pure predicates. No LLM, no network, no I/O, no clock -- everything time-
related arrives in TriggerContext, which is what lets a 45-second idle
trigger and a 30-minute session both be tested in milliseconds.

THE HARD PART IS NOT MAKING HIM TALK, IT IS MAKING HIM SHUT UP.

A counterpart who speaks whenever a condition fires is worse than a
passive one: he talks over the trainee, interrupts their thinking, and
buries a critical report in chatter. So this module decides not only
whether a trigger fires but whether it should be allowed to right now, and
records the suppression either way (FR-C8) -- because "why didn't he warn
me about the fuel?" is undebuggable if a suppressed firing leaves no
trace.

One interface, three built-in kinds. A fourth kind (for example "three
missed readbacks -> he gets impatient") is a new class and nothing else
changes (FR-C9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from core.derived import ExpressionError, evaluate_bool
from core.models import (
    IdleTrigger,
    Mission,
    Priority,
    ThresholdTrigger,
    TimelineTrigger,
    TriggerEffect,
)

# Ordering for "is this more important than what is already happening".
_PRIORITY_RANK = {
    Priority.LOW: 0,
    Priority.NORMAL: 1,
    Priority.HIGH: 2,
    Priority.CRITICAL: 3,
}


@dataclass(frozen=True)
class TriggerContext:
    """Everything a trigger needs to decide, passed in rather than read.

    `trainee_composing` is fed by the typing indicator in text and by VAD
    in voice -- the SAME predicate from two sources, so the suppression
    rule is written once and works for both channels.
    """

    state: Mapping[str, Any]
    mission_seconds: float
    last_trainee_message_at: float | None = None
    last_counterpart_message_at: float | None = None
    fired_counts: Mapping[str, int] = field(default_factory=dict)
    delivery_active: bool = False
    active_priority: Priority | None = None
    trainee_composing: bool = False

    def silence_seconds(self) -> float:
        """How long since the trainee last transmitted.

        Measured from the start of the session when they have not
        transmitted at all, so an idle prompt can fire for a trainee who
        never checks in -- a realistic and worth-training failure.
        """
        reference = self.last_trainee_message_at
        if reference is None:
            return self.mission_seconds
        return max(0.0, self.mission_seconds - reference)


@dataclass(frozen=True)
class TriggerFiring:
    """A trigger that wants to produce an utterance.

    `say_intent` is an INTENT, never a line of dialogue (FR-C7): it reaches
    the model as labelled data so he phrases it in his own voice. A literal
    script would be read out identically every time and immediately feel
    canned.
    """

    trigger_id: str
    priority: Priority
    say_intent: str | None
    instructions: str | None = None
    effects: tuple[TriggerEffect, ...] = ()
    label: str | None = None

    @property
    def is_silent(self) -> bool:
        """A checkpoint that only changes the world and says nothing.

        Useful for setting up a situation the trainee has to notice
        themselves.
        """
        return not self.say_intent


@dataclass(frozen=True)
class Suppression:
    """A firing that was withheld, and why (FR-C8).

    Deferred firings are NOT dropped: a bingo-fuel call withheld because he
    was mid-sentence must still happen a moment later. Dropping it would be
    the simulation lying to the trainee.
    """

    trigger_id: str
    reason: str
    deferred: bool


class Trigger(Protocol):
    """One interface for every kind."""

    id: str
    priority: Priority

    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None: ...


# -- the three built-in kinds ---------------------------------------------


class TimelineTriggerRule:
    """Fires at a mission time, optionally gated by a condition.

    `when` is what stops a timeline being a fixed script. A checkpoint that
    only makes sense if the trainee did -- or failed to do -- something must
    carry the condition, or it fires regardless and teaches the opposite of
    the intended lesson: a "target escapes" beat that fires even after the
    trainee tracked correctly is worse than no beat at all.

    `after` chains checkpoints, so a sequence stays ordered even when an
    earlier step was delayed by its own condition.
    """

    def __init__(self, spec: TimelineTrigger) -> None:
        self.spec = spec
        self.id = spec.id
        self.priority = spec.priority

    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None:
        if ctx.fired_counts.get(self.id, 0) > 0:
            return None                      # timeline checkpoints fire once
        if ctx.mission_seconds < self.spec.at_mission_seconds:
            return None
        if self.spec.after and ctx.fired_counts.get(self.spec.after, 0) == 0:
            return None                      # waiting on its predecessor
        if self.spec.when and not _condition_holds(self.spec.when, ctx.state):
            return None
        return _firing(self.spec)


class ThresholdTriggerRule:
    """Fires when a condition over mission state becomes true."""

    def __init__(self, spec: ThresholdTrigger) -> None:
        self.spec = spec
        self.id = spec.id
        self.priority = spec.priority

    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None:
        if self.spec.once and ctx.fired_counts.get(self.id, 0) > 0:
            return None
        if not _condition_holds(self.spec.when, ctx.state):
            return None
        return _firing(self.spec)


class IdleTriggerRule:
    """Fires after the trainee has been silent long enough.

    `max_fires` matters more than it looks: a check-in prompt that repeats
    indefinitely stops reading as an operator checking in and starts
    reading as nagging, which trains the trainee to ignore him.
    """

    def __init__(self, spec: IdleTrigger) -> None:
        self.spec = spec
        self.id = spec.id
        self.priority = spec.priority

    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None:
        if ctx.fired_counts.get(self.id, 0) >= self.spec.max_fires:
            return None
        if ctx.silence_seconds() < self.spec.after_silence_seconds:
            return None
        # Measured from his OWN last transmission too, so he does not
        # immediately prompt again having just spoken.
        if ctx.last_counterpart_message_at is not None:
            since_he_spoke = ctx.mission_seconds - ctx.last_counterpart_message_at
            if since_he_spoke < self.spec.after_silence_seconds:
                return None
        return _firing(self.spec)


def build_triggers(mission: Mission) -> list[Trigger]:
    """All of a mission's triggers, as one uniform list."""
    triggers: list[Trigger] = []
    triggers.extend(TimelineTriggerRule(spec) for spec in mission.triggers.timeline)
    triggers.extend(ThresholdTriggerRule(spec) for spec in mission.triggers.thresholds)
    triggers.extend(IdleTriggerRule(spec) for spec in mission.triggers.idle)
    return triggers


# -- selection and suppression -------------------------------------------


def select_firing(
    triggers: Sequence[Trigger],
    ctx: TriggerContext,
) -> tuple[TriggerFiring | None, list[Suppression]]:
    """Choose at most ONE firing, and report what was withheld.

    The four guards, in order (docs/specs/agent-initiative.md §4):

      1. Single flight -- one utterance at a time, enforced by the caller.
      2. Deferred not dropped -- a lower-priority firing during delivery is
         withheld with deferred=True so the caller can retry it.
      3. Composition window -- non-critical initiative waits while the
         trainee is composing.
      4. Highest priority wins when several fire at once, so a bingo call
         is never buried behind a weather remark.
    """
    candidates: list[TriggerFiring] = []
    suppressions: list[Suppression] = []

    for trigger in triggers:
        firing = trigger.evaluate(ctx)
        if firing is None:
            continue

        # Guard 3: he waits while the trainee is still talking. Critical
        # overrides -- see the asymmetry note in TriggerFiring.
        if ctx.trainee_composing and firing.priority is not Priority.CRITICAL:
            suppressions.append(Suppression(
                trigger_id=firing.trigger_id,
                reason="trainee_composing",
                deferred=True,
            ))
            continue

        # Guard 2: something is already being said. Equal or lower priority
        # defers; higher interrupts.
        if ctx.delivery_active:
            active_rank = _PRIORITY_RANK.get(ctx.active_priority or Priority.NORMAL, 1)
            if _PRIORITY_RANK[firing.priority] <= active_rank:
                suppressions.append(Suppression(
                    trigger_id=firing.trigger_id,
                    reason="delivery_active",
                    deferred=True,
                ))
                continue

        candidates.append(firing)

    if not candidates:
        return None, suppressions

    # Guard 4: highest priority wins; a silent world-change checkpoint
    # loses to anything that needs saying, since it can be applied anyway.
    candidates.sort(key=lambda f: (_PRIORITY_RANK[f.priority], not f.is_silent),
                    reverse=True)
    chosen = candidates[0]

    for other in candidates[1:]:
        suppressions.append(Suppression(
            trigger_id=other.trigger_id,
            reason="lower_priority_than_chosen",
            deferred=True,
        ))

    return chosen, suppressions


# -- internals -----------------------------------------------------------


def _firing(spec: Any) -> TriggerFiring:
    return TriggerFiring(
        trigger_id=spec.id,
        priority=spec.priority,
        say_intent=spec.say_intent,
        instructions=spec.instructions,
        effects=tuple(spec.effects),
        label=spec.label,
    )


def _condition_holds(expression: str, state: Mapping[str, Any]) -> bool:
    """Evaluate a trigger condition against state.

    A condition that cannot be evaluated returns False rather than raising:
    a derived value may be temporarily unavailable (division by a
    drained-to-zero quantity), and ending the session over it would be
    worse than not firing. The mission loader has already validated that
    every reference resolves, so this is a runtime arithmetic case rather
    than an authoring error.
    """
    try:
        return evaluate_bool(expression, state)
    except ExpressionError:
        return False
