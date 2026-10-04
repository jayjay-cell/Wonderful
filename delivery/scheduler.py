"""Executes a DeliveryPlan against a clock, emitting channel events.

The scheduler knows nothing about text or audio. It walks a plan's
segments in time and hands each transition to a DeliveryChannel. That is
the seam that makes voice additive: phase 2 adds one file implementing
DeliveryChannel plus a VAD hook that sets the barge-in event, and nothing
in core/ or sim/ changes.

INTERRUPTION. The barge-in event is set by whichever channel detects the
trainee starting to transmit -- a sent message in text, VAD in voice. The
scheduler then stops at the next INTERRUPTIBLE boundary rather than
instantly, because cutting off mid-number leaves the trainee holding half
a figure, which is worse than hearing all of it or none of it.

The exception is critical initiative, which rides through: a real operator
shouting "BINGO FUEL" does not stop because you started talking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from delivery.plan import DeliveryOutcome, DeliveryPlan, DeliverySegment


# -- events ----------------------------------------------------------------


@dataclass(frozen=True)
class PlanStarted:
    plan: DeliveryPlan


@dataclass(frozen=True)
class LeadInStarted:
    """He has heard the transmission and is composing a reply.

    In text this is the typing indicator; in voice it is the gap before
    audio. Emitted as its own event because a counterpart who starts
    speaking the instant the trainee stops is the single most obvious
    tell that it is software.
    """

    duration_ms: int


@dataclass(frozen=True)
class SegmentStarted:
    segment: DeliverySegment


@dataclass(frozen=True)
class SegmentEnded:
    segment: DeliverySegment


@dataclass(frozen=True)
class PlanInterrupted:
    plan: DeliveryPlan
    at_segment: int
    delivered_text: str


@dataclass(frozen=True)
class PlanCompleted:
    plan: DeliveryPlan


DeliveryEvent = (
    PlanStarted | LeadInStarted | SegmentStarted | SegmentEnded
    | PlanInterrupted | PlanCompleted
)


class DeliveryChannel(Protocol):
    """Anything that can render a delivery: SSE now, audio in phase 2."""

    async def on_event(self, event: DeliveryEvent) -> None: ...


# -- execution -------------------------------------------------------------


async def execute(
    plan: DeliveryPlan,
    channel: DeliveryChannel,
    barge_in: asyncio.Event | None = None,
    speed: float = 1.0,
) -> DeliveryOutcome:
    """Deliver a plan, honouring barge-in.

    `speed` compresses real waiting for tests and for accelerated sessions:
    at speed=0 a plan executes instantly, so a 30-second utterance is
    testable in microseconds while the event ORDER stays identical. That is
    what makes timing behaviour assertable rather than merely observable.
    """
    started = time.monotonic()
    await channel.on_event(PlanStarted(plan))

    if plan.lead_in_ms:
        await channel.on_event(LeadInStarted(plan.lead_in_ms))
        interrupted = await _wait(plan.lead_in_ms, barge_in, speed)
        # Being cut off during the lead-in means he had not started
        # speaking yet, so nothing was delivered.
        if interrupted and plan.yield_on_trainee_speech:
            await channel.on_event(PlanInterrupted(plan, -1, ""))
            return DeliveryOutcome(
                plan_id=plan.plan_id, status="abandoned", delivered_text="",
                delivered_segments=0, total_segments=len(plan.segments),
                elapsed_ms=_elapsed_ms(started),
            )

    delivered_through = -1

    for segment in plan.segments:
        # Check BEFORE starting a segment: if the trainee is already
        # talking and this segment may be skipped, do not begin it.
        if (barge_in is not None and barge_in.is_set()
                and plan.yield_on_trainee_speech and segment.interruptible):
            return await _interrupt(plan, channel, delivered_through, started)

        await channel.on_event(SegmentStarted(segment))
        interrupted = await _wait(segment.duration_ms, barge_in, speed)
        delivered_through = segment.index
        await channel.on_event(SegmentEnded(segment))

        if interrupted and plan.yield_on_trainee_speech:
            if segment.interruptible:
                return await _interrupt(plan, channel, delivered_through, started)
            # A non-interruptible segment (one carrying a number) finishes,
            # then delivery stops. The trainee hears the whole figure.
            return await _interrupt(plan, channel, delivered_through, started)

    await channel.on_event(PlanCompleted(plan))
    return DeliveryOutcome(
        plan_id=plan.plan_id,
        status="completed",
        delivered_text=plan.clean_text,
        delivered_segments=len(plan.segments),
        total_segments=len(plan.segments),
        elapsed_ms=_elapsed_ms(started),
    )


async def _interrupt(
    plan: DeliveryPlan,
    channel: DeliveryChannel,
    delivered_through: int,
    started: float,
) -> DeliveryOutcome:
    """Stop delivery and report what was ACTUALLY said.

    Recording the full intended text would make the debrief a lie: the
    trainee would be shown words they never heard.
    """
    delivered = plan.text_delivered_through(delivered_through)
    await channel.on_event(PlanInterrupted(plan, delivered_through, delivered))
    return DeliveryOutcome(
        plan_id=plan.plan_id,
        status="interrupted",
        delivered_text=delivered,
        delivered_segments=delivered_through + 1,
        total_segments=len(plan.segments),
        elapsed_ms=_elapsed_ms(started),
    )


async def _wait(duration_ms: int, barge_in: asyncio.Event | None, speed: float) -> bool:
    """Wait out a duration, returning True if barge-in fired first.

    Waiting on the EVENT rather than sleeping and checking afterwards is
    what makes interruption feel immediate: the trainee's transmission
    cuts in at once instead of after the current segment's full duration.
    """
    if speed <= 0 or duration_ms <= 0:
        # Instant mode for tests: still report a pre-existing barge-in so
        # ordering assertions hold.
        return barge_in is not None and barge_in.is_set()

    seconds = (duration_ms / 1000.0) / speed
    if barge_in is None:
        await asyncio.sleep(seconds)
        return False

    try:
        await asyncio.wait_for(barge_in.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
