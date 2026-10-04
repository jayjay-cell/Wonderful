"""Text rendering of a delivery plan: events -> SSE-shaped dicts.

One of two DeliveryChannel implementations (the other, voice, is phase 2).
Everything here is a TRANSLATION of a scheduler event into something a
browser can render -- no timing decisions, which belong to the plan.

The event names are deliberately channel-neutral in meaning ("typing" is
the text rendering of a lead-in, not a concept the plan knows about), so
the UI contract stays stable when voice is added alongside.
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from delivery.scheduler import (
    DeliveryEvent,
    LeadInStarted,
    PlanCompleted,
    PlanInterrupted,
    PlanStarted,
    SegmentEnded,
    SegmentStarted,
)

Emit = Callable[[dict[str, Any]], Awaitable[None]]


class TextChannel:
    """Renders delivery events as dicts for an SSE stream.

    Collects `delivered` as it goes so the caller can persist what was
    actually said without re-deriving it from the plan.
    """

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self.delivered: list[str] = []

    async def on_event(self, event: DeliveryEvent) -> None:
        if isinstance(event, PlanStarted):
            await self._emit({
                "type": "utterance_start",
                "plan_id": event.plan.plan_id,
                "origin": event.plan.origin,
                "trigger_id": event.plan.trigger_id,
            })

        elif isinstance(event, LeadInStarted):
            # The typing indicator. Its DURATION is the realism: a reply
            # that appears instantly is the clearest tell that there is no
            # person on the other end.
            await self._emit({
                "type": "typing_start",
                "duration_ms": event.duration_ms,
            })

        elif isinstance(event, SegmentStarted):
            segment = event.segment
            if segment.kind == "tone_shift":
                # Zero-duration: a styling hint only. The voice channel
                # will map the same event to TTS prosody.
                await self._emit({
                    "type": "tone_shift",
                    "tone": segment.tone.name,
                    "pace": segment.tone.pace,
                })
            elif segment.kind == "pause":
                # Emitted rather than silently waited out, so the UI can
                # keep the typing indicator alive during a hesitation
                # instead of appearing to freeze.
                await self._emit({
                    "type": "pause",
                    "duration_ms": segment.duration_ms,
                })
            elif segment.kind == "selfcorrect":
                await self._emit({"type": "self_correction"})
            else:
                self.delivered.append(segment.text)
                await self._emit({
                    "type": "chunk",
                    "text": segment.text,
                    "garbled": segment.garbled,
                    "kind": segment.kind,
                    "tone": segment.tone.name,
                })

        elif isinstance(event, SegmentEnded):
            pass  # text needs no per-segment end signal

        elif isinstance(event, PlanInterrupted):
            await self._emit({
                "type": "interrupted",
                "delivered_text": event.delivered_text,
                "at_segment": event.at_segment,
            })

        elif isinstance(event, PlanCompleted):
            await self._emit({
                "type": "utterance_end",
                "text": event.plan.clean_text,
            })

    @property
    def delivered_text(self) -> str:
        return " ".join(part for part in self.delivered if part).strip()


class CollectingChannel:
    """A channel that records events instead of emitting them.

    For tests and for the terminal client: lets the full event SEQUENCE be
    asserted, which is what catches ordering bugs (a chunk before its
    typing indicator, an end event after an interruption).
    """

    def __init__(self) -> None:
        self.events: list[DeliveryEvent] = []

    async def on_event(self, event: DeliveryEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[str]:
        return [type(event).__name__ for event in self.events]

    def spoken_text(self) -> str:
        parts = [
            event.segment.text for event in self.events
            if isinstance(event, SegmentStarted)
            and event.segment.kind in {"speech", "filler", "garble"}
        ]
        return " ".join(part for part in parts if part).strip()
