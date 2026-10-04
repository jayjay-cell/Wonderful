"""Voice rendering of a delivery plan: events -> streamed audio.

THIS IS THE FILE THE WHOLE ARCHITECTURE WAS SHAPED AROUND. The claim made
back in docs/ARCHITECTURE.md was that adding voice would mean one new
DeliveryChannel implementation, with core/, tools/, agent/ and sim/
untouched. This is that file, and that claim held.

HOW THE REALISM LAYER BECOMES AUDIBLE:

  speech segment  -> synthesized and streamed
  pause segment   -> REAL SILENCE of the planned duration
  filler          -> synthesized like speech ("אה" is a sound, not a tag)
  garble          -> routed through a degradation filter
  tone_shift      -> changes the voice settings for subsequent segments
  interruption    -> stops synthesizing; the rest is never generated

Per-segment synthesis is what makes that work. Synthesizing a whole turn
in one call would collapse the plan's pacing into whatever prosody the TTS
engine chose, and an interrupted utterance would already have been paid
for and generated in full.

DRIFT CORRECTION: a plan's duration_ms for speech is an ESTIMATE, from a
characters-per-second constant. Real TTS output differs. Rather than let
the scheduler's clock and the audio diverge over a long utterance, actual
durations are reported back via `measured_ms` so the caller can re-anchor.
Without it, a plan's pauses would land in the wrong places by the end of a
long transmission.
"""

from __future__ import annotations

import asyncio
import math
import struct
from typing import Any, Awaitable, Callable

from delivery.plan import DeliverySegment
from delivery.scheduler import (
    DeliveryEvent,
    LeadInStarted,
    PlanCompleted,
    PlanInterrupted,
    PlanStarted,
    SegmentEnded,
    SegmentStarted,
)
from obs.logging import get_logger
from providers.base import VoiceSettings

logger = get_logger("delivery.voice")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2   # PCM s16le

# Audio sink: receives raw PCM plus a frame kind, so a transport can label
# or prioritise frames (an interruption frame must jump any queue).
EmitAudio = Callable[[bytes, str], Awaitable[None]]
EmitEvent = Callable[[dict[str, Any]], Awaitable[None]]


class VoiceChannel:
    """Renders delivery events as streamed PCM audio.

    Takes a TTS provider rather than constructing one, so a local engine
    swaps in without touching this file.
    """

    def __init__(
        self,
        tts: Any,
        emit_audio: EmitAudio,
        emit_event: EmitEvent | None = None,
        voice_id: str | None = None,
    ) -> None:
        self._tts = tts
        self._emit_audio = emit_audio
        self._emit_event = emit_event
        self._voice_id = voice_id
        self._tone_pace = 1.0
        self.delivered: list[str] = []
        # Actual versus planned durations, for drift correction.
        self.measured_ms: dict[int, int] = {}

    async def on_event(self, event: DeliveryEvent) -> None:
        if isinstance(event, PlanStarted):
            self.delivered.clear()
            self.measured_ms.clear()
            self._tone_pace = 1.0
            await self._event({
                "type": "utterance_start",
                "plan_id": event.plan.plan_id,
                "origin": event.plan.origin,
                "trigger_id": event.plan.trigger_id,
            })

        elif isinstance(event, LeadInStarted):
            # No audio during the lead-in. The scheduler is already waiting
            # out the duration, so emitting silence here would double it.
            # A UI cue goes out instead, which is the audio equivalent of
            # the typing indicator.
            await self._event({"type": "speaking_soon",
                               "duration_ms": event.duration_ms})

        elif isinstance(event, SegmentStarted):
            await self._render(event.segment)

        elif isinstance(event, SegmentEnded):
            pass

        elif isinstance(event, PlanInterrupted):
            # A stop frame rather than merely ceasing to send: the client
            # must DISCARD buffered audio, or the counterpart keeps talking
            # from the buffer after being cut off -- which is exactly the
            # talk-over the design forbids.
            await self._emit_audio(b"", "interrupt")
            await self._event({"type": "interrupted",
                               "delivered_text": event.delivered_text})

        elif isinstance(event, PlanCompleted):
            await self._event({"type": "utterance_end",
                               "text": event.plan.clean_text})

    async def _render(self, segment: DeliverySegment) -> None:
        """Turn one segment into audio."""
        if segment.kind == "tone_shift":
            # Applies to SUBSEQUENT segments, so a mid-utterance shift to
            # urgent is audible rather than only visible in a transcript.
            self._tone_pace = segment.tone.pace
            await self._event({"type": "tone_shift", "tone": segment.tone.name})
            return

        if segment.kind == "pause":
            # Real silence of the planned length. Sending actual samples
            # rather than nothing keeps the client's audio clock continuous
            # -- a gap in the stream makes some players resynchronize with
            # an audible click.
            await self._emit_audio(_silence(segment.duration_ms), "silence")
            return

        if segment.kind == "selfcorrect":
            await self._emit_audio(_silence(segment.duration_ms), "silence")
            return

        if not segment.text.strip():
            return

        started = asyncio.get_running_loop().time()
        total_bytes = 0
        try:
            settings = VoiceSettings(
                voice_id=self._voice_id or "", pace=self._tone_pace,
            )
            async for chunk in self._tts.synthesize_stream(segment.text, settings):
                audio = _degrade(chunk) if segment.garbled else chunk
                total_bytes += len(audio)
                await self._emit_audio(audio, "speech")
        except Exception as err:
            # A TTS failure must not end the session: the trainee hears a
            # dropped transmission, which is a thing that happens on a real
            # net, rather than the simulator dying mid-exercise.
            logger.error("voice.segment_failed", segment_count=segment.index,
                         error_code=type(err).__name__, status="error")
            await self._emit_audio(_silence(240), "silence")
            return

        elapsed_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        # Prefer the audio's own length over wall-clock time: network
        # jitter would otherwise be recorded as speech duration.
        audio_ms = int(total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000)
        self.measured_ms[segment.index] = audio_ms or elapsed_ms
        self.delivered.append(segment.text)

    async def _event(self, payload: dict[str, Any]) -> None:
        if self._emit_event is not None:
            await self._emit_event(payload)

    @property
    def delivered_text(self) -> str:
        return " ".join(part for part in self.delivered if part).strip()

    def drift_ms(self, plan: Any) -> int:
        """How far actual audio ran from the plan's estimate.

        Exposed so a caller can re-anchor subsequent timing. Reported
        rather than corrected here, because the scheduler owns the clock
        and this channel owns only rendering.
        """
        planned = sum(s.duration_ms for s in plan.segments
                      if s.index in self.measured_ms)
        actual = sum(self.measured_ms.values())
        return actual - planned


# -- audio helpers ---------------------------------------------------------


def _silence(duration_ms: int) -> bytes:
    samples = max(0, int(SAMPLE_RATE * duration_ms / 1000))
    return b"\x00\x00" * samples


def _degrade(pcm: bytes, noise_level: float = 0.18) -> bytes:
    """Make a span sound like a degraded transmission.

    Deliberately simple: additive noise plus mild clipping. A convincing
    radio-degradation filter would need bandpass filtering and codec
    artefacts, which is a real DSP job and would pull numpy into the
    runtime. This is honest placeholder quality -- audibly rougher, enough
    to convey "poor comms", and marked as such rather than pretending to be
    a radio model.
    """
    if not pcm or len(pcm) % 2:
        return pcm

    import random

    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm)
    amplitude = int(32767 * noise_level)
    out = []
    for sample in samples:
        noisy = sample + random.randint(-amplitude, amplitude)
        # Soft clip, so loud passages distort rather than wrap around --
        # wrapping produces a loud crack, not a degraded signal.
        if noisy > 32767:
            noisy = 32767
        elif noisy < -32768:
            noisy = -32768
        out.append(noisy)
    return struct.pack(f"<{count}h", *out)


def pcm_duration_ms(pcm: bytes) -> int:
    return int(len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000)
