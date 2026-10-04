"""The DeliveryPlan: a resolved, channel-agnostic score for one utterance.

THE CENTRAL DECISION (docs/specs/realism-engine.md):

    The model annotates INTENT. Deterministic code owns ALL timing.

Both naive alternatives fail. If the model emits `<pause 2s>` tags, pacing
is untestable, drifts, and the model cannot know wall-clock facts like how
long the trainee has been silent. If a scheduler paces generically, pauses
land at meaningless points -- but a hesitation has to come BEFORE the
number he is unsure of, not at a random word.

So the model emits semantic markers and never a duration; core/realism.py
compiles those into the millisecond-precise plan defined here.

Three properties follow, all load-bearing:

  * REPRODUCIBLE -- same markers plus same seed gives an identical plan, so
    timing is unit-testable even though the model is not.
  * CHANNEL-AGNOSTIC -- nothing here names text or audio. Phase 2 adds one
    file implementing DeliveryChannel; core/ and sim/ are untouched.
  * BOUNDED MODEL ROLE -- it never emits a number, the same discipline
    applied to mission state.

Pure dataclasses: no framework, no I/O, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SegmentKind = Literal[
    "speech",       # words to say
    "pause",        # silence: a hesitation or a breath
    "filler",       # "אה", "רגע" -- spoken, but not content
    "selfcorrect",  # a correction follows; marks the seam
    "garble",       # this span was a degraded transmission
    "tone_shift",   # manner changes from here on; zero duration
]

ResumePolicy = Literal[
    "restart_segment",  # say this segment again from the start
    "skip_segment",     # drop it and continue
    "abandon_plan",     # stop entirely
]


@dataclass(frozen=True)
class ToneState:
    """The manner in force during a segment.

    Carried per segment rather than per plan because tone shifts
    MID-utterance -- he starts calm and becomes urgent on reading a gauge.
    The text channel uses this for styling; the voice channel will map it
    to TTS prosody, which is why it travels with the segment rather than
    being applied once at the start.
    """

    name: str = "baseline"
    pace: float = 1.0            # >1 is faster
    terseness: float = 0.5
    filler_multiplier: float = 1.0


@dataclass(frozen=True)
class DeliverySegment:
    """One timed piece of an utterance."""

    index: int
    kind: SegmentKind
    text: str                    # "" for pause and tone_shift

    # Timing, fully resolved. The single source of truth for both channels.
    start_ms: int                # offset from plan start
    duration_ms: int             # text: dwell time; voice: expected TTS duration

    # Interruption semantics. A clause boundary is a natural place to be
    # cut off; mid-number is not, because a half-spoken figure is worse
    # than either the whole figure or none of it.
    interruptible: bool = True
    resume_policy: ResumePolicy = "abandon_plan"

    tone: ToneState = field(default_factory=ToneState)
    garbled: bool = False

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class DeliveryPlan:
    """A complete utterance, resolved to milliseconds.

    `clean_text` is what gets persisted: marker-free prose. Markers must
    never reach the transcript or the replayed history, or the model starts
    imitating its own marker density and drifts within a few turns.
    """

    plan_id: str
    session_id: str
    turn_id: str
    origin: Literal["reactive", "initiated"]

    lead_in_ms: int              # the "thinking" delay BEFORE segment 0
    segments: tuple[DeliverySegment, ...]
    clean_text: str
    seed: int

    # Critical initiative does not yield to trainee speech. A real operator
    # shouting "BINGO FUEL" does not stop because you started talking, and
    # that asymmetry is pedagogically correct -- the trainee should
    # experience being talked over when something urgent is happening.
    yield_on_trainee_speech: bool = True

    trigger_id: str | None = None

    @property
    def total_ms(self) -> int:
        if not self.segments:
            return self.lead_in_ms
        return self.lead_in_ms + self.segments[-1].end_ms

    @property
    def speech_segments(self) -> tuple[DeliverySegment, ...]:
        return tuple(s for s in self.segments if s.kind in {"speech", "filler", "garble"})

    def text_delivered_through(self, segment_index: int) -> str:
        """What was ACTUALLY said if delivery stopped after this segment.

        Used when an utterance is interrupted: the transcript must record
        what the trainee heard, not what was planned. Recording the full
        intended text would make the debrief a lie -- the trainee would be
        shown words that were never spoken.
        """
        parts = [
            s.text for s in self.segments
            if s.index <= segment_index and s.kind in {"speech", "filler", "garble"}
        ]
        return " ".join(p for p in parts if p).strip()


@dataclass(frozen=True)
class DeliveryOutcome:
    """What actually happened when a plan was executed.

    Planned-versus-delivered is persisted (FR-H3), so an interrupted
    utterance is auditable afterwards.
    """

    plan_id: str
    status: Literal["completed", "interrupted", "abandoned"]
    delivered_text: str
    delivered_segments: int
    total_segments: int
    elapsed_ms: int

    @property
    def was_cut_short(self) -> bool:
        return self.delivered_segments < self.total_segments
