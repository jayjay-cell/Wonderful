"""Marker compilation: semantic markers -> a timed DeliveryPlan.

Pure. No LLM, no network, no I/O, no clock. Given the same markers and the
same seed this is a mathematical function, which is what makes the
simulator's timing unit-testable despite the model being non-deterministic.

THE MARKER VOCABULARY (closed set, taught in the system prompt):

    «hesitate»  searching for a word, or checking an instrument
    «correct»   a self-correction follows: wrong «correct» right
    «filler»    a filler sound, drawn from the mission's configured list
    «breath»    a natural clause boundary -- the cheapest interruption point
    «urgent»    tone shift: faster, clipped, from here on
    «calm»      tone shift: release back toward baseline
    «garble»    this span was a degraded transmission

Closed deliberately. An open vocabulary invites invention, and a model
given room to invent timing notation will drift into emitting forty tags a
turn.

WHY LOGNORMAL DELAYS: real reply delays cluster short with a long tail.
Uniform delay is the single biggest reason naive simulators read as network
lag rather than as a person thinking -- a uniform draw produces as many
2.4-second pauses as 0.7-second ones, which no human conversation does.
"""

from __future__ import annotations

import math
import random
import re
import secrets
from typing import Any, Mapping

from core.models import Mission, RealismConfig, ToneProfile
from delivery.plan import DeliveryPlan, DeliverySegment, SegmentKind, ToneState

# The closed marker set. Written with guillemets because they are
# vanishingly rare in Hebrew and English prose, so a marker is never
# confused with content the model meant to say.
MARKER_PATTERN = re.compile(r"«(hesitate|correct|filler|breath|urgent|calm|garble)»")

ALL_MARKERS = ("hesitate", "correct", "filler", "breath", "urgent", "calm", "garble")

# How fast a span of text is "delivered".
#
# THIS WAS A DESIGN ERROR, now fixed: the original value paced TEXT at
# SPEAKING rate (~14 chars/sec), so a 59-character transmission took 4.2
# seconds to appear. In voice that is correct -- the sentence genuinely
# takes that long to say. In text the trainee reads it at a glance, so the
# pacing was pure waiting on top of the model's own latency, and it is what
# made replies feel slow even on a fast model.
#
# So the rate is now per channel:
#
#   TEXT  — fast enough to feel like a chat message landing, slow enough
#           that chunk boundaries and pauses are still perceptible.
#   VOICE — real speaking rate, replaced at runtime by the TTS engine's
#           actual measured duration (see delivery/plan.py on re-anchoring).
#
# Missions can override via realism.chars_per_second when a persona should
# read faster or slower than the default.
_TEXT_CHARS_PER_SECOND = 45.0
_SPEECH_CHARS_PER_SECOND = 14.0
_MIN_SPEECH_MS = 120

# Test markers: force a specific path on demand rather than waiting for a
# probability to land. Without these, garbling could only be exercised by
# retrying until a 15% chance fired -- i.e. a flaky test (NFR-4).
FORCE_GARBLE = "__FORCE_GARBLE__"
FORCE_STALL = "__FORCE_STALL__"
FORCE_NO_REALISM = "__FORCE_NO_REALISM__"
TEST_MARKERS = (FORCE_GARBLE, FORCE_STALL, FORCE_NO_REALISM)


def strip_markers(text: str) -> str:
    """Remove every marker, leaving clean prose.

    Applied before persistence and before replaying history (FR-A9). Two
    reasons: transcripts must be readable for a debrief, and markers
    accumulating in replayed history make the model imitate its own marker
    density, which drifts within a few turns.
    """
    cleaned = MARKER_PATTERN.sub(" ", text)
    for marker in TEST_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def new_seed() -> int:
    """A fresh seed for a session.

    secrets rather than random: a session seed is stored and can be used to
    replay a session, so it should not be predictable from another
    session's seed.
    """
    return secrets.randbelow(2**31)


def compile_plan(
    marked_text: str,
    *,
    mission: Mission,
    session_id: str,
    turn_id: str,
    plan_id: str,
    origin: str = "reactive",
    seed: int | None = None,
    state: Mapping[str, Any] | None = None,
    tone_name: str | None = None,
    yield_on_trainee_speech: bool = True,
    trigger_id: str | None = None,
    for_voice: bool = False,
) -> DeliveryPlan:
    """Compile marked model output into a resolved delivery plan.

    `state` is live mission state, used only for state-dependent garbling
    (FR-A6): poor comms quality produces more degradation, and the model
    never decides that.

    `for_voice` selects the real speaking rate instead of the text rate --
    the one place the plan is channel-aware, because the DURATION of a
    spoken sentence is a physical fact while text pacing is a UX choice.
    The voice channel still re-anchors against the TTS engine's actual
    measured duration (see delivery/plan.py).
    """
    realism = mission.realism
    rate = realism.chars_per_second or (
        _SPEECH_CHARS_PER_SECOND if for_voice else _TEXT_CHARS_PER_SECOND
    )
    effective_seed = seed if seed is not None else (realism.seed or new_seed())
    rng = random.Random(effective_seed)

    tone = _resolve_tone(mission, tone_name)

    # A test marker disables realism entirely, so a test can assert on
    # content without timing noise.
    if FORCE_NO_REALISM in marked_text:
        clean = strip_markers(marked_text)
        return DeliveryPlan(
            plan_id=plan_id, session_id=session_id, turn_id=turn_id,
            origin=origin,  # type: ignore[arg-type]
            lead_in_ms=0,
            segments=(DeliverySegment(
                index=0, kind="speech", text=clean,
                start_ms=0, duration_ms=_speech_duration(clean, tone),
                interruptible=True, tone=tone,
            ),) if clean else (),
            clean_text=clean, seed=effective_seed,
            yield_on_trainee_speech=yield_on_trainee_speech,
            trigger_id=trigger_id,
        )

    garble_probability = _garble_probability(realism, state or {})
    if FORCE_GARBLE in marked_text:
        garble_probability = 1.0
    force_stall = FORCE_STALL in marked_text

    tokens = _tokenize(marked_text)
    segments = _build_segments(
        tokens, realism=realism, tone=tone, rng=rng,
        garble_probability=garble_probability, force_stall=force_stall,
        rate=rate,
    )

    return DeliveryPlan(
        plan_id=plan_id,
        session_id=session_id,
        turn_id=turn_id,
        origin=origin,  # type: ignore[arg-type]
        lead_in_ms=_lead_in_ms(realism, rng, origin, trigger_id, mission),
        segments=tuple(segments),
        clean_text=strip_markers(marked_text),
        seed=effective_seed,
        yield_on_trainee_speech=yield_on_trainee_speech,
        trigger_id=trigger_id,
    )


# -- tokenizing ------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Split into ('marker', name) and ('text', span) pairs, in order.

    Order is what makes markers meaningful: «hesitate» before a number is
    the whole point, so position must survive tokenizing.
    """
    tokens: list[tuple[str, str]] = []
    position = 0
    for match in MARKER_PATTERN.finditer(text):
        if match.start() > position:
            span = text[position:match.start()]
            if span.strip():
                tokens.append(("text", span.strip()))
        tokens.append(("marker", match.group(1)))
        position = match.end()
    tail = text[position:]
    if tail.strip():
        tokens.append(("text", tail.strip()))
    return tokens


# -- segment construction --------------------------------------------------


def _build_segments(
    tokens: list[tuple[str, str]],
    *,
    realism: RealismConfig,
    tone: ToneState,
    rng: random.Random,
    garble_probability: float,
    force_stall: bool,
    rate: float = _TEXT_CHARS_PER_SECOND,
) -> list[DeliverySegment]:
    segments: list[DeliverySegment] = []
    cursor_ms = 0
    index = 0
    current_tone = tone
    pending_correction = False

    # Pause markers that would stack with an adjacent pause. Two pauses
    # back to back read as a dropped connection rather than as someone
    # thinking -- observed as 620ms + 1490ms of unbroken dead air.
    _PAUSE_MARKERS = {"hesitate", "breath"}

    def _last_was_pause() -> bool:
        return bool(segments) and segments[-1].kind == "pause"

    def _next_is_pause(position: int) -> bool:
        nxt = tokens[position + 1] if position + 1 < len(tokens) else None
        return bool(nxt and nxt[0] == "marker" and nxt[1] in _PAUSE_MARKERS)

    for position, (kind, value) in enumerate(tokens):
        if kind == "marker":
            if value in {"urgent", "calm"}:
                current_tone = _shift_tone(current_tone, value)
                segments.append(DeliverySegment(
                    index=index, kind="tone_shift", text="",
                    start_ms=cursor_ms, duration_ms=0,
                    interruptible=True, tone=current_tone,
                ))
                index += 1
                continue

            if value == "hesitate":
                if _last_was_pause():
                    continue        # never stack two pauses
                duration = _draw_ms(realism.stall_duration_ms, rng)
                segments.append(DeliverySegment(
                    index=index, kind="pause", text="",
                    start_ms=cursor_ms, duration_ms=duration,
                    # A hesitation is a natural moment to be cut off: the
                    # trainee hears silence and fills it.
                    interruptible=True, resume_policy="abandon_plan",
                    tone=current_tone,
                ))
                cursor_ms += duration
                index += 1
                continue

            if value == "breath":
                if _last_was_pause():
                    continue        # never stack two pauses
                # Short by design: a clause boundary, not a hesitation.
                duration = max(120, int(_draw_ms(realism.stall_duration_ms, rng) * 0.35))
                segments.append(DeliverySegment(
                    index=index, kind="pause", text="",
                    start_ms=cursor_ms, duration_ms=duration,
                    interruptible=True, resume_policy="abandon_plan",
                    tone=current_tone,
                ))
                cursor_ms += duration
                index += 1
                continue

            if value == "filler":
                sound = _pick_filler(realism, rng)
                duration = _speech_duration(sound, current_tone, rate)
                segments.append(DeliverySegment(
                    index=index, kind="filler", text=sound,
                    start_ms=cursor_ms, duration_ms=duration,
                    interruptible=True, tone=current_tone,
                ))
                cursor_ms += duration
                index += 1
                continue

            if value == "correct":
                segments.append(DeliverySegment(
                    index=index, kind="selfcorrect", text="",
                    start_ms=cursor_ms, duration_ms=160,
                    interruptible=False, tone=current_tone,
                ))
                cursor_ms += 160
                index += 1
                pending_correction = True
                continue

            if value == "garble":
                # Handled on the following text span.
                pending_correction = pending_correction
                continue

        else:  # text span
            garbled = rng.random() < garble_probability
            duration = _speech_duration(value, current_tone, rate)
            segments.append(DeliverySegment(
                index=index,
                kind="garble" if garbled else "speech",
                text=value,
                start_ms=cursor_ms,
                duration_ms=duration,
                # A span carrying a number is NOT interruptible: half a
                # figure is worse than all of it or none of it, because the
                # trainee may act on the fragment.
                interruptible=not _contains_number(value),
                resume_policy="restart_segment" if _contains_number(value) else "abandon_plan",
                tone=current_tone,
                garbled=garbled,
            ))
            cursor_ms += duration
            index += 1

            # An unmarked stall: inserted probabilistically at clause ends,
            # so hesitation appears even when the model emitted no marker.
            if (not _next_is_pause(position)
                    and (force_stall or rng.random() < realism.stall_probability)
                    and value.endswith((",", "،"))):
                duration = _draw_ms(realism.stall_duration_ms, rng)
                segments.append(DeliverySegment(
                    index=index, kind="pause", text="",
                    start_ms=cursor_ms, duration_ms=duration,
                    interruptible=True, tone=current_tone,
                ))
                cursor_ms += duration
                index += 1

    return segments


# -- timing ----------------------------------------------------------------


def _lead_in_ms(
    realism: RealismConfig,
    rng: random.Random,
    origin: str,
    trigger_id: str | None,
    mission: Mission,
) -> int:
    """The delay before he starts speaking.

    Scales with priority for initiated turns: a critical report comes fast,
    an idle check-in comes slowly. His urgency is therefore audible in the
    timing before a single word is understood -- which is how it works on a
    real net.
    """
    base = _draw_ms(realism.response_delay_ms, rng)

    if origin != "initiated" or trigger_id is None:
        return base

    priority = _trigger_priority(mission, trigger_id)
    scale = {"critical": 0.3, "high": 0.6, "normal": 1.0, "low": 1.6}.get(priority, 1.0)
    return int(base * scale)


def _draw_ms(delay_range: Any, rng: random.Random) -> int:
    """Draw a duration from a configured range.

    Lognormal by default: real delays cluster short with a long tail.
    Clamped to the configured bounds so a long tail cannot produce an
    absurd pause.
    """
    low, high = delay_range.min, delay_range.max
    if high <= low:
        return int(low)

    if delay_range.distribution == "uniform":
        return int(rng.uniform(low, high))

    # Lognormal shaped so the median sits a third of the way up the range:
    # mostly prompt replies, occasionally a long think.
    median = low + (high - low) * 0.33
    sigma = 0.5
    drawn = rng.lognormvariate(math.log(max(median, 1.0)), sigma)
    return int(max(low, min(high, drawn)))


def _speech_duration(text: str, tone: ToneState,
                     chars_per_second: float = _TEXT_CHARS_PER_SECOND) -> int:
    """How long a span takes to deliver, adjusted for the tone's pace.

    `chars_per_second` is the channel's rate: fast for text, real speaking
    rate for voice. An urgent tone raises pace, so a clipped transmission
    lands faster -- which is audible before a single word is understood.
    """
    if not text:
        return 0
    seconds = len(text) / (max(chars_per_second, 1.0) * max(tone.pace, 0.1))
    return max(_MIN_SPEECH_MS, int(seconds * 1000))


# -- tone ------------------------------------------------------------------


def _resolve_tone(mission: Mission, tone_name: str | None) -> ToneState:
    name = tone_name or mission.tone.baseline
    profile = mission.tone.profiles.get(name) or ToneProfile()
    return ToneState(
        name=name,
        pace=profile.pace,
        terseness=profile.terseness,
        filler_multiplier=profile.filler_multiplier,
    )


def _shift_tone(current: ToneState, marker: str) -> ToneState:
    """Apply an inline tone shift.

    Note filler_multiplier DROPS when urgent: stressed people use fewer
    fillers, not more. Getting this backwards is a common and immediately
    noticeable mistake -- an agitated operator saying "um" more often reads
    as confusion rather than urgency.
    """
    if marker == "urgent":
        return ToneState(
            name="urgent",
            pace=min(current.pace * 1.3, 2.0),
            terseness=min(current.terseness + 0.2, 1.0),
            filler_multiplier=current.filler_multiplier * 0.4,
        )
    return ToneState(name="calm", pace=1.0, terseness=current.terseness,
                     filler_multiplier=1.0)


def _pick_filler(realism: RealismConfig, rng: random.Random) -> str:
    sounds = realism.filler_sounds or ["אה"]
    return rng.choice(sounds)


# -- state-dependent garbling ---------------------------------------------


def _garble_probability(realism: RealismConfig, state: Mapping[str, Any]) -> float:
    """Degradation probability, read from live mission state (FR-A6).

    Tied to state rather than left to the model, so comms quality actually
    governs how broken transmissions sound.
    """
    garble = realism.garble_by
    if garble is None:
        return 0.0
    current = state.get(garble.parameter)
    if current is None:
        return 0.0
    return float(garble.probabilities.get(str(current), 0.0))


def _trigger_priority(mission: Mission, trigger_id: str) -> str:
    for group in (mission.triggers.timeline, mission.triggers.thresholds,
                  mission.triggers.idle):
        for trigger in group:
            if trigger.id == trigger_id:
                return trigger.priority.value
    return "normal"


def _contains_number(text: str) -> bool:
    return any(character.isdigit() for character in text)
