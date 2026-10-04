"""Turn detection: when has the trainee finished transmitting?

THE HARDEST UX PROBLEM IN VOICE AI, and the one that decides whether the
system feels fast. Two failure modes, pulling in opposite directions:

  WAIT TOO LONG   every reply feels sluggish even when the model is quick,
                  because the trainee has stopped talking and nothing is
                  happening.
  CUT IN TOO SOON the counterpart replies to half a sentence, which is
                  worse than slow -- it answers the wrong question.

So this is not a single timeout. Three signals combine:

  1. ENERGY VAD        is there speech in the audio right now?
  2. SILENCE DURATION  how long since speech stopped?
  3. TRANSCRIPT SHAPE  does the partial transcript look complete?

Signal 3 is what buys speed. "מפקדה, נחשון 3, דווח" is clearly unfinished
and deserves patience; "...דווח מצב דלק" is clearly complete and deserves
an immediate reply. A pure timeout treats both identically and must pick a
compromise that is wrong for one of them.

BARGE-IN IS SEPARATE AND MUST BE FAST. Detecting that the trainee has
STARTED talking has nothing to do with knowing what they said, so it fires
on energy alone, within one frame. Waiting for a transcript before
yielding would let the counterpart talk over them for a full second.

Pure, except for reading a clock: all audio arrives as arguments, so this
is testable with synthetic frames and no microphone.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum

SAMPLE_RATE = 16000


class TurnState(str, Enum):
    SILENT = "silent"            # nobody talking
    TRAINEE_SPEAKING = "trainee_speaking"
    TRAINEE_FINISHED = "trainee_finished"   # ready to reply


@dataclass
class TurnConfig:
    """Tunable thresholds. Defaults chosen for a radio net, where
    transmissions are short and deliberate rather than conversational."""

    # RMS above this counts as speech. 500 of 32767 is roughly a quiet
    # room floor; a noisy environment needs it raised.
    energy_threshold: int = 500

    # Speech must persist this long to count as a transmission, so a cough
    # or a chair creak does not trigger a turn.
    min_speech_ms: int = 180

    # Silence before the turn is considered over, when the transcript
    # looks COMPLETE. Short, because waiting is what feels slow.
    end_silence_ms: int = 420

    # Silence required when the transcript looks UNFINISHED. Longer,
    # because cutting in on a half-finished order is worse than a pause.
    end_silence_unsure_ms: int = 900

    # Hard ceiling: reply regardless once this much silence has passed.
    # Guards against transcript heuristics that never read as complete.
    max_silence_ms: int = 1600

    # Speech this long while the counterpart is talking is a barge-in.
    # Deliberately short -- talking over the trainee is the worst failure.
    barge_in_ms: int = 220


@dataclass
class TurnDetector:
    """Frame-by-frame turn detection over a PCM stream."""

    config: TurnConfig = field(default_factory=TurnConfig)

    state: TurnState = TurnState.SILENT
    _speech_ms: int = 0
    _silence_ms: int = 0
    _partial: str = ""
    _counterpart_speaking: bool = False
    _barge_in_ms: int = 0

    def set_counterpart_speaking(self, speaking: bool) -> None:
        """Tell the detector whether the counterpart currently holds the
        net. Barge-in only has meaning while he is talking."""
        self._counterpart_speaking = speaking
        if not speaking:
            self._barge_in_ms = 0

    def update_partial(self, text: str) -> None:
        """Latest interim transcript, used only to judge completeness."""
        self._partial = text or ""

    def feed(self, pcm: bytes) -> tuple[TurnState, bool]:
        """Process one audio frame.

        Returns (state, barge_in_detected). The barge-in flag is separate
        from the state because it must act immediately, before anything is
        known about what was said.
        """
        frame_ms = int(len(pcm) / (SAMPLE_RATE * 2) * 1000)
        if frame_ms <= 0:
            return self.state, False

        is_speech = rms(pcm) >= self.config.energy_threshold

        barge_in = False
        if self._counterpart_speaking:
            if is_speech:
                self._barge_in_ms += frame_ms
                if self._barge_in_ms >= self.config.barge_in_ms:
                    barge_in = True
            else:
                # Reset on any gap: a single loud frame is a noise, not an
                # interruption, and reacting to it would make the
                # counterpart stop mid-word for no reason.
                self._barge_in_ms = 0

        if is_speech:
            self._speech_ms += frame_ms
            self._silence_ms = 0
            if self._speech_ms >= self.config.min_speech_ms:
                self.state = TurnState.TRAINEE_SPEAKING
        else:
            self._silence_ms += frame_ms
            if self.state is TurnState.TRAINEE_SPEAKING:
                if self._silence_ms >= self._required_silence_ms():
                    self.state = TurnState.TRAINEE_FINISHED
            elif self.state is TurnState.SILENT:
                # Brief speech that never reached the threshold: discard,
                # so noise does not accumulate across a long silence.
                self._speech_ms = 0

        return self.state, barge_in

    def _required_silence_ms(self) -> int:
        """How much silence this particular utterance needs.

        The transcript's shape decides: a clearly finished transmission
        gets a short wait, an unfinished one gets a longer one. This is
        where the speed comes from -- a fixed timeout has to be wrong for
        one of the two cases.
        """
        if looks_complete(self._partial):
            return self.config.end_silence_ms
        # Unfinished-looking transcripts wait longer, but never past the
        # hard ceiling -- otherwise a transcript that never reads as
        # complete (or an STT that returns nothing) would hang the turn.
        return min(self.config.end_silence_unsure_ms, self.config.max_silence_ms)

    def consume_turn(self) -> None:
        """Reset after a finished turn has been handed to the agent."""
        self.state = TurnState.SILENT
        self._speech_ms = 0
        self._silence_ms = 0
        self._partial = ""
        self._barge_in_ms = 0


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of a PCM frame.

    Energy VAD rather than a neural model deliberately: it costs
    microseconds, needs no model file, and for deciding "is anyone talking
    at all" it is sufficient. A learned VAD is worth adding only if a noisy
    environment proves this inadequate -- which is an empirical question,
    not an assumption.
    """
    if len(pcm) < 2:
        return 0.0
    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    total = sum(float(s) * float(s) for s in samples)
    return (total / count) ** 0.5


# Words that end a transmission on this kind of net. A transcript ending
# in one of these is almost certainly complete, so the detector can reply
# fast instead of waiting out the full timeout.
_TERMINAL_WORDS = frozenset({
    "רות", "וילקו", "שלילי", "סוף", "עבור", "קץ", "נגטיב",
    "roger", "wilco", "negative", "out", "over", "copy",
})

# An unfinished clause: a transmission ending here is mid-sentence, and
# replying to it would answer the wrong question.
_CONTINUATION_WORDS = frozenset({
    "ו", "אבל", "אם", "כי", "של", "את", "עם", "על", "מה", "איפה",
    "and", "but", "if", "the", "a", "to", "for", "with",
})


# Verbs that take an object. A transmission ending on one is still being
# spoken -- the thing being asked for has not been said yet.
_REQUEST_VERBS = frozenset({
    "דווח", "דווחי", "תדווח", "עדכן", "תעדכן", "אשר", "תאשר",
    # NOTE: "עבור" is deliberately NOT here. It means both "over" (a
    # terminal word, already in _TERMINAL_WORDS) and "switch to" (needs an
    # object). The terminal reading wins: treating it as open would add
    # ~480ms to every properly signed-off transmission, which is the
    # common case, to save a rare early reply on "עבור למצב...".
    "תעבור", "עלה", "תעלה", "ירד", "תרד", "כוון", "תכוון",
    "report", "update", "confirm", "switch", "climb", "descend", "slew",
})


def looks_complete(partial: str) -> bool:
    """Whether a partial transcript reads as a finished transmission.

    A heuristic, and openly so: it is a speed optimisation, not a
    correctness guarantee. Being wrong costs at most the difference
    between a short and a long silence threshold, and max_silence_ms
    bounds the damage either way.
    """
    text = (partial or "").strip()
    if not text:
        return False

    # Explicit terminal punctuation.
    if text[-1] in ".?!":
        return True

    words = text.replace(",", " ").split()
    if not words:
        return False

    last = words[-1].strip(".,!?").lower()

    if last in _TERMINAL_WORDS:
        return True
    if last in _CONTINUATION_WORDS:
        return False

    # A trailing number is usually mid-readback ("climb to one eight") and
    # worth waiting on.
    if any(character.isdigit() for character in last):
        return False

    # A request verb as the LAST word means its object is still coming:
    # "...נחשון 3, דווח" is mid-sentence, while "דווח מצב דלק" is a
    # complete order because the verb is followed by what it asks for.
    # Only the trailing position matters -- an earlier check on any verb
    # in the transmission made every order read as unfinished.
    if last in _REQUEST_VERBS:
        return False

    # A request verb WITH an object after it is a complete order, however
    # short: "דווח מצב" is two words and entirely finished. Radio traffic
    # is terse, so a word-count floor alone is the wrong test.
    if any(w.strip(".,!?").lower() in _REQUEST_VERBS for w in words[:-1]):
        return True

    # Otherwise: a substantial transmission that is not obviously
    # unfinished. By this point the trainee has already fallen silent.
    return len(words) >= 3
