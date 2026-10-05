"""Runtime state the CONVERSATION produces -- not a fourth author input.

Pure. No LLM, no I/O, no clock.

Tracks what has actually happened between Madbeka and Glok: whether
contact is established, whether a briefing took place, what Madbeka has
shared, and when each side last spoke. The authored sources say what COULD
happen; this says what DID.

TWO THINGS HERE EXIST BECAUSE A PROMPT CANNOT BE TRUSTED WITH THEM:

  * Radio addressing. The convention is addressee-then-caller on
    establishing or re-establishing contact, and NOT on every
    transmission. Whether this transmission is a re-establishment is a
    fact about elapsed silence, so code decides it and the prompt is told
    the answer.
  * Crew availability during handover. Enforced from the timeline, not
    suggested in prose, because "do not reply normally" is exactly the
    kind of instruction a model drops under pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Silence after which addressing becomes formal again. An INITIAL TUNING
# DEFAULT chosen for this prototype, not an asserted real-world rule --
# the mission file overrides it.
DEFAULT_REESTABLISH_SILENCE = 30.0


class BriefingStage(str, Enum):
    """How far the opening exchange has got.

    Deliberately coarse. A finer state machine would turn the briefing
    into a checklist the exercise is blocked behind, which the brief
    explicitly rules out.
    """

    NONE = "none"                  # nothing yet
    CONTACT = "contact"            # radio check done, no mission content
    BRIEFED = "briefed"            # Madbeka has given mission context


@dataclass
class ConversationState:
    """What the dialogue has established."""

    contact_established: bool = False
    briefing_stage: BriefingStage = BriefingStage.NONE

    last_trainee_at: float | None = None
    last_operator_at: float | None = None

    # Facts Madbeka has told Glok. Glok may use these in reasoning, with
    # attribution -- but they never overwrite timeline truth, because a
    # trainee's claim is not an observation.
    shared_by_trainee: list[str] = field(default_factory=list)

    turn_count: int = 0
    briefing_requested: bool = False      # Glok has asked for a skipped briefing

    reestablish_silence: float = DEFAULT_REESTABLISH_SILENCE

    # -- recording --------------------------------------------------------

    def trainee_spoke(self, at: float, text: str = "") -> None:
        """Record a trainee transmission; the first one establishes contact."""
        self.last_trainee_at = at
        self.turn_count += 1
        if not self.contact_established:
            self.contact_established = True
            if self.briefing_stage is BriefingStage.NONE:
                self.briefing_stage = BriefingStage.CONTACT

    def operator_spoke(self, at: float) -> None:
        """Record that Glok transmitted, which resets the silence used to decide re-addressing."""
        self.last_operator_at = at

    def note_shared(self, fact: str) -> None:
        """Record context Madbeka supplied, de-duplicated."""
        fact = fact.strip()
        if fact and fact not in self.shared_by_trainee:
            self.shared_by_trainee.append(fact)

    def mark_briefed(self) -> None:
        """Note that the opening briefing happened, so the crew stops asking for it."""
        self.briefing_stage = BriefingStage.BRIEFED

    # -- addressing -------------------------------------------------------

    def needs_formal_addressing(self, at: float) -> bool:
        """Whether this transmission should name both callsigns.

        True on first contact, and again after a meaningful break. False
        in flowing dialogue -- repeating callsigns every line is the
        single most obvious tell that nobody real is on the net.
        """
        if not self.contact_established:
            return True
        last = max(
            [t for t in (self.last_trainee_at, self.last_operator_at) if t is not None],
            default=None,
        )
        if last is None:
            return True
        return (at - last) >= self.reestablish_silence

    def silence_seconds(self, at: float) -> float:
        """How long since the trainee last transmitted.

        Measured from exercise start when they have not spoken at all, so
        a trainee who never checks in is a situation the crew can react
        to.
        """
        if self.last_trainee_at is None:
            return max(0.0, at)
        return max(0.0, at - self.last_trainee_at)

    # -- briefing -------------------------------------------------------

    def should_request_briefing(
        self,
        at: float,
        initiative: float,
        grace_seconds: float,
    ) -> bool:
        """Whether a proactive crew should ask for a skipped briefing.

        Gated on `initiative` so this varies by exercise: an active crew
        asks, a passive one waits. Asked at most once -- a crew that keeps
        requesting a briefing reads as nagging, not initiative.
        """
        if self.briefing_requested:
            return False
        if self.briefing_stage is BriefingStage.BRIEFED:
            return False
        if not self.contact_established:
            return False
        if initiative < 0.5:
            return False
        return self.silence_seconds(at) >= grace_seconds

    def mark_briefing_requested(self) -> None:
        """Note that the crew already asked for the briefing, so it asks only once."""
        self.briefing_requested = True
