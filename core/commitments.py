"""Reporting commitments: what Glok has agreed to tell Madbeka about.

Pure. No LLM, no I/O, no clock.

WHY THIS IS CODE AND NOT MODEL MEMORY. If "tell me about every vehicle"
lived only in the conversation history, the model would have to remember
it across a 40-minute exercise and correctly notice each of three authored
appearances. It will not do that reliably. So the LLM interprets the
REQUEST, and code stores, matches and executes it against the timeline's
authored tags and entity ids.

The consequence that matters: given three authored vehicle appearances
where only one is proactive by default, "report every vehicle" + Glok's
agreement means all three get reported when their time arrives.

Two rules that keep agreements honest:

  * A subscription defaults to FUTURE events. Agreeing to watch for
    vehicles is not a request to recap what already happened -- that is a
    separate question, answered from revealed facts.
  * Registering requires explicit agreement. A request Glok clarified,
    pushed back on, or left hanging must not silently become a
    commitment, or the trainee is held to an agreement that was never
    made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from core.timeline import ReportingPolicy, Timeline, TimelineEvent


class CommitmentStatus(str, Enum):
    """Whether an agreement still stands. Cancelled and superseded are kept for the debrief."""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"      # narrowed or replaced by a later agreement


@dataclass
class Commitment:
    """One agreed reporting obligation.

    `from_time` is the moment of agreement, so matching naturally excludes
    events that had already passed.
    """

    commitment_id: str
    tags: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    from_time: float = 0.0
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    description: str = ""          # how Glok acknowledged it, for the debrief

    @property
    def is_active(self) -> bool:
        """True while this agreement still obliges reports."""
        return self.status is CommitmentStatus.ACTIVE

    def covers(self, event: TimelineEvent) -> bool:
        """Whether this agreement obliges a report for that event."""
        if not self.is_active:
            return False
        if event.start_time <= self.from_time:
            return False           # agreed after this event; not covered
        return event.matches(self.tags, self.entity_ids)


@dataclass
class CommitmentLedger:
    """Every commitment made this exercise, and which reports are done.

    Keeps cancelled and superseded entries rather than deleting them: a
    debrief needs to show that the trainee asked for something and then
    narrowed it.
    """

    commitments: list[Commitment] = field(default_factory=list)
    reported: set[str] = field(default_factory=set)
    _counter: int = 0

    # -- registering ------------------------------------------------------

    def register(
        self,
        timeline: Timeline,
        at: float,
        tags: Iterable[str] = (),
        entity_ids: Iterable[str] = (),
        description: str = "",
        replaces: Iterable[str] = (),
    ) -> tuple[Commitment, tuple[TimelineEvent, ...]]:
        """Record an agreement, returning it and the events it will cover.

        The event list is returned so the caller can acknowledge honestly
        ("understood, I'll report those") WITHOUT telling Glok what or
        when they are -- the caller gets a count, never the contents.
        Handing the model the matched events would leak the future.
        """
        self._counter += 1
        tag_tuple = tuple(t.strip() for t in tags if t and t.strip())
        id_tuple = tuple(e.strip() for e in entity_ids if e and e.strip())

        for commitment_id in replaces:
            self.supersede(commitment_id)

        commitment = Commitment(
            commitment_id=f"c{self._counter}",
            tags=tag_tuple,
            entity_ids=id_tuple,
            from_time=at,
            description=description,
        )
        self.commitments.append(commitment)

        covered = timeline.matching(tag_tuple, id_tuple, after=at)
        return commitment, covered

    def cancel(self, commitment_id: str) -> bool:
        """Drop one agreement by id; False if it was unknown or already inactive."""
        for commitment in self.commitments:
            if commitment.commitment_id == commitment_id and commitment.is_active:
                commitment.status = CommitmentStatus.CANCELLED
                return True
        return False

    def cancel_all(self) -> int:
        """Stop every standing agreement -- "stop updating me"."""
        count = 0
        for commitment in self.commitments:
            if commitment.is_active:
                commitment.status = CommitmentStatus.CANCELLED
                count += 1
        return count

    def supersede(self, commitment_id: str) -> bool:
        """Retire one agreement because a narrower or newer one replaced it."""
        for commitment in self.commitments:
            if commitment.commitment_id == commitment_id and commitment.is_active:
                commitment.status = CommitmentStatus.SUPERSEDED
                return True
        return False

    @property
    def active(self) -> tuple[Commitment, ...]:
        """Every agreement still in force, for the prompt and the trainer's rail."""
        return tuple(c for c in self.commitments if c.is_active)

    # -- matching ---------------------------------------------------------

    def should_report(self, event: TimelineEvent) -> bool:
        """Whether this event is owed a proactive report.

        Deduplication lives here, not at the call site: a baseline
        REQUIRED event that is ALSO covered by a subscription must produce
        one report, not two. `reported` is keyed by event id, so a second
        match is a no-op.
        """
        if event.event_id in self.reported:
            return False
        if event.reporting_policy is ReportingPolicy.REQUIRED:
            return True
        if event.reporting_policy is ReportingPolicy.SUBSCRIPTION:
            return any(c.covers(event) for c in self.active)
        # ON_REQUEST is never volunteered -- but an explicit agreement
        # covering it makes it reportable, which is the whole point of
        # "tell me about every vehicle".
        return any(c.covers(event) for c in self.active)

    def mark_reported(self, event_id: str) -> None:
        """Record a report as DELIVERED.

        Called only after delivery actually happened. Marking on queueing
        would silently drop a report that was cancelled by a pause or an
        interruption.
        """
        self.reported.add(event_id)


