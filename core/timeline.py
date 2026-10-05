"""The authored mission timeline, and what is known at a given moment.

Pure: no LLM, no network, no I/O, no clock. Elapsed time is always passed
in, which is what lets a 40-minute exercise be tested in milliseconds.

THE CENTRAL RULE: the engine holds the whole timeline; the model only ever
sees `operator_information` from events that have already been REVEALED.
Author-facing `description`, future events, private notes and the schedule
itself never reach a prompt. Everything here is built so that leak cannot
happen by accident -- `revealed()` is the only accessor that yields
operator-visible text, and it is bounded by elapsed time.

TEMPORAL SEMANTICS, which are the reason this is a fact store rather than
a queue of lines to read out:

  POINT       an occurrence. "A vehicle stopped at 04:10." True as a past
              event forever after; never a claim about the current picture.
  PERSISTENT  a fact that holds from its start until superseded by a later
              event touching the same key, or explicitly cleared.
  INTERVAL    current only inside [start, end). Outside it the observation
              is expired and must not be described as current.
  HANDOVER    crew rotation. Blocks ordinary conversation for its duration.

A query for "what is true now" is therefore a fold over revealed events,
not a lookup of the most recent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class EventType(str, Enum):
    """How an event behaves in time. Semantics are in the module docstring."""
    POINT = "point"
    PERSISTENT = "persistent"
    INTERVAL = "interval"
    HANDOVER = "handover"


class ReportingPolicy(str, Enum):
    """Whether Glok volunteers this event.

    REQUIRED      reported proactively when it becomes known.
    ON_REQUEST    available if asked, never volunteered.
    SUBSCRIPTION  volunteered only if a dialogue agreement covers it.
    """

    REQUIRED = "required"
    ON_REQUEST = "on_request"
    SUBSCRIPTION = "subscription"


class Priority(str, Enum):
    """How urgently a report is owed. Only urgent interrupts or crosses a handover."""
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass(frozen=True)
class TimelineEvent:
    """One authored moment in the recording."""

    event_id: str
    start_time: float                      # seconds since recording start
    event_type: EventType = EventType.POINT

    # Author-facing. NEVER reaches the model: it may describe intent,
    # the solution, or what the trainee is supposed to notice.
    description: str = ""

    # What Glok may know once this is revealed. The only field that
    # reaches a prompt.
    operator_information: str = ""

    end_time: float | None = None          # interval only
    state_updates: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()             # for subscription matching
    entity_ids: tuple[str, ...] = ()       # e.g. a specific vehicle
    reporting_policy: ReportingPolicy = ReportingPolicy.ON_REQUEST
    priority: Priority = Priority.NORMAL
    instructions: str = ""                 # in-character guidance for this beat
    expires_at: float | None = None        # after this, a report is stale

    def is_revealed_at(self, elapsed: float) -> bool:
        return elapsed >= self.start_time

    def is_current_at(self, elapsed: float) -> bool:
        """Whether this describes the picture RIGHT NOW.

        A point event is never 'current' -- it happened. Treating it as
        current is how a simulator ends up claiming a vehicle is still
        stopping ten minutes after it stopped.
        """
        if not self.is_revealed_at(elapsed):
            return False
        if self.event_type is EventType.POINT:
            return False
        if self.event_type is EventType.INTERVAL:
            return self.end_time is None or elapsed < self.end_time
        return True                         # persistent / handover

    def is_stale_at(self, elapsed: float) -> bool:
        """Whether an undelivered report about this has lost its value.

        Used when a report was queued behind speech: an interval that has
        closed, or an explicit expiry that has passed, should not be
        delivered as news.
        """
        if self.expires_at is not None and elapsed >= self.expires_at:
            return True
        if (self.event_type is EventType.INTERVAL
                and self.end_time is not None and elapsed >= self.end_time):
            return True
        return False

    def matches(self, tags: Iterable[str] = (), entity_ids: Iterable[str] = ()) -> bool:
        """Whether a subscription over these tags/entities covers this event."""
        wanted_tags = {t.strip().lower() for t in tags if t}
        wanted_ids = {e.strip().lower() for e in entity_ids if e}
        if wanted_ids and {e.lower() for e in self.entity_ids} & wanted_ids:
            return True
        if wanted_tags and {t.lower() for t in self.tags} & wanted_tags:
            return True
        return False


@dataclass(frozen=True)
class RevealedFact:
    """One operator-visible fact, with enough context to phrase it."""

    event_id: str
    text: str
    at: float
    is_current: bool
    event_type: EventType
    tags: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()


class Timeline:
    """The authored timeline, queryable by elapsed time.

    Events are sorted once at construction, so every query is a scan in
    authored order -- which keeps `current_facts` deterministic when two
    events touch the same state key.
    """

    def __init__(self, events: Iterable[TimelineEvent]) -> None:
        self._events = tuple(sorted(events, key=lambda e: (e.start_time, e.event_id)))

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    @property
    def events(self) -> tuple[TimelineEvent, ...]:
        """Engine-only access to the full timeline, future included."""
        return self._events

    def event(self, event_id: str) -> TimelineEvent | None:
        return next((e for e in self._events if e.event_id == event_id), None)

    @property
    def duration(self) -> float:
        """Latest authored moment, for a default exercise length."""
        if not self._events:
            return 0.0
        return max(e.end_time or e.start_time for e in self._events)

    # -- revelation -------------------------------------------------------

    def revealed(self, elapsed: float) -> tuple[RevealedFact, ...]:
        """Operator-visible facts known by `elapsed`, oldest first.

        The ONLY accessor that yields prompt-safe text. Anything the model
        is allowed to know comes through here, which is what makes the
        no-future-leak property checkable in one place.
        """
        return tuple(
            RevealedFact(
                event_id=e.event_id,
                text=e.operator_information,
                at=e.start_time,
                is_current=e.is_current_at(elapsed),
                event_type=e.event_type,
                tags=e.tags,
                entity_ids=e.entity_ids,
            )
            for e in self._events
            if e.is_revealed_at(elapsed) and e.operator_information
        )

    def newly_revealed(self, since: float, until: float) -> tuple[TimelineEvent, ...]:
        """Events crossing their start time in (since, until].

        Half-open at the lower bound so a tick never re-reports the event
        it just handled, and closed at the upper bound so an event exactly
        on a tick boundary is not skipped.
        """
        return tuple(
            e for e in self._events if since < e.start_time <= until
        )

    def current_facts(self, elapsed: float) -> dict[str, Any]:
        """State keys as they stand now, folding revealed updates in order.

        Later events win over earlier ones for the same key -- that is what
        'persistent until superseded' means. An interval's updates apply
        only while it is current, so a closed interval's facts fall away
        rather than lingering.
        """
        facts: dict[str, Any] = {}
        for event in self._events:
            if not event.is_revealed_at(elapsed) or not event.state_updates:
                continue
            if event.event_type is EventType.INTERVAL and not event.is_current_at(elapsed):
                continue
            if event.event_type is EventType.POINT:
                # A point event may still set a fact (a vehicle's last
                # known position), which then persists as the last thing
                # observed. It just is not itself "current".
                facts.update(event.state_updates)
                continue
            facts.update(event.state_updates)
        return facts

    def past_observations(self, elapsed: float) -> tuple[RevealedFact, ...]:
        """Revealed facts that are no longer current.

        Recallable with past-tense phrasing; never presentable as the
        current picture.
        """
        return tuple(f for f in self.revealed(elapsed) if not f.is_current)

    # -- handover ---------------------------------------------------------

    def handover_at(self, elapsed: float) -> TimelineEvent | None:
        """The crew handover in progress, if any.

        Returned rather than a bool so the caller can read its duration
        and authored instructions. Availability is enforced in code from
        this, not suggested in a prompt.
        """
        for event in self._events:
            if event.event_type is not EventType.HANDOVER:
                continue
            if not event.is_revealed_at(elapsed):
                continue
            end = event.end_time
            if end is None or elapsed < end:
                return event
        return None

    def matching(
        self,
        tags: Iterable[str] = (),
        entity_ids: Iterable[str] = (),
        after: float | None = None,
    ) -> tuple[TimelineEvent, ...]:
        """Events a subscription covers.

        `after` restricts to future events, which is the default for a new
        agreement: asking to be told about vehicles means from now on, not
        a recap of what already happened.
        """
        return tuple(
            e for e in self._events
            if e.matches(tags, entity_ids)
            and (after is None or e.start_time > after)
        )


def build_timeline(rows: Iterable[Mapping[str, Any]]) -> Timeline:
    """Build a Timeline from normalized rows (see core/timeline_import.py).

    Kept separate from parsing so the importer owns file formats and this
    module owns semantics.
    """
    return Timeline(TimelineEvent(**dict(row)) for row in rows)
