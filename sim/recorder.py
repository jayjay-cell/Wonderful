"""The one place exercise history is written to the store.

WHY THIS FILE EXISTS. Persistence used to live in two places: `Session`
wrote utterances, reveals and agreements, while `api/live_voice.py` wrote
its own utterances and reveals directly -- with slightly different field
choices, so a voice transcript and a text transcript did not record the
same things. Every call site also repeated `if store is not None`.

So all writes moved here. A channel records an event; it never decides how
that event is stored. The `None` store (tests, a dry run) is handled once,
in one place.

Deduplication also lives here, because it is a storage concern: an event
revealed on one tick must not be written again on the next, and an
agreement is re-written only when its STATUS changes, so a later cancel is
not lost behind the original "active" row.
"""

from __future__ import annotations

from typing import Any

from sim.exercise import Exercise, Utterance


class ExerciseRecorder:
    """Writes one exercise's history. Owned by the Session that created it.

    `store=None` makes every method a no-op, which is what tests and a
    model-free validation run use.
    """

    def __init__(self, exercise: Exercise, store: Any = None,
                 channel: str = "text") -> None:
        """Hold what to record, where to write it, and how faithful the text is."""
        self.exercise = exercise
        self.store = store
        self.channel = channel

        # Reveals are keyed by event AND disposition, so one event can
        # legitimately progress pending -> delivered without repeating.
        self._reveals: set[str] = set()
        # commitment_id -> last status written.
        self._agreements: dict[str, str] = {}

    # -- session-level ----------------------------------------------------

    def status(self, status: str, ended_at: str | None = None) -> None:
        """Record a lifecycle transition."""
        if self.store is None:
            return
        if ended_at is None:
            self.store.set_status(self.exercise.session_id, status)
        else:
            self.store.set_status(self.exercise.session_id, status, ended_at)

    def channel_used(self) -> None:
        """Record which channel actually carried the exercise."""
        if self.store is None:
            return
        self.store.set_channel(self.exercise.session_id, self.channel)

    # -- speech -----------------------------------------------------------

    def utterance(self, utterance: Utterance) -> None:
        """Record something that was actually said.

        Never called for undelivered model output: an abandoned turn is
        stored with empty text and a status, so a debrief cannot show words
        the trainee never heard.
        """
        if self.store is None:
            return
        self.store.add_utterance(
            self.exercise.session_id,
            speaker=utterance.speaker,
            text=utterance.text,
            mission_seconds=utterance.at,
            origin=utterance.origin,
            event_id=utterance.event_id,
            status=utterance.status,
            delivery=self.delivery_kind,
        )

    @property
    def delivery_kind(self) -> str:
        """How faithfully the stored text reflects what was heard.

        'streamed' for native speech-to-speech: the model transcribes its
        own speech, so if the trainee talked over it what they actually
        heard can only be estimated. Marked rather than implying a
        precision we do not have.
        """
        return "streamed" if self.channel == "gemini_live" else "paced"

    # -- timeline and agreements ------------------------------------------

    def revealed(self, event_id: str, reported: bool = False,
                 disposition: str = "pending") -> None:
        """Record that an event became known, and what became of its report."""
        if self.store is None:
            return
        key = f"{event_id}:{disposition}"
        if key in self._reveals:
            return
        self._reveals.add(key)
        self.store.add_revealed(
            self.exercise.session_id, event_id, self.exercise.clock.now(),
            reported=reported, disposition=disposition,
        )

    def dropped_reports(self) -> None:
        """Record reports dropped as stale, so a debrief sees them.

        A report that expired before it could be spoken is a fact about the
        exercise; left only in the in-memory log it vanished with the
        process.
        """
        for _at, event_id, reason in self.exercise.log.dropped:
            self.revealed(event_id, disposition=reason)

    def agreements(self) -> None:
        """Record reporting agreements, including later status changes.

        Polled from the ledger rather than hooked into the tool, so the
        tool stays pure and every channel persists identically.
        """
        if self.store is None:
            return
        for commitment in self.exercise.ledger.commitments:
            if self._agreements.get(commitment.commitment_id) == \
                    commitment.status.value:
                continue
            self._agreements[commitment.commitment_id] = \
                commitment.status.value
            self.store.add_agreement(
                self.exercise.session_id, commitment.commitment_id,
                self.exercise.clock.now(), list(commitment.tags),
                list(commitment.entity_ids), commitment.description,
                commitment.status.value,
            )
