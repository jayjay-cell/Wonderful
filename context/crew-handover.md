# Crew handover

Crews generally rotate roughly every four hours. The kamak does not
rotate.

## Timing is authored, never automatic

A recording can start partway through a shift, so handover is **not**
scheduled four hours after the recording begins. The actual moment is
authored as a `handover` event in the mission timeline, with an
`end_time` for its duration.

Handover is relevant only to some exercises. Most timelines will not
contain one.

## What happens during it

Ordinary dialogue is unavailable. A representative announcement:

> אנחנו בהחלפת צוותים, שתי דקות אנחנו איתך.

If called mid-handover, the crew replies briefly, in character, that
handover is in progress (`handover.busy_reply`).

**Enforced in code**, not suggested in the prompt — availability is a fact
about the timeline, and a model under pressure drops prose instructions.

## Urgent exceptions

Explicitly authored urgent events may still come through
(`handover.allow_urgent`, with the event marked `priority: urgent`).
Nothing routine does.

## What the incoming crew inherits

Everything. One exercise keeps one timeline and one agreement ledger, so
established facts and active reporting agreements carry straight across a
rotation, and no fresh briefing is required.

This is not configurable. Settings for selective inheritance
(`inherit_facts`, `inherit_commitments`, `expects_rebrief`) were
previously documented here and declared in the mission schema, but the
engine never read them: there is no separate incoming-crew memory to
withhold anything from. They have been removed rather than left to look
like they work.

Modelling a cold handover — where the incoming crew genuinely does not
know what was established — would need a second conversation state, and
is not part of the current MVP.
