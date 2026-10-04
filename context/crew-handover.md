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

Configurable per exercise. Defaults:

| Setting | Default | Meaning |
|---|---|---|
| `inherit_facts` | true | established mission facts carry over |
| `inherit_commitments` | true | active reporting agreements are honoured |
| `expects_rebrief` | false | no fresh briefing is required |

Set `inherit_facts: false` for an exercise about the cost of a cold
handover — the incoming crew then starts from the authored prior briefing
alone.
