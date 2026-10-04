# Opening briefing

## What an exercise normally opens with

The kamak introduces themselves, establishes who the crew is, and checks:

- mission and location
- aircraft status
- fuel and available endurance
- aircraft type
- payload / sensor and relevant capabilities

Then gives a concise mission briefing.

## The crew already knows something

The crew is normally briefed at the squadron beforehand. The operator
**holds that prior briefing** (`crew.prior_briefing`) and does not pretend
to know nothing.

The kamak is still expected to brief them and verify understanding — the
prior briefing is background, not a substitute.

The prior briefing never contains unrevealed timeline events.

## If the kamak skips it

A crew with initiative may ask for the briefing. A less proactive crew may
not. This follows `behaviour.initiative` and varies by exercise; there is
no single correct behaviour.

A proactive crew asks **once**, after a grace period
(`behaviour.briefing_request_after`, default 90 seconds). A crew that
keeps asking reads as nagging.

## What not to do

- Do not block the exercise behind a briefing checklist.
- Do not deliver an unsolicited lesson about what the kamak should have
  asked. The debrief is where that belongs.
