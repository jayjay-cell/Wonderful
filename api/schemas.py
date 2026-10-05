"""Request and response shapes for the HTTP boundary.

Separate from core/mission.py: these describe the WIRE format, which may
need to change for a UI reason without touching the domain model.

NOTE WHAT IS ABSENT. There is no `delivery_speed`. The exercise runs
alongside a video playing in a separate player, and accelerating mission
time would desynchronise them -- so real time is the only option. The old
field travelled under four different names (element id `speed`, request
`delivery_speed`, response `mission_speed`, clock `multiplier`) for a
feature that could not correctly exist.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    """Prepare a session. Does NOT start the exercise clock."""

    mission_file: str = Field(description="file name inside missions/")

    # Which transport carries the conversation. Domain behaviour is
    # identical across both; only delivery differs.
    channel: str = Field(default="text", pattern="^(text|gemini_live)$")


class SessionCreated(BaseModel):
    """What the UI needs after preparing: callsigns, briefing, duration, phase."""
    session_id: str
    mission_id: str
    title: str
    operator_callsign: str
    trainee_callsign: str
    controller_callsign: str | None = None
    trainee_briefing: str | None = None
    language: str = "he"
    duration_seconds: float | None = None
    phase: str = "preparing"


class MessageRequest(BaseModel):
    # Bounded: a transmission is a radio call, not an essay, and an
    # unbounded field is an easy way to exhaust the context window.
    """One trainee transmission."""
    text: str = Field(min_length=1, max_length=2000)


class SharedContextRequest(BaseModel):
    """Context the trainee is passing to the crew.

    Recorded as something they SAID, not as an observation -- a trainee
    claim never overwrites what the recording shows.
    """

    fact: str = Field(min_length=1, max_length=2000)

    # True when this IS the opening briefing, which marks the crew as
    # briefed so a proactive crew stops asking for it.
    briefing: bool = False
