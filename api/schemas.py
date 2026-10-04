"""Request and response shapes for the HTTP boundary.

Kept separate from core/models.py deliberately: these describe the WIRE
format, which may need to change for a UI reason without touching the
domain model, and vice versa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    mission_file: str = Field(description="file name inside missions/")

    # MISSION-TIME multiplier. At x60, one real second is one mission
    # minute, so a bingo-fuel scenario is reachable in minutes instead of
    # an hour. It does NOT speed up his speech: that stays natural, since
    # a rushed voice is the opposite of what the realism layer is for.
    #
    # Kept under the old field name for compatibility with any saved
    # client state; `mission_speed` is the accurate alias.
    delivery_speed: float = Field(default=1.0, gt=0.0, le=200.0)

    # Fixing the seed replays a session's timing exactly, which is what
    # makes "run that again and watch what you missed" possible.
    seed: int | None = None


class SessionCreated(BaseModel):
    session_id: str
    mission_id: str
    title: str
    counterpart: str
    trainee_callsign: str
    counterpart_callsign: str
    briefing: str | None = None
    language: str = "he"


class MessageRequest(BaseModel):
    # Bounded: a transmission is a radio call, not an essay, and an
    # unbounded field is an easy way to blow the context window.
    text: str = Field(min_length=1, max_length=2000)


class ComposingRequest(BaseModel):
    composing: bool
