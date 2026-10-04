"""Request and response shapes for the HTTP boundary.

Kept separate from core/models.py deliberately: these describe the WIRE
format, which may need to change for a UI reason without touching the
domain model, and vice versa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    mission_file: str = Field(description="file name inside missions/")

    # Mission-time multiplier for delivery. Lets a trainer reach a
    # fuel-emergency scenario in minutes rather than sitting at a terminal
    # for most of an hour.
    delivery_speed: float = Field(default=1.0, gt=0.0, le=100.0)

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
