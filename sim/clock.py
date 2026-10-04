"""Mission clocks.

Two implementations behind one interface. The virtual one is not a testing
afterthought -- it is what makes the simulator's timing testable at all: a
45-second idle trigger and a 30-minute session both assert in milliseconds
instead of actually waiting (NFR-3).

Everything downstream takes a Clock rather than calling time.monotonic()
itself, so no module needs a special test mode.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Mission time in seconds since the session began.

    Deliberately not wall-clock time: a mission can be paused, and tests
    advance time by hand.
    """

    def now(self) -> float: ...


class RealClock:
    """Wall-clock mission time, from monotonic time so a system clock
    adjustment mid-session cannot make mission time jump or run backwards --
    MissionStateEngine.advance_to rejects a rewind, so that would otherwise
    end the session."""

    def __init__(self) -> None:
        self._started = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._started

    def reset(self) -> None:
        self._started = time.monotonic()


class ScaledClock:
    """Mission time running faster than wall-clock time.

    WHY THIS EXISTS: a bingo-fuel scenario is an hour away at real time,
    so testing one meant sitting at a terminal for most of an hour. The
    session's `delivery_speed` previously compressed only SPEECH PACING,
    which left the mission clock at 1x -- so fuel never visibly drained
    and timed checkpoints never arrived. A trainer setting "x60" saw
    nothing happen, which looked like broken triggers.

    Separate from VirtualClock: that one only moves when told, for tests.
    This one runs on its own, just faster.
    """

    def __init__(self, multiplier: float = 1.0) -> None:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        self._started = time.monotonic()
        self._multiplier = float(multiplier)

    def now(self) -> float:
        return (time.monotonic() - self._started) * self._multiplier

    @property
    def multiplier(self) -> float:
        return self._multiplier

    def reset(self) -> None:
        self._started = time.monotonic()


class VirtualClock:
    """Mission time that only moves when told to.

    Lets a test assert exact ordering ("the idle trigger fires at T+45,
    before the timeline trigger at T+180") with no sleeping and no
    flakiness.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        """Move forward. Rejects negative input rather than silently
        clamping, since time running backwards indicates a caller bug."""
        if seconds < 0:
            raise ValueError(f"VirtualClock.advance requires seconds >= 0, got {seconds}")
        self._now += seconds
        return self._now

    def set(self, seconds: float) -> float:
        if seconds < self._now:
            raise ValueError(
                f"VirtualClock.set({seconds}) would rewind from {self._now}; "
                "mission time is monotonic"
            )
        self._now = seconds
        return self._now
