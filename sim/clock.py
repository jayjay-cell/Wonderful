"""Clocks.

PRODUCTION USES core/lifecycle.py's ExerciseClock, and only that. Real
time, no multiplier: the exercise runs alongside a video playing in a
separate player, and any scaling would desynchronise them. The previous
ScaledClock and AcceleratedClock are gone with the speed selector.

This module holds the TEST clock: same surface, moves only when told, so a
20-minute exercise asserts in milliseconds with exact ordering.
"""

from __future__ import annotations

from core.lifecycle import LifecycleError, Phase


class VirtualClock:
    """An ExerciseClock whose time only moves on command.

    Mirrors the real lifecycle exactly -- including time staying zero
    until start() -- so a test exercises the same transitions production
    does, just without waiting.
    """

    def __init__(self) -> None:
        self._phase = Phase.PREPARING
        self._now = 0.0
        self._frozen = 0.0
        self._duration: float | None = None

    # -- the ExerciseClock surface ---------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_running(self) -> bool:
        return self._phase is Phase.RUNNING

    @property
    def is_ended(self) -> bool:
        return self._phase is Phase.ENDED

    @property
    def duration(self) -> float | None:
        return self._duration

    def now(self) -> float:
        """Elapsed mission time: zero before start, frozen while paused."""
        if self._phase in (Phase.PREPARING, Phase.READY):
            return 0.0
        if self._phase in (Phase.PAUSED, Phase.ENDED):
            return self._frozen
        return self._now

    def is_past_duration(self) -> bool:
        return self._duration is not None and self.now() >= self._duration

    def mark_ready(self, duration: float | None = None) -> None:
        """Preparation done. Mission time is still zero."""
        if self._phase is not Phase.PREPARING:
            raise LifecycleError(f"cannot become ready from {self._phase.value}")
        self._duration = duration
        self._phase = Phase.READY

    def start(self) -> None:
        """Begin the exercise."""
        if self._phase is not Phase.READY:
            raise LifecycleError(f"cannot start from {self._phase.value}")
        self._now = 0.0
        self._phase = Phase.RUNNING

    def pause(self) -> None:
        """Freeze mission time where it stands."""
        if self._phase is not Phase.RUNNING:
            raise LifecycleError(f"cannot pause from {self._phase.value}")
        self._frozen = self._now
        self._phase = Phase.PAUSED

    def resume(self) -> None:
        """Continue from the frozen time."""
        if self._phase is not Phase.PAUSED:
            raise LifecycleError(f"cannot resume from {self._phase.value}")
        self._now = self._frozen
        self._phase = Phase.RUNNING

    def end(self) -> None:
        """Stop for good. Idempotent."""
        if self._phase is Phase.ENDED:
            return
        self._frozen = self.now()
        self._phase = Phase.ENDED

    # -- test control ----------------------------------------------------

    def set(self, seconds: float) -> None:
        """Jump mission time. Monotonic: rewinding is a test bug, and
        silently allowing it would hide an ordering mistake."""
        if seconds < self._now:
            raise ValueError(
                f"set({seconds}) would rewind from {self._now}; mission time "
                f"is monotonic"
            )
        self._now = seconds

    def advance(self, seconds: float) -> float:
        """Move mission time forward by this many seconds."""
        if seconds < 0:
            raise ValueError(f"advance requires seconds >= 0, got {seconds}")
        self._now += seconds
        return self._now
