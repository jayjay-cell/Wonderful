"""Exercise lifecycle and the mission clock.

Pure apart from reading a monotonic clock.

WHY PREPARING AND RUNNING ARE SEPARATE STATES. The trainer starts an
external video player by hand at roughly the same moment as the exercise.
Connecting a voice socket and validating a timeline take time, and if the
clock started at session creation, that setup time would silently become
mission time -- so by the time the trainer pressed play on the video, the
exercise would already think it was 20 seconds in. Elapsed time therefore
stays exactly zero until an explicit Start.

    PREPARING --ready--> READY (t=0) --start--> RUNNING
                                                  | pause / resume
                                                  v
                                                PAUSED
    any state ----------------------------------> ENDED

Pausing freezes mission time rather than stopping the loop that reads it,
so a resumed exercise continues from where it was instead of jumping
forward by the length of the break.
"""

from __future__ import annotations

import time
from enum import Enum


class Phase(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class LifecycleError(RuntimeError):
    """An illegal transition. Raised rather than ignored: silently
    refusing to start an exercise would look like a hung session."""


class ExerciseClock:
    """Mission time, in seconds since the exercise started.

    Monotonic-based so a system clock adjustment mid-exercise cannot make
    mission time jump or run backwards.

    Real time only. A multiplier would desynchronise the exercise from the
    video playing beside it, which is the one thing that must stay true.
    Tests inject their own clock instead (sim/clock.py VirtualClock).
    """

    def __init__(self) -> None:
        self._phase = Phase.PREPARING
        self._started_at: float | None = None      # monotonic at Start
        self._frozen_elapsed: float = 0.0          # mission time at pause
        self._duration: float | None = None        # authored end, seconds

    # -- queries ----------------------------------------------------------

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_running(self) -> bool:
        return self._phase is Phase.RUNNING

    @property
    def is_ended(self) -> bool:
        return self._phase is Phase.ENDED

    def now(self) -> float:
        """Elapsed mission time.

        Zero while preparing or ready -- that is the property the whole
        split exists for. Frozen while paused.
        """
        if self._phase in (Phase.PREPARING, Phase.READY):
            return 0.0
        if self._phase in (Phase.PAUSED, Phase.ENDED) or self._started_at is None:
            return self._frozen_elapsed
        return self._frozen_elapsed + (time.monotonic() - self._started_at)

    @property
    def duration(self) -> float | None:
        return self._duration

    def is_past_duration(self) -> bool:
        return self._duration is not None and self.now() >= self._duration

    # -- transitions ------------------------------------------------------

    def mark_ready(self, duration: float | None = None) -> None:
        """Preparation finished. Mission time is still zero."""
        if self._phase is not Phase.PREPARING:
            raise LifecycleError(f"cannot become ready from {self._phase.value}")
        self._duration = duration
        self._phase = Phase.READY

    def start(self) -> None:
        """Begin the exercise, alongside the trainer's manual video start."""
        if self._phase is not Phase.READY:
            raise LifecycleError(f"cannot start from {self._phase.value}")
        self._started_at = time.monotonic()
        self._frozen_elapsed = 0.0
        self._phase = Phase.RUNNING

    def pause(self) -> None:
        """Freeze mission time, keeping everything established so far."""
        if self._phase is not Phase.RUNNING:
            raise LifecycleError(f"cannot pause from {self._phase.value}")
        # Capture elapsed BEFORE changing phase, or now() would already be
        # reading the frozen value and the pause would lose the interval.
        self._frozen_elapsed = self.now()
        self._started_at = None
        self._phase = Phase.PAUSED

    def resume(self) -> None:
        if self._phase is not Phase.PAUSED:
            raise LifecycleError(f"cannot resume from {self._phase.value}")
        self._started_at = time.monotonic()
        self._phase = Phase.RUNNING

    def end(self) -> None:
        """Stop for good. Idempotent, since both an explicit End and the
        duration check can reach it."""
        if self._phase is Phase.ENDED:
            return
        self._frozen_elapsed = self.now()
        self._started_at = None
        self._phase = Phase.ENDED
