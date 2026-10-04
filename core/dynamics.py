"""How a parameter's value changes as mission time advances.

Pure functions. No LLM, no network, no I/O, no clock -- elapsed time is
always passed in, never read, which is what lets a 30-minute session be
tested in milliseconds against a virtual clock (NFR-3).

Each kind is deliberately tiny and separately tested. Adding a kind is
ADDITIVE (FR-E10): a new function plus one registry entry, with nothing
existing touched. This is the only situation in which a new domain needs
code rather than just a new mission file -- everything else about a
scenario is authored in YAML.

A note on the signature: every kind takes the same arguments even when it
ignores most of them. That uniformity is what lets the engine apply
dynamics generically, without knowing which kind it is dealing with.
"""

from __future__ import annotations

from typing import Any, Callable

from core.models import Dynamics, DynamicsKind, Parameter

# A dynamics function maps current value + elapsed time -> new value.
#
#   current:  the parameter's value now
#   elapsed:  seconds of mission time since the last advance (>= 0)
#   param:    the declaration, for min/max/type
#   config:   the mission's dynamics block for this parameter
#   target:   a commanded destination, for rate_toward_target only
#   now:      absolute mission time in seconds, for scripted keyframes only
DynamicsFn = Callable[..., Any]


def _hold(current: Any, elapsed: float, param: Parameter, config: Dynamics,
          target: Any = None, now: float = 0.0) -> Any:
    """Unchanged by the passage of time. Altitude holds until commanded."""
    return current


def _stepped(current: Any, elapsed: float, param: Parameter, config: Dynamics,
             target: Any = None, now: float = 0.0) -> Any:
    """Changes only on an explicit command -- sensor mode, for instance.

    Identical behaviour to `hold`, kept as its own kind because it
    documents authorial intent: `hold` means "a continuous quantity that
    happens to be steady", `stepped` means "a discrete setting". A future
    UI can reasonably render them differently.
    """
    return current


def _linear_drain(current: Any, elapsed: float, param: Parameter, config: Dynamics,
                  target: Any = None, now: float = 0.0) -> Any:
    """Decreases at a constant rate, never below the declared floor.

    The floor is clamped rather than allowed to go negative: fuel reaching
    zero is a mission condition for a trigger to detect, not an arithmetic
    error to propagate into every derived value that divides by it.
    """
    rate_per_second = (config.rate_per_hour or 0.0) / 3600.0
    new_value = current - rate_per_second * elapsed
    floor = config.floor if config.floor is not None else param.min
    if floor is not None:
        new_value = max(new_value, floor)
    return new_value


def _linear_fill(current: Any, elapsed: float, param: Parameter, config: Dynamics,
                 target: Any = None, now: float = 0.0) -> Any:
    """Increases at a constant rate, never above the declared ceiling."""
    rate_per_second = (config.rate_per_hour or 0.0) / 3600.0
    new_value = current + rate_per_second * elapsed
    ceiling = config.ceiling if config.ceiling is not None else param.max
    if ceiling is not None:
        new_value = min(new_value, ceiling)
    return new_value


def _rate_toward_target(current: Any, elapsed: float, param: Parameter, config: Dynamics,
                        target: Any = None, now: float = 0.0) -> Any:
    """Moves toward a commanded target at a maximum rate, never past it.

    This is what makes a commanded change take TIME rather than snapping.
    Asking for 20,000 ft does not teleport the aircraft there; it climbs,
    and the counterpart can truthfully report being mid-climb. Without
    this, every commanded change would be instantaneous and the simulation
    would lose the delay that makes mission management a skill.
    """
    if target is None:
        return current
    rate_per_second = (config.rate_per_minute or 0.0) / 60.0
    max_delta = rate_per_second * elapsed
    difference = target - current
    if abs(difference) <= max_delta:
        return target
    return current + max_delta * (1 if difference > 0 else -1)


def _scripted(current: Any, elapsed: float, param: Parameter, config: Dynamics,
              target: Any = None, now: float = 0.0) -> Any:
    """Follows mission-time keyframes -- weather deteriorating on schedule.

    Returns the value of the latest keyframe at or before `now`, so the
    result depends only on absolute mission time. That makes advancing
    idempotent (FR-D4): advancing to T+600 twice gives the same answer, and
    advancing in one jump matches advancing in many small steps.
    """
    applicable = [kf for kf in config.keyframes if kf.at <= now]
    if not applicable:
        return current
    return max(applicable, key=lambda kf: kf.at).value


_REGISTRY: dict[DynamicsKind, DynamicsFn] = {
    DynamicsKind.HOLD: _hold,
    DynamicsKind.STEPPED: _stepped,
    DynamicsKind.LINEAR_DRAIN: _linear_drain,
    DynamicsKind.LINEAR_FILL: _linear_fill,
    DynamicsKind.RATE_TOWARD_TARGET: _rate_toward_target,
    DynamicsKind.SCRIPTED: _scripted,
}

# Every DynamicsKind must have an implementation. Checked at import time so
# adding a kind to the enum without writing its function fails immediately
# and unmistakably, rather than at runtime inside a mission nobody is
# watching closely.
_missing = set(DynamicsKind) - set(_REGISTRY)
if _missing:
    raise RuntimeError(
        f"dynamics kinds declared but not implemented: {sorted(k.value for k in _missing)}"
    )


def apply_dynamics(current: Any, elapsed_seconds: float, param: Parameter,
                   target: Any = None, mission_seconds: float = 0.0) -> Any:
    """Advance one parameter by `elapsed_seconds` of mission time.

    A parameter with no declared dynamics is constant. Negative elapsed
    time is rejected rather than quietly treated as zero -- time running
    backwards means a caller bug, and silently absorbing it would hide the
    monotonicity violation that MissionStateEngine.advance_to guards
    against (FR-D4).
    """
    if elapsed_seconds < 0:
        raise ValueError(
            f"apply_dynamics: elapsed_seconds must be >= 0, got {elapsed_seconds}"
        )
    if param.dynamics is None:
        return current

    fn = _REGISTRY[param.dynamics.kind]
    new_value = fn(current, elapsed_seconds, param, param.dynamics,
                   target=target, now=mission_seconds)

    # A declared min/max binds regardless of kind, so a mission author's
    # range is honoured even if a dynamics kind's own clamp is absent or
    # looser than the parameter's declaration.
    if param.min is not None and isinstance(new_value, (int, float)):
        new_value = max(new_value, param.min)
    if param.max is not None and isinstance(new_value, (int, float)):
        new_value = min(new_value, param.max)
    return new_value


def kind_requires_target(kind: DynamicsKind) -> bool:
    """Whether a command for this kind sets a TARGET to move toward rather
    than a value to take effect immediately. The engine uses this to decide
    how to interpret a StateCommand."""
    return kind is DynamicsKind.RATE_TOWARD_TARGET
