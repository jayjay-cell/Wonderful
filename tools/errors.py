"""Typed tool errors (FR-G3).

A tool failure must never reach the model or the trainee as a raw
exception. Two reasons, and the second is the one that matters here:

  1. A stack trace leaks internals -- file paths, module names, library
     versions (FR-G7).
  2. The counterpart may SPEAK this message. "Traceback (most recent call
     last)" in the middle of a radio transmission destroys the simulation
     instantly, and a trainee cannot un-see it.

So every message here is written to be sayable in character: it states
what cannot be done, in the mission's own terms, with no engineering
vocabulary.

The codes are stable and machine-readable so the agent layer can log
`error_code` without logging the message text (FR-G5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorCode = Literal[
    "INVALID_ARG",             # malformed or missing argument
    "OUT_OF_RANGE",            # numerically outside the declared range
    "INVALID_VALUE",           # not one of a declared enum's values
    "UNKNOWN_PARAMETER",       # no such parameter in this mission
    "NOT_VISIBLE_TO_PERSONA",  # exists, but outside the counterpart's knowledge
    "NOT_PERMITTED",           # possible, but not allowed in current state
]


@dataclass(frozen=True)
class ToolError:
    """A failure a tool returns rather than raises.

    Returned, not raised, deliberately: the agent loop should hand the
    counterpart a fact it can respond to in character ("I can't take her
    above 25,000"), not catch an exception and fall back to a generic
    apology. A tool failure is usually a legitimate simulation event, not
    a system fault.
    """

    code: ErrorCode
    message: str

    def to_dict(self) -> dict[str, str]:
        """The shape the model sees. Deliberately minimal -- no field here
        carries anything the counterpart should not be able to say."""
        return {"error": self.code, "message": self.message}


def not_visible(parameter_id: str) -> ToolError:
    """The counterpart asked for state outside its knowledge boundary.

    The message deliberately does NOT confirm the parameter exists. "I
    don't have that" is what an operator would say; "that exists but I
    can't see it" would leak the existence of hidden mission state, which
    is exactly what the boundary is for (FR-D6).
    """
    return ToolError(
        code="NOT_VISIBLE_TO_PERSONA",
        message="That is not something I have access to from here.",
    )


def unknown_parameter(parameter_id: str, known: list[str]) -> ToolError:
    """An unknown id. Lists what IS available, since this usually means the
    model guessed a name rather than reading the state list."""
    available = ", ".join(sorted(known)) if known else "none"
    return ToolError(
        code="UNKNOWN_PARAMETER",
        message=f"I have no reading called that. What I can report: {available}.",
    )
