"""Per-session conversation state.

`messages` holds the raw conversation and is replayed to the model IN FULL
every turn (NFR-6), rather than being condensed into a summary. A
summarization step is one more place the record can silently drift from
what was actually said -- and in a training simulator the transcript is
the thing being assessed, so drift is not acceptable.

Deliberately a plain TypedDict with no framework dependency beyond the
message annotation: the session layer owns persistence, so this carries no
storage concerns.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SessionState(TypedDict):
    """One training session's conversation and bookkeeping."""

    messages: Annotated[list, add_messages]

    # Set when the last turn failed (provider outage, step limit reached),
    # so the session layer can log the cause without the failure needing
    # to still be in the message window.
    last_failure: Optional[str]

    # Mission ids the trainee has been told and which the mission's
    # procedure requires them to read back. Tracked here rather than in
    # core/ because it is a fact about the CONVERSATION, not about the
    # simulated world -- core/procedure.py stays pure by taking it as an
    # argument.
    awaiting_readback_for: list[str]

    # Whether any trainee transmission has happened yet, so the
    # first-transmission callsign rule applies exactly once.
    trainee_has_transmitted: bool


def initial_state() -> SessionState:
    """A fresh session: no messages, no failure, nothing awaiting readback."""
    return SessionState(
        messages=[],
        last_failure=None,
        awaiting_readback_for=[],
        trainee_has_transmitted=False,
    )


def append_trainee_message(state: SessionState, text: str) -> None:
    """Add a trainee transmission and record that contact has now been made."""
    state["messages"].append({"role": "user", "content": text})
    state["trainee_has_transmitted"] = True


def append_counterpart_message(state: SessionState, text: str) -> None:
    """Add one of Glok's replies to the history."""
    state["messages"].append({"role": "assistant", "content": text})


def message_count(state: SessionState) -> int:
    """How many messages the history holds."""
    return len(state["messages"])


def tool_calls_in_last_turn(state: SessionState) -> list[dict[str, Any]]:
    """Tool calls made since the most recent trainee message.

    Used by the numeric-provenance check: every number the counterpart
    says must appear in a tool result from THIS turn (FR-D1). Scanning
    only back to the last trainee message is what makes "this turn"
    precise.
    """
    calls: list[dict[str, Any]] = []
    for message in reversed(state["messages"]):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in {"user", "human"}:
            break
        raw_calls = (message.get("tool_calls") if isinstance(message, dict)
                     else getattr(message, "tool_calls", None))
        if raw_calls:
            calls.extend(raw_calls)
    return calls
