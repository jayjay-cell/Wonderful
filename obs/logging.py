"""Structured logging with an explicit field ALLOWLIST.

An allowlist, not a scrub-after-the-fact denylist (FR-G5). The difference
matters: with a denylist, every new field a future caller passes is logged
by default and someone has to remember to exclude it. With an allowlist, a
new field is dropped by default and someone has to deliberately permit it.
That turns "we never log message text" from a habit into a structural
property.

What must never reach a log here: trainee message text, counterpart
speech, tool arguments, mission briefing content. In a training context
these are the whole substance of a session, and a log is the easiest place
for them to leak somewhere they were not meant to go.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# The complete set of fields a log line may carry. Anything else is
# dropped silently -- deliberately silently, since raising would turn a
# logging mistake into a session-ending error.
ALLOWED_FIELDS: frozenset[str] = frozenset({
    "event",              # what happened, e.g. "turn.completed"
    "request_id",         # correlates an API request with its server-side error
    "session_id",
    "mission_id",
    "mission_version",
    "turn_id",
    "plan_id",
    "trigger_id",
    "tool_name",
    "parameter_id",       # which parameter, never its value
    "provider",           # which provider class, never a key
    "status",             # ok | error | rejected | deferred | suppressed
    "error_code",         # a typed code, never an exception message
    "reason",             # a short enum-like reason, never free text
    "latency_ms",
    "step_count",
    "segment_count",
    "mission_seconds",
    "priority",
    "origin",             # reactive | initiated
    "outcome",            # completed | interrupted | abandoned
})

_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def configure(level: str | None = None) -> None:
    """Install the root logging configuration. Call once at entry point."""
    logging.basicConfig(
        level=(level or _LEVEL),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


class StructuredLogger:
    """Emits one key=value line per event, allowlist-filtered.

    Deliberately not a JSON logger: a single-trainer local deployment reads
    logs in a terminal, and key=value stays greppable. The allowlist is the
    part that matters, not the serialization format.
    """

    def __init__(self, name: str) -> None:
        """Wrap the standard logger for this module name."""
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        """Filter fields through the allowlist and write one key=value line."""
        safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS and v is not None}
        safe.pop("event", None)
        rendered = " ".join(f"{k}={v}" for k, v in sorted(safe.items()))
        self._logger.log(level, "%s %s", event, rendered)

    def info(self, event: str, **fields: Any) -> None:
        """Log a normal event."""
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Log something recoverable but worth noticing."""
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        """Logs an event, never an exception message.

        A raw exception string can contain a file path, a SQL fragment or a
        snippet of user content. Pass `error_code` instead, and let the
        global API handler log the full traceback server-side against a
        request_id (FR-G4).
        """
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        """Logs with a traceback, for the server-side-only path.

        The traceback goes to the server log; the client sees only a
        generic message plus a request_id.
        """
        safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS and v is not None}
        rendered = " ".join(f"{k}={v}" for k, v in sorted(safe.items()))
        self._logger.exception("%s %s", event, rendered)


def get_logger(name: str) -> StructuredLogger:
    """A logger for one module, filtered through the field allowlist."""
    return StructuredLogger(name)
