"""SQLite session store, behind a repository interface.

Single trainer, single process: Redis or Postgres would be unjustified
here, and SQLite needs no service to install -- which matters because the
deployment target is a closed network where "just run Postgres" is not a
casual request.

WHAT IS STORED AND WHY EACH TABLE EXISTS:

  sessions          the mission, its version, and the realism SEED, so a
                    session can be replayed exactly (FR-H4)
  utterances        what was ACTUALLY said, alongside what was planned, so
                    an interrupted transmission is auditable (FR-H3)
  trainee_messages  the trainee's side of the transcript
  state_snapshots   append-only state with the CAUSE of each change --
                    answers "what did he know at 03:12?" (FR-D7)
  trigger_firings   including SUPPRESSED firings and why, because "why
                    didn't he warn me about the fuel?" is undebuggable
                    when the absence of an event leaves no trace (FR-C8)

Together these make later scoring purely additive: it reads history rather
than re-simulating (FR-H5).

Transcripts are marker-free prose. Markers are stripped before they ever
reach here, so a debrief reads as a conversation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    mission_id        TEXT NOT NULL,
    mission_version   INTEGER NOT NULL,
    mission_title     TEXT,
    realism_seed      INTEGER NOT NULL,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    status            TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS utterances (
    session_id          TEXT NOT NULL,
    seq                 INTEGER NOT NULL,
    plan_id             TEXT,
    turn_id             TEXT,
    origin              TEXT NOT NULL,      -- reactive | initiated
    trigger_id          TEXT,
    text                TEXT NOT NULL,      -- what was ACTUALLY said
    planned_text        TEXT,               -- what was intended
    status              TEXT NOT NULL,      -- completed | interrupted | abandoned
    delivered_segments  INTEGER,
    total_segments      INTEGER,
    mission_seconds     REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS trainee_messages (
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    text             TEXT NOT NULL,
    mission_seconds  REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    mission_seconds  REAL NOT NULL,
    state_json       TEXT NOT NULL,
    cause            TEXT NOT NULL,         -- tick | tool:<name> | trigger:<id>
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS trigger_firings (
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    trigger_id       TEXT NOT NULL,
    mission_seconds  REAL NOT NULL,
    suppressed       INTEGER NOT NULL DEFAULT 0,
    reason           TEXT,
    PRIMARY KEY (session_id, seq)
);
"""


@dataclass
class SessionSummary:
    session_id: str
    mission_id: str
    mission_title: str | None
    started_at: str
    ended_at: str | None
    status: str
    utterance_count: int = 0


class SessionStore(Protocol):
    """The interface the API depends on.

    Exists so Postgres is additive later, and so tests can run against an
    in-memory database with no file system.
    """

    def create_session(self, **fields: Any) -> None: ...
    def end_session(self, session_id: str, status: str) -> None: ...
    def add_utterance(self, session_id: str, **fields: Any) -> None: ...
    def add_trainee_message(self, session_id: str, text: str, mission_seconds: float) -> None: ...
    def add_snapshot(self, session_id: str, mission_seconds: float,
                     state: dict, cause: str) -> None: ...
    def add_trigger_firing(self, session_id: str, trigger_id: str,
                           mission_seconds: float, suppressed: bool,
                           reason: str | None) -> None: ...


class SqliteSessionStore:
    """SQLite implementation.

    `path=":memory:"` is used by tests. Note each connection is opened per
    operation rather than held: a long-lived connection shared across
    asyncio tasks is a classic source of "database is locked", and the
    write volume here (a handful of rows per turn) makes connection reuse
    an irrelevant optimisation.
    """

    def __init__(self, path: str | Path = "./maslul.db") -> None:
        self.path = str(path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.path == ":memory:":
            # An in-memory database vanishes when its connection closes,
            # so a single connection must be held for the store's lifetime.
            self._memory_connection = sqlite3.connect(self.path)
            self._memory_connection.row_factory = sqlite3.Row
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_connection is not None:
            yield self._memory_connection
            self._memory_connection.commit()
            return
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _next_seq(self, connection: sqlite3.Connection, table: str, session_id: str) -> int:
        row = connection.execute(
            f"SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM {table} WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["next"])

    # -- writes -----------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        mission_id: str,
        mission_version: int,
        realism_seed: int,
        started_at: str,
        mission_title: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, mission_id, mission_version, mission_title,
                    realism_seed, started_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'running')""",
                (session_id, mission_id, mission_version, mission_title,
                 realism_seed, started_at),
            )

    def end_session(self, session_id: str, status: str = "completed",
                    ended_at: str | None = None) -> None:
        from datetime import datetime, timezone
        stamp = ended_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET ended_at = ?, status = ? WHERE session_id = ?",
                (stamp, status, session_id),
            )

    def add_utterance(
        self,
        session_id: str,
        text: str,
        origin: str,
        mission_seconds: float,
        status: str = "completed",
        plan_id: str | None = None,
        turn_id: str | None = None,
        trigger_id: str | None = None,
        planned_text: str | None = None,
        delivered_segments: int | None = None,
        total_segments: int | None = None,
    ) -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "utterances", session_id)
            connection.execute(
                """INSERT INTO utterances
                   (session_id, seq, plan_id, turn_id, origin, trigger_id, text,
                    planned_text, status, delivered_segments, total_segments,
                    mission_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, seq, plan_id, turn_id, origin, trigger_id, text,
                 planned_text, status, delivered_segments, total_segments,
                 mission_seconds),
            )

    def add_trainee_message(self, session_id: str, text: str,
                            mission_seconds: float) -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "trainee_messages", session_id)
            connection.execute(
                """INSERT INTO trainee_messages (session_id, seq, text, mission_seconds)
                   VALUES (?, ?, ?, ?)""",
                (session_id, seq, text, mission_seconds),
            )

    def add_snapshot(self, session_id: str, mission_seconds: float,
                     state: dict, cause: str) -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "state_snapshots", session_id)
            connection.execute(
                """INSERT INTO state_snapshots
                   (session_id, seq, mission_seconds, state_json, cause)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, seq, mission_seconds,
                 json.dumps(state, ensure_ascii=False, default=str), cause),
            )

    def add_trigger_firing(self, session_id: str, trigger_id: str,
                           mission_seconds: float, suppressed: bool = False,
                           reason: str | None = None) -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "trigger_firings", session_id)
            connection.execute(
                """INSERT INTO trigger_firings
                   (session_id, seq, trigger_id, mission_seconds, suppressed, reason)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, seq, trigger_id, mission_seconds,
                 1 if suppressed else 0, reason),
            )

    # -- reads ------------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.*, (SELECT COUNT(*) FROM utterances u
                                WHERE u.session_id = s.session_id) AS utterance_count
                   FROM sessions s
                   ORDER BY s.started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"], mission_id=row["mission_id"],
                mission_title=row["mission_title"], started_at=row["started_at"],
                ended_at=row["ended_at"], status=row["status"],
                utterance_count=row["utterance_count"],
            )
            for row in rows
        ]

    def transcript(self, session_id: str) -> list[dict[str, Any]]:
        """The interleaved conversation, in mission-time order.

        Interleaved here rather than in the UI so every consumer -- review
        page, future scoring, an export -- sees the same ordering.
        """
        with self._connect() as connection:
            trainee = connection.execute(
                """SELECT text, mission_seconds FROM trainee_messages
                   WHERE session_id = ? ORDER BY seq""",
                (session_id,),
            ).fetchall()
            counterpart = connection.execute(
                """SELECT text, planned_text, origin, trigger_id, status,
                          mission_seconds
                   FROM utterances WHERE session_id = ? ORDER BY seq""",
                (session_id,),
            ).fetchall()

        entries: list[dict[str, Any]] = []
        entries.extend({
            "speaker": "trainee", "text": row["text"],
            "mission_seconds": row["mission_seconds"],
        } for row in trainee)
        entries.extend({
            "speaker": "counterpart", "text": row["text"],
            "planned_text": row["planned_text"], "origin": row["origin"],
            "trigger_id": row["trigger_id"], "status": row["status"],
            "mission_seconds": row["mission_seconds"],
        } for row in counterpart)

        entries.sort(key=lambda entry: entry["mission_seconds"])
        return entries

    def firings(self, session_id: str) -> list[dict[str, Any]]:
        """Every firing, including suppressed ones with their reason."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT trigger_id, mission_seconds, suppressed, reason
                   FROM trigger_firings WHERE session_id = ? ORDER BY seq""",
                (session_id,),
            ).fetchall()
        return [
            {"trigger_id": row["trigger_id"],
             "mission_seconds": row["mission_seconds"],
             "suppressed": bool(row["suppressed"]), "reason": row["reason"]}
            for row in rows
        ]

    def snapshots(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT mission_seconds, state_json, cause
                   FROM state_snapshots WHERE session_id = ? ORDER BY seq""",
                (session_id,),
            ).fetchall()
        return [
            {"mission_seconds": row["mission_seconds"],
             "state": json.loads(row["state_json"]), "cause": row["cause"]}
            for row in rows
        ]

    def session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
        return dict(row) if row else None
