"""SQLite session store.

Single trainer, single process: SQLite needs no service to install, and
Redis or Postgres would be unjustified. The repository interface keeps a
later swap additive.

WHAT CHANGED AND WHY IT MATTERED. The previous version was written only
from api/main.py, so neither voice path persisted anything: a Gemini Live
session left no transcript at all, and the ElevenLabs path received an
utterance record and dropped it. `GET /review` returned an empty
transcript for every voice session, which made them undebriefable --
exactly the thing a training tool exists to support.

Persistence now hangs off the shared session layer, so all three channels
record the same way.

HONEST LIMIT ON NATIVE STREAMING VOICE. With Gemini Live the model
generates audio directly and transcribes its own speech. If the trainee
talks over it, what they actually HEARD can only be estimated -- the
transcript is of what was generated, not of what reached the ear. Rows
from that path carry `delivery = "streamed"` to mark the distinction
rather than implying a precision we do not have.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    mission_id      TEXT NOT NULL,
    mission_version INTEGER NOT NULL,
    mission_title   TEXT,
    channel         TEXT,              -- text | elevenlabs | gemini_live
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    status          TEXT NOT NULL DEFAULT 'preparing'
);

CREATE TABLE IF NOT EXISTS utterances (
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    speaker         TEXT NOT NULL,     -- trainee | operator
    text            TEXT NOT NULL,     -- what was ACTUALLY said
    planned_text    TEXT,              -- what was intended, if cut short
    origin          TEXT NOT NULL,     -- reactive | report | briefing_request
    event_id        TEXT,              -- the timeline event, for a report
    status          TEXT NOT NULL,     -- completed | interrupted | abandoned
    delivery        TEXT,              -- paced | streamed (see module docstring)
    mission_seconds REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS revealed_events (
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    event_id        TEXT NOT NULL,
    mission_seconds REAL NOT NULL,
    reported        INTEGER NOT NULL DEFAULT 0,
    disposition     TEXT,              -- delivered | dropped_stale | pending
    PRIMARY KEY (session_id, seq)
);

CREATE TABLE IF NOT EXISTS agreements (
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    commitment_id   TEXT NOT NULL,
    tags            TEXT,
    entity_ids      TEXT,
    description     TEXT,
    status          TEXT NOT NULL,     -- active | cancelled | superseded
    mission_seconds REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);
"""


@dataclass
class SessionSummary:
    """One row of the session list."""
    session_id: str
    mission_id: str
    mission_title: str | None
    channel: str | None
    started_at: str
    ended_at: str | None
    status: str
    utterance_count: int = 0


class SqliteSessionStore:
    """SQLite implementation.

    A connection per operation rather than one held open: a long-lived
    connection shared across asyncio tasks is a classic source of
    "database is locked", and the write volume here (a few rows per turn)
    makes reuse an irrelevant optimisation.
    """

    def __init__(self, path: str | Path = "./maslul.db") -> None:
        self.path = str(path)
        self._memory: sqlite3.Connection | None = None
        if self.path == ":memory:":
            # An in-memory database vanishes with its connection, so one
            # must be held for the store's lifetime.
            self._memory = sqlite3.connect(self.path, check_same_thread=False)
            self._memory.row_factory = sqlite3.Row
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Add columns a pre-existing database is missing.

        CREATE TABLE IF NOT EXISTS silently leaves an older table alone,
        so a database from a previous schema stays stale and every insert
        fails with "no such column" -- which is what happened on the first
        run after this refactor. Checked per column rather than per
        version, because the alternative is a version table that also has
        to be migrated into existence.
        """
        expected = {
            "sessions": {"channel": "TEXT"},
            "utterances": {"delivery": "TEXT", "planned_text": "TEXT",
                           "event_id": "TEXT"},
            "revealed_events": {"disposition": "TEXT"},
        }
        for table, columns in expected.items():
            try:
                existing = {
                    row[1] for row in
                    connection.execute(f"PRAGMA table_info({table})")
                }
            except sqlite3.Error:
                continue
            if not existing:
                continue
            for column, kind in columns.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {kind}"
                    )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection per operation; an in-memory store holds one for its lifetime."""
        if self._memory is not None:
            yield self._memory
            self._memory.commit()
            return
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _next_seq(self, connection: sqlite3.Connection, table: str,
                  session_id: str) -> int:
        row = connection.execute(
            f"SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM {table} "
            f"WHERE session_id = ?", (session_id,),
        ).fetchone()
        return int(row["n"])

    # -- writes -----------------------------------------------------------

    def create_session(self, session_id: str, mission_id: str,
                       mission_version: int, started_at: str,
                       mission_title: str | None = None,
                       channel: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, mission_id, mission_version, mission_title,
                    channel, started_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'preparing')""",
                (session_id, mission_id, mission_version, mission_title,
                 channel, started_at),
            )

    def set_status(self, session_id: str, status: str,
                   ended_at: str | None = None) -> None:
        with self._connect() as connection:
            if ended_at:
                connection.execute(
                    "UPDATE sessions SET status = ?, ended_at = ? "
                    "WHERE session_id = ?", (status, ended_at, session_id),
                )
            else:
                connection.execute(
                    "UPDATE sessions SET status = ? WHERE session_id = ?",
                    (status, session_id),
                )

    def set_channel(self, session_id: str, channel: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET channel = ? WHERE session_id = ?",
                (channel, session_id),
            )

    def add_utterance(self, session_id: str, speaker: str, text: str,
                      mission_seconds: float, origin: str = "reactive",
                      event_id: str | None = None, status: str = "completed",
                      planned_text: str | None = None,
                      delivery: str | None = None) -> None:
        """Record something actually said.

        Never called for undelivered model output: an abandoned turn is
        recorded with status `abandoned` and empty text, so a debrief does
        not show words the trainee never heard.
        """
        with self._connect() as connection:
            seq = self._next_seq(connection, "utterances", session_id)
            connection.execute(
                """INSERT INTO utterances
                   (session_id, seq, speaker, text, planned_text, origin,
                    event_id, status, delivery, mission_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, seq, speaker, text, planned_text, origin,
                 event_id, status, delivery, mission_seconds),
            )

    def add_revealed(self, session_id: str, event_id: str,
                     mission_seconds: float, reported: bool = False,
                     disposition: str = "pending") -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "revealed_events", session_id)
            connection.execute(
                """INSERT INTO revealed_events
                   (session_id, seq, event_id, mission_seconds, reported,
                    disposition)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, seq, event_id, mission_seconds,
                 1 if reported else 0, disposition),
            )

    def add_agreement(self, session_id: str, commitment_id: str,
                      mission_seconds: float, tags: list[str] | None = None,
                      entity_ids: list[str] | None = None,
                      description: str = "", status: str = "active") -> None:
        with self._connect() as connection:
            seq = self._next_seq(connection, "agreements", session_id)
            connection.execute(
                """INSERT INTO agreements
                   (session_id, seq, commitment_id, tags, entity_ids,
                    description, status, mission_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, seq, commitment_id,
                 ",".join(tags or []), ",".join(entity_ids or []),
                 description, status, mission_seconds),
            )

    # -- reads ------------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.*, (SELECT COUNT(*) FROM utterances u
                                WHERE u.session_id = s.session_id) AS n
                   FROM sessions s ORDER BY s.started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=r["session_id"], mission_id=r["mission_id"],
                mission_title=r["mission_title"], channel=r["channel"],
                started_at=r["started_at"], ended_at=r["ended_at"],
                status=r["status"], utterance_count=r["n"],
            )
            for r in rows
        ]

    def session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def transcript(self, session_id: str) -> list[dict[str, Any]]:
        """The conversation in mission-time order.

        Ordered here rather than in the UI, so every consumer -- review
        page, future scoring, an export -- sees the same exchange.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM utterances WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            {
                "speaker": r["speaker"], "text": r["text"],
                "planned_text": r["planned_text"], "origin": r["origin"],
                "event_id": r["event_id"], "status": r["status"],
                "delivery": r["delivery"],
                "mission_seconds": r["mission_seconds"],
            }
            for r in rows
        ]

    def revealed(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM revealed_events WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            {"event_id": r["event_id"], "mission_seconds": r["mission_seconds"],
             "reported": bool(r["reported"]), "disposition": r["disposition"]}
            for r in rows
        ]

    def agreements(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agreements WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return [
            {"commitment_id": r["commitment_id"],
             "tags": [t for t in (r["tags"] or "").split(",") if t],
             "entity_ids": [e for e in (r["entity_ids"] or "").split(",") if e],
             "description": r["description"], "status": r["status"],
             "mission_seconds": r["mission_seconds"]}
            for r in rows
        ]
