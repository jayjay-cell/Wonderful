"""Running one turn: the agent, its tools, and what was actually said.

Sits between Exercise (which owns facts and timing) and the channels
(which own audio). Every channel drives this, so domain behaviour cannot
diverge between text, the ElevenLabs cascade and Gemini Live -- the
previous version had the Live path reimplementing turns, and it silently
lost two procedure rules and a whole tool.

A turn can fail. A provider outage, a step limit, or a pause mid-flight
all produce a safe outcome rather than an exception: the exercise
continues, and nothing undelivered is recorded as heard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from agent.prompts import (
    build_briefing_request_cue,
    build_report_cue,
    build_system_prompt,
)
from obs.logging import get_logger
from sim.exercise import Exercise, Utterance

logger = get_logger("sim.turns")

# Spoken when a turn fails. In character deliberately: a trainee hearing
# "an error occurred" has left the exercise, whereas a missed transmission
# is a thing that happens on a real net.
FALLBACK = {
    "he": "לא קיבלתי. אמור שוב.",
    "en": "Say again, I did not receive that.",
}


@dataclass
class TurnResult:
    text: str
    origin: Literal["reactive", "report", "briefing_request"] = "reactive"
    event_id: str | None = None
    tools_used: list[str] = None            # type: ignore[assignment]
    latency_ms: int = 0
    failed: bool = False
    failure: str | None = None
    abandoned: bool = False                 # paused or ended mid-flight

    def __post_init__(self) -> None:
        if self.tools_used is None:
            self.tools_used = []


class TurnRunner:
    """Runs agent turns for one exercise."""

    def __init__(self, exercise: Exercise, model: Any, for_speech: bool = False) -> None:
        self.exercise = exercise
        self.for_speech = for_speech
        self._model = model
        self._history: list[dict[str, Any]] = []

    # -- public -----------------------------------------------------------

    async def trainee_turn(self, text: str) -> TurnResult:
        """The trainee transmitted; the operator replies.

        During a handover the operator says only that handover is in
        progress. Enforced here rather than left to the prompt, because
        availability is a fact about the timeline.
        """
        now = self.exercise.clock.now()
        self.exercise.record_utterance(
            Utterance(speaker="trainee", text=text, at=now)
        )

        if not self.exercise.crew_available():
            reply = (self.exercise.mission.handover.busy_reply
                     or FALLBACK.get(self.exercise.mission.language, FALLBACK["en"]))
            return TurnResult(text=reply, origin="reactive")

        self._history.append({"role": "user", "content": text})
        return await self._run("reactive")

    async def report_turn(self, information: str, event_id: str,
                          instructions: str = "", priority: Any = None) -> TurnResult:
        """The operator reports something it has just seen.

        The cue is DATA, so the operator phrases the report itself rather
        than reading an authored line aloud.
        """
        from core.timeline import Priority

        cue = build_report_cue(
            information, instructions, priority or Priority.NORMAL
        )
        self._history.append({"role": "user", "content": cue})
        return await self._run("report", event_id=event_id)

    async def briefing_request_turn(self) -> TurnResult:
        """A proactive crew asks for a briefing the kamak skipped."""
        self.exercise.state.mark_briefing_requested()
        cue = build_briefing_request_cue(self.exercise.mission.callsigns.trainee)
        self._history.append({"role": "user", "content": cue})
        return await self._run("briefing_request")

    # -- internals --------------------------------------------------------

    async def _run(self, origin: str, event_id: str | None = None) -> TurnResult:
        """One agent invocation, with the pause/end guard around it."""
        from agent.loop import build_agent, extract_reply_text, strip_system_messages

        generation = self.exercise.generation()
        started = time.monotonic()

        # Rebuilt per turn: the prompt carries live revealed facts, and the
        # tools close over the exercise.
        agent = build_agent(
            model=self._model,
            tools=self._tools(),
            system_prompt=build_system_prompt(self.exercise, self.for_speech),
            step_limit=None,
        )

        try:
            result = await agent.ainvoke({"messages": self._history})
        except Exception as err:
            logger.error("turn.failed", session_id=self.exercise.session_id,
                         origin=origin, error_code=type(err).__name__,
                         status="error")
            return self._fallback(origin, type(err).__name__, started)

        # A late provider result must not be spoken into a paused or ended
        # exercise. Checked AFTER the await, which is the only moment it
        # can be known.
        if not self.exercise.is_current_generation(generation):
            logger.info("turn.abandoned", session_id=self.exercise.session_id,
                        origin=origin, reason="paused_or_ended", status="rejected")
            return TurnResult(text="", origin=origin, event_id=event_id,
                              abandoned=True,
                              latency_ms=int((time.monotonic() - started) * 1000))

        messages = result.get("messages", [])
        text = extract_reply_text(messages)
        if not text.strip():
            # Usually the step limit hit mid-tool-call. A silent operator
            # is indistinguishable from a broken exercise.
            return self._fallback(origin, "EMPTY_REPLY", started)

        self._history = strip_system_messages(messages)
        return TurnResult(
            text=text,
            origin=origin,                   # type: ignore[arg-type]
            event_id=event_id,
            tools_used=_tool_names(messages),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _tools(self) -> list[Any]:
        from tools.mission_tools import build_mission_tools
        return build_mission_tools(self.exercise)

    def _fallback(self, origin: str, failure: str, started: float) -> TurnResult:
        """A safe in-character reply. History is left untouched so the next
        turn retries from clean state rather than inheriting a half-turn."""
        language = self.exercise.mission.language
        return TurnResult(
            text=FALLBACK.get(language, FALLBACK["en"]),
            origin=origin,                   # type: ignore[arg-type]
            failed=True,
            failure=failure,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def _tool_names(messages: list[Any]) -> list[str]:
    """Tool names called this turn. Names only -- arguments are never
    logged, since they can carry mission content."""
    names: list[str] = []
    for message in messages:
        calls = (message.get("tool_calls") if isinstance(message, dict)
                 else getattr(message, "tool_calls", None))
        for call in calls or []:
            name = (call.get("name") if isinstance(call, dict)
                    else getattr(call, "name", None))
            if name:
                names.append(name)
    return names
