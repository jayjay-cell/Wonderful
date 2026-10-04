"""Session orchestration: one turn, end to end.

Owns the sequence that a turn actually is -- advance mission time, run the
agent, record what was said -- and the failure handling around it.

WHAT IS HERE NOW (step 3): reactive turns. The trainee transmits, the
counterpart answers.

WHAT COMES LATER, and why the seams exist already:
  * step 4 -- the reply becomes a DeliveryPlan (stalls, pauses, pacing)
    instead of a plain string. run_turn already returns a TurnResult rather
    than a bare string, so that change does not ripple outward.
  * step 5 -- self-initiated turns and interruption. run_turn takes an
    `origin` so an initiated turn goes through the SAME path, which is
    what keeps realism uniform between the two.

FAILURE POLICY: a turn never crashes the session. A provider outage or a
reached step limit produces a safe in-character transmission and leaves
the conversation history untouched, so the next turn can retry cleanly
(FR-G2). The real cause is logged server-side with a typed code only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.loop import build_agent, extract_reply_text, strip_system_messages
from agent.prompts import build_initiative_cue
from agent.state import (
    SessionState,
    append_trainee_message,
    initial_state,
)
from core.models import Mission
from core.procedure import check_transmission
from core.state import MissionStateEngine
from obs.logging import get_logger
from sim.clock import Clock, RealClock

logger = get_logger("sim.session")

# Spoken when a turn fails. In character deliberately: a trainee hearing
# "an error occurred" has left the simulation, whereas a garbled-comms
# transmission is a thing that happens on a real net and keeps them in it.
FALLBACK_REPLY = "מפקדה, נחשון 3, לא קיבלתי. אמור שוב."
FALLBACK_REPLY_BY_LANGUAGE = {
    "he": FALLBACK_REPLY,
    "en": "Command, say again. I did not receive that.",
}


@dataclass
class TurnResult:
    """The outcome of one turn.

    A structured result rather than a bare string, so step 4 can add a
    delivery plan without changing every caller.
    """

    text: str
    origin: Literal["reactive", "initiated"] = "reactive"
    trigger_id: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    step_count: int = 0
    latency_ms: int = 0
    failed: bool = False
    failure_code: str | None = None


class Session:
    """One training session: a mission, its state, and the conversation."""

    def __init__(
        self,
        mission: Mission,
        model: Any,
        session_id: str,
        clock: Clock | None = None,
    ) -> None:
        self.mission = mission
        self.session_id = session_id
        self.clock = clock or RealClock()
        self.engine = MissionStateEngine(mission)
        self.state: SessionState = initial_state()
        self._model = model

        # Mission-time stamps of the last transmission each way. Needed by
        # the idle trigger, and tracked here rather than derived from the
        # message list because history holds no timing.
        self.last_trainee_at: float | None = None
        self.last_counterpart_at: float | None = None

        logger.info(
            "session.started",
            session_id=session_id,
            mission_id=mission.id,
            mission_version=mission.version,
        )

    # -- turns ------------------------------------------------------------

    async def run_trainee_turn(self, transmission: str) -> TurnResult:
        """Handle a transmission from the trainee."""
        procedure_report = check_transmission(
            self.mission,
            transmission,
            is_first_transmission=not self.state["trainee_has_transmitted"],
            awaiting_readback_for=self.state["awaiting_readback_for"],
        )
        if procedure_report.violations:
            logger.info(
                "procedure.violation",
                session_id=self.session_id,
                reason=procedure_report.violations[0].kind,
            )

        append_trainee_message(self.state, transmission)
        self.last_trainee_at = self.clock.now()

        # A readback obligation is discharged by the trainee's next
        # transmission regardless of outcome -- the counterpart challenges
        # once and moves on, rather than holding a grudge for the rest of
        # the session.
        self.state["awaiting_readback_for"] = []

        return await self._run_agent(origin="reactive")

    async def run_initiated_turn(
        self,
        say_intent: str,
        trigger_id: str,
        instructions: str | None = None,
    ) -> TurnResult:
        """Have the counterpart transmit unprompted.

        The cue enters history as DATA, not as an instruction, so the
        counterpart phrases it in its own voice rather than reading the
        intent string aloud (FR-C7).

        Step 5 adds the suppression and deferral rules around WHEN this is
        called; the mechanics of the turn itself are the same as a
        reactive one, which is what keeps realism uniform.
        """
        self.state["messages"].append({
            "role": "user",
            "content": build_initiative_cue(say_intent, instructions),
        })
        return await self._run_agent(origin="initiated", trigger_id=trigger_id)

    async def _run_agent(
        self,
        origin: Literal["reactive", "initiated"],
        trigger_id: str | None = None,
    ) -> TurnResult:
        """Run one agent turn and record the result."""
        # Mission time advances BEFORE the model runs, so the counterpart
        # reasons over current state rather than state as of the last turn.
        self.engine.advance_to(self.clock.now())

        started = time.monotonic()
        # Rebuilt per turn: the system prompt embeds live state, and the
        # tools close over the engine. The stable prefix of the prompt is
        # byte-identical between turns, so caching still applies.
        agent = build_agent(self.mission, self.engine, self._model)

        try:
            result = await agent.ainvoke({"messages": self.state["messages"]})
        except Exception as err:
            return self._fallback(origin, trigger_id, err, started)

        messages = result.get("messages", [])
        text = extract_reply_text(messages)

        if not text.strip():
            # An empty reply usually means the step limit was reached
            # mid-tool-call. Treated as a recoverable failure: a silent
            # counterpart is indistinguishable from a broken simulator.
            return self._fallback(origin, trigger_id, None, started,
                                  code="EMPTY_REPLY")

        self.state["messages"] = strip_system_messages(messages)
        self.state["last_failure"] = None
        self.last_counterpart_at = self.clock.now()
        self._note_readback_obligation(text)

        latency_ms = int((time.monotonic() - started) * 1000)
        tool_names = _tool_names(messages)

        logger.info(
            "turn.completed",
            session_id=self.session_id,
            origin=origin,
            trigger_id=trigger_id,
            latency_ms=latency_ms,
            step_count=len(messages),
            status="ok",
        )

        return TurnResult(
            text=text,
            origin=origin,
            trigger_id=trigger_id,
            tool_calls=tool_names,
            step_count=len(messages),
            latency_ms=latency_ms,
        )

    def _fallback(
        self,
        origin: Literal["reactive", "initiated"],
        trigger_id: str | None,
        err: Exception | None,
        started: float,
        code: str | None = None,
    ) -> TurnResult:
        """A safe in-character reply, with history left untouched.

        Not appending the failed exchange is deliberate: the next turn
        retries against clean state rather than inheriting a half-finished
        turn the model would try to continue.
        """
        failure_code = code or (type(err).__name__ if err else "UNKNOWN")
        self.state["last_failure"] = failure_code

        logger.error(
            "turn.failed",
            session_id=self.session_id,
            origin=origin,
            trigger_id=trigger_id,
            error_code=failure_code,
            status="error",
        )

        return TurnResult(
            text=FALLBACK_REPLY_BY_LANGUAGE.get(self.mission.language, FALLBACK_REPLY),
            origin=origin,
            trigger_id=trigger_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            failed=True,
            failure_code=failure_code,
        )

    def replace_last_counterpart_text(self, delivered: str) -> None:
        """Rewrite the last reply to what was ACTUALLY delivered.

        Called when an utterance was interrupted. Without this the model's
        history holds words the trainee never heard, and it will later
        refer back to a figure it never finished saying -- which reads as
        the counterpart being confused or lying.
        """
        for index in range(len(self.state["messages"]) - 1, -1, -1):
            message = self.state["messages"][index]
            role = (message.get("role") if isinstance(message, dict)
                    else getattr(message, "type", None))
            if role in {"assistant", "ai"}:
                suffix = " —" if delivered else ""
                new_text = (delivered + suffix) if delivered else "[נקטע]"
                if isinstance(message, dict):
                    message["content"] = new_text
                else:
                    self.state["messages"][index] = {
                        "role": "assistant", "content": new_text,
                    }
                return

    def _note_readback_obligation(self, text: str) -> None:
        """Record which reported values the trainee must read back.

        Set when the counterpart has just stated a parameter the mission's
        procedure requires reading back -- so the obligation is only ever
        asserted after it was actually reported, never pre-emptively.
        """
        required: list[str] = []
        for rule in self.mission.procedure.rules:
            for parameter_id in rule.requires_readback_for:
                parameter = self.mission.parameter(parameter_id)
                if parameter is None:
                    continue
                mentioned = (parameter.display_name in text
                             or (parameter.unit and parameter.unit in text))
                if mentioned and any(c.isdigit() for c in text):
                    required.append(parameter_id)
        self.state["awaiting_readback_for"] = required


def _tool_names(messages: list[Any]) -> list[str]:
    """Tool names called during a turn, for logging and tests.

    Names only -- arguments are never logged (FR-G5).
    """
    names: list[str] = []
    for message in messages:
        calls = (message.get("tool_calls") if isinstance(message, dict)
                 else getattr(message, "tool_calls", None))
        for call in calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(name)
    return names
