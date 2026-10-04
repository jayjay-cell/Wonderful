"""Gemini Live bridge: browser audio <-> Live session <-> mission tools.

The alternative voice path to api/voice.py's cascade. Both exist so the
choice can be made on measured evidence rather than a guess -- see
providers/cloud/gemini_live.py for what each gives up.

THE CRITICAL PART IS TOOL EXECUTION. Gemini Live decides when to call a
tool, but the tool itself runs HERE, against the same MissionStateEngine
the text agent uses. That is what preserves the guarantee the whole
project rests on: every number the counterpart says comes from the engine,
and hidden state stays hidden, even though the model is now choosing its
own words and timing.

What this path does NOT use: core/realism.py. Gemini Live owns its own
prosody and pauses, so there is no DeliveryPlan, no marker placement and
no state-driven garbling. The trade is deliberate and measured (~1.3s to
first audio versus 1.5-2.5s for the cascade).

Triggers still work. A self-initiated report is injected as a text turn
into the Live session, so the counterpart speaks it in his own voice --
the same intent-not-script discipline as the text path.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.models import StateCommand
from core.procedure import check_transmission
from obs.logging import get_logger

logger = get_logger("api.live_voice")


class LiveVoiceBridge:
    """One browser socket, one Live session, one mission."""

    def __init__(self, socket: WebSocket, live: Any, mission: Any) -> None:
        self.socket = socket
        self.live = live
        self.runner = live.runner
        self.mission = mission
        self.engine = self.runner.session.engine

        self._session: Any = None
        self._stopped = asyncio.Event()
        self._speaking = False

    # -- tool execution ---------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool against the real mission engine.

        Synchronous and pure-ish: the engine is in-memory, so there is
        nothing to await, and keeping it sync means a tool can never
        introduce latency into the audio path.
        """
        try:
            if name == "read_state":
                return self._read_state(args.get("parameter_ids"))
            if name == "set_parameter":
                return self._set_parameter(args)
            if name == "check_procedure":
                report = check_transmission(
                    self.mission, str(args.get("transmission", "")),
                )
                return {
                    "compliant": report.compliant,
                    "should_challenge": report.should_challenge,
                    "violations": [v.kind for v in report.violations],
                }
        except Exception as err:
            logger.error("live.tool_failed", tool_name=name,
                         error_code=type(err).__name__, status="error")
            return {"error": "TOOL_FAILED",
                    "message": "That reading is not coming through."}
        return {"error": "UNKNOWN_TOOL", "message": "I cannot do that."}

    def _read_state(self, parameter_ids: Any) -> dict[str, Any]:
        """Persona-filtered state. The knowledge boundary applies here
        exactly as it does for the text agent -- hidden parameters are
        absent, not merely discouraged."""
        self.engine.advance_to(self.runner.clock.now())
        snapshot = self.engine.snapshot(for_persona=True)

        wanted = parameter_ids if isinstance(parameter_ids, list) else None
        readings = []
        for row in self.engine.describe_for_prompt(for_persona=True):
            if wanted and row["id"] not in wanted:
                continue
            reading = {"id": row["id"], "name": row["name"], "value": row["value"]}
            if row.get("unit"):
                reading["unit"] = row["unit"]
            # The spoken form, so he does not say "idle" on a Hebrew net.
            if row.get("say_as"):
                reading["say_as"] = row["say_as"]
            readings.append(reading)

        if wanted and not readings:
            available = ", ".join(sorted(snapshot))
            return {"error": "UNKNOWN_PARAMETER",
                    "message": f"I have no such reading. Available: {available}."}
        return {"readings": readings,
                "mission_seconds": round(self.runner.clock.now(), 1)}

    def _set_parameter(self, args: dict[str, Any]) -> dict[str, Any]:
        parameter_id = str(args.get("parameter_id", ""))
        parameter = self.mission.parameter(parameter_id)
        if parameter is None:
            return {"error": "UNKNOWN_PARAMETER",
                    "message": "There is no such setting."}
        if not parameter.visible_to_persona:
            return {"error": "NOT_PERMITTED",
                    "message": "That is not something I control."}

        # The engine coerces "18000" and "18,000" to a number: Live sends
        # tool arguments as strings, so without coercion a legal order
        # would be refused -- the same bug the text path hit.
        result = self.engine.apply(StateCommand(
            parameter_id=parameter_id, value=args.get("value"), source="tool",
        ))
        if not result.accepted:
            return {"error": result.reason_code or "NOT_PERMITTED",
                    "message": result.message or "I cannot do that."}

        target = self.engine.target_of(parameter_id)
        if target is not None:
            return {"accepted": True, "in_transit_to": target,
                    "note": "commanded; in transit, not yet reached"}
        return {"accepted": True, "value": self.engine.value(parameter_id)}

    # -- session ----------------------------------------------------------

    async def run(self, provider: Any, system_prompt: str) -> None:
        """Open the Live session and pump audio both ways."""
        from providers.cloud.gemini_live import tool_declarations_for

        session_cm = await provider.connect(
            system_prompt, tool_declarations_for(self.mission),
        )

        async with session_cm as session:
            self._session = session
            await self._send_event({
                "type": "voice_ready",
                "mode": "gemini_live",
                "input_sample_rate": 16000,
                "output_sample_rate": 24000,
                "language": self.mission.language,
            })

            tasks = [
                asyncio.create_task(self._receive_from_browser()),
                asyncio.create_task(self._receive_from_model(session)),
                asyncio.create_task(self._trigger_loop(session)),
            ]
            try:
                await self._stopped.wait()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _receive_from_browser(self) -> None:
        """Forward microphone audio into the Live session."""
        try:
            while not self._stopped.is_set():
                message = await self.socket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if (pcm := message.get("bytes")) is not None:
                    if self._session is not None and pcm:
                        await self._session.send_realtime_input(
                            audio={"data": pcm, "mime_type": "audio/pcm;rate=16000"},
                        )
                elif (text := message.get("text")) is not None:
                    await self._on_control(json.loads(text))
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as err:
            logger.error("live.browser_loop_failed",
                         error_code=type(err).__name__, status="error")
        finally:
            self._stopped.set()

    async def _on_control(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "text" and self._session is not None:
            # A typed transmission, for when a brevity code is misheard.
            await self._session.send_client_content(
                turns={"role": "user",
                       "parts": [{"text": str(message.get("text", ""))}]},
                turn_complete=True,
            )
        elif kind == "stop":
            self._stopped.set()

    async def _receive_from_model(self, session: Any) -> None:
        """Stream model audio to the browser and run its tool calls.

        SDK BEHAVIOUR THAT MATTERS: session.receive() yields a SINGLE model
        turn and then stops -- it breaks on turn_complete by design (see
        google.genai.live.AsyncSession.receive). So this loops, taking a
        fresh iterator per turn.

        Holding one iterator for the whole session looks natural and fails
        silently: the first turn works and every later one returns nothing.
        Observed exactly that -- turn 1 spoke, turns 2-4 were mute, with no
        error anywhere.
        """
        try:
            while not self._stopped.is_set():
                # A fresh iterator per turn, per the SDK contract above.
                async for message in session.receive():
                    if self._stopped.is_set():
                        return
                    await self._handle_message(session, message)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error("live.model_loop_failed",
                         error_code=type(err).__name__, status="error")
            await self._send_event({
                "type": "error",
                "message": "החיבור הקולי נפל. אפשר להקליד.",
            })
            self._stopped.set()

    async def _handle_message(self, session: Any, message: Any) -> None:
        """Handle one message from the Live session."""
        # Tool calls first: the model is waiting on the result, so any
        # delay here is audible as a gap mid-transmission.
        if message.tool_call:
            await self._handle_tool_calls(session, message.tool_call)
            return

        content = message.server_content
        if content is None:
            return

        # Interruption is detected inside the model. The browser must
        # DISCARD buffered audio, or he keeps talking from the buffer
        # after being cut off.
        if getattr(content, "interrupted", False):
            await self._send_event({"type": "flush_audio"})
            await self._send_event({"type": "barge_in"})
            self._speaking = False
            return

        if content.model_turn:
            for part in content.model_turn.parts or []:
                data = getattr(part.inline_data, "data", None) if part.inline_data else None
                # A frame must hold at least one whole 16-bit sample, and
                # be an even number of bytes. The API emits a 2-byte
                # leading frame, and an odd-length or 1-sample buffer makes
                # Web Audio throw on createBuffer -- which silently kills
                # playback for the whole turn, not just that frame.
                if not data:
                    continue
                if len(data) < 4 or len(data) % 2 != 0:
                    # The API sends a 2-byte leading frame. Web Audio
                    # THROWS on createBuffer for a zero-length or
                    # odd-length buffer, and an uncaught throw there kills
                    # playback for the whole turn -- so one stray frame
                    # silenced every reply.
                    continue
                self._speaking = True
                await self.socket.send_bytes(data)

        if content.input_transcription and content.input_transcription.text:
            await self._send_event({
                "type": "final_transcript",
                "text": content.input_transcription.text,
            })

        if content.output_transcription and content.output_transcription.text:
            # The transcript is what makes a voice session debriefable at
            # all -- without it there is no record of what was said.
            await self._send_event({
                "type": "counterpart_text",
                "text": content.output_transcription.text,
            })

        if content.turn_complete:
            self._speaking = False
            await self._send_event({"type": "turn_complete"})

    async def _handle_tool_calls(self, session: Any, tool_call: Any) -> None:
        """Execute tool calls against the mission engine and reply."""
        from providers.cloud.gemini_live import build_tool_response

        responses = []
        for call in tool_call.function_calls or []:
            args = dict(call.args or {})
            result = self._execute_tool(call.name, args)
            logger.info("live.tool_called", tool_name=call.name,
                        status="error" if "error" in result else "ok")
            responses.append(build_tool_response(call.id, call.name, result))
        if responses:
            await session.send_tool_response(function_responses=responses)

    async def _trigger_loop(self, session: Any) -> None:
        """Fire mission triggers into the Live session.

        A trigger's intent is injected as a text turn, so the counterpart
        speaks it in his own voice rather than reading a script -- the same
        discipline as the text path (FR-C7). The realism layer is not
        involved, so priority affects only WHETHER it fires, not its pacing.
        """
        from core.triggers import select_firing
        from sim.runner import TICK_SECONDS

        try:
            while not self._stopped.is_set():
                now = self.runner.clock.now()
                self.engine.advance_to(now)

                context = self.runner._context(now)
                firing, suppressions = select_firing(self.runner.triggers, context)

                for suppression in suppressions:
                    self.runner.log.suppressions.append((now, suppression))

                if firing is not None:
                    self.runner.log.fired_counts[firing.trigger_id] = (
                        self.runner.log.fired_counts.get(firing.trigger_id, 0) + 1
                    )
                    for effect in firing.effects:
                        self.engine.apply(StateCommand(
                            parameter_id=effect.parameter, value=effect.value,
                            source="trigger",
                        ))
                    if firing.say_intent and not self._speaking:
                        cue = (
                            "[SIMULATION CUE — not a transmission from the trainee, "
                            "and not words to quote. Say this yourself, briefly, in "
                            f"your own voice.] {firing.say_intent}"
                        )
                        if firing.instructions:
                            cue += f" ({firing.instructions})"
                        await session.send_client_content(
                            turns={"role": "user", "parts": [{"text": cue}]},
                            turn_complete=True,
                        )
                        await self._send_event({
                            "type": "initiated", "trigger_id": firing.trigger_id,
                        })

                await asyncio.sleep(TICK_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error("live.trigger_loop_failed",
                         error_code=type(err).__name__, status="error")

    async def _send_event(self, payload: dict[str, Any]) -> None:
        try:
            await self.socket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._stopped.set()
