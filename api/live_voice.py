"""Gemini Live: native Hebrew speech-to-speech, over the shared domain layer.

One model hears and speaks. No separate STT or TTS, and no second vendor
key -- it uses the GEMINI_API_KEY the text path already needs.

WHAT CHANGED. The previous version reimplemented the domain layer here:
its own tool shim (which omitted a whole tool and never gave the procedure
checker its context) and its own trigger loop (which dropped deferral and
double-counted every event against shared bookkeeping). This file is now a
TRANSPORT. Facts, commitments, handover, lifecycle and persistence come
from sim/session.py, exactly as they do for text.

What it still owns, because it genuinely differs:

  * audio framing -- 16 kHz in, 24 kHz out
  * tool execution through the Live API's own function-call protocol
  * interruption, which the model detects internally

PROSODY IS THE MODEL'S. Pauses, pacing and emphasis come from Gemini Live
itself; this file never shapes them. Stated because it is a real limit on
how much of the delivery is authorable.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.lifecycle import Phase
from obs.logging import get_logger

logger = get_logger("api.live_voice")


async def run_live(socket: WebSocket, live: Any) -> None:
    """Attach a Gemini Live socket to an already-prepared session."""
    from agent.prompts import build_system_prompt
    from providers.base import ProviderError
    from providers.cloud.gemini_live import GeminiLiveProvider

    exercise = live.session.exercise

    try:
        provider = GeminiLiveProvider(language=exercise.mission.language)
        provider._require_key()
    except ProviderError as err:
        await socket.send_text(json.dumps({"type": "error", "message": str(err)}))
        await socket.close()
        return

    bridge = LiveBridge(socket, live, provider)
    # Vocal direction included: a native speech-to-speech model needs
    # delivery stated, or it reads the words correctly and sounds like a
    # narrator.
    prompt = build_system_prompt(exercise, for_speech=True)

    logger.info("live_voice.connected", session_id=exercise.session_id)
    try:
        await bridge.run(prompt)
    finally:
        logger.info("live_voice.disconnected", session_id=exercise.session_id)


class LiveBridge:
    """Browser audio <-> Live session <-> the shared domain layer."""

    def __init__(self, socket: WebSocket, live: Any, provider: Any) -> None:
        """Wire the socket, session and Live provider together and start with no Live session open."""
        self.socket = socket
        self.live = live
        self.session = live.session
        self.exercise = live.session.exercise
        self.provider = provider

        self._live_session: Any = None
        self._stopped = asyncio.Event()
        self._speaking = False
        self._buffer = ""
        # The report this turn is delivering, resolved on turn_complete.
        self._awaiting: Any = None
        # Last lifecycle phase seen, to detect a pause and flush playback.
        self._last_phase = self.exercise.phase

    # -- tools ------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool from the SHARED tool set.

        Dispatches by name into the same tools the text path uses rather
        than reimplementing them. That is what keeps the paths from
        diverging: one definition of what each tool returns.
        """
        from tools.mission_tools import build_mission_tools

        tools = {t.name: t for t in build_mission_tools(self.exercise)}
        tool = tools.get(name)
        if tool is None:
            return {"error": "UNKNOWN_TOOL", "message": "I cannot do that."}
        try:
            return tool.invoke(args or {})
        except Exception as err:
            logger.error("live.tool_failed", tool_name=name,
                         error_code=type(err).__name__, status="error")
            return {"error": "TOOL_FAILED",
                    "message": "That is not coming through right now."}

    # -- run --------------------------------------------------------------

    async def run(self, prompt: str) -> None:
        """Open the Live session and run its three loops until the socket closes."""
        from providers.cloud.gemini_live import tool_declarations_for

        session_cm = await self.provider.connect(
            prompt, tool_declarations_for(self.exercise),
        )
        async with session_cm as live_session:
            self._live_session = live_session
            # Take over report delivery for as long as this bridge lives,
            # so the Session's speak loop stands down and one queue has
            # exactly one consumer.
            self.exercise.reports_owner = "gemini_live"
            await self._event({
                "type": "voice_ready", "mode": "gemini_live",
                "input_sample_rate": 16000, "output_sample_rate": 24000,
                "language": self.exercise.mission.language,
            })

            tasks = [
                asyncio.create_task(self._from_browser()),
                asyncio.create_task(self._from_model(live_session)),
                asyncio.create_task(self._speak_reports(live_session)),
            ]
            try:
                await self._stopped.wait()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # Hand reports back, so a session that keeps running after
                # the socket drops is not left with no consumer at all.
                self.exercise.reports_owner = "session"
                if self._awaiting is not None:
                    self.exercise.release_report(self._awaiting)
                    self._awaiting = None

    async def _from_browser(self) -> None:
        """Forward microphone audio into the Live session."""
        try:
            while not self._stopped.is_set():
                message = await self.socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if (pcm := message.get("bytes")) is not None:
                    # Dropped, not buffered, outside a running exercise:
                    # audio sent before Start or during a Pause would be
                    # answered as though the exercise were live, and a
                    # backlog would burst on resume.
                    if (pcm and self._live_session is not None
                            and self._accepting_input()):
                        await self._live_session.send_realtime_input(
                            audio={"data": pcm, "mime_type": "audio/pcm;rate=16000"},
                        )
                elif (text := message.get("text")) is not None:
                    await self._control(json.loads(text))
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as err:
            logger.error("live.browser_loop_failed",
                         error_code=type(err).__name__, status="error")
        finally:
            self._stopped.set()

    def _accepting_input(self) -> bool:
        """Whether trainee input may reach the model right now.

        False before Start, while Paused and after End, and during a crew
        handover -- the same code-enforced availability the text channel
        uses, so voice cannot talk to a crew that is mid-rotation.
        """
        return self.exercise.clock.is_running and self.exercise.crew_available()

    async def _control(self, message: dict[str, Any]) -> None:
        """Handle a non-audio message -- a typed transmission, an interruption or a stop."""
        kind = message.get("type")
        if kind == "text" and self._live_session is not None:
            # A typed transmission, for when a term is misheard. Recorded
            # through the shared state so it appears in the transcript
            # exactly as a spoken one would.
            if not self._accepting_input():
                return
            text = str(message.get("text", ""))
            self.session.record_trainee(text)
            await self._live_session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True,
            )
        elif kind == "stop":
            self._stopped.set()

    async def _from_model(self, live_session: Any) -> None:
        """Stream model audio to the browser and run its tool calls.

        session.receive() yields a SINGLE turn and then stops -- it breaks
        on turn_complete by design. So this loops, taking a fresh iterator
        per turn. Holding one for the whole session looks natural and
        fails silently: the first turn works and every later one is mute.
        """
        try:
            while not self._stopped.is_set():
                async for message in live_session.receive():
                    if self._stopped.is_set():
                        return
                    await self._handle(live_session, message)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error("live.model_loop_failed",
                         error_code=type(err).__name__, status="error")
            await self._event({"type": "error",
                               "message": "החיבור הקולי נפל. אפשר להקליד."})
            self._stopped.set()

    async def _handle(self, live_session: Any, message: Any) -> None:
        """Route one Live message: a tool call, audio, a transcript, or end of turn."""
        from providers.cloud.gemini_live import build_tool_response

        if message.tool_call:
            responses = []
            for call in message.tool_call.function_calls or []:
                result = self._execute_tool(call.name, dict(call.args or {}))
                logger.info("live.tool_called", tool_name=call.name,
                            status="error" if "error" in result else "ok")
                responses.append(build_tool_response(call.id, call.name, result))
            if responses:
                await live_session.send_tool_response(function_responses=responses)
            return

        content = message.server_content
        if content is None:
            return

        if getattr(content, "interrupted", False):
            # The browser must DISCARD buffered audio, or he keeps talking
            # from the buffer after being cut off.
            await self._event({"type": "flush_audio"})
            await self._event({"type": "barge_in"})
            self._speaking = False
            # Persist what was actually said before the cut, and clear the
            # buffer. Carrying it forward would splice the unsaid
            # remainder onto the next turn's transcript.
            self._flush_transcript(status="interrupted")
            return

        if content.model_turn:
            for part in content.model_turn.parts or []:
                data = (getattr(part.inline_data, "data", None)
                        if part.inline_data else None)
                # A frame must hold at least one whole 16-bit sample and be
                # an even number of bytes. The API emits a 2-byte leading
                # frame, and Web Audio throws on a buffer that small --
                # which silently kills playback for the whole turn.
                if data and len(data) >= 4 and len(data) % 2 == 0:
                    self._speaking = True
                    await self.socket.send_bytes(data)

        if content.input_transcription and content.input_transcription.text:
            # The trainee's own speech. Persisted and pushed into the
            # shared conversation state, so a voice session has a
            # transcript to debrief and the addressing and briefing rules
            # see that contact was made.
            await self._event({"type": "final_transcript",
                               "text": content.input_transcription.text})
            self.session.record_trainee(content.input_transcription.text)

        if content.output_transcription and content.output_transcription.text:
            await self._event({"type": "counterpart_text",
                               "text": content.output_transcription.text})
            self._buffer += content.output_transcription.text

        if content.turn_complete:
            self._speaking = False
            self._flush_transcript()
            await self._event({"type": "turn_complete"})

    def _flush_transcript(self, status: str = "completed") -> None:
        """Close out the operator's turn: record what was said, resolve any
        report it was delivering.

        Recorded through the Session, not the store, so a voice transcript
        holds the same fields as a text one.
        """
        text = self._buffer.strip()
        self._buffer = ""
        report, self._awaiting = self._awaiting, None

        if not text:
            # Nothing was said, so an owed report is still owed.
            if report is not None:
                self.exercise.release_report(report)
            return

        # A report counts as delivered only here, and only if the turn ran
        # to completion. Cut off part-way, it is retried instead.
        delivered = report is not None and status == "completed"
        if report is not None:
            if delivered:
                self.exercise.complete_report(report)
            else:
                self.exercise.release_report(report)
            self.session.recorder.revealed(
                report.event.event_id, reported=delivered,
                disposition="delivered" if delivered else "pending",
            )

        self.session.record_operator(
            text,
            origin="report" if report is not None else "reactive",
            event_id=report.event.event_id if report is not None else None,
            status=status,
        )

    # -- reports ----------------------------------------------------------

    async def _speak_reports(self, live_session: Any) -> None:
        """Inject owed reports as text turns.

        The SHARED exercise decides what is owed, when it is stale and
        whether the crew is available; this only delivers. Injected as a
        cue rather than a line so the operator phrases it itself.
        """
        from agent.prompts import build_report_cue
        from sim.exercise import TICK_SECONDS

        try:
            while not self._stopped.is_set():
                await asyncio.sleep(TICK_SECONDS)

                # On leaving RUNNING, tell the browser to drop whatever it
                # has buffered: audio generated before a pause must not
                # keep playing after it, or resume into silence it already
                # heard.
                phase = self.exercise.phase
                if phase is not self._last_phase:
                    self._last_phase = phase
                    if phase is not Phase.RUNNING:
                        self._speaking = False
                        await self._event({"type": "flush_audio"})
                        self._flush_transcript(status="abandoned")

                if not self.exercise.clock.is_running or self._speaking:
                    continue

                report = self.exercise.claim_report()
                if report is None:
                    continue

                cue = build_report_cue(
                    report.event.operator_information,
                    report.event.instructions,
                    report.event.priority,
                )
                try:
                    await live_session.send_client_content(
                        turns={"role": "user", "parts": [{"text": cue}]},
                        turn_complete=True,
                    )
                except Exception:
                    # The cue never reached the model, so nothing was
                    # said. Put it back rather than recording a delivery.
                    self.exercise.release_report(report)
                    raise

                # Sending a cue is not delivery: the model still has to
                # speak it. _flush_transcript resolves this report once a
                # turn actually completes.
                self._awaiting = report
                await self._event({"type": "report",
                                   "event_id": report.event.event_id})
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error("live.report_loop_failed",
                         error_code=type(err).__name__, status="error")

    async def _event(self, payload: dict[str, Any]) -> None:
        """Send one event to the browser, stopping the bridge if the socket is gone."""
        try:
            await self.socket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._stopped.set()
