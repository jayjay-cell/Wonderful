"""ElevenLabs cascade: STT -> shared domain layer -> TTS.

The other voice transport. Unlike Gemini Live it keeps the realism
layer's pacing, because the text is produced before it is spoken -- so
authored stalls and the response delay apply. That is the real difference
between the two paths, and it is why both are kept.

This file owns only transport: microphone framing, turn detection, and
pushing synthesized audio back. Facts, commitments, handover, lifecycle
and persistence all come from sim/session.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from obs.logging import get_logger
from sim.turn_detection import TurnDetector, TurnState

logger = get_logger("api.voice")


async def run_cascade(socket: WebSocket, live: Any) -> None:
    """Attach an ElevenLabs cascade socket to a prepared session."""
    from providers.base import ProviderError, build_stt_provider, build_tts_provider

    exercise = live.session.exercise

    try:
        stt = build_stt_provider(exercise.mission)
        tts = build_tts_provider(exercise.mission)
    except ProviderError as err:
        await socket.send_text(json.dumps({"type": "error", "message": str(err)}))
        await socket.close()
        return

    bridge = CascadeBridge(socket, live, tts)
    await socket.send_text(json.dumps({
        "type": "voice_ready", "mode": "elevenlabs",
        "input_sample_rate": 16000, "output_sample_rate": 16000,
        "language": exercise.mission.language,
    }))

    logger.info("voice.connected", session_id=exercise.session_id)
    try:
        await bridge.run(stt)
    finally:
        logger.info("voice.disconnected", session_id=exercise.session_id)


class CascadeBridge:
    """Microphone in, synthesized speech out."""

    def __init__(self, socket: WebSocket, live: Any, tts: Any) -> None:
        self.socket = socket
        self.live = live
        self.session = live.session
        self.exercise = live.session.exercise
        self.tts = tts

        self.detector = TurnDetector()
        self._mic: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._stopped = asyncio.Event()
        self._turn_in_flight = False

        # The session speaks through us, so utterances it produces -- a
        # report, a briefing request -- reach the speaker too.
        self._previous_hook = self.session.on_utterance
        self.session.on_utterance = self._on_utterance

    # -- run --------------------------------------------------------------

    async def run(self, stt: Any) -> None:
        tasks = [
            asyncio.create_task(self._from_browser()),
            asyncio.create_task(self._transcribe(stt)),
        ]
        try:
            await self._stopped.wait()
        finally:
            self.session.on_utterance = self._previous_hook
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _from_browser(self) -> None:
        try:
            while not self._stopped.is_set():
                message = await self.socket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (pcm := message.get("bytes")) is not None:
                    await self._on_audio(pcm)
                elif (text := message.get("text")) is not None:
                    await self._control(json.loads(text))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self._stopped.set()
            await self._mic.put(None)

    async def _on_audio(self, pcm: bytes) -> None:
        """Handle one microphone frame: detect turns, forward to STT."""
        self.detector.set_counterpart_speaking(self._turn_in_flight)
        state, barge_in = self.detector.feed(pcm)

        if barge_in:
            # Immediate, before any transcript exists. Waiting to know
            # WHAT was said would let the operator talk over the trainee
            # for a full second.
            await self._event({"type": "barge_in"})

        try:
            self._mic.put_nowait(pcm)
        except asyncio.QueueFull:
            # Drop the oldest rather than block the socket: a backed-up
            # queue means STT stalled, and old audio is worth less than
            # staying responsive.
            try:
                self._mic.get_nowait()
                self._mic.put_nowait(pcm)
            except Exception:
                pass

    async def _control(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "text":
            await self._transmission(str(message.get("text", "")))
        elif kind == "stop":
            self._stopped.set()

    async def _transcribe(self, stt: Any) -> None:
        async def audio_source():
            while not self._stopped.is_set():
                chunk = await self._mic.get()
                if chunk is None:
                    return
                yield chunk

        try:
            async for transcript in stt.transcribe_stream(audio_source()):
                if self._stopped.is_set():
                    return
                if not transcript.is_final:
                    # Interim text steers the silence threshold and shows
                    # the trainee what was heard. Never acted on as
                    # content -- it changes as more audio arrives.
                    self.detector.update_partial(transcript.text)
                    await self._event({"type": "partial_transcript",
                                       "text": transcript.text})
                    continue

                await self._event({"type": "final_transcript",
                                   "text": transcript.text})
                if self.detector.state is TurnState.TRAINEE_FINISHED:
                    await self._transmission(transcript.text)
        except Exception as err:
            logger.error("voice.stt_failed", error_code=type(err).__name__,
                         status="error")
            await self._event({"type": "error",
                               "message": "התמלול נכשל. אפשר להקליד."})

    async def _transmission(self, text: str) -> None:
        """Run one turn through the SHARED session."""
        text = text.strip()
        if not text or self._turn_in_flight:
            return
        self._turn_in_flight = True
        self.detector.consume_turn()
        try:
            await self.session.trainee_says(text)
        finally:
            self._turn_in_flight = False

    # -- speaking ---------------------------------------------------------

    def _on_utterance(self, utterance: Any) -> Any:
        """Speak anything the session records from the operator."""
        if self._previous_hook is not None:
            maybe = self._previous_hook(utterance)
            if asyncio.iscoroutine(maybe):
                asyncio.create_task(maybe)
        if utterance.speaker != "operator" or not utterance.text.strip():
            return None
        return self._speak(utterance.text)

    async def _speak(self, text: str) -> None:
        """Synthesize and stream one utterance."""
        from providers.base import VoiceSettings

        try:
            async for chunk in self.tts.synthesize_stream(
                text, VoiceSettings(voice_id="")
            ):
                if self._stopped.is_set():
                    return
                if chunk:
                    await self.socket.send_bytes(chunk)
        except Exception as err:
            # A TTS failure must not end the exercise: the trainee hears a
            # dropped transmission, which happens on a real net, rather
            # than the session dying.
            logger.error("voice.tts_failed", error_code=type(err).__name__,
                         status="error")

    async def _event(self, payload: dict[str, Any]) -> None:
        try:
            await self.socket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._stopped.set()
