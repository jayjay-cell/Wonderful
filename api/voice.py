"""Voice WebSocket: browser audio in, counterpart audio out.

ONE socket carries both directions. Binary frames are PCM audio; text
frames are JSON control messages. A second socket for control would need
its own lifecycle and could desynchronize from the audio, which matters
because an interrupt must be ordered relative to the audio around it.

THE LATENCY PATH, and where each millisecond goes:

    browser mic
      -> PCM frames (20ms each)           ~20ms
      -> turn detection                   <1ms  (energy VAD, local)
      -> Scribe v2 Realtime STT          ~150ms
      -> [turn considered over]           420-900ms  (silence threshold)
      -> agent + tools + mission state   ~1000ms
      -> realism plan                     <1ms
      -> ElevenLabs Flash v2.5 TTS       ~250ms to first audio byte
      -> browser speaker

The silence threshold and the model call dominate. Both are already as
tuned as they can be without changing what the product is: the threshold
adapts to transcript shape (sim/turn_detection.py), and the model is the
fastest one measured at equal quality.

What CANNOT be optimised away is the persona's own delay -- and should not
be, because an operator who replies the instant you stop talking is the
single clearest tell that there is no person there. The realism lead-in
absorbs the model's latency rather than adding to it (see
sim/runner.py), so real cost hides inside modelled hesitation.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from obs.logging import get_logger
from sim.turn_detection import TurnDetector, TurnState

logger = get_logger("api.voice")

# 20ms of 16kHz mono PCM. Small enough that barge-in detection is prompt,
# large enough that per-frame overhead stays irrelevant.
FRAME_BYTES = 640


class VoiceBridge:
    """Connects one browser audio socket to one live session.

    Three concurrent concerns, each its own task because they have
    independent rhythms: receiving microphone audio (fixed clock),
    transcribing (bursty), and delivering the counterpart's audio
    (driven by the delivery plan). Interleaving them in one loop would
    make each wait on the others.
    """

    def __init__(self, socket: WebSocket, live: Any, mission: Any) -> None:
        self.socket = socket
        self.live = live
        self.runner = live.runner
        self.mission = mission

        self.detector = TurnDetector()
        self._mic: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        self._stopped = asyncio.Event()
        self._turn_in_flight = False

    # -- outbound ---------------------------------------------------------

    async def send_audio(self, pcm: bytes, kind: str) -> None:
        """Send counterpart audio, or a control frame, to the browser."""
        try:
            if kind == "interrupt":
                # Text frame, not binary: the client must DISCARD its
                # buffer. Sending nothing would leave queued audio playing
                # after he was cut off -- the exact talk-over the design
                # forbids.
                await self.socket.send_text(json.dumps({"type": "flush_audio"}))
                return
            if pcm:
                await self.socket.send_bytes(pcm)
        except Exception:
            self._stopped.set()

    async def send_event(self, payload: dict[str, Any]) -> None:
        try:
            await self.socket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            self._stopped.set()

    # -- inbound ----------------------------------------------------------

    async def receive_loop(self) -> None:
        """Read frames from the browser until it disconnects."""
        try:
            while not self._stopped.is_set():
                message = await self.socket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if (data := message.get("bytes")) is not None:
                    await self._on_audio(data)
                elif (text := message.get("text")) is not None:
                    await self._on_control(json.loads(text))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self._stopped.set()
            await self._mic.put(None)

    async def _on_audio(self, pcm: bytes) -> None:
        """Handle one microphone frame: detect turns, forward to STT."""
        self.detector.set_counterpart_speaking(self.runner.is_delivering)
        state, barge_in = self.detector.feed(pcm)

        if barge_in:
            # Immediate, before any transcript exists. Waiting to know
            # WHAT was said would let him talk over the trainee for a full
            # second -- detecting that someone started is a separate and
            # much faster question than knowing what they said.
            self.runner.interrupt()
            await self.send_event({"type": "barge_in"})

        if state is TurnState.TRAINEE_SPEAKING:
            # Feeds the suppression guard: non-critical initiative waits
            # while the trainee holds the net. Same predicate the text UI
            # drives from its typing indicator.
            self.runner.set_composing(True)

        try:
            self._mic.put_nowait(pcm)
        except asyncio.QueueFull:
            # Drop the oldest frame rather than blocking the socket: a
            # backed-up queue means STT has stalled, and old audio is
            # worth less than staying responsive.
            try:
                self._mic.get_nowait()
                self._mic.put_nowait(pcm)
            except Exception:
                pass

    async def _on_control(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "interrupt":
            self.runner.interrupt()
        elif kind == "text":
            # A typed transmission during a voice session: useful when a
            # brevity code is misrecognised and the trainer wants to move on.
            await self._handle_transmission(str(message.get("text", "")))
        elif kind == "stop":
            self._stopped.set()

    # -- transcription ----------------------------------------------------

    async def transcribe_loop(self, stt: Any) -> None:
        """Stream microphone audio to STT and act on finished turns."""
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
                    # content -- an interim transcript changes as more
                    # audio arrives.
                    self.detector.update_partial(transcript.text)
                    await self.send_event({
                        "type": "partial_transcript", "text": transcript.text,
                    })
                    continue

                await self.send_event({
                    "type": "final_transcript", "text": transcript.text,
                })

                if self.detector.state is TurnState.TRAINEE_FINISHED:
                    await self._handle_transmission(transcript.text)
        except Exception as err:
            logger.error("voice.stt_loop_failed",
                         error_code=type(err).__name__, status="error")
            await self.send_event({
                "type": "error",
                "message": "התמלול נכשל. אפשר להקליד במקום.",
            })

    async def _handle_transmission(self, text: str) -> None:
        """Run one reactive turn from a completed transmission."""
        text = text.strip()
        if not text or self._turn_in_flight:
            return

        self._turn_in_flight = True
        self.detector.consume_turn()
        self.runner.set_composing(False)
        try:
            record = await self.runner.handle_trainee_message(text)
            if record is not None:
                await self.send_event({
                    "type": "turn_complete",
                    "text": record.text,
                    "status": record.status,
                })
        finally:
            self._turn_in_flight = False

    # -- lifecycle --------------------------------------------------------

    async def run(self, stt: Any) -> None:
        """Run until the browser disconnects."""
        tasks = [
            asyncio.create_task(self.receive_loop()),
            asyncio.create_task(self.transcribe_loop(stt)),
        ]
        try:
            await self._stopped.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
