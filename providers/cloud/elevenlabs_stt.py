"""ElevenLabs Scribe v2 Realtime speech-to-text over WebSocket.

~150ms latency, 90+ languages including Hebrew.

WHY PARTIAL TRANSCRIPTS MATTER HERE, and not just for display: turn
detection needs to know the trainee has STARTED talking long before it
knows what they said. A batch STT that returns only finished utterances
would make barge-in impossible to detect in time -- the counterpart would
keep talking over them until they stopped, which is the exact failure the
whole suppression design exists to prevent.

So this yields two kinds of result:

  is_final=False   an interim guess. Used ONLY as a speech-detected
                   signal, never acted on as content.
  is_final=True    a committed transcript, safe to send to the agent.

HEBREW BREVITY CODES ARE THE RISK. Military brevity in Hebrew is close to
worst case for any ASR: short, unusual words, often clipped. Scribe
supports keyterm conditioning, so the mission's own brevity dictionary and
callsigns are passed as hints -- which is the single cheapest accuracy win
available, and only possible because the mission file already declares
them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from obs.logging import get_logger
from providers.base import ProviderError

logger = get_logger("providers.elevenlabs_stt")

_DEFAULT_MODEL = "scribe_v2_realtime"
_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

# Must match what the browser captures and what we tell the API.
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Transcript:
    text: str
    is_final: bool
    confidence: float | None = None


class ElevenLabsSTT:
    """Streaming Hebrew STT over a WebSocket."""

    name = "elevenlabs"

    def __init__(self, language: str = "he", keyterms: Sequence[str] = ()) -> None:
        self.language = language
        self._api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        self._model = os.environ.get("ELEVENLABS_STT_MODEL", "").strip() or _DEFAULT_MODEL
        # Brevity codes and callsigns from the mission file. Biasing the
        # model toward words it should expect is the cheapest accuracy win
        # available for domain vocabulary.
        self._keyterms = list(keyterms)[:100]

    @property
    def requires_network(self) -> bool:
        return True

    def _require_key(self) -> str:
        if not self._api_key:
            raise ProviderError(
                "ELEVENLABS_API_KEY is not set. Get one at https://elevenlabs.io, "
                "or set STT_PROVIDER=whisper_local for an offline Hebrew model."
            )
        return self._api_key

    async def transcribe_stream(
        self, audio: AsyncIterator[bytes]
    ) -> AsyncIterator[Transcript]:
        """Transcribe a live audio stream, yielding partials then finals.

        Audio is pushed by a concurrent task rather than interleaved with
        receiving, because the two have independent rhythms: the microphone
        produces chunks on a fixed clock while the model emits transcripts
        when it has something to say. Interleaving them in one loop would
        make each wait on the other.
        """
        import websockets

        api_key = self._require_key()
        url = f"{_WS_URL}?model_id={self._model}"

        try:
            async with websockets.connect(
                url, additional_headers={"xi-api-key": api_key}
            ) as socket:
                config: dict[str, object] = {
                    "type": "configure",
                    "language": self.language,
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate": SAMPLE_RATE,
                    },
                }
                if self._keyterms:
                    config["keyterms"] = self._keyterms
                await socket.send(json.dumps(config))

                async def pump() -> None:
                    """Forward microphone audio until the stream ends."""
                    try:
                        async for chunk in audio:
                            if not chunk:
                                continue
                            await socket.send(json.dumps({
                                "type": "audio",
                                "audio": base64.b64encode(chunk).decode(),
                            }))
                        await socket.send(json.dumps({"type": "commit"}))
                    except Exception as err:
                        logger.warning("stt.pump_stopped",
                                       error_code=type(err).__name__, status="error")

                pump_task = asyncio.create_task(pump())
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=60.0)
                        except asyncio.TimeoutError:
                            return

                        message = json.loads(raw)
                        kind = message.get("type", "")

                        if kind in {"partial_transcript", "interim_transcript"}:
                            text = (message.get("text") or "").strip()
                            if text:
                                yield Transcript(text=text, is_final=False)
                        elif kind in {"final_transcript", "committed_transcript",
                                      "transcript"}:
                            text = (message.get("text") or "").strip()
                            if text:
                                yield Transcript(
                                    text=text, is_final=True,
                                    confidence=message.get("confidence"),
                                )
                        elif kind == "error":
                            raise ProviderError(
                                f"ElevenLabs STT error: {message.get('message', 'unknown')}"
                            )
                finally:
                    pump_task.cancel()
        except ProviderError:
            raise
        except Exception as err:
            logger.error("stt.failed", provider=self.name,
                         error_code=type(err).__name__, status="error")
            raise ProviderError(f"STT stream failed: {type(err).__name__}") from err

    def is_unavailable_error(self, err: Exception) -> bool:
        message = str(err).lower()
        if any(fatal in message for fatal in ("401", "invalid api key", "quota_exceeded")):
            return False
        return any(marker in message for marker in (
            "429", "rate limit", "timeout", "503", "502", "connection",
        ))


def mission_keyterms(mission) -> list[str]:
    """Brevity codes, callsigns and parameter labels from a mission.

    Lives here rather than in core/ because it is an STT concern, but it
    reads only the mission's own declarations -- so a new scenario's
    vocabulary biases the recognizer with no code change.
    """
    terms: list[str] = [
        mission.procedure.callsigns.counterpart,
        mission.procedure.callsigns.trainee,
    ]
    terms.extend(term.term for term in mission.procedure.brevity)
    terms.extend(p.label for p in mission.parameters if p.label)
    # Spoken enum forms too: "עוקב" is a word the trainee will say.
    for parameter in mission.parameters:
        terms.extend(parameter.value_labels.values())
    # Deduplicated while preserving order, so the most important terms
    # (callsigns) stay at the front if the list is ever truncated.
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return unique
