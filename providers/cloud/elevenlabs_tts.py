"""ElevenLabs text-to-speech over WebSocket, streaming.

MODEL CHOICE IS LOAD-BEARING FOR HEBREW. Measured by others and widely
reported: eleven_v3 takes roughly SEVEN SECONDS per Hebrew sentence, while
eleven_flash_v2_5 takes about one. v3 has the better voices and the
expressive audio tags, but it is built for pre-rendered audio; ElevenLabs
themselves recommend Flash for real-time conversational use.

For a radio net, where a multi-second gap before every transmission is
conspicuous, that is not a close call. Flash v2.5 it is, overridable per
deployment if a future model changes the tradeoff.

WHY WEBSOCKET AND NOT HTTP: the WebSocket endpoint streams audio as it is
generated and supports mid-utterance interruption. The HTTP endpoint
returns a finished file, which would mean waiting for the whole
transmission before the trainee hears a word -- and would make barge-in
impossible, since there would be nothing to cut off.

SYNTHESIS IS PER DELIVERY SEGMENT, not per turn. That is what carries the
realism layer into audio: a pause between segments becomes real silence,
and an interrupted plan simply stops without having synthesized the rest.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import AsyncIterator

from obs.logging import get_logger
from providers.base import ProviderError, VoiceSettings

logger = get_logger("providers.elevenlabs_tts")

# Flash v2.5: ~75ms generation, 32 languages including Hebrew.
_DEFAULT_MODEL = "eleven_flash_v2_5"

# A default multilingual voice. Hebrew quality varies a lot by voice, so
# this is the first thing worth changing by ear -- see ELEVENLABS_VOICE_ID.
_DEFAULT_VOICE = "pNInz6obpgDQGcFmaJgB"

_WS_URL = (
    "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    "?model_id={model_id}&output_format=pcm_16000&inactivity_timeout=20"
)

# 16kHz mono PCM: enough for speech, and a quarter the bytes of 48kHz.
# The browser resamples on playback.
SAMPLE_RATE = 16000


class ElevenLabsTTS:
    """Streaming Hebrew TTS.

    Holds one WebSocket per utterance rather than one per session: a
    per-session socket would need careful state resets between utterances
    and leaks a connection whenever a session is abandoned mid-transmission,
    which happens constantly with barge-in.
    """

    name = "elevenlabs"

    def __init__(self, language: str = "he") -> None:
        self.language = language
        self._api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        self._model = os.environ.get("ELEVENLABS_TTS_MODEL", "").strip() or _DEFAULT_MODEL
        self._voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or _DEFAULT_VOICE

    @property
    def requires_network(self) -> bool:
        return True

    def _require_key(self) -> str:
        if not self._api_key:
            raise ProviderError(
                "ELEVENLABS_API_KEY is not set. Get one at https://elevenlabs.io "
                "(the free tier is enough for testing), or set TTS_PROVIDER=piper "
                "for a local voice."
            )
        return self._api_key

    async def synthesize_stream(
        self, text: str, voice: VoiceSettings | None = None
    ) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks for one span of text.

        Called once per delivery segment, so the caller controls pacing and
        can stop between segments. `voice.pace` maps to the model's speed
        setting, which is how a tone shift reaches the audio rather than
        only the transcript.
        """
        import websockets

        api_key = self._require_key()
        settings = voice or VoiceSettings(voice_id=self._voice_id)
        voice_id = settings.voice_id or self._voice_id

        url = _WS_URL.format(voice_id=voice_id, model_id=self._model)

        try:
            async with websockets.connect(
                url, additional_headers={"xi-api-key": api_key}
            ) as socket:
                await socket.send(json.dumps({
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.45,
                        "similarity_boost": 0.75,
                        # Clamped: the API rejects values outside 0.7-1.2,
                        # and an urgent tone can push pace past that.
                        "speed": max(0.7, min(1.2, settings.pace)),
                    },
                    # Lowest-latency chunking: emit audio after very little
                    # text rather than waiting for a natural boundary.
                    "generation_config": {"chunk_length_schedule": [50, 120, 160, 290]},
                }))
                await socket.send(json.dumps({"text": text}))
                await socket.send(json.dumps({"text": "", "flush": True}))

                while True:
                    try:
                        raw = await asyncio.wait_for(socket.recv(), timeout=20.0)
                    except asyncio.TimeoutError:
                        logger.warning("tts.timeout", provider=self.name, status="error")
                        return

                    message = json.loads(raw)
                    if message.get("audio"):
                        yield base64.b64decode(message["audio"])
                    if message.get("isFinal"):
                        return
                    if message.get("error"):
                        # Surfaced as a provider error rather than silence:
                        # a mute counterpart is indistinguishable from a
                        # broken simulator.
                        raise ProviderError(
                            f"ElevenLabs TTS error: {message.get('message', 'unknown')}"
                        )
        except ProviderError:
            raise
        except Exception as err:
            logger.error("tts.failed", provider=self.name,
                         error_code=type(err).__name__, status="error")
            raise ProviderError(f"TTS stream failed: {type(err).__name__}") from err

    def is_unavailable_error(self, err: Exception) -> bool:
        """True only for a recognised transient failure; a real bug propagates."""
        message = str(err).lower()
        if any(fatal in message for fatal in ("401", "invalid api key", "quota_exceeded")):
            return False
        return any(marker in message for marker in (
            "429", "rate limit", "timeout", "503", "502", "connection",
        ))
