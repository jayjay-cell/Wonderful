"""Gemini Live: native Hebrew speech-to-speech.

One model hears audio and speaks audio. No separate STT or TTS, and no
other vendor key -- it uses the GEMINI_API_KEY already configured.

MEASURED on this project (gemini-3.1-flash-live-preview):

    tool call issued      547 ms
    first Hebrew audio   1312 ms
    tool result honoured  yes -- it reported 176.3 from the tool rather
                          than inventing a figure

That last line is the one that mattered. The concern with a native
speech-to-speech model was that the mission-state guarantee would weaken:
if the model will not reliably call a tool before stating a number, the
simulator can teach a wrong figure, which is worse than no training. It
does call the tool, so the guarantee holds.

WHAT THIS PATH GIVES UP, stated plainly:

  * The realism layer. Gemini Live owns its own prosody and pauses, so
    core/realism.py's DeliveryPlan is NOT used here. There are no
    controlled stalls, no marker placement, no state-driven garbling.
  * Deterministic pacing. Timing is the model's, so it is not
    reproducible from a seed.

WHAT IT GAINS:

  * Roughly 1.3s to first audio, versus the cascade's 1.5-2.5s.
  * Natural interruption handled inside the model.
  * No second vendor, no second key.

Both paths therefore stay in the codebase. This is the honest comparison
the architecture was meant to permit, not a replacement: see
docs/ARCHITECTURE.md ADR-2 for why the cascade remains the default.

AUDIO FORMATS, which differ per direction and are easy to get wrong:
    input   16 kHz, 16-bit PCM, mono
    output  24 kHz, 16-bit PCM, mono
The browser resamples on playback.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from obs.logging import get_logger
from providers.base import ProviderError

logger = get_logger("providers.gemini_live")

# Verified working against this account. Live model names change often, so
# a failure here is a model-availability problem rather than a code bug --
# hence the candidate list.
_DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
_FALLBACK_MODELS = ("gemini-3.8-live", "gemini-live-2.5-flash-preview")

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


class GeminiLiveProvider:
    """Native speech-to-speech session factory."""

    name = "gemini_live"

    def __init__(self, language: str = "he") -> None:
        self.language = language
        self._api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        self._model = os.environ.get("GEMINI_LIVE_MODEL", "").strip() or _DEFAULT_MODEL
        # Charon, chosen by ear from a 10-voice Hebrew comparison (see
        # voice_samples/compare.html). Puck -- the API default -- reads
        # Hebrew with a noticeably English delivery.
        self._voice = os.environ.get("GEMINI_LIVE_VOICE", "").strip() or "Charon"

    @property
    def requires_network(self) -> bool:
        return True

    def _require_key(self) -> str:
        if not self._api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. It is the same key the text agent "
                "uses -- see .env."
            )
        return self._api_key

    def build_config(self, system_prompt: str, tool_declarations: list[Any]) -> Any:
        """Assemble the Live session config.

        Both transcriptions are enabled deliberately: without them a voice
        session leaves no transcript, and a training session with no record
        cannot be debriefed. They are the audio equivalent of the text
        channel's message log.
        """
        from google.genai import types

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_prompt,
            # What the trainee said, and what the counterpart said.
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice,
                    )
                ),
                language_code="he-IL" if self.language == "he" else None,
            ),
            tools=tool_declarations or None,
        )

    def client(self) -> Any:
        from google import genai
        return genai.Client(api_key=self._require_key())

    async def connect(self, system_prompt: str, tool_declarations: list[Any]):
        """Open a Live session, trying fallback models if the first is gone.

        Live model names are renamed and retired frequently, so a single
        hardcoded name turns a vendor change into an outage. The fallback
        list keeps a session openable without a code edit.
        """
        client = self.client()
        config = self.build_config(system_prompt, tool_declarations)

        candidates = [self._model, *(m for m in _FALLBACK_MODELS if m != self._model)]
        last_error: Exception | None = None

        for model in candidates:
            try:
                session = client.aio.live.connect(model=model, config=config)
                if model != self._model:
                    logger.warning("live.model_fallback", provider=self.name,
                                   reason="primary_unavailable")
                return session
            except Exception as err:
                last_error = err
                continue

        raise ProviderError(
            f"could not open a Gemini Live session on any of {candidates}: "
            f"{type(last_error).__name__}"
        ) from last_error

    def is_unavailable_error(self, err: Exception) -> bool:
        """True only for a recognised transient failure; a real bug propagates."""
        message = str(err).lower()
        if any(fatal in message for fatal in (
            "api key not valid", "permission denied", "invalid argument",
        )):
            return False
        return any(marker in message for marker in (
            "429", "quota", "resource exhausted", "unavailable",
            "timeout", "503", "502", "connection", "not found",
        ))


def build_tool_response(call_id: Any, name: str, result: dict[str, Any]) -> Any:
    """Wrap a tool result in the SDK's FunctionResponse type.

    Exists so api/live_voice.py never imports the Google SDK: the
    architecture guard confines cloud SDKs to providers/cloud/, and it
    caught this exact leak. Keeping the SDK here is what makes a provider
    genuinely swappable rather than nominally so.
    """
    from google.genai import types
    return types.FunctionResponse(id=call_id, name=name, response=result)


def tool_declarations_for(exercise: Any) -> list[Any]:
    """Declare the shared tool set to the Live API.

    Written by hand rather than reflected off the LangChain tools: the
    Live API wants its own Schema objects, and reflecting would couple two
    SDKs' schema formats for no gain. The tool SET is small and stable;
    what matters is that the names and semantics match
    tools/mission_tools.py exactly, because both paths execute the same
    implementations.
    """
    from google.genai import types

    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="current_information",
            description=(
                "Everything you can see right now: your readings and the "
                "current picture. Use it when you need a fact you were not "
                "just given. If something is absent you do not have it — say "
                "so rather than estimating."
            ),
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="recall_observation",
            description=(
                "Look back at something you observed earlier this sortie. "
                "Results marked is_current false HAPPENED and are over — "
                "describe them in the past tense, never as the current "
                "picture."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"about": types.Schema(type="STRING")},
            ),
        ),
        types.FunctionDeclaration(
            name="agree_to_report",
            description=(
                "Record that you have AGREED to report on something. Call it "
                "only AFTER actually agreeing — not while still clarifying, "
                "and not if you pushed back and it was unresolved. Covers "
                "events from now on."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "tags": types.Schema(type="ARRAY",
                                         items=types.Schema(type="STRING")),
                    "entity_ids": types.Schema(type="ARRAY",
                                               items=types.Schema(type="STRING")),
                    "description": types.Schema(type="STRING"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="cancel_reporting",
            description=(
                "Stop reporting on something agreed earlier. everything=true "
                "for 'stop updating me'."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "commitment_id": types.Schema(type="STRING"),
                    "everything": types.Schema(type="BOOLEAN"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="listed_commitments",
            description="What you have currently agreed to report on.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="cannot_comply",
            description=(
                "Look up why you cannot do something asked of you — change "
                "the camera angle, zoom further, change altitude. Returns the "
                "real reason from your mission constraints. Say it in your own "
                "words; never claim you did the thing."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"topic": types.Schema(type="STRING")},
            ),
        ),
    ])]
