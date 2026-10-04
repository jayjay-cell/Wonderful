"""Provider interfaces -- THE AIR-GAP SEAM (docs/ARCHITECTURE.md §4).

Every external dependency the system has -- the language model, and later
speech-to-text and text-to-speech -- is reached only through one of these
protocols. Switching between a cloud provider during development and a
local model in a closed network is an environment variable, never a code
change (FR-F3).

Why this exists from the first commit rather than being added later:
retrofitting an air gap means touching every layer that ever called a
cloud SDK directly, and by then the call sites are spread through the
codebase. Built in from the start it costs one small file per provider
type. The deployment target has no internet, so this is a hard
requirement, not a nicety.

Two mechanisms keep it honest, both automated:
  * tests/test_architecture.py fails if any module outside providers/cloud/
    imports a cloud SDK (FR-F4).
  * providers/offline.py raises on any outbound connection when
    MASLUL_OFFLINE=1, so a stray cloud call fails loudly here instead of
    silently in a room with no internet (FR-F6).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Import only for type checking so this module stays importable in an
    # air-gapped install where langchain may not be present.
    from langchain_core.language_models import BaseChatModel


@runtime_checkable
class LLMProvider(Protocol):
    """A source of a chat model.

    `is_unavailable_error` is what makes provider fallback safe: only a
    RECOGNIZED transient failure (rate limit, timeout, connection refused)
    should move on to the next provider. A real bug -- bad request, auth
    failure, wrong model name -- must propagate, or a fallback loop
    silently reports a code defect as a provider outage. This distinction
    is carried over from the airport project's providers.py, where it was
    learned the hard way.
    """

    name: str

    def chat_model(self) -> "BaseChatModel": ...

    def is_unavailable_error(self, err: Exception) -> bool: ...

    @property
    def requires_network(self) -> bool: ...


@runtime_checkable
class STTProvider(Protocol):
    """Speech to text. Phase 2.

    Streaming rather than batch, because turn detection needs partial
    transcripts as the trainee speaks -- waiting for a complete utterance
    would make barge-in impossible to detect in time.
    """

    name: str
    language: str

    async def transcribe_stream(
        self, audio: AsyncIterator[bytes]
    ) -> AsyncIterator["Transcript"]: ...

    @property
    def requires_network(self) -> bool: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text to speech. Phase 2.

    Synthesis is per DELIVERY SEGMENT, not per turn, so the realism
    layer's pacing survives into audio: a pause between segments is real
    silence, and an interrupted plan stops mid-utterance without having
    synthesized the rest.
    """

    name: str
    language: str

    async def synthesize_stream(
        self, text: str, voice: "VoiceSettings"
    ) -> AsyncIterator[bytes]: ...

    @property
    def requires_network(self) -> bool: ...


# -- shapes used by the phase-2 protocols ----------------------------------


class Transcript:
    """One STT result. `is_final` distinguishes a stable transcript from an
    interim guess, which turn detection needs in order to avoid treating a
    mid-word partial as the end of a turn."""

    def __init__(self, text: str, is_final: bool, confidence: float | None = None) -> None:
        self.text = text
        self.is_final = is_final
        self.confidence = confidence


class VoiceSettings:
    """Per-utterance synthesis settings.

    `pace` and `pitch` are driven by the delivery plan's tone profile, so a
    mid-utterance tone shift reaches the voice rather than only the text.
    """

    def __init__(self, voice_id: str, pace: float = 1.0, pitch: float = 1.0) -> None:
        self.voice_id = voice_id
        self.pace = pace
        self.pitch = pitch


# -- selection -------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider cannot be selected or constructed.

    Distinct from a provider being temporarily unavailable: this means
    misconfiguration, which should fail loudly at startup rather than
    falling back and masking the mistake.
    """


def selected_llm_provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", "gemini").strip().lower()


def build_llm_provider(name: str | None = None) -> LLMProvider:
    """Construct the configured LLM provider.

    Imports lazily and per-branch, so an air-gapped install that has no
    cloud SDK installed can still import this module and run the local
    provider. Eager imports at module top would make a missing optional
    dependency fatal for every provider, not just the one requiring it.
    """
    name = (name or selected_llm_provider_name()).strip().lower()

    if name == "gemini":
        from providers.cloud.gemini import GeminiProvider
        return GeminiProvider()
    if name == "anthropic":
        from providers.cloud.anthropic import AnthropicProvider
        return AnthropicProvider()
    if name == "ollama":
        from providers.local.ollama import OllamaProvider
        return OllamaProvider()

    raise ProviderError(
        f"unknown LLM_PROVIDER {name!r}; expected one of: gemini, anthropic, ollama"
    )


def build_stt_provider(mission: Any = None, name: str | None = None) -> Any:
    """Construct the configured speech-to-text provider.

    Takes the mission so its brevity codes and callsigns can bias the
    recognizer -- the cheapest accuracy win available for Hebrew military
    vocabulary, and only possible because the mission file declares them.
    """
    name = (name or os.environ.get("STT_PROVIDER", "elevenlabs")).strip().lower()

    keyterms: list[str] = []
    if mission is not None:
        from providers.cloud.elevenlabs_stt import mission_keyterms
        keyterms = mission_keyterms(mission)

    if name == "elevenlabs":
        from providers.cloud.elevenlabs_stt import ElevenLabsSTT
        return ElevenLabsSTT(
            language=getattr(mission, "language", "he"), keyterms=keyterms,
        )

    raise ProviderError(
        f"unknown STT_PROVIDER {name!r}; expected: elevenlabs"
    )


def build_tts_provider(mission: Any = None, name: str | None = None) -> Any:
    """Construct the configured text-to-speech provider."""
    name = (name or os.environ.get("TTS_PROVIDER", "elevenlabs")).strip().lower()

    if name == "elevenlabs":
        from providers.cloud.elevenlabs_tts import ElevenLabsTTS
        return ElevenLabsTTS(language=getattr(mission, "language", "he"))

    raise ProviderError(
        f"unknown TTS_PROVIDER {name!r}; expected: elevenlabs"
    )
