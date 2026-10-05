"""Provider interfaces -- THE AIR-GAP SEAM (docs/ARCHITECTURE.md §4).

Every external dependency the system has -- currently the language model
-- is reached only through this protocol. Speech needs no protocol here:
Gemini Live is speech-to-speech in one session, so its audio never passes
through a separate STT or TTS provider. Switching between a cloud provider during development and a
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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

    def chat_model(self) -> "BaseChatModel":
        """The LangChain chat model to run turns with."""
        ...

    def is_unavailable_error(self, err: Exception) -> bool:
        """Whether this error is a transient outage rather than a bug."""
        ...

    @property
    def requires_network(self) -> bool:
        """Whether this provider needs internet access."""
        ...


# -- selection -------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a provider cannot be selected or constructed.

    Distinct from a provider being temporarily unavailable: this means
    misconfiguration, which should fail loudly at startup rather than
    falling back and masking the mistake.
    """


def selected_llm_provider_name() -> str:
    """Which LLM provider the environment asks for, defaulting to gemini."""
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
