"""Anthropic LLM provider -- DEVELOPMENT ONLY.

The only place in the codebase permitted to import a cloud SDK
(tests/test_architecture.py enforces that). The deployment target is
air-gapped and will use providers/local/ollama.py instead; this exists so
persona quality and realism can be developed against a strong model before
local-model limitations are tackled.

Import is lazy inside chat_model() so an air-gapped install without
langchain-anthropic can still import this module. Eager imports would make
a missing optional dependency fatal for every provider rather than just
this one.
"""

from __future__ import annotations

import os

from providers.base import ProviderError

# Short, so a stuck provider is abandoned in seconds rather than retried
# into the same wall. SDK-level retries are disabled for the same reason:
# fallback logic belongs in one place, not layered invisibly underneath.
_REQUEST_TIMEOUT_SECONDS = 30.0

# Verified against the installed langchain-anthropic; see
# docs/ARCHITECTURE.md §9. Re-check before changing.
_DEFAULT_MODEL = "claude-opus-5"


class AnthropicProvider:
    """LLMProvider backed by the Anthropic API."""

    name = "anthropic"

    def __init__(self) -> None:
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        # Blank is treated as unset: `ANTHROPIC_MODEL=` in .env would
        # otherwise override the default with "" and fail inside the SDK.
        self._model = os.environ.get("ANTHROPIC_MODEL", "").strip() or _DEFAULT_MODEL

    @property
    def requires_network(self) -> bool:
        return True

    def chat_model(self):
        if not self._api_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Set it in .env for development, "
                "or select the local provider with LLM_PROVIDER=ollama."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as err:
            raise ProviderError(
                "langchain-anthropic is not installed. It is an optional "
                "development dependency; the air-gapped deployment does not need "
                "it. Install with: pip install langchain-anthropic"
            ) from err

        return ChatAnthropic(
            model=self._model,
            api_key=self._api_key,
            temperature=0.7,     # roleplay wants variation, unlike an analytics agent
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,       # fallback is handled once, in the agent layer
        )

    def is_unavailable_error(self, err: Exception) -> bool:
        """True only for a RECOGNIZED transient failure.

        A real bug -- bad request, auth failure, unknown model name -- must
        propagate. Letting a fallback loop swallow it would report a code
        defect as a provider outage, which is a lesson carried over from
        the airport project's providers.py.
        """
        try:
            import anthropic
            if isinstance(err, (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.InternalServerError,
            )):
                return True
            # Explicitly NOT transient, listed so the intent is unmistakable.
            if isinstance(err, (
                anthropic.AuthenticationError,
                anthropic.BadRequestError,
                anthropic.NotFoundError,
                anthropic.PermissionDeniedError,
            )):
                return False
        except ImportError:
            pass

        message = str(err).lower()
        return any(marker in message for marker in (
            "rate limit", "429", "timeout", "timed out",
            "connection", "overloaded", "503", "502",
        ))
