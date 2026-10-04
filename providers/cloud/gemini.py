"""Google Gemini LLM provider -- DEVELOPMENT ONLY.

One of two files permitted to import a cloud SDK
(tests/test_architecture.py enforces that). The air-gapped deployment uses
providers/local/ollama.py instead; this exists so persona quality can be
developed against a capable model first.

MODEL CHOICE: gemini-3.7-flash by default -- the current stable general-
purpose Flash model, and a free tier exists for it. Two notes that matter:

  * The Gemini 2.5 family reaches end of life on 16 October 2026, so it is
    deliberately NOT the default here even though much example code still
    uses it.
  * Override with GEMINI_MODEL if you want Pro for better roleplay, or a
    Flash-Lite for lower cost. Roleplay quality is this project's core
    risk, so Flash versus Pro is worth measuring rather than assuming.

Import is lazy inside chat_model() so an air-gapped install without the
Google SDK can still import this module.
"""

from __future__ import annotations

import os

from providers.base import ProviderError

# Short, so a stuck provider is abandoned in seconds rather than retried
# into the same wall. SDK retries are disabled because fallback belongs in
# one place, not layered invisibly underneath.
_REQUEST_TIMEOUT_SECONDS = 30.0

# MEASURED, not assumed. On this project's prompt (~2,900 tokens of Hebrew
# persona, plan and comms schema) with four tools bound:
#
#   gemini-3.5-flash-lite   0.8-1.8 s per turn
#   gemini-3.7-flash        4.2-9.1 s per turn
#
# Quality was INDISTINGUISHABLE on the roleplay tests -- same formats, same
# brevity codes, same refusal to identify a hidden target. For a radio net,
# where a 4-second gap before every reply is conspicuous, a 3-5x latency
# win at no measurable quality cost is the right default.
#
# Revisit if the persona starts feeling flat: roleplay quality is this
# project's core risk, and a stronger model is one env var away.
# Note gemini-2.5-* is end-of-life 16 Oct 2026, so it is not an option.
_DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiProvider:
    """LLMProvider backed by the Gemini Developer API."""

    name = "gemini"

    def __init__(self) -> None:
        # GOOGLE_API_KEY is accepted as an alias because the Google SDKs
        # and much documentation use that name; silently ignoring a key the
        # user has already set under the other name would be a confusing
        # failure.
        self._api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        # `or _DEFAULT_MODEL` rather than a dict default: an env var that is
        # PRESENT BUT EMPTY (`GEMINI_MODEL=` in .env, which is how the
        # template ships) would otherwise override the default with "" and
        # fail deep inside the SDK as "model is required." Treating blank
        # as unset is what makes a commented-out override behave the way
        # anyone editing .env expects.
        self._model = os.environ.get("GEMINI_MODEL", "").strip() or _DEFAULT_MODEL

        # Thinking tokens are latency with no benefit for short in-character
        # radio traffic -- the model is not solving a problem, it is reading
        # an instrument and phrasing one sentence. Note "minimal" is
        # REJECTED by gemini-3.7-flash ("Thinking level MINIMAL is not
        # supported"), so "low" is the safe floor across models.
        self._reasoning_effort = os.environ.get("GEMINI_REASONING_EFFORT", "low").strip()

    @property
    def requires_network(self) -> bool:
        return True

    def chat_model(self):
        if not self._api_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to .env (get one at "
                "https://aistudio.google.com/apikey), or select the local "
                "provider with LLM_PROVIDER=ollama."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as err:
            raise ProviderError(
                "langchain-google-genai is not installed. It is an optional "
                "development dependency; the air-gapped deployment does not need "
                "it. Install with: pip install langchain-google-genai"
            ) from err

        extra: dict[str, object] = {}
        if self._reasoning_effort and self._reasoning_effort.lower() != "default":
            extra["reasoning_effort"] = self._reasoning_effort

        return ChatGoogleGenerativeAI(
            model=self._model,
            google_api_key=self._api_key,
            **extra,
            # Roleplay wants variation, unlike an analytics agent where 0
            # would be right: an operator who phrases things identically
            # every time stops feeling like a person.
            temperature=0.7,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def is_unavailable_error(self, err: Exception) -> bool:
        """True only for a RECOGNIZED transient failure.

        A real bug -- bad request, auth failure, unknown model name -- must
        propagate rather than being disguised as an outage by a fallback
        loop. Gemini's free tier rate-limits readily, so 429 handling here
        is load-bearing rather than theoretical.
        """
        try:
            from google.genai.errors import ClientError, ServerError
            if isinstance(err, ClientError):
                # 429 is a rate limit: transient. Other 4xx are real bugs.
                return getattr(err, "code", None) == 429
            if isinstance(err, ServerError):
                return True
        except ImportError:
            pass

        message = str(err).lower()
        if any(fatal in message for fatal in (
            "api key not valid", "permission denied", "invalid argument",
            "not found", "401", "403", "400", "404",
        )):
            return False
        return any(marker in message for marker in (
            "rate limit", "quota", "429", "resource exhausted",
            "timeout", "timed out", "deadline", "unavailable",
            "503", "502", "500", "internal error",
        ))
