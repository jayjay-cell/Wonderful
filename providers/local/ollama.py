"""Local LLM via Ollama -- the air-gapped deployment path.

Reached over plain HTTP on localhost, so it needs no vendor SDK and no
internet. This is the provider that will actually run in the closed
network (FR-F5).

MODEL CHOICE IS THE PROJECT'S BIGGEST OPEN RISK. Local models are
substantially weaker at sustained roleplay than frontier cloud models, and
Hebrew narrows the field further. A small model may break character,
flatten the persona, or lose fluency mid-session. No amount of
architecture fixes that -- which is why docs/PLAN.md tests a candidate
local model at step 3 rather than step 7, while there is still time to
change which model is targeted.

OLLAMA_MODEL is deliberately NOT defaulted. A silent default would mean
discovering the wrong model is in use halfway through a training session;
failing at startup with a list of what is installed is far cheaper.
"""

from __future__ import annotations

import os

from providers.base import ProviderError

# Keep the per-request timeout generous relative to the cloud provider: a
# local model on modest hardware is slower, and a timeout here means a
# dropped turn rather than a retry.
_REQUEST_TIMEOUT_SECONDS = 120.0


class OllamaProvider:
    """LLMProvider backed by a local Ollama server."""

    name = "ollama"

    def __init__(self) -> None:
        # Blank treated as unset, as with the model name elsewhere.
        """Read the model name and host from the environment."""
        self._base_url = (
            os.environ.get("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
        ).rstrip("/")
        self._model = os.environ.get("OLLAMA_MODEL", "").strip()

    @property
    def requires_network(self) -> bool:
        """False: localhost only, which is why the offline guard permits
        loopback connections."""
        return False

    def chat_model(self):
        """Build the chat model; raises ProviderError if no model is chosen."""
        if not self._model:
            raise ProviderError(
                "OLLAMA_MODEL is not set. Choose a model explicitly rather than "
                "relying on a default -- the wrong model silently in use is worse "
                "than a startup failure. List what is installed with: ollama list"
            )
        try:
            from langchain_ollama import ChatOllama
        except ImportError as err:
            raise ProviderError(
                "langchain-ollama is not installed. In a closed network, install "
                "it from the vendored wheels: "
                "pip install --no-index --find-links vendor/ langchain-ollama"
            ) from err

        return ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=0.7,   # roleplay wants variation, unlike an analytics agent
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    def is_unavailable_error(self, err: Exception) -> bool:
        """Only a RECOGNIZED transient failure should trigger fallback.

        For a local server the realistic transient cases are "not running
        yet" and "timed out under load". A 404 (model not pulled) is
        deliberately NOT transient: it is a configuration error, and
        treating it as an outage would hide the real cause behind a
        confusing fallback.
        """
        message = str(err).lower()
        transient_markers = (
            "connection refused", "connection error", "timeout", "timed out",
            "cannot connect", "max retries", "503", "502",
        )
        return any(marker in message for marker in transient_markers)

    def health_check_url(self) -> str:
        """For step 7's air-gap verification: a cheap liveness probe that
        avoids loading the model."""
        return f"{self._base_url}/api/tags"
