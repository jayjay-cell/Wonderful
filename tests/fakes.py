"""Fake models for testing the agent loop with no network and no API key.

Lets the whole agent layer be verified offline (NFR-2), and -- more
importantly -- lets failure paths be triggered ON DEMAND rather than by
waiting for a real provider to misbehave (NFR-4). A step-limit test that
depends on a real model choosing to loop is not a test.

Built on langchain's FakeMessagesListChatModel where a scripted reply is
enough, and on a hand-rolled model where the test needs to control tool
calls precisely.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Returns a fixed sequence of replies, one per invocation.

    Each entry is either a string (plain text reply) or a list of tool-call
    dicts (the model deciding to call tools). This is what makes a
    multi-step turn testable deterministically.
    """

    replies: list[Any] = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        """LangChain's model-type tag."""
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next scripted reply, repeating the last once the script runs out."""
        index = min(self.call_count, len(self.replies) - 1) if self.replies else 0
        self.call_count += 1
        reply = self.replies[index] if self.replies else ""

        if isinstance(reply, list):
            message = AIMessage(content="", tool_calls=[
                {"name": c["name"], "args": c.get("args", {}),
                 "id": c.get("id", f"call_{self.call_count}")}
                for c in reply
            ])
        else:
            message = AIMessage(content=reply)

        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        """Tool binding is a no-op: this model's behaviour is scripted, so
        it does not need the schemas. Required because create_agent binds
        tools to whatever model it is given."""
        return self


class NeverStopsCallingToolsModel(BaseChatModel):
    """Always asks for another tool call, never produces a final answer.

    Exists to prove the step limit is genuinely enforced (FR-G1/G2).
    Without a model like this, that test would depend on a real model
    happening to loop -- which it mostly does not, so the limit would go
    unverified until it mattered.
    """

    tool_name: str = "read_state"
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        """LangChain's model-type tag."""
        return "never_stops"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Always return another tool call, never a final answer."""
        self.call_count += 1
        message = AIMessage(content="", tool_calls=[{
            "name": self.tool_name,
            "args": {},
            "id": f"loop_{self.call_count}",
        }])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "NeverStopsCallingToolsModel":
        """Ignore the tools and return self -- this model never varies its reply."""
        return self


class FailingModel(BaseChatModel):
    """Raises a chosen exception on every call.

    `error` is configurable so a test can distinguish a transient failure
    (which should fall back) from a real bug (which must propagate) --
    the distinction the provider layer depends on.
    """

    error: Exception = RuntimeError("provider unavailable")

    @property
    def _llm_type(self) -> str:
        """LangChain's model-type tag."""
        return "failing"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        """Always raise, standing in for a provider outage."""
        raise self.error

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FailingModel":
        """Ignore the tools and return self."""
        return self


class EmptyReplyModel(BaseChatModel):
    """Returns an empty reply.

    A silent counterpart is indistinguishable from a broken simulator, so
    the session layer must treat this as a recoverable failure rather than
    passing the silence through.
    """

    @property
    def _llm_type(self) -> str:
        """LangChain's model-type tag."""
        return "empty"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None,
                  **kwargs: Any) -> ChatResult:
        """Return whitespace, standing in for a model that says nothing."""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="   "))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "EmptyReplyModel":
        """Ignore the tools and return self."""
        return self


def numbers_in(text: str) -> set[str]:
    """Extract numeric literals from text, for the provenance check.

    Hebrew numerals are written as digits, so digit extraction is
    sufficient here. Spelled-out numbers ("ארבעים") are not caught -- a
    known limitation, documented rather than hidden, and the reason the
    prompt pushes hard toward reading values from tools.
    """
    import re
    return set(re.findall(r"\d+(?:\.\d+)?", text))
