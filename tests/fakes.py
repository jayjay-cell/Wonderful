"""A fake model, so the whole suite runs with no network and no API key.

ScriptedModel returns replies from a list, one per call. That is enough
for every test here: what is being verified is the engine's bookkeeping --
timing, revelation, reports, persistence -- not the model's wording.
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
