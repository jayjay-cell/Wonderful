"""The agent loop: LangChain's create_agent wired to the mission tools.

API NOTE -- READ BEFORE COPYING FROM AN OLDER PROJECT. The installed
langchain is 1.4.0, where the constructor is:

    from langchain.agents import create_agent
    create_agent(model, tools=..., system_prompt=..., middleware=...)

The airport project at C:\\dev\\Wonderful Airport Investment Intelligence
Agent uses `create_react_agent(prompt=...)` from langgraph.prebuilt, which
is DEPRECATED here and raises TypeError on the `prompt=` kwarg. Verified by
introspection; see docs/ARCHITECTURE.md §9. Re-check `pip show langchain`
before changing any of this.

Three mandated behaviours are native middleware in this version, so they
are configured rather than hand-rolled:

    ModelCallLimitMiddleware  -- the step limit, with a GRACEFUL stop
                                 (better than the airport project's
                                 GraphRecursionError catch)
    ToolErrorMiddleware       -- structured tool errors to the model
    ModelFallbackMiddleware   -- provider fallback

MEMORY: the full conversation is replayed to the model every turn
(NFR-6). No summarization step, because a summary is one more place the
record can drift from what was actually said -- and in a training
simulator, "what was actually said" is the thing being assessed.

SYSTEM PROMPT: passed via `system_prompt=`, never prepended to the
persisted message list. Prepending it manually and then saving the
returned list re-adds it every turn -- an unbounded duplicate-system-
message bug the airport project hit and documented. _strip_system_messages
below is the second line of defence (FR-G8).
"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
)

from obs.logging import get_logger

logger = get_logger("agent.loop")

# The real, enforced ceiling on model/tool steps in one turn (FR-G1).
# Reaching it ends the turn gracefully with whatever the model has, rather
# than raising -- see exit_behavior below.
DEFAULT_MAX_STEPS = 8


def max_steps() -> int:
    """Step limit: env var, else the default.

    Env-configurable so it can be lowered in testing without touching a
    mission file.
    """
    from_env = os.environ.get("AGENT_MAX_STEPS", "").strip()
    if from_env.isdigit() and int(from_env) > 0:
        return int(from_env)
    return DEFAULT_MAX_STEPS


def _on_tool_error(error: Exception, *args: Any, **kwargs: Any) -> str:
    """Turn an unexpected tool exception into something sayable.

    Tools return ToolError for EXPECTED failures (out of range, not
    visible), so reaching here means a genuine defect. The model still
    gets a usable in-character fact rather than a stack trace (FR-G3), and
    the real exception is logged server-side with a typed code only
    (FR-G5).
    """
    logger.error("tool.unexpected_error", error_code=type(error).__name__, status="error")
    return (
        "error: TOOL_FAILED — that reading is not coming through right now. "
        "Report that you cannot get it rather than guessing a value."
    )


def build_agent(
    model: Any,
    tools: list[Any],
    system_prompt: str,
    step_limit: int | None = None,
):
    """Construct the agent for one turn.

    Tools and prompt are passed in rather than built here: they close over
    the live exercise, and the caller (sim/turns.py) is what knows it.
    Keeping this layer ignorant of the domain is what lets every channel
    share one turn implementation.

    `model` is injected so the provider layer owns provider selection and
    tests can pass a fake model with no network.
    """
    limit = step_limit if step_limit is not None else max_steps()

    return create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            # exit_behavior="end" stops cleanly at the limit and returns
            # what the model has so far. "error" would raise and force
            # every caller to catch it -- a worse default mid-exercise,
            # where a degraded reply beats a dropped transmission.
            ModelCallLimitMiddleware(run_limit=limit, exit_behavior="end"),
            ToolErrorMiddleware(on_error=_on_tool_error),
        ],
    )


def strip_system_messages(messages: list[Any]) -> list[Any]:
    """Drop system-role messages before persisting conversation state.

    The system prompt is supplied fresh via `system_prompt=` on every call
    and must never be stored (FR-G8). This is the defensive second layer;
    the first is simply not prepending it.
    """
    def is_system(message: Any) -> bool:
        if isinstance(message, dict):
            return message.get("role") == "system"
        return getattr(message, "type", None) == "system"

    return [m for m in messages if not is_system(m)]


def extract_reply_text(messages: list[Any]) -> str:
    """Pull the counterpart's spoken text out of the final message.

    Handles both dict-shaped and object-shaped messages, and content that
    arrives as a list of blocks rather than a plain string -- which is
    normal for models that interleave reasoning with text.
    """
    if not messages:
        return ""

    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()

    return (content or "").strip()
