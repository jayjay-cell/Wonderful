"""Terminal chat with the counterpart -- the step-3 deliverable.

    python -m sim.chat --mission missions/uav_operator_he.yaml
    python -m sim.chat --mission missions/uav_operator_he.yaml --dry-run
    python -m sim.chat --mission missions/uav_operator_he.yaml --speed 60

No realism layer yet: replies arrive whole and instantly. Stalls, pauses
and interruption are step 4. The question this tool answers is narrower
and comes first: does the counterpart sound like an operator, and does it
read its numbers from the engine instead of inventing them?

--dry-run needs no API key: it prints the assembled system prompt and the
tools the model would be given. Useful for checking the persona and the
knowledge boundary before spending a single token.

--speed compresses mission time, so a 45-minute fuel situation can be
reached in a few minutes of real conversation. Without it, testing a
bingo-fuel exchange would mean sitting at a terminal for most of an hour.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv

# Must precede any module that reads os.environ for a key: uvicorn and a
# bare python invocation do not load .env themselves.
load_dotenv()

from core.mission import MissionError, load_mission  # noqa: E402
from obs.logging import configure as configure_logging  # noqa: E402
from providers.base import ProviderError, build_llm_provider  # noqa: E402
from sim.clock import Clock  # noqa: E402
from sim.session import Session  # noqa: E402


def _force_utf8_output() -> None:
    """UTF-8 regardless of the Windows console codepage.

    Not cosmetic: the conversation is in Hebrew, and cp1252 raises on the
    first Hebrew character."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class AcceleratedClock:
    """Mission time running faster than real time.

    Exists so a fuel-emergency scenario is reachable in a test session.
    Implements the same Clock protocol, so nothing downstream knows the
    difference.
    """

    def __init__(self, multiplier: float = 1.0) -> None:
        import time
        self._time = time
        self._started = time.monotonic()
        self._multiplier = multiplier

    def now(self) -> float:
        return (self._time.monotonic() - self._started) * self._multiplier


def _print_briefing(mission) -> None:
    print()
    print(f"  {mission.title}")
    print(f"  {'─' * 68}")
    print(f"  {mission.setting.place}")
    print(f"  You are: {mission.setting.trainee_role}")
    print(f"  Callsigns: you = {mission.procedure.callsigns.trainee}, "
          f"them = {mission.procedure.callsigns.counterpart}")
    if mission.setting.briefing:
        print()
        for line in mission.setting.briefing.strip().splitlines():
            print(f"  {line}")
    print()
    print("  Commands: /state  /quit")
    print(f"  {'─' * 68}")
    print()


def _print_state(session: Session) -> None:
    """Show the full simulation state -- the trainer's view.

    Deliberately the UNFILTERED view, including anything hidden from the
    counterpart: as the trainer you need to see what it cannot, in order
    to judge whether its answers were honest.
    """
    session.engine.advance_to(session.clock.now())
    print(f"\n  [trainer view — T+{int(session.engine.mission_seconds // 60):02d}:"
          f"{int(session.engine.mission_seconds % 60):02d}]")
    for row in session.engine.describe_for_prompt(for_persona=False):
        value = row["value"]
        if isinstance(value, float):
            value = round(value, 1)
        unit = f" {row['unit']}" if row.get("unit") else ""
        transit = (f"  → {row['in_transit_to']}"
                   if row.get("in_transit_to") is not None else "")
        hidden = "" if row["id"] in session.engine.snapshot(for_persona=True) else "   (hidden)"
        print(f"    {row['name']}: {value}{unit}{transit}{hidden}")
    print()


def _dry_run(mission) -> int:
    """Print the assembled prompt and tool list. No model, no key."""
    from agent.prompts import build_system_prompt
    from core.state import MissionStateEngine
    from tools.mission_tools import build_mission_tools

    engine = MissionStateEngine(mission)
    print("\n" + "=" * 72)
    print("  SYSTEM PROMPT (assembled from the mission file)")
    print("=" * 72 + "\n")
    print(build_system_prompt(mission, engine))

    print("\n" + "=" * 72)
    print("  TOOLS")
    print("=" * 72 + "\n")
    for tool in build_mission_tools(mission, engine):
        first_line = (tool.description or "").strip().splitlines()[0]
        print(f"  {tool.name}  —  {first_line}")

    print("\n" + "=" * 72)
    print("  WHAT THE COUNTERPART CAN AND CANNOT SEE")
    print("=" * 72 + "\n")
    visible = set(engine.snapshot(for_persona=True))
    for parameter in mission.parameters:
        mark = "visible" if parameter.id in visible else "HIDDEN "
        print(f"  [{mark}]  {parameter.display_name}  ({parameter.id})")
    print()
    return 0


async def _chat_loop(session: Session, mission) -> int:
    counterpart = mission.procedure.callsigns.counterpart
    trainee = mission.procedure.callsigns.trainee

    while True:
        try:
            line = input(f"  {trainee} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  session ended\n")
            return 0

        if not line:
            continue
        if line in {"/quit", "/exit"}:
            print("\n  session ended\n")
            return 0
        if line == "/state":
            _print_state(session)
            continue

        result = await session.run_trainee_turn(line)

        print(f"\n  {counterpart} > {result.text}")
        # Shown so you can verify numbers were READ rather than invented --
        # the central question this tool exists to answer.
        detail = f"      [{result.latency_ms}ms"
        if result.tool_calls:
            detail += f", tools: {', '.join(result.tool_calls)}"
        else:
            detail += ", no tools called"
        if result.failed:
            detail += f", FAILED: {result.failure_code}"
        print(detail + "]\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sim.chat",
        description="Talk to a mission's counterpart in the terminal.",
    )
    parser.add_argument("--mission", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the system prompt and tools, then exit (no API key needed)")
    parser.add_argument("--speed", type=float, default=1.0, metavar="N",
                        help="mission time multiplier, e.g. 60 for one minute per second")
    parser.add_argument("--provider", default=None,
                        help="override LLM_PROVIDER for this run")
    args = parser.parse_args(argv)

    _force_utf8_output()
    configure_logging()

    try:
        mission = load_mission(Path(args.mission))
    except MissionError as err:
        print(f"\n  ✗ mission did not load\n    {err}\n", file=sys.stderr)
        return 1

    if args.dry_run:
        return _dry_run(mission)

    try:
        provider = build_llm_provider(args.provider)
        model = provider.chat_model()
    except ProviderError as err:
        print(f"\n  ✗ {err}\n", file=sys.stderr)
        print("    Tip: --dry-run inspects the prompt and tools with no API key.\n",
              file=sys.stderr)
        return 1

    clock: Clock = AcceleratedClock(args.speed)
    session = Session(mission, model, secrets.token_urlsafe(8), clock=clock)

    _print_briefing(mission)
    if args.speed != 1.0:
        print(f"  mission time running at {args.speed:g}x\n")
    print(f"  provider: {provider.name}\n")

    return asyncio.run(_chat_loop(session, mission))


if __name__ == "__main__":
    raise SystemExit(main())
