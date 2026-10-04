"""Validate and inspect an exercise without a model or a network.

    python -m sim.cli --mission missions/synthetic_he.yaml
    python -m sim.cli --mission missions/synthetic_he.yaml --walk 20

`--walk N` steps a virtual clock through N minutes and prints what the
operator learns, when. This is how an author checks a timeline before
anyone speaks to it: whether the beats land where intended, whether a
report would be owed, and whether the handover sits where expected.

Needs no API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mission import MissionError, load_context, load_mission  # noqa: E402
from core.timeline import EventType, Timeline  # noqa: E402
from core.timeline_import import TimelineError, load_timeline  # noqa: E402


def _force_utf8() -> None:
    """UTF-8 regardless of the Windows console codepage.

    Not cosmetic: the content is Hebrew, and cp1252 raises on the first
    Hebrew character, which would make this tool unusable for its purpose.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _stamp(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def _summary(mission, timeline: Timeline, context_chars: int) -> None:
    print(f"\n  {mission.title}")
    print(f"  {'-' * 68}")
    print(f"  id           {mission.id}  (v{mission.version}, {mission.language})")
    print(f"  callsigns    operator {mission.callsigns.operator} · "
          f"trainee {mission.callsigns.trainee}"
          + (f" · controller {mission.callsigns.controller}"
             if mission.callsigns.controller else ""))
    duration = mission.duration_seconds or timeline.duration
    print(f"  duration     {_stamp(duration)}"
          + ("" if mission.duration_seconds else "  (from timeline)"))
    print(f"  behaviour    initiative {mission.behaviour.initiative} · "
          f"challenge {mission.behaviour.challenge} · "
          f"verbosity {mission.behaviour.verbosity}")

    facts = mission.initial_facts
    print(f"  initial facts {len(facts)}"
          + (f"  ({', '.join(facts)})" if facts else "  (none authored)"))
    if mission.hidden_facts:
        print(f"  hidden       {', '.join(mission.hidden_facts)}")

    print(f"  context      {len(mission.context_files)} files, {context_chars} chars")
    print(f"  timeline     {len(timeline)} events")

    counts: dict[str, int] = {}
    for event in timeline:
        counts[event.reporting_policy.value] = counts.get(event.reporting_policy.value, 0) + 1
    print("               " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    handovers = [e for e in timeline if e.event_type is EventType.HANDOVER]
    if handovers:
        for handover in handovers:
            end = _stamp(handover.end_time) if handover.end_time else "?"
            print(f"  handover     {_stamp(handover.start_time)} – {end}")

    if mission.private.solution or mission.private.debrief_points:
        print(f"  private      present, hidden from the operator")
    print()


def _events(timeline: Timeline) -> None:
    print(f"  Timeline")
    print(f"  {'-' * 68}")
    print(f"  {'time':<7} {'type':<11} {'report':<13} {'pri':<7} id")
    for event in timeline:
        window = _stamp(event.start_time)
        if event.end_time:
            window += f"–{_stamp(event.end_time)}"
        print(f"  {window:<7} {event.event_type.value:<11} "
              f"{event.reporting_policy.value:<13} {event.priority.value:<7} "
              f"{event.event_id}")
        if event.tags or event.entity_ids:
            marks = list(event.tags) + [f"#{e}" for e in event.entity_ids]
            print(f"          tags: {', '.join(marks)}")
    print()


def _walk(mission, timeline: Timeline, minutes: int) -> None:
    """Step a virtual clock and show what the operator learns.

    Uses the real Exercise, so this exercises the same revelation code
    production does -- an author checking a timeline here is checking the
    behaviour they will get.
    """
    from sim.clock import VirtualClock
    from sim.exercise import Exercise

    clock = VirtualClock()
    exercise = Exercise(mission, timeline, model=None, clock=clock)
    exercise.mark_ready()
    clock.set(0.0)
    exercise.start()

    print(f"  Walkthrough — {minutes} min of mission time")
    print(f"  {'-' * 68}")

    step = 15.0
    elapsed = 0.0
    while elapsed <= minutes * 60:
        clock.set(elapsed)
        newly = exercise.advance()
        for event in newly:
            owed = exercise.ledger.should_report(event)
            flag = "REPORT" if owed else "      "
            label = event.operator_information or f"({event.event_type.value})"
            print(f"  {_stamp(elapsed)}  {flag}  {label[:54]}")
        elapsed += step

    clock.set(minutes * 60)
    print()
    print(f"  At {_stamp(minutes * 60)}:")
    facts = exercise.current_information()
    for key, value in facts.items():
        print(f"    {key}: {value}")
    current = exercise.revealed_observations(current_only=True)
    if current:
        print("    currently: " + "; ".join(o["information"] for o in current))
    pending = [p.event.event_id for p in exercise.pending_reports()]
    if pending:
        print(f"    reports owed: {', '.join(pending)}")
    if exercise.handover_active() is not None:
        print("    crew is mid-handover")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sim.cli",
        description="Validate an exercise. No model or network required.",
    )
    parser.add_argument("--mission", required=True)
    parser.add_argument("--timeline", default=None,
                        help="override the mission's timeline_file")
    parser.add_argument("--events", action="store_true",
                        help="list every timeline event")
    parser.add_argument("--walk", type=int, metavar="MINUTES",
                        help="step a virtual clock and show what is revealed")
    args = parser.parse_args(argv)

    _force_utf8()

    try:
        mission = load_mission(args.mission)
    except MissionError as err:
        print(f"\n  x mission did not load\n    {err}\n", file=sys.stderr)
        return 1

    timeline_name = args.timeline or mission.timeline_file
    if not timeline_name:
        print("\n  x no timeline: set timeline_file in the mission, or pass "
              "--timeline\n", file=sys.stderr)
        return 1

    timeline_path = Path(timeline_name)
    if not timeline_path.is_absolute():
        timeline_path = ROOT / "timelines" / timeline_name
    try:
        timeline = load_timeline(timeline_path)
    except TimelineError as err:
        print(f"\n  x timeline did not load\n    {err}\n", file=sys.stderr)
        return 1

    try:
        context = load_context(mission.context_files, ROOT / "context")
    except MissionError as err:
        print(f"\n  x context did not load\n    {err}\n", file=sys.stderr)
        return 1

    print(f"\n  ok  all three sources loaded")
    _summary(mission, timeline, len(context))

    if args.events:
        _events(timeline)
    if args.walk:
        _walk(mission, timeline, args.walk)

    placeholders = Path(args.mission).read_text(encoding="utf-8").count("SYNTHETIC")
    if placeholders:
        print(f"  !  {placeholders} SYNTHETIC marker(s) in the mission file — "
              f"invented values awaiting replacement.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
