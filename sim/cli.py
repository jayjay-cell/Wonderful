"""Mission inspection CLI -- the step-2 deliverable.

Needs no API key, no model and no network. Two jobs:

    --validate        does this mission file load cleanly?
    --timeline N      print N minutes of mission state

The timeline is how a mission author checks the numbers are plausible
before any model is involved. Fuel burn, endurance, a scheduled comms
degradation and a commanded climb are all visible as plain arithmetic, so
"this platform behaves implausibly" is caught in seconds rather than
discovered by a trainee mid-session.

Usage:
    python -m sim.cli --mission missions/uav_operator_he.yaml --validate
    python -m sim.cli --mission missions/uav_operator_he.yaml --timeline 10
    python -m sim.cli --mission missions/uav_operator_he.yaml --timeline 20 --step 2
    python -m sim.cli --mission missions/uav_operator_he.yaml --show-persona-view
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.mission import MissionError, load_mission
from core.models import Mission
from core.state import MissionStateEngine


def _force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 regardless of the Windows console codepage.

    Not cosmetic: mission content is Hebrew, and the default cp1252
    codepage on a Windows console raises UnicodeEncodeError on the first
    Hebrew character -- which would make this tool unusable for its actual
    purpose. Must run before any print().
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # A redirected or wrapped stream may refuse; "replace"
                # keeps output readable rather than crashing the tool.
                pass


def _format_value(value: object) -> str:
    """Round floats for display only -- never for storage or arithmetic."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}" if abs(value) < 1000 else f"{value:,.0f}"
    if value is None:
        return "—"
    return str(value)


def _print_summary(mission: Mission) -> None:
    print(f"\n  {mission.title}")
    print(f"  {'─' * 68}")
    print(f"  id              {mission.id}  (version {mission.version}, lang {mission.language})")
    print(f"  purpose         {mission.setting.purpose}")
    print(f"  place           {mission.setting.place}")
    print(f"  trainee role    {mission.setting.trainee_role}")
    print(f"  counterpart     {mission.persona.name} — {mission.persona.role}")
    print(f"  parameters      {len(mission.parameters)}  ({', '.join(mission.parameter_ids())})")
    if mission.derived:
        print(f"  derived         {len(mission.derived)}  ({', '.join(mission.derived_ids())})")
    triggers = mission.triggers
    print(f"  triggers        {len(triggers.all_ids())}  "
          f"(timeline {len(triggers.timeline)}, "
          f"threshold {len(triggers.thresholds)}, idle {len(triggers.idle)})")
    hidden = [p.id for p in mission.parameters if not p.visible_to_persona]
    if hidden:
        print(f"  hidden from him {', '.join(hidden)}")
    print()


def _print_timeline(mission: Mission, minutes: int, step_minutes: int,
                    persona_view: bool) -> None:
    """Advance mission state and print it at intervals.

    No LLM and no clock: time is advanced explicitly, so this is pure
    arithmetic and reproducible.
    """
    engine = MissionStateEngine(mission)

    rows = engine.describe_for_prompt(for_persona=persona_view)
    headers = [r["name"] for r in rows]
    widths = [max(len(h), 9) for h in headers]

    view_label = "as the counterpart sees it" if persona_view else "full state"
    print(f"  Timeline — {minutes} min, every {step_minutes} min ({view_label})")
    print(f"  {'─' * 68}")
    print("  " + "TIME".ljust(9) + "".join(h.ljust(w + 2) for h, w in zip(headers, widths)))

    trigger_notes: list[str] = []
    for elapsed in range(0, minutes * 60 + 1, step_minutes * 60):
        effects = engine.advance_to(float(elapsed))
        for effect in effects:
            if effect.detail and "floor" in effect.detail or (effect.detail or "").startswith("at "):
                trigger_notes.append(
                    f"T+{elapsed // 60:02d}:{elapsed % 60:02d}  "
                    f"{effect.parameter_id} {effect.detail}"
                )

        values = engine.describe_for_prompt(for_persona=persona_view)
        stamp = f"T+{elapsed // 60:02d}:{elapsed % 60:02d}"
        cells = "".join(_format_value(v["value"]).ljust(w + 2) for v, w in zip(values, widths))
        print(f"  {stamp.ljust(9)}{cells}")

    units = [f"{r['name']} [{r.get('unit', '-')}]" for r in rows if r.get("unit")]
    if units:
        print(f"\n  units: {', '.join(units)}")

    if trigger_notes:
        print("\n  limits reached:")
        for note in dict.fromkeys(trigger_notes):
            print(f"    {note}")

    # Threshold triggers are evaluated here as a preview, so an author can
    # see which conditions a straight run would cross. Full trigger
    # handling (deferral, suppression, priority) arrives in step 5.
    if mission.triggers.thresholds:
        from core.derived import ExpressionError, evaluate_bool

        snapshot = engine.snapshot()
        fired = []
        for trigger in mission.triggers.thresholds:
            try:
                if evaluate_bool(trigger.when, snapshot):
                    fired.append(f"{trigger.id}  [{trigger.priority.value}]  {trigger.when}")
            except ExpressionError as err:
                fired.append(f"{trigger.id}  — could not evaluate: {err}")
        if fired:
            print(f"\n  threshold conditions true at T+{minutes:02d}:00:")
            for f in fired:
                print(f"    {f}")

    print()


def _print_placeholders(mission_path: Path) -> None:
    """Count the invented values still awaiting review.

    Surfaced on every validate so placeholders cannot quietly become
    permanent -- they are the values most likely to make a trainee
    disbelieve the simulation.
    """
    text = mission_path.read_text(encoding="utf-8")
    count = text.count("PLACEHOLDER")
    if count:
        print(f"  ⚠  {count} value(s) still marked PLACEHOLDER — invented, awaiting your review.")
        print(f"     grep -n PLACEHOLDER {mission_path.as_posix()}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sim.cli",
        description="Inspect a Maslul mission file. No model or network required.",
    )
    parser.add_argument("--mission", required=True, help="path to a mission YAML file")
    parser.add_argument("--validate", action="store_true",
                        help="load and cross-validate the mission, then report")
    parser.add_argument("--timeline", type=int, metavar="MINUTES",
                        help="print mission state over this many minutes")
    parser.add_argument("--step", type=int, default=5, metavar="MINUTES",
                        help="timeline interval in minutes (default 5)")
    parser.add_argument("--show-persona-view", action="store_true",
                        help="show only what the counterpart can see (the knowledge boundary)")
    args = parser.parse_args(argv)

    _force_utf8_output()
    mission_path = Path(args.mission)

    try:
        mission = load_mission(mission_path)
    except MissionError as err:
        # Loud and specific: the code, the field path and a suggestion.
        print(f"\n  ✗ mission did not load\n    {err}\n", file=sys.stderr)
        return 1

    print(f"\n  ✓ mission loaded and validated: {mission_path}")

    if args.validate or not args.timeline:
        _print_summary(mission)
        _print_placeholders(mission_path)

    if args.timeline:
        if args.step < 1:
            print("  ✗ --step must be at least 1 minute\n", file=sys.stderr)
            return 1
        _print_timeline(mission, args.timeline, args.step, args.show_persona_view)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
