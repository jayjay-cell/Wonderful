"""Numeric provenance: every number spoken must come from a tool (FR-D1).

This is the test that turns "the model might invent a figure" from an
unfalsifiable worry into something that fails CI.

Why it earns its place: a hallucinated fuel reading does not merely look
wrong, it teaches a wrong habit, which is worse than no training at all. A
trainee who learns to trust a number that was never measured has been
actively mistrained.

KNOWN LIMITATION, stated rather than hidden: this checks digits. A number
spelled out in words ("ארבעים דקות") is not caught. Closing that gap
would need Hebrew number-word parsing, which is its own project; the
mitigation meanwhile is that the system prompt pushes hard toward reading
values from tools, and radio brevity favours digits anyway.
"""

from __future__ import annotations

import json
import re

import pytest

from core.mission import load_mission_dict
from core.state import MissionStateEngine
from tools.mission_tools import build_mission_tools


def numbers_in(text: str, callsigns: list[str] | None = None) -> set[str]:
    """Numeric literals in a piece of text, normalized.

    Trailing '.0' is stripped so "70" and "70.0" compare equal -- the
    engine's float and the counterpart's spoken form are the same fact.

    Callsigns are removed FIRST. A callsign like "נחשון 3" contains a
    digit that is part of the counterpart's own name, not a reading, and
    counting it as an unexplained number would make every single
    transmission fail the check -- the sort of false positive that gets a
    useful test deleted.
    """
    for callsign in callsigns or []:
        text = text.replace(callsign, " ")

    found = set()
    for raw in re.findall(r"\d+(?:\.\d+)?", text):
        found.add(raw[:-2] if raw.endswith(".0") else raw)
    return found


def spoken_numbers(text: str, mission) -> set[str]:
    """Numbers in a transmission, excluding this mission's callsigns."""
    callsigns = [
        mission.procedure.callsigns.counterpart,
        mission.procedure.callsigns.trainee,
    ]
    return numbers_in(text, callsigns=callsigns)


def numbers_available_from_tools(tool_results: list[object]) -> set[str]:
    """Every number a turn's tool results made available.

    Includes rounded forms, because an operator says "a hundred and
    seventy-six" for 176.33 and that must count as provenance rather than
    as invention.
    """
    available: set[str] = set()
    for result in tool_results:
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        for raw in re.findall(r"\d+(?:\.\d+)?", serialized):
            value = float(raw)
            available.add(raw[:-2] if raw.endswith(".0") else raw)
            available.add(str(int(value)))             # 176.3 -> "176"
            available.add(str(round(value)))
            available.add(str(round(value, 1)))
            if value >= 100:                            # 12000 -> "12"
                available.add(str(int(value // 1000)))
    return available


@pytest.fixture
def mission(mission_with_everything):
    return load_mission_dict(mission_with_everything)


class TestProvenanceChecker:
    """The checker itself must be correct before it can police anything."""

    def test_extracts_digits(self) -> None:
        assert numbers_in("fuel 176.3, alt 12000") == {"176.3", "12000"}

    def test_normalizes_trailing_zero(self) -> None:
        assert numbers_in("70.0 lb") == {"70"}

    def test_callsign_digits_are_not_treated_as_readings(self) -> None:
        """Otherwise every transmission fails the check, and a test that
        always fails is a test that gets deleted."""
        assert numbers_in("נחשון 3, נותרו 70", callsigns=["נחשון 3"]) == {"70"}

    def test_accepts_rounded_report(self) -> None:
        """The engine says 176.33; the operator says 176. Provenance holds."""
        available = numbers_available_from_tools([{"readings": [{"value": 176.33}]}])
        assert "176" in available

    def test_rejects_an_invented_number(self) -> None:
        available = numbers_available_from_tools([{"readings": [{"value": 176.33}]}])
        assert "42" not in available


class TestEveryReportedNumberIsTraceable:
    def test_tool_results_cover_a_truthful_report(self, mission) -> None:
        engine = MissionStateEngine(mission)
        engine.advance_to(1800.0)
        tools = {t.name: t for t in build_mission_tools(mission, engine)}

        result = tools["read_state"].invoke({})
        # Uses this mission's own callsigns, so the check exercises
        # callsign exclusion against real config rather than hardcoded text.
        spoken = (f"{mission.procedure.callsigns.trainee}, "
                  f"{mission.procedure.callsigns.counterpart}, "
                  f"נותרו 70 ליטר, זמן שהייה 70 דקות.")

        unexplained = spoken_numbers(spoken, mission) - numbers_available_from_tools([result])
        assert not unexplained, f"numbers with no tool provenance: {unexplained}"

    def test_an_invented_number_is_detected(self, mission) -> None:
        """The check must actually fail when it should -- otherwise it is
        decoration."""
        engine = MissionStateEngine(mission)
        tools = {t.name: t for t in build_mission_tools(mission, engine)}
        result = tools["read_state"].invoke({})

        spoken = "נותרו 999 ליטר."
        unexplained = spoken_numbers(spoken, mission) - numbers_available_from_tools([result])
        assert "999" in unexplained

    def test_no_tool_call_means_no_number_is_defensible(self, mission) -> None:
        """A turn with no tool call cannot justify any figure at all."""
        spoken = "נותרו 40 דקות."
        assert spoken_numbers(spoken, mission) - numbers_available_from_tools([]) == {"40"}

    def test_derived_values_count_as_provenance(self, mission) -> None:
        """Endurance comes from the engine, not from the model's
        arithmetic, so quoting it is legitimate."""
        engine = MissionStateEngine(mission)
        tools = {t.name: t for t in build_mission_tools(mission, engine)}
        result = tools["read_state"].invoke({"parameter_ids": ["endurance_min"]})
        assert "100" in numbers_available_from_tools([result])

    def test_hidden_state_is_not_available_as_provenance(self, mission) -> None:
        """A number the counterpart should not know must not become
        quotable just because it exists in the simulation."""
        engine = MissionStateEngine(mission)
        tools = {t.name: t for t in build_mission_tools(mission, engine)}
        result = tools["read_state"].invoke({})
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        assert "hidden truth" not in serialized
