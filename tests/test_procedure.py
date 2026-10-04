"""Procedure checking tests.

The asymmetry here is deliberate and worth stating: a FALSE challenge is
worse than a MISSED one.

If the counterpart challenges a correct transmission, the trainee knows
they were right, concludes the simulator is arbitrary, and stops taking
the assessment seriously. If it misses a sloppy call, the trainee merely
gets away with something once. So the checker is lenient, and these tests
pin that leniency down rather than treating it as an accident.
"""

from __future__ import annotations

import pytest

from core.mission import load_mission_dict
from core.procedure import brevity_terms, check_transmission


@pytest.fixture
def mission(mission_with_everything):
    """The shared fixture plus a callsign rule.

    Added here rather than in conftest because procedure rules are
    mission-authored: a mission with no callsign rule should NOT flag a
    missing callsign, and tests elsewhere depend on that.
    """
    mission_with_everything["procedure"]["rules"].append(
        {"id": "callsign_required", "applies_to": "first_transmission",
         "on_violation": "challenge"}
    )
    return load_mission_dict(mission_with_everything)


class TestCallsignRule:
    def test_first_transmission_without_callsign_is_flagged(self, mission) -> None:
        report = check_transmission(mission, "what's your fuel",
                                    is_first_transmission=True)
        assert not report.compliant
        assert report.violations[0].kind == "missing_callsign"

    def test_first_transmission_with_callsign_is_compliant(self, mission) -> None:
        report = check_transmission(mission, "Op, HQ, report fuel",
                                    is_first_transmission=True)
        assert report.compliant

    def test_later_transmissions_need_no_callsign(self, mission) -> None:
        """Requiring it on every message would flag normal mid-conversation
        traffic, which no real net does."""
        report = check_transmission(mission, "and your altitude",
                                    is_first_transmission=False)
        assert report.compliant

    def test_multiword_callsign_is_recognized(self, mission_with_everything) -> None:
        """A callsign with a space and a digit will not survive naive token
        comparison, so this would silently flag every opening call."""
        mission_with_everything["procedure"]["callsigns"]["counterpart"] = "נחשון 3"
        mission = load_mission_dict(mission_with_everything)
        report = check_transmission(mission, "נחשון 3, מפקדה, דווח דלק",
                                    is_first_transmission=True)
        assert report.compliant


class TestReadbackRule:
    def test_missing_readback_is_flagged(self, mission) -> None:
        report = check_transmission(mission, "roger",
                                    awaiting_readback_for=["alt"])
        assert not report.compliant
        assert report.violations[0].kind == "missing_readback"

    def test_numeric_readback_is_accepted(self, mission) -> None:
        """A real readback is often just the number, with context implied."""
        report = check_transmission(mission, "roger, 5000",
                                    awaiting_readback_for=["alt"])
        assert report.compliant

    def test_named_readback_is_accepted(self, mission) -> None:
        report = check_transmission(mission, "roger, alt 5000",
                                    awaiting_readback_for=["alt"])
        assert report.compliant

    def test_no_obligation_means_no_violation(self, mission) -> None:
        report = check_transmission(mission, "roger", awaiting_readback_for=[])
        assert report.compliant

    def test_obligation_for_another_parameter_is_not_flagged(self, mission) -> None:
        """The rule only covers 'alt', so an outstanding 'fuel' readback
        must not trip it."""
        report = check_transmission(mission, "roger",
                                    awaiting_readback_for=["fuel"])
        assert report.compliant


class TestLeniency:
    """A false challenge destroys trust faster than a missed one."""

    def test_empty_transmission_is_not_flagged(self, mission) -> None:
        assert check_transmission(mission, "").compliant
        assert check_transmission(mission, "   ").compliant

    def test_punctuation_does_not_defeat_matching(self, mission) -> None:
        report = check_transmission(mission, "Op -- HQ. Report fuel!",
                                    is_first_transmission=True)
        assert report.compliant

    def test_ignore_rule_never_flags(self, mission_with_everything) -> None:
        mission_with_everything["procedure"]["rules"] = [
            {"id": "rb", "requires_readback_for": ["alt"], "on_violation": "ignore"}
        ]
        mission = load_mission_dict(mission_with_everything)
        report = check_transmission(mission, "roger", awaiting_readback_for=["alt"])
        assert report.compliant

    def test_accept_with_note_flags_but_does_not_challenge(self, mission_with_everything) -> None:
        """A violation worth recording for the debrief but not worth
        interrupting the exercise over."""
        mission_with_everything["procedure"]["rules"] = [
            {"id": "rb", "requires_readback_for": ["alt"],
             "on_violation": "accept_with_note"}
        ]
        mission = load_mission_dict(mission_with_everything)
        report = check_transmission(mission, "roger", awaiting_readback_for=["alt"])
        assert not report.compliant
        assert not report.should_challenge


class TestBrevityDictionary:
    def test_terms_are_exposed_for_the_prompt(self, mission) -> None:
        terms = brevity_terms(mission)
        assert terms["ROGER"] == "understood"

    def test_empty_dictionary_is_fine(self, minimal_mission_dict) -> None:
        mission = load_mission_dict(minimal_mission_dict)
        assert brevity_terms(mission) == {}
