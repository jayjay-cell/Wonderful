"""Mission state engine tests -- the correctness keystone (FR-D1..D7).

The properties here are what make the simulator trustworthy enough to
train on. A hallucinated or self-contradictory fuel figure does not merely
look bad: it teaches a wrong habit, which is worse than no training. So
these tests are deliberately pedantic about monotonicity, idempotence,
rejection semantics and the knowledge boundary.
"""

from __future__ import annotations

import pytest

from core.mission import load_mission_dict
from core.models import StateCommand
from core.state import MissionStateEngine


@pytest.fixture
def engine(mission_with_everything) -> MissionStateEngine:
    return MissionStateEngine(load_mission_dict(mission_with_everything))


class TestTimeAdvance:
    def test_initial_state_matches_declared_initials(self, engine) -> None:
        snapshot = engine.snapshot()
        assert snapshot["fuel"] == 100.0
        assert snapshot["mode"] == "idle"
        assert engine.mission_seconds == 0.0

    def test_linear_drain_is_exact(self, engine) -> None:
        """60 lb/hr for 30 minutes is exactly 30 lb. Fuel arithmetic being
        exact matters because the trainee is taught to reason from it."""
        engine.advance_to(1800.0)
        assert engine.snapshot()["fuel"] == pytest.approx(70.0)

    def test_advance_is_idempotent(self, engine) -> None:
        """Advancing to a time already reached changes nothing (FR-D4)."""
        engine.advance_to(600.0)
        first = engine.snapshot()
        effects = engine.advance_to(600.0)
        assert effects == []
        assert engine.snapshot() == first

    def test_many_small_steps_equal_one_jump(self, mission_with_everything) -> None:
        """Otherwise the fuel figure would depend on how often the session
        loop happened to tick -- a trainee could get a different mission by
        the server being busy."""
        stepwise = MissionStateEngine(load_mission_dict(mission_with_everything))
        for second in range(0, 601, 10):
            stepwise.advance_to(float(second))

        single = MissionStateEngine(load_mission_dict(mission_with_everything))
        single.advance_to(600.0)

        assert stepwise.snapshot()["fuel"] == pytest.approx(single.snapshot()["fuel"])
        assert stepwise.snapshot()["comms"] == single.snapshot()["comms"]

    def test_rewinding_time_is_rejected(self, engine) -> None:
        """Time running backwards means a caller bug. Absorbing it quietly
        would corrupt the numbers while hiding the cause."""
        engine.advance_to(600.0)
        with pytest.raises(ValueError, match="monotonic"):
            engine.advance_to(300.0)

    def test_drain_clamps_at_floor(self, engine) -> None:
        """Fuel reaching zero is a mission condition for a trigger to
        detect, not a negative number to propagate into every derived
        value that divides by it."""
        engine.advance_to(100 * 3600.0)
        assert engine.snapshot()["fuel"] == 0.0

    def test_scripted_keyframe_applies_at_its_time(self, engine) -> None:
        engine.advance_to(299.0)
        assert engine.snapshot()["comms"] == "good"
        engine.advance_to(301.0)
        assert engine.snapshot()["comms"] == "bad"

    def test_hold_and_stepped_ignore_time(self, engine) -> None:
        engine.advance_to(7200.0)
        assert engine.snapshot()["mode"] == "idle"


class TestCommands:
    def test_valid_command_applies(self, engine) -> None:
        result = engine.apply(StateCommand(parameter_id="mode", value="active"))
        assert result.accepted
        assert engine.snapshot()["mode"] == "active"

    def test_rejected_enum_value_mutates_nothing(self, engine) -> None:
        """FR-D3: validation happens entirely before assignment, so there
        is no partial-application path and no rollback to get wrong."""
        before = engine.snapshot()
        result = engine.apply(StateCommand(parameter_id="mode", value="turbo"))
        assert not result.accepted
        assert result.reason_code == "INVALID_VALUE"
        assert engine.snapshot() == before

    def test_rejected_out_of_range_mutates_nothing(self, engine) -> None:
        before = engine.snapshot()
        result = engine.apply(StateCommand(parameter_id="alt", value=99999.0))
        assert not result.accepted
        assert result.reason_code == "OUT_OF_RANGE"
        assert engine.snapshot() == before

    def test_unknown_parameter_is_rejected_not_created(self, engine) -> None:
        result = engine.apply(StateCommand(parameter_id="nonexistent", value=1))
        assert not result.accepted
        assert result.reason_code == "UNKNOWN_PARAMETER"
        assert "nonexistent" not in engine.snapshot()

    def test_rejection_message_leaks_no_internals(self, engine) -> None:
        """The counterpart may speak this message aloud, so it must carry
        no file path, stack frame or module name (FR-G3)."""
        result = engine.apply(StateCommand(parameter_id="alt", value=99999.0))
        message = result.message or ""
        for leak in ("Traceback", ".py", "core.", "__", "Error("):
            assert leak not in message

    def test_rate_toward_target_takes_time(self, engine) -> None:
        """A commanded climb must not teleport: that delay is the thing
        mission management is a skill at managing."""
        result = engine.apply(StateCommand(parameter_id="alt", value=5000.0))
        assert result.accepted
        assert engine.snapshot()["alt"] == 1000.0   # has not moved yet

        engine.advance_to(60.0)
        assert engine.snapshot()["alt"] == pytest.approx(1500.0)   # 500 ft/min

        engine.advance_to(600.0)
        assert engine.snapshot()["alt"] == pytest.approx(5000.0)   # arrived, no overshoot

    def test_target_cleared_once_reached(self, engine) -> None:
        engine.apply(StateCommand(parameter_id="alt", value=2000.0))
        engine.advance_to(600.0)
        assert engine.target_of("alt") is None


class TestDerivedValues:
    def test_derived_recomputes_from_current_state(self, engine) -> None:
        assert engine.snapshot()["endurance_min"] == pytest.approx(100.0)
        engine.advance_to(1800.0)
        assert engine.snapshot()["endurance_min"] == pytest.approx(70.0)

    def test_derived_cannot_contradict_its_input(self, engine) -> None:
        """FR-D2 made structural: endurance has no stored existence, so it
        cannot drift out of sync with fuel. This is the property that stops
        the counterpart saying '40 minutes' and then '12 minutes' for the
        same fuel reading."""
        for seconds in (0, 300, 900, 1800, 3600):
            engine.advance_to(float(seconds))
            snapshot = engine.snapshot()
            assert snapshot["endurance_min"] == pytest.approx(snapshot["fuel"] / 60 * 60)

    def test_derived_can_depend_on_derived(self, engine) -> None:
        snapshot = engine.snapshot()
        assert snapshot["half_endurance"] == pytest.approx(snapshot["endurance_min"] / 2)


class TestKnowledgeBoundary:
    """FR-D6: the counterpart cannot read hidden state. Enforced by
    omission from its snapshot, not by a prompt instruction -- so it holds
    even under an adversarial trainee message."""

    def test_hidden_parameter_absent_from_persona_view(self, engine) -> None:
        assert "secret" in engine.snapshot()
        assert "secret" not in engine.snapshot(for_persona=True)

    def test_hidden_value_not_leaked_through_arithmetic(self, mission_with_everything) -> None:
        """A derived value computed from hidden state must also be hidden,
        or the hidden figure is recoverable from the derived one."""
        mission_with_everything["derived"].append(
            {"id": "secret_length", "expr": "fuel + 0", "unit": "x"}
        )
        mission_with_everything["parameters"].append(
            {"id": "hidden_num", "type": "number", "initial": 42.0,
             "visible_to_persona": False}
        )
        mission_with_everything["derived"].append(
            {"id": "derived_from_hidden", "expr": "hidden_num * 2", "unit": "x"}
        )
        engine = MissionStateEngine(load_mission_dict(mission_with_everything))

        persona_view = engine.snapshot(for_persona=True)
        assert "hidden_num" not in persona_view
        assert "derived_from_hidden" not in persona_view
        # Still computed correctly for the simulation itself.
        assert engine.snapshot()["derived_from_hidden"] == pytest.approx(84.0)

    def test_knows_list_restricts_further(self, engine) -> None:
        """persona.knows is an allowlist when non-empty, so 'alt' and
        'comms' are excluded despite being visible parameters."""
        persona_view = engine.snapshot(for_persona=True)
        assert "fuel" in persona_view
        assert "alt" not in persona_view

    def test_describe_for_prompt_respects_the_boundary(self, engine) -> None:
        ids = {row["id"] for row in engine.describe_for_prompt(for_persona=True)}
        assert "secret" not in ids
        assert "fuel" in ids


class TestEffects:
    def test_value_change_emits_effect(self, engine) -> None:
        effects = engine.advance_to(600.0)
        changed = [e for e in effects if e.parameter_id == "fuel"]
        assert changed and changed[0].old_value == 100.0

    def test_floor_emits_limit_reached(self, engine) -> None:
        """So a trigger can react to 'fuel is empty' without polling."""
        effects = engine.advance_to(100 * 3600.0)
        assert any(e.kind.value == "limit_reached" and e.parameter_id == "fuel"
                   for e in effects)


class TestValueCoercion:
    """An LLM sends stringly-typed tool arguments in practice, whatever the
    declared schema says (FR-D3 regression).

    OBSERVED LIVE: "climb to 18,000" with a 25,000 ft ceiling produced
    "negative, unable" because the value arrived as the string "18000".
    The counterpart then TRUTHFULLY reported a refusal that should never
    have happened -- worse than a crash, because it silently teaches the
    trainee that a legal instruction was impossible.
    """

    @pytest.mark.parametrize("sent,expected", [
        (5000, 5000),
        ("5000", 5000),
        ("5,000", 5000),        # digit grouping, as a model often writes it
        ("5000.0", 5000),
        (" 5000 ", 5000),
    ])
    def test_numeric_strings_are_accepted(self, engine, sent, expected) -> None:
        result = engine.apply(StateCommand(parameter_id="alt", value=sent))
        assert result.accepted, f"{sent!r} should be accepted"
        assert engine.target_of("alt") == expected

    @pytest.mark.parametrize("sent", ["high", "ABOVE", "", "abc"])
    def test_non_numeric_strings_are_still_rejected(self, engine, sent) -> None:
        """Coercion must not become a licence to accept nonsense -- a
        genuinely non-numeric value should still fail rather than silently
        becoming zero."""
        result = engine.apply(StateCommand(parameter_id="alt", value=sent))
        assert not result.accepted
        assert result.reason_code == "INVALID_VALUE"

    @pytest.mark.parametrize("sent,expected", [
        ("active", "active"),
        ("ACTIVE", "active"),       # case-insensitive match
        ("Active", "active"),
        (" active ", "active"),
    ])
    def test_enum_case_is_normalized(self, engine, sent, expected) -> None:
        result = engine.apply(StateCommand(parameter_id="mode", value=sent))
        assert result.accepted
        assert engine.snapshot()["mode"] == expected

    def test_unknown_enum_value_still_rejected(self, engine) -> None:
        result = engine.apply(StateCommand(parameter_id="mode", value="TURBO"))
        assert not result.accepted

    def test_out_of_range_numeric_string_still_rejected(self, engine) -> None:
        """Coercion happens BEFORE range validation, so limits still bind."""
        result = engine.apply(StateCommand(parameter_id="alt", value="99999"))
        assert not result.accepted
        assert result.reason_code == "OUT_OF_RANGE"
