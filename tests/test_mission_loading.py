"""Mission loader tests -- FAIL LOUDLY (FR-E8), and stay flexible (FR-E1..E10).

Two groups, both essential:

  * Validation: an invalid mission must never load with silent defaults.
    A mission that quietly ignored a mistyped trigger id would run a
    scenario missing a safety-critical report, the trainee would be
    assessed on it, and nobody would know the trigger was never armed.
    Silence is the dangerous failure here, not noise.

  * Flexibility: adding a parameter, removing one, or running an entirely
    different domain must work with no code change. These tests are the
    executable form of the product's core requirement.
"""

from __future__ import annotations

import pytest

from core.mission import MissionError, load_mission, load_mission_dict
from core.state import MissionStateEngine


class TestValidMissions:
    def test_minimal_mission_loads(self, minimal_mission_dict) -> None:
        mission = load_mission_dict(minimal_mission_dict)
        assert mission.id == "test_mission"
        assert len(mission.parameters) == 1

    def test_full_mission_loads(self, mission_with_everything) -> None:
        mission = load_mission_dict(mission_with_everything)
        assert len(mission.triggers.all_ids()) == 3

    def test_reference_mission_loads(self, reference_mission_path) -> None:
        """The real UAV file, so breakage is caught in what you actually
        author against -- not only in synthetic fixtures."""
        mission = load_mission(reference_mission_path)
        assert mission.language == "he"
        assert mission.persona.name
        assert any(p.id == "fuel_lb" for p in mission.parameters)

    def test_defaults_fill_in_optional_sections(self, minimal_mission_dict) -> None:
        mission = load_mission_dict(minimal_mission_dict)
        assert mission.version == 1
        assert mission.limits.max_steps == 8
        assert mission.realism.stall_probability > 0


class TestLoudFailures:
    """Every case names the problem and its location."""

    def test_unknown_parameter_in_derived_expression(self, minimal_mission_dict) -> None:
        minimal_mission_dict["derived"] = [{"id": "d", "expr": "nonexistent * 2"}]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "UNKNOWN_PARAMETER_REF"
        assert "nonexistent" in str(err.value)

    def test_typo_suggests_the_intended_name(self, minimal_mission_dict) -> None:
        """Most mission-file errors in practice are a typo or a rename;
        naming the likely intent turns a hunt into a glance."""
        minimal_mission_dict["derived"] = [{"id": "d", "expr": "fule * 2"}]
        minimal_mission_dict["persona"]["knows"] = ["fule"]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert "Did you mean 'fuel'?" in str(err.value)

    def test_duplicate_parameter_id(self, minimal_mission_dict) -> None:
        minimal_mission_dict["parameters"].append(
            {"id": "fuel", "type": "number", "initial": 5.0}
        )
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "DUPLICATE_ID"

    def test_parameter_and_derived_sharing_an_id(self, minimal_mission_dict) -> None:
        minimal_mission_dict["derived"] = [{"id": "fuel", "expr": "fuel * 1"}]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "DUPLICATE_ID"

    def test_enum_initial_outside_declared_values(self, minimal_mission_dict) -> None:
        minimal_mission_dict["parameters"].append(
            {"id": "mode", "type": "enum", "values": ["a", "b"], "initial": "c"}
        )
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "INVALID_FIELD"

    def test_numeric_initial_below_min(self, minimal_mission_dict) -> None:
        minimal_mission_dict["parameters"].append(
            {"id": "x", "type": "number", "initial": -5.0, "min": 0}
        )
        with pytest.raises(MissionError):
            load_mission_dict(minimal_mission_dict)

    def test_dynamics_missing_required_config(self, minimal_mission_dict) -> None:
        """A linear_drain with no rate would silently never drain -- the
        parameter would look fine and behave wrongly."""
        minimal_mission_dict["parameters"].append(
            {"id": "y", "type": "number", "initial": 10.0,
             "dynamics": {"kind": "linear_drain"}}
        )
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert "rate_per_hour" in str(err.value)

    def test_unknown_dynamics_kind(self, minimal_mission_dict) -> None:
        minimal_mission_dict["parameters"].append(
            {"id": "z", "type": "number", "initial": 1.0,
             "dynamics": {"kind": "teleport"}}
        )
        with pytest.raises(MissionError):
            load_mission_dict(minimal_mission_dict)

    def test_tone_shift_naming_unknown_trigger(self, mission_with_everything) -> None:
        mission_with_everything["tone"]["shifts"] = [
            {"when": "no_such_trigger", "to": "urgent"}
        ]
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert err.value.code == "UNKNOWN_TRIGGER_REF"

    def test_tone_shift_naming_unknown_tone(self, mission_with_everything) -> None:
        mission_with_everything["tone"]["shifts"] = [
            {"when": "low_fuel", "to": "hysterical"}
        ]
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert err.value.code == "UNKNOWN_TONE"

    def test_persona_knows_unknown_parameter(self, minimal_mission_dict) -> None:
        minimal_mission_dict["persona"]["knows"] = ["fuel", "imaginary"]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "UNKNOWN_PARAMETER_REF"

    def test_trigger_effect_on_unknown_parameter(self, minimal_mission_dict) -> None:
        minimal_mission_dict["triggers"] = {
            "timeline": [{"id": "t", "at_mission_seconds": 10.0, "say_intent": "x",
                          "effects": [{"parameter": "ghost", "value": 1}]}]
        }
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "UNKNOWN_PARAMETER_REF"

    def test_trigger_effect_with_invalid_enum_value(self, mission_with_everything) -> None:
        mission_with_everything["triggers"]["timeline"][0]["effects"] = [
            {"parameter": "mode", "value": "nonsense"}
        ]
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert err.value.code == "INVALID_VALUE"

    def test_garble_source_must_be_enum(self, mission_with_everything) -> None:
        mission_with_everything["realism"]["garble_by"] = {
            "parameter": "fuel", "probabilities": {"good": 0.1}
        }
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert err.value.code == "INVALID_GARBLE_SOURCE"

    def test_garble_must_cover_every_enum_value(self, mission_with_everything) -> None:
        """A missing entry would silently mean 'never garble in that
        state', which is a plausible-looking wrong behaviour."""
        mission_with_everything["realism"]["garble_by"]["probabilities"] = {"good": 0.0}
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert "bad" in str(err.value)

    def test_circular_derived_dependency(self, minimal_mission_dict) -> None:
        minimal_mission_dict["derived"] = [
            {"id": "a", "expr": "b + 1"},
            {"id": "b", "expr": "a + 1"},
        ]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code == "CIRCULAR_REFERENCE"

    def test_self_referential_derived(self, minimal_mission_dict) -> None:
        minimal_mission_dict["derived"] = [{"id": "a", "expr": "a + 1"}]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert err.value.code in {"CIRCULAR_REFERENCE", "UNKNOWN_PARAMETER_REF"}

    def test_missing_required_section(self, minimal_mission_dict) -> None:
        del minimal_mission_dict["persona"]
        with pytest.raises(MissionError) as err:
            load_mission_dict(minimal_mission_dict)
        assert "persona" in str(err.value)

    def test_nonexistent_file(self) -> None:
        with pytest.raises(MissionError) as err:
            load_mission("missions/does_not_exist.yaml")
        assert err.value.code == "FILE_NOT_FOUND"


class TestExpressionSafety:
    """Mission files are the natural 'import a mission someone sent me'
    path, so an expression must never be able to execute code."""

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo pwned')",
        "(1).__class__.__bases__",
        "fuel.__class__",
        "open('secret.txt').read()",
        "[x for x in range(10)]",
        "lambda: 1",
        "exec('x=1')",
    ])
    def test_dangerous_expressions_rejected(self, minimal_mission_dict, expr) -> None:
        minimal_mission_dict["derived"] = [{"id": "evil", "expr": expr}]
        with pytest.raises(MissionError):
            load_mission_dict(minimal_mission_dict)

    def test_threshold_condition_is_also_validated(self, minimal_mission_dict) -> None:
        minimal_mission_dict["triggers"] = {
            "thresholds": [{"id": "t", "when": "__import__('os')", "say_intent": "x"}]
        }
        with pytest.raises(MissionError):
            load_mission_dict(minimal_mission_dict)


class TestFlexibility:
    """FR-E2/E3/E7 -- the product's core requirement, as executable tests."""

    def test_adding_a_parameter_needs_no_code_change(self, minimal_mission_dict) -> None:
        minimal_mission_dict["parameters"].append({
            "id": "brand_new_thing", "label": "Something", "type": "number",
            "initial": 7.0, "unit": "widgets",
            "dynamics": {"kind": "linear_fill", "rate_per_hour": 10.0, "ceiling": 100},
        })
        engine = MissionStateEngine(load_mission_dict(minimal_mission_dict))
        engine.advance_to(3600.0)
        assert engine.snapshot()["brand_new_thing"] == pytest.approx(17.0)

    def test_removing_a_parameter_needs_no_code_change(self, mission_with_everything) -> None:
        """Deleting a parameter must not crash -- but anything referring to
        it must then fail loudly rather than silently ignoring the gap."""
        mission_with_everything["parameters"] = [
            p for p in mission_with_everything["parameters"] if p["id"] != "alt"
        ]
        mission_with_everything["procedure"]["rules"] = []
        engine = MissionStateEngine(load_mission_dict(mission_with_everything))
        assert "alt" not in engine.snapshot()
        assert "fuel" in engine.snapshot()

    def test_dangling_reference_after_removal_fails_loudly(self, mission_with_everything) -> None:
        mission_with_everything["parameters"] = [
            p for p in mission_with_everything["parameters"] if p["id"] != "alt"
        ]
        # procedure.rules still references 'alt'
        with pytest.raises(MissionError) as err:
            load_mission_dict(mission_with_everything)
        assert err.value.code == "UNKNOWN_PARAMETER_REF"

    def test_entirely_different_domain(self) -> None:
        """FR-E7: a scenario with no aircraft, no fuel and no sensor runs
        with zero code changes. This is the test that justifies the whole
        declaration-driven design."""
        naval = {
            "id": "naval", "title": "Naval watch", "language": "en",
            "setting": {"purpose": "p", "place": "bridge", "trainee_role": "officer"},
            "persona": {"name": "Helm", "role": "helmsman"},
            "parameters": [
                {"id": "speed_kt", "type": "number", "initial": 12.0, "min": 0, "max": 30,
                 "dynamics": {"kind": "rate_toward_target", "rate_per_minute": 2.0}},
                {"id": "sea_state", "type": "enum", "values": ["calm", "rough"],
                 "initial": "calm",
                 "dynamics": {"kind": "scripted",
                              "keyframes": [{"at": 300, "value": "rough"}]}},
                {"id": "crew_rested_pct", "type": "number", "initial": 100.0, "min": 0,
                 "dynamics": {"kind": "linear_drain", "rate_per_hour": 8.0, "floor": 0}},
            ],
            "derived": [{"id": "half_speed", "expr": "speed_kt / 2", "unit": "kt"}],
            "procedure": {"callsigns": {"counterpart": "Helm", "trainee": "Bridge"}},
            "triggers": {"thresholds": [
                {"id": "rough", "when": "sea_state == 'rough'",
                 "say_intent": "report rough seas", "priority": "high"}
            ]},
        }
        engine = MissionStateEngine(load_mission_dict(naval))
        engine.advance_to(600.0)
        snapshot = engine.snapshot()
        assert snapshot["sea_state"] == "rough"
        assert snapshot["crew_rested_pct"] == pytest.approx(100 - 8 * (600 / 3600))
        assert snapshot["half_speed"] == pytest.approx(6.0)

    def test_realism_tuning_needs_no_code_change(self, minimal_mission_dict) -> None:
        minimal_mission_dict["realism"] = {
            "stall_probability": 0.9,
            "filler_probability": 0.05,
            "response_delay_ms": {"min": 100, "max": 200, "distribution": "uniform"},
        }
        mission = load_mission_dict(minimal_mission_dict)
        assert mission.realism.stall_probability == 0.9
        assert mission.realism.response_delay_ms.distribution == "uniform"

    def test_language_is_mission_declared(self, minimal_mission_dict) -> None:
        """FR-E9: the engine is language-agnostic."""
        minimal_mission_dict["language"] = "ar"
        assert load_mission_dict(minimal_mission_dict).language == "ar"
