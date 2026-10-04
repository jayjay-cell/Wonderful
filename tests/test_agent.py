"""Agent loop and tool tests -- all offline, no API key (NFR-2).

Covers the three things that must hold before a real model is ever
pointed at this:

  * the step limit is genuinely enforced (FR-G1/G2)
  * tools cannot be made to leak hidden state or reach another session
  * a failed turn produces a safe in-character reply, not a crash
"""

from __future__ import annotations

import pytest

from agent.loop import build_agent, extract_reply_text, max_steps, strip_system_messages
from agent.prompts import build_initiative_cue, build_system_prompt
from core.mission import load_mission_dict
from core.state import MissionStateEngine
from sim.clock import VirtualClock
from sim.session import Session
from tests.fakes import (
    EmptyReplyModel,
    FailingModel,
    NeverStopsCallingToolsModel,
    ScriptedModel,
)
from tools.mission_tools import build_mission_tools


@pytest.fixture
def mission(mission_with_everything):
    return load_mission_dict(mission_with_everything)


@pytest.fixture
def engine(mission):
    return MissionStateEngine(mission)


@pytest.fixture
def tools_by_name(mission, engine):
    return {t.name: t for t in build_mission_tools(mission, engine)}


class TestToolsRespectTheKnowledgeBoundary:
    """The boundary is structural: there is no code path by which a hidden
    value reaches the model (FR-D6)."""

    def test_read_state_omits_hidden_parameters(self, tools_by_name) -> None:
        result = tools_by_name["read_state"].invoke({})
        ids = {r["id"] for r in result["readings"]}
        assert "fuel" in ids
        assert "secret" not in ids

    def test_reading_a_hidden_parameter_is_refused(self, tools_by_name) -> None:
        result = tools_by_name["read_state"].invoke({"parameter_ids": ["secret"]})
        assert result["error"] == "NOT_VISIBLE_TO_PERSONA"

    def test_refusal_does_not_confirm_the_parameter_exists(self, tools_by_name) -> None:
        """Saying 'that exists but is hidden' would leak the existence of
        hidden mission state, which is what the boundary is for."""
        hidden = tools_by_name["read_state"].invoke({"parameter_ids": ["secret"]})
        unknown = tools_by_name["read_state"].invoke({"parameter_ids": ["nonexistent"]})
        assert "secret" not in hidden["message"]
        assert hidden["message"] != unknown["message"]

    def test_hidden_world_state_cannot_be_commanded(self, tools_by_name) -> None:
        """`visible_to_persona: false` is hidden world state -- the
        counterpart can neither read nor change it."""
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "secret", "value": "changed"}
        )
        assert result["error"] == "NOT_VISIBLE_TO_PERSONA"

    def test_unreported_parameter_is_still_controllable(self, tools_by_name, engine) -> None:
        """Reading and controlling are DIFFERENT permissions.

        'alt' is absent from persona.knows, so the counterpart does not
        quote an altimeter reading -- but it can still be ordered to climb
        and carry that out. Treating `knows` as an access list would make
        any parameter used purely as a reporting filter uncommandable,
        silently breaking such missions.
        """
        assert "alt" not in engine.snapshot(for_persona=True)
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "alt", "value": 5000}
        )
        assert result.get("accepted") is True

    def test_list_readings_omits_hidden(self, tools_by_name) -> None:
        result = tools_by_name["list_readings"].invoke({})
        assert "secret" not in {r["id"] for r in result["readings"]}


class TestToolsHaveNoSessionArgument:
    """Authorization by structure, not by checking: the model has no field
    to populate with another session's identity."""

    def test_no_tool_accepts_a_session_or_identity_argument(self, tools_by_name) -> None:
        forbidden = {"session_id", "session", "user_id", "trainee_id", "mission_id"}
        for name, tool in tools_by_name.items():
            fields = set(tool.args_schema.model_fields) if tool.args_schema else set()
            leaked = fields & forbidden
            assert not leaked, f"tool {name!r} exposes identity argument(s): {leaked}"


class TestToolsReturnDataNotProse:
    """A tool must never hand the model a sentence it can simply echo --
    that is what keeps numbers traceable (FR-D1)."""

    def test_read_state_returns_structured_readings(self, tools_by_name) -> None:
        result = tools_by_name["read_state"].invoke({})
        assert isinstance(result["readings"], list)
        assert all({"id", "value"} <= set(r) for r in result["readings"])

    def test_readings_carry_name_and_unit(self, tools_by_name) -> None:
        """A bare number is easy to misreport as the wrong quantity."""
        result = tools_by_name["read_state"].invoke({"parameter_ids": ["fuel"]})
        reading = result["readings"][0]
        assert reading["name"] and reading["unit"] == "lb"


class TestSetParameter:
    def test_valid_change_applies(self, tools_by_name, engine) -> None:
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "mode", "value": "active"}
        )
        assert result["accepted"]
        assert engine.snapshot()["mode"] == "active"

    def test_invalid_enum_refused_with_sayable_message(self, tools_by_name) -> None:
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "mode", "value": "turbo"}
        )
        assert result["error"] == "INVALID_VALUE"
        for leak in ("Traceback", ".py", "__", "core."):
            assert leak not in result["message"]

    def test_out_of_range_refused(self, tools_by_name) -> None:
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "alt", "value": 99999}
        )
        assert result["error"] == "OUT_OF_RANGE"

    def test_in_transit_is_reported_honestly(self, tools_by_name) -> None:
        """Claiming a commanded altitude has been reached when the aircraft
        is still climbing would be the counterpart lying about state."""
        result = tools_by_name["set_parameter"].invoke(
            {"parameter_id": "alt", "value": 5000}
        )
        assert result["accepted"]
        assert result["in_transit_to"] == 5000
        assert "value" not in result


class TestStepLimit:
    """FR-G1/G2 -- proven with a model that never stops calling tools, so
    the limit is verified rather than assumed."""

    @pytest.mark.asyncio
    async def test_step_limit_ends_turn_gracefully(self, mission, engine) -> None:
        model = NeverStopsCallingToolsModel()
        agent = build_agent(mission, engine, model, step_limit=3)

        result = await agent.ainvoke({"messages": [{"role": "user", "content": "status"}]})

        # Stopped, not raised, and well short of runaway.
        assert model.call_count <= 4
        assert "messages" in result

    def test_env_var_overrides_mission_limit(self, mission, monkeypatch) -> None:
        monkeypatch.setenv("AGENT_MAX_STEPS", "3")
        assert max_steps(mission) == 3

    def test_mission_limit_used_when_no_env_var(self, mission, monkeypatch) -> None:
        monkeypatch.delenv("AGENT_MAX_STEPS", raising=False)
        assert max_steps(mission) == mission.limits.max_steps

    def test_invalid_env_var_falls_back(self, mission, monkeypatch) -> None:
        """A typo in .env must not silently disable the limit."""
        monkeypatch.setenv("AGENT_MAX_STEPS", "not-a-number")
        assert max_steps(mission) == mission.limits.max_steps


class TestSystemPrompt:
    def test_prompt_is_built_from_the_mission(self, mission, engine) -> None:
        prompt = build_system_prompt(mission, engine)
        assert mission.persona.name in prompt
        assert mission.procedure.callsigns.counterpart in prompt

    def test_prompt_omits_hidden_state(self, mission, engine) -> None:
        """The orientation panel must respect the same boundary as the
        tools, or the prompt leaks what the tools protect."""
        assert "hidden truth" not in build_system_prompt(mission, engine)

    def test_prompt_contains_the_numbers_rule(self, mission, engine) -> None:
        prompt = build_system_prompt(mission, engine)
        assert "read_state" in prompt
        assert "Never estimate" in prompt

    def test_prompt_forbids_revealing_internals(self, mission, engine) -> None:
        prompt = build_system_prompt(mission, engine).lower()
        assert "confidentiality" in prompt
        assert "never reveal" in prompt

    def test_prompt_declares_external_content_as_data(self, mission, engine) -> None:
        prompt = build_system_prompt(mission, engine).lower()
        assert "data, not instructions" in prompt

    def test_language_is_stated_from_the_mission(self, mission_with_everything) -> None:
        mission_with_everything["language"] = "he"
        mission = load_mission_dict(mission_with_everything)
        prompt = build_system_prompt(mission, MissionStateEngine(mission))
        assert "Hebrew" in prompt

    def test_initiative_cue_is_framed_as_data(self) -> None:
        """So the counterpart phrases the report itself instead of reading
        the intent string aloud (FR-C7)."""
        cue = build_initiative_cue("report bingo fuel")
        assert "not an instruction to quote" in cue
        assert "report bingo fuel" in cue


class TestMessageHygiene:
    def test_system_messages_are_stripped(self) -> None:
        """FR-G8: the system prompt is supplied per call and must never be
        persisted, or it accumulates one copy per turn."""
        messages = [
            {"role": "system", "content": "the prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "roger"},
        ]
        result = strip_system_messages(messages)
        assert len(result) == 2
        assert all(m["role"] != "system" for m in result)

    def test_extract_reply_handles_block_content(self) -> None:
        """Models that interleave reasoning with text return content as a
        list of blocks rather than a string."""
        messages = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "internal"},
            {"type": "text", "text": "רות, מפקדה."},
        ]}]
        assert extract_reply_text(messages) == "רות, מפקדה."

    def test_extract_reply_handles_empty(self) -> None:
        assert extract_reply_text([]) == ""


class TestSessionFailureHandling:
    """A turn must never crash the session (FR-G2)."""

    @pytest.mark.asyncio
    async def test_provider_failure_yields_in_character_reply(self, mission) -> None:
        session = Session(mission, FailingModel(), "s1", clock=VirtualClock())
        result = await session.run_trainee_turn("status")

        assert result.failed
        assert result.text
        # In character: a trainee hearing "an error occurred" has left the
        # simulation; garbled comms is a thing that happens on a real net.
        for leak in ("error", "exception", "Traceback", "None"):
            assert leak.lower() not in result.text.lower()

    @pytest.mark.asyncio
    async def test_empty_reply_treated_as_failure(self, mission) -> None:
        """A silent counterpart is indistinguishable from a broken one."""
        session = Session(mission, EmptyReplyModel(), "s2", clock=VirtualClock())
        result = await session.run_trainee_turn("status")
        assert result.failed
        assert result.failure_code == "EMPTY_REPLY"
        assert result.text.strip()

    @pytest.mark.asyncio
    async def test_failed_turn_leaves_history_retryable(self, mission) -> None:
        """History is left untouched so the next turn retries cleanly
        rather than inheriting a half-finished exchange."""
        session = Session(mission, FailingModel(), "s3", clock=VirtualClock())
        await session.run_trainee_turn("first")
        assert session.state["last_failure"]
        assert not any(
            m.get("role") == "assistant" for m in session.state["messages"]
        )

    @pytest.mark.asyncio
    async def test_successful_turn_records_the_exchange(self, mission) -> None:
        session = Session(mission, ScriptedModel(replies=["רות, מפקדה."]),
                          "s4", clock=VirtualClock())
        result = await session.run_trainee_turn("מה מצב הדלק")

        assert not result.failed
        assert result.text == "רות, מפקדה."
        assert session.state["last_failure"] is None

    @pytest.mark.asyncio
    async def test_mission_time_advances_before_the_model_runs(self, mission) -> None:
        """So the counterpart reasons over current state, not state as of
        the previous turn."""
        clock = VirtualClock()
        session = Session(mission, ScriptedModel(replies=["רות"]), "s5", clock=clock)

        clock.advance(1800.0)
        await session.run_trainee_turn("status")

        assert session.engine.mission_seconds == pytest.approx(1800.0)
        assert session.engine.snapshot()["fuel"] == pytest.approx(70.0)

    @pytest.mark.asyncio
    async def test_initiated_turn_uses_the_same_path(self, mission) -> None:
        """Realism must be uniform between reactive and self-initiated
        turns, which is why both go through _run_agent."""
        session = Session(mission, ScriptedModel(replies=["בינגו דלק"]),
                          "s6", clock=VirtualClock())
        result = await session.run_initiated_turn("report bingo fuel", "fuel_below_bingo")

        assert result.origin == "initiated"
        assert result.trigger_id == "fuel_below_bingo"
        assert result.text == "בינגו דלק"


class TestAuthorFreeTextReachesThePrompt:
    """Free-text authoring (FR-E4): no fixed schema anticipates every
    instruction, so the author's own words must survive into the prompt."""

    @pytest.fixture
    def authored(self, mission_with_everything):
        mission_with_everything["persona"].update({
            "background": "AUTHORED_BACKGROUND",
            "speech_guide": "AUTHORED_SPEECH_GUIDE",
            "behaviour_guide": "AUTHORED_BEHAVIOUR_GUIDE",
            "speech_examples": ["AUTHORED_EXAMPLE_ONE", "AUTHORED_EXAMPLE_TWO"],
        })
        mission_with_everything["setting"].update({
            "situation": "AUTHORED_SITUATION",
            "extra_instructions": "AUTHORED_EXTRA",
        })
        return load_mission_dict(mission_with_everything)

    def test_every_free_text_field_appears(self, authored) -> None:
        prompt = build_system_prompt(authored, MissionStateEngine(authored))
        for marker in ("AUTHORED_BACKGROUND", "AUTHORED_SPEECH_GUIDE",
                       "AUTHORED_BEHAVIOUR_GUIDE", "AUTHORED_EXAMPLE_ONE",
                       "AUTHORED_EXAMPLE_TWO", "AUTHORED_SITUATION",
                       "AUTHORED_EXTRA"):
            assert marker in prompt, f"{marker} did not reach the prompt"

    def test_extra_instructions_come_after_generated_sections(self, authored) -> None:
        """So an author can countermand generated guidance -- 'he never
        uses brevity codes' must beat the procedure section."""
        prompt = build_system_prompt(authored, MissionStateEngine(authored))
        assert prompt.index("AUTHORED_EXTRA") > prompt.index("COMMS PROCEDURE")

    def test_trainee_briefing_is_never_shown_to_the_model(self, mission_with_everything) -> None:
        """The briefing may contain intent the operator must not know;
        leaking it would breach the knowledge boundary."""
        mission_with_everything["setting"]["briefing"] = "SECRET_TRAINEE_BRIEFING"
        mission = load_mission_dict(mission_with_everything)
        prompt = build_system_prompt(mission, MissionStateEngine(mission))
        assert "SECRET_TRAINEE_BRIEFING" not in prompt

    def test_omitted_fields_add_nothing(self, mission) -> None:
        prompt = build_system_prompt(mission, MissionStateEngine(mission))
        assert "YOUR BACKGROUND" not in prompt
        assert "ADDITIONAL INSTRUCTIONS" not in prompt


class TestCheckpointInstructions:
    def test_per_checkpoint_instructions_reach_the_cue(self) -> None:
        cue = build_initiative_cue("report movement", "he is unsure here")
        assert "he is unsure here" in cue
        assert "How to play this moment" in cue

    def test_cue_without_instructions_is_unchanged(self) -> None:
        cue = build_initiative_cue("report movement")
        assert "How to play this moment" not in cue

    @pytest.mark.asyncio
    async def test_session_passes_instructions_through(self, mission) -> None:
        session = Session(mission, ScriptedModel(replies=["ok"]), "s7",
                          clock=VirtualClock())
        await session.run_initiated_turn("report X", "trig", "PER_MOMENT_NOTE")
        # Messages come back as LangChain objects, not the dicts that went
        # in, so read content via getattr rather than subscripting.
        first = session.state["messages"][0]
        cue = first["content"] if isinstance(first, dict) else first.content
        assert "PER_MOMENT_NOTE" in cue


class TestMissionPlanBriefing:
    """The plan is what he was BRIEFED on, as distinct from triggers, which
    are what the simulation does to him. Without it he treats every event
    as a surprise and never volunteers a scheduled report."""

    @pytest.fixture
    def briefed(self, mission_with_everything):
        mission_with_everything["plan"] = {
            "overview": "PLAN_OVERVIEW",
            "reporting_schedule": "REPORT_EVERY_TEN",
            "steps": [{"at": "H+5", "what": "STEP_ON_STATION", "note": "STEP_NOTE"}],
            "expected_events": ["EXPECTED_WEATHER"],
            "standing_orders": ["STANDING_NO_DESCENT"],
        }
        return load_mission_dict(mission_with_everything)

    def test_plan_reaches_the_prompt(self, briefed) -> None:
        prompt = build_system_prompt(briefed, MissionStateEngine(briefed))
        for marker in ("PLAN_OVERVIEW", "REPORT_EVERY_TEN", "STEP_ON_STATION",
                       "STEP_NOTE", "EXPECTED_WEATHER", "STANDING_NO_DESCENT"):
            assert marker in prompt, f"{marker} missing from prompt"

    def test_reporting_schedule_framed_as_an_obligation(self, briefed) -> None:
        """A schedule he merely knows about does not make him proactive."""
        prompt = build_system_prompt(briefed, MissionStateEngine(briefed))
        assert "without being asked" in prompt

    def test_no_plan_adds_no_section(self, mission) -> None:
        prompt = build_system_prompt(mission, MissionStateEngine(mission))
        assert "MISSION BRIEFING" not in prompt

    def test_trigger_timings_are_not_leaked_as_plan(self, briefed) -> None:
        """He is briefed on the plan, not on the simulation's script: an
        event the author left out of expected_events should still surprise
        him."""
        prompt = build_system_prompt(briefed, MissionStateEngine(briefed))
        assert "at_mission_seconds" not in prompt
        assert "120.0" not in prompt


class TestCommsSchema:
    """An agreed message STRUCTURE, not just vocabulary. A model given a
    word list alone invents its own transmission format."""

    @pytest.fixture
    def with_formats(self, mission_with_everything):
        mission_with_everything["procedure"]["report_formats"] = [{
            "id": "status",
            "when": "WHEN_TO_USE_STATUS",
            "template": "TEMPLATE_SHAPE",
            "required_fields": ["FIELD_A", "FIELD_B"],
            "example": "EXAMPLE_TRANSMISSION",
        }]
        mission_with_everything["procedure"]["comms_guide"] = "NET_DISCIPLINE_TEXT"
        return load_mission_dict(mission_with_everything)

    def test_format_details_reach_the_prompt(self, with_formats) -> None:
        prompt = build_system_prompt(with_formats, MissionStateEngine(with_formats))
        for marker in ("WHEN_TO_USE_STATUS", "TEMPLATE_SHAPE", "FIELD_A",
                       "EXAMPLE_TRANSMISSION", "NET_DISCIPLINE_TEXT"):
            assert marker in prompt

    def test_deviation_is_stated_as_an_error(self, with_formats) -> None:
        prompt = build_system_prompt(with_formats, MissionStateEngine(with_formats))
        assert "Never invent your own phrasing" in prompt

    def test_comms_guide_comes_after_generated_procedure(self, with_formats) -> None:
        """So authored doctrine overrides the generated guidance."""
        prompt = build_system_prompt(with_formats, MissionStateEngine(with_formats))
        assert prompt.index("NET_DISCIPLINE_TEXT") > prompt.index("AGREED TERMS")


class TestSpokenEnumLabels:
    """Enum ids stay ASCII because conditions reference them, but the
    counterpart must not say "idle" aloud on a Hebrew net -- observed live,
    and it breaks immersion instantly."""

    @pytest.fixture
    def labelled(self, mission_with_everything):
        for parameter in mission_with_everything["parameters"]:
            if parameter["id"] == "mode":
                parameter["value_labels"] = {"idle": "ללא מיקוד", "active": "עוקב"}
        return load_mission_dict(mission_with_everything)

    def test_spoken_form_offered_to_the_model(self, labelled) -> None:
        engine = MissionStateEngine(labelled)
        tools = {t.name: t for t in build_mission_tools(labelled, engine)}
        result = tools["read_state"].invoke({"parameter_ids": ["mode"]})
        assert result["readings"][0]["say_as"] == "ללא מיקוד"

    def test_machine_value_still_available(self, labelled) -> None:
        """Conditions and effects reference the id, so it must survive."""
        engine = MissionStateEngine(labelled)
        tools = {t.name: t for t in build_mission_tools(labelled, engine)}
        result = tools["read_state"].invoke({"parameter_ids": ["mode"]})
        assert result["readings"][0]["value"] == "idle"

    def test_prompt_panel_shows_spoken_form(self, labelled) -> None:
        prompt = build_system_prompt(labelled, MissionStateEngine(labelled))
        assert "ללא מיקוד" in prompt

    def test_unlabelled_values_unchanged(self, mission) -> None:
        engine = MissionStateEngine(mission)
        tools = {t.name: t for t in build_mission_tools(mission, engine)}
        result = tools["read_state"].invoke({"parameter_ids": ["mode"]})
        assert "say_as" not in result["readings"][0]
