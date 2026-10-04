"""Voice layer tests: turn detection, audio rendering, provider wiring.

All offline -- no microphone, no API key (NFR-2). Audio is synthesized as
PCM frames, which is what makes turn detection assertable at all: real
speech would make every threshold test a judgement call.

THE CLAIM BEING CHECKED HERE is the architectural one from
docs/ARCHITECTURE.md: that voice is additive. These tests exercise the
voice channel against the SAME DeliveryPlan the text channel consumes, so
if the seam had leaked, they would not compile.
"""

from __future__ import annotations

import math
import struct

import pytest

from core.mission import load_mission_dict
from core.realism import compile_plan
from delivery.scheduler import execute
from delivery.voice_channel import SAMPLE_RATE, VoiceChannel, _degrade, _silence
from sim.turn_detection import TurnConfig, TurnDetector, TurnState, looks_complete, rms


def pcm_frame(ms: int = 20, amplitude: int = 0) -> bytes:
    """One PCM frame: silence at amplitude 0, a tone otherwise."""
    count = int(SAMPLE_RATE * ms / 1000)
    return struct.pack(
        f"<{count}h",
        *[int(amplitude * math.sin(i * 0.3)) for i in range(count)],
    )


SPEECH = pcm_frame(20, 6000)
SILENCE = pcm_frame(20, 0)


@pytest.fixture
def mission(mission_with_everything):
    return load_mission_dict(mission_with_everything)


class TestEnergyDetection:
    def test_silence_reads_as_silent(self) -> None:
        assert rms(SILENCE) < 100

    def test_speech_reads_as_loud(self) -> None:
        assert rms(SPEECH) > 1000

    def test_empty_frame_does_not_crash(self) -> None:
        assert rms(b"") == 0.0


class TestTurnCompleteness:
    """The heuristic that buys SPEED: a clearly finished transmission gets
    a short silence threshold, an unfinished one gets a longer wait. A
    fixed timeout has to be wrong for one of the two cases."""

    @pytest.mark.parametrize("text,expected", [
        ("מפקדה, נחשון 3, דווח", False),      # trailing verb: object still coming
        ("דווח מצב דלק", True),                # verb WITH its object
        ("דווח מצב", True),
        ("נחשון 3, רות", True),                # terminal brevity word
        ("עלה לגובה 18", False),               # trailing number: mid-readback
        ("עלה לגובה 18000 רגל", True),
        ("מה מצב הדלק?", True),                # explicit punctuation
        ("נחשון 3, מפקדה, עבור", True),        # "over" reading wins
        ("", False),
    ])
    def test_completeness_judgement(self, text, expected) -> None:
        assert looks_complete(text) is expected

    def test_trailing_continuation_word_is_open(self) -> None:
        assert not looks_complete("דווח מצב דלק ו")


class TestTurnDetector:
    def test_sustained_speech_starts_a_turn(self) -> None:
        detector = TurnDetector()
        for _ in range(15):
            detector.feed(SPEECH)
        assert detector.state is TurnState.TRAINEE_SPEAKING

    def test_brief_noise_does_not_start_a_turn(self) -> None:
        """A cough or a chair creak must not trigger a reply."""
        detector = TurnDetector()
        detector.feed(SPEECH)      # 20ms, below min_speech_ms
        assert detector.state is TurnState.SILENT

    def test_complete_transcript_replies_sooner(self) -> None:
        """The whole point of the heuristic: measurably less waiting."""
        quick = self._silence_until_finished("דווח מצב דלק")
        slow = self._silence_until_finished("מפקדה, נחשון 3, דווח")
        assert quick < slow
        assert quick <= TurnConfig().end_silence_ms + 40

    def _silence_until_finished(self, partial: str) -> int:
        detector = TurnDetector()
        detector.update_partial(partial)
        for _ in range(15):
            detector.feed(SPEECH)
        elapsed = 0
        while detector.state is not TurnState.TRAINEE_FINISHED and elapsed < 4000:
            detector.feed(SILENCE)
            elapsed += 20
        return elapsed

    def test_turn_always_ends_eventually(self) -> None:
        """A transcript that never reads as complete -- or an STT that
        returns nothing at all -- must not hang the turn forever."""
        assert self._silence_until_finished("") <= TurnConfig().max_silence_ms + 60

    def test_consume_resets_for_the_next_turn(self) -> None:
        detector = TurnDetector()
        detector.update_partial("רות")
        for _ in range(15):
            detector.feed(SPEECH)
        for _ in range(40):
            detector.feed(SILENCE)
        detector.consume_turn()
        assert detector.state is TurnState.SILENT


class TestBargeIn:
    """Detecting that the trainee STARTED talking is a separate and much
    faster question than knowing what they said -- so it fires on energy
    alone, within a couple of frames."""

    def test_barge_in_fires_quickly(self) -> None:
        detector = TurnDetector()
        detector.set_counterpart_speaking(True)
        fired_at = None
        for i in range(1, 30):
            _, barge_in = detector.feed(SPEECH)
            if barge_in:
                fired_at = i * 20
                break
        assert fired_at is not None
        assert fired_at <= TurnConfig().barge_in_ms + 40

    def test_no_barge_in_when_counterpart_is_silent(self) -> None:
        detector = TurnDetector()
        detector.set_counterpart_speaking(False)
        for _ in range(30):
            _, barge_in = detector.feed(SPEECH)
            assert not barge_in

    def test_isolated_noise_does_not_barge_in(self) -> None:
        """A single loud frame is a noise, not an interruption. Reacting
        would make him stop mid-word for no reason."""
        detector = TurnDetector()
        detector.set_counterpart_speaking(True)
        for _ in range(20):
            detector.feed(SPEECH)      # one frame
            _, barge_in = detector.feed(SILENCE)
            assert not barge_in


class TestAudioHelpers:
    def test_silence_duration_is_accurate(self) -> None:
        pcm = _silence(500)
        assert abs(len(pcm) / (SAMPLE_RATE * 2) * 1000 - 500) < 2

    def test_degrade_preserves_length(self) -> None:
        """A garbled span must occupy the same time as a clean one, or the
        plan's timing drifts."""
        original = pcm_frame(100, 8000)
        assert len(_degrade(original)) == len(original)

    def test_degrade_changes_the_signal(self) -> None:
        original = pcm_frame(100, 8000)
        assert _degrade(original) != original

    def test_degrade_does_not_wrap(self) -> None:
        """Soft clipping: a wrapped sample is a loud crack, not a degraded
        signal."""
        loud = struct.pack("<4h", 32700, 32700, -32700, -32700)
        samples = struct.unpack("<4h", _degrade(loud))
        assert all(-32768 <= s <= 32767 for s in samples)
        # Signs preserved means no wraparound occurred.
        assert samples[0] > 0 and samples[2] < 0


class TestVoiceChannel:
    """Rendered from the SAME DeliveryPlan the text channel consumes --
    which is the architectural claim being checked."""

    class FakeTTS:
        """Returns a fixed amount of audio proportional to text length."""

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def synthesize_stream(self, text, voice=None):
            self.calls.append(text)
            yield pcm_frame(max(20, len(text) * 4), 5000)

    @pytest.mark.asyncio
    async def test_plan_renders_to_audio(self, mission) -> None:
        emitted: list[tuple[int, str]] = []

        async def emit_audio(pcm: bytes, kind: str) -> None:
            emitted.append((len(pcm), kind))

        tts = self.FakeTTS()
        channel = VoiceChannel(tts=tts, emit_audio=emit_audio)
        plan = compile_plan("מפקדה, «hesitate» רות", mission=mission,
                            session_id="s", turn_id="t", plan_id="p",
                            seed=1, for_voice=True)

        await execute(plan, channel, speed=0)

        assert tts.calls, "TTS was never called"
        assert any(kind == "speech" for _, kind in emitted)
        # A pause becomes REAL SILENCE, not an absence of frames: a gap in
        # the stream makes some players resynchronize audibly.
        assert any(kind == "silence" for _, kind in emitted)

    @pytest.mark.asyncio
    async def test_pause_duration_matches_the_plan(self, mission) -> None:
        durations: list[int] = []

        async def emit_audio(pcm: bytes, kind: str) -> None:
            if kind == "silence":
                durations.append(int(len(pcm) / (SAMPLE_RATE * 2) * 1000))

        plan = compile_plan("א «hesitate» ב", mission=mission, session_id="s",
                            turn_id="t", plan_id="p", seed=1, for_voice=True)
        await execute(plan, VoiceChannel(tts=self.FakeTTS(), emit_audio=emit_audio),
                      speed=0)

        planned = [s.duration_ms for s in plan.segments if s.kind == "pause"]
        assert durations and planned
        assert abs(durations[0] - planned[0]) < 25

    @pytest.mark.asyncio
    async def test_interruption_sends_a_flush(self, mission) -> None:
        """Merely stopping would leave queued audio playing after he was
        cut off -- the exact talk-over the design forbids."""
        import asyncio
        kinds: list[str] = []

        async def emit_audio(pcm: bytes, kind: str) -> None:
            kinds.append(kind)

        plan = compile_plan("מפקדה, נחשון 3, דלק 180, ראות 8", mission=mission,
                            session_id="s", turn_id="t", plan_id="p",
                            seed=1, for_voice=True)
        barge_in = asyncio.Event()
        barge_in.set()
        await execute(plan, VoiceChannel(tts=self.FakeTTS(), emit_audio=emit_audio),
                      barge_in=barge_in, speed=0)
        assert "interrupt" in kinds

    @pytest.mark.asyncio
    async def test_tts_failure_does_not_end_the_session(self, mission) -> None:
        """The trainee hears a dropped transmission -- a thing that happens
        on a real net -- rather than the simulator dying mid-exercise."""
        class BrokenTTS:
            async def synthesize_stream(self, text, voice=None):
                raise RuntimeError("tts exploded")
                yield b""   # pragma: no cover

        emitted: list[str] = []

        async def emit_audio(pcm: bytes, kind: str) -> None:
            emitted.append(kind)

        plan = compile_plan("מפקדה, רות", mission=mission, session_id="s",
                            turn_id="t", plan_id="p", seed=1, for_voice=True)
        outcome = await execute(plan, VoiceChannel(tts=BrokenTTS(),
                                                   emit_audio=emit_audio), speed=0)
        assert outcome.status == "completed"
        assert "silence" in emitted

    @pytest.mark.asyncio
    async def test_voice_pacing_is_slower_than_text(self, mission) -> None:
        """Speech duration is a physical fact; text pacing is a UX choice."""
        text_plan = compile_plan("מפקדה, נחשון 3, דלק 180 ליברות.",
                                 mission=mission, session_id="s", turn_id="t",
                                 plan_id="p", seed=1, for_voice=False)
        voice_plan = compile_plan("מפקדה, נחשון 3, דלק 180 ליברות.",
                                  mission=mission, session_id="s", turn_id="t",
                                  plan_id="p", seed=1, for_voice=True)
        assert voice_plan.total_ms > text_plan.total_ms


class TestProviderWiring:
    def test_stt_keyterms_come_from_the_mission(self, mission) -> None:
        """Biasing the recognizer with the mission's own brevity codes is
        the cheapest Hebrew accuracy win available."""
        from providers.cloud.elevenlabs_stt import mission_keyterms
        terms = mission_keyterms(mission)
        assert mission.procedure.callsigns.counterpart in terms
        assert any(t.term in terms for t in mission.procedure.brevity)

    def test_keyterms_are_deduplicated(self, mission) -> None:
        from providers.cloud.elevenlabs_stt import mission_keyterms
        terms = mission_keyterms(mission)
        assert len(terms) == len(set(terms))

    def test_missing_key_raises_a_clear_error(self, monkeypatch) -> None:
        """Two likely causes -- no key, or a refused mic -- need different
        fixes, so the message must say which."""
        from providers.base import ProviderError
        from providers.cloud.elevenlabs_tts import ElevenLabsTTS

        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTTS()._require_key()

    def test_providers_declare_network_need(self) -> None:
        """So the offline guard and a future local provider can be
        distinguished without inspecting their class."""
        from providers.cloud.elevenlabs_stt import ElevenLabsSTT
        from providers.cloud.elevenlabs_tts import ElevenLabsTTS
        assert ElevenLabsTTS().requires_network
        assert ElevenLabsSTT().requires_network

    def test_flash_model_is_the_tts_default(self) -> None:
        """eleven_v3 takes ~7s per Hebrew sentence versus ~1s for Flash
        v2.5. For a radio net that is not a close call."""
        from providers.cloud.elevenlabs_tts import ElevenLabsTTS
        assert "flash" in ElevenLabsTTS()._model.lower()


class TestSpokenValueResolution:
    """Regression: a legal order in Hebrew was refused.

    The counterpart is told to SAY "סורק" rather than "scanning", so when
    the trainee orders it in Hebrew the model naturally passes the Hebrew
    word back. Without resolution the engine refused it -- and the
    counterpart then recited the English ids aloud ("idle, tracking,
    scanning"), breaking character twice over.
    """

    @pytest.fixture
    def uav(self, reference_mission_path):
        from core.mission import load_mission
        return load_mission(reference_mission_path)

    @pytest.mark.parametrize("spoken,expected", [
        ("scanning", "scanning"),       # machine value
        ("SCANNING", "scanning"),       # case-insensitive
        ("סורק", "scanning"),           # the canonical spoken label
        ("סריקה", "scanning"),          # a different Hebrew word form
        ("לסרוק", "scanning"),
        ("עוקב", "tracking"),
        ("מעקב", "tracking"),
    ])
    def test_spoken_forms_resolve(self, uav, spoken, expected) -> None:
        from core.models import StateCommand
        from core.state import MissionStateEngine

        engine = MissionStateEngine(uav)
        result = engine.apply(StateCommand(parameter_id="sensor_mode", value=spoken))
        assert result.accepted, f"{spoken!r} should be accepted"
        assert engine.snapshot()["sensor_mode"] == expected

    def test_nonsense_is_still_refused(self, uav) -> None:
        """Resolution must not become a licence to accept anything."""
        from core.models import StateCommand
        from core.state import MissionStateEngine

        result = MissionStateEngine(uav).apply(
            StateCommand(parameter_id="sensor_mode", value="turbo")
        )
        assert not result.accepted

    def test_refusal_lists_spoken_options_not_ids(self, uav) -> None:
        """Reciting machine ids aloud on a Hebrew net breaks character."""
        from core.models import StateCommand
        from core.state import MissionStateEngine

        result = MissionStateEngine(uav).apply(
            StateCommand(parameter_id="sensor_mode", value="turbo")
        )
        assert "idle" not in (result.message or "")
        assert "ללא מיקוד" in (result.message or "")

    def test_aliases_are_validated_at_load(self, reference_mission_path) -> None:
        """A typo'd alias key must fail loudly, like value_labels."""
        import yaml
        from core.mission import MissionError, load_mission_dict

        raw = yaml.safe_load(reference_mission_path.read_text(encoding="utf-8"))
        for parameter in raw["parameters"]:
            if parameter["id"] == "sensor_mode":
                parameter["value_aliases"] = {"scaning": ["x"]}   # typo
        with pytest.raises(MissionError):
            load_mission_dict(raw)


class TestGeminiLiveProvider:
    """The native speech-to-speech alternative. Measured at ~1.1s to first
    Hebrew audio, and it DOES call read_state before quoting a figure --
    which is why this path is viable at all."""

    def test_tool_declarations_cover_the_tool_surface(self, reference_mission_path) -> None:
        from core.mission import load_mission
        from providers.cloud.gemini_live import tool_declarations_for

        tools = tool_declarations_for(load_mission(reference_mission_path))
        names = {fn.name for tool in tools for fn in tool.function_declarations}
        assert {"read_state", "set_parameter", "check_procedure"} <= names

    def test_read_state_declaration_forbids_invented_numbers(self, reference_mission_path) -> None:
        """The instruction has to be IN the declaration: it is the only
        place a native speech-to-speech model reliably reads it."""
        from core.mission import load_mission
        from providers.cloud.gemini_live import tool_declarations_for

        tools = tool_declarations_for(load_mission(reference_mission_path))
        read_state = next(fn for tool in tools for fn in tool.function_declarations
                          if fn.name == "read_state")
        assert "MUST call this before" in read_state.description
        assert "fuel_lb" in read_state.description   # valid ids listed

    def test_enum_hint_leads_with_the_spoken_form(self, reference_mission_path) -> None:
        """Regression: he said "idle" aloud on a Hebrew net.

        The hint listed the English id first, so that is what he read out.
        The spoken form must come first, since it is what he says and what
        the trainee orders it with -- the machine id is only for the tool
        call.
        """
        from core.mission import load_mission
        from providers.cloud.gemini_live import tool_declarations_for

        tools = tool_declarations_for(load_mission(reference_mission_path))
        set_parameter = next(fn for tool in tools for fn in tool.function_declarations
                             if fn.name == "set_parameter")
        description = set_parameter.description

        assert "סורק" in description
        assert "pass as scanning" in description
        # The Hebrew word must precede its machine id.
        assert description.index("סורק") < description.index("pass as scanning")

    def test_read_state_tells_it_to_say_the_hebrew(self, reference_mission_path) -> None:
        from core.mission import load_mission
        from providers.cloud.gemini_live import tool_declarations_for

        tools = tool_declarations_for(load_mission(reference_mission_path))
        read_state = next(fn for tool in tools for fn in tool.function_declarations
                          if fn.name == "read_state")
        assert "say_as" in read_state.description

    def test_hidden_ids_are_not_named_in_declarations(self, reference_mission_path) -> None:
        """Naming a hidden parameter tells the model it EXISTS, which is a
        boundary leak even though read_state filters the values -- it would
        start asking about a target identity it should not know of."""
        from core.mission import load_mission
        from providers.cloud.gemini_live import tool_declarations_for

        mission = load_mission(reference_mission_path)
        tools = tool_declarations_for(mission)
        text = " ".join(fn.description for tool in tools
                        for fn in tool.function_declarations)

        hidden = [p.id for p in mission.parameters if not p.visible_to_persona]
        assert hidden, "reference mission should hide at least one parameter"
        for parameter_id in hidden:
            assert parameter_id not in text

    def test_hidden_state_stays_hidden_through_live_tools(self, reference_mission_path) -> None:
        """The knowledge boundary must hold on the Live path too.

        Checked against the engine rather than the declaration text: the
        declaration lists ids for the model's convenience, but what
        actually protects the boundary is read_state filtering its output.
        """
        from core.mission import load_mission
        from core.state import MissionStateEngine

        mission = load_mission(reference_mission_path)
        engine = MissionStateEngine(mission)
        persona_view = engine.snapshot(for_persona=True)

        hidden = [p.id for p in mission.parameters if not p.visible_to_persona]
        assert hidden, "reference mission should hide at least one parameter"
        for parameter_id in hidden:
            assert parameter_id not in persona_view

    def test_missing_key_raises_clearly(self, monkeypatch) -> None:
        from providers.base import ProviderError
        from providers.cloud.gemini_live import GeminiLiveProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
            GeminiLiveProvider()._require_key()

    def test_audio_rates_differ_by_direction(self) -> None:
        """Live takes 16kHz in and emits 24kHz out. Playing at the wrong
        rate makes the voice sound chipmunked or slurred."""
        from providers.cloud.gemini_live import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
        assert INPUT_SAMPLE_RATE == 16000
        assert OUTPUT_SAMPLE_RATE == 24000
        assert INPUT_SAMPLE_RATE != OUTPUT_SAMPLE_RATE


class TestPromptMarkerModes:
    """Regression: the counterpart said "hesitate" out loud.

    Gemini Live owns its own prosody, so the marker vocabulary must be
    OMITTED for it -- stripping the guillemets afterwards leaves the bare
    word, which it then reads aloud.
    """

    def test_markers_present_by_default(self, reference_mission_path) -> None:
        from agent.prompts import build_system_prompt
        from core.mission import load_mission
        from core.state import MissionStateEngine

        mission = load_mission(reference_mission_path)
        prompt = build_system_prompt(mission, MissionStateEngine(mission))
        assert "«hesitate»" in prompt

    def test_markers_absent_when_disabled(self, reference_mission_path) -> None:
        from agent.prompts import build_system_prompt
        from core.mission import load_mission
        from core.state import MissionStateEngine

        mission = load_mission(reference_mission_path)
        prompt = build_system_prompt(mission, MissionStateEngine(mission),
                                     with_markers=False)
        # Neither the marker nor the bare word may survive.
        assert "hesitate" not in prompt
        assert "«" not in prompt
