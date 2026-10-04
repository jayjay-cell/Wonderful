"""Typed domain models for the deterministic core.

No LLM, no LangChain, no network, no I/O. tests/test_architecture.py walks
imports and fails if anything in core/ reaches a framework or an HTTP
client -- without that test the layering erodes within weeks.

THE CENTRAL DESIGN DECISION (docs/ARCHITECTURE.md §3): mission state is
NOT typed domain fields. There is no `fuel_lb: float` anywhere in this
file, because the engine must not know what fuel is. A mission file
DECLARES its parameters and this module describes the shape of such a
declaration.

The obvious alternative --

    class MissionState(BaseModel):     # <- rejected
        fuel_lb: float
        altitude_ft: float

-- fails the product's core requirement (FR-E2/E3/E7): adding a parameter
would mean editing this model, the tools and the prompt, and a naval
scenario with no fuel at all would be a rewrite rather than a new file.

The cost of this choice is losing static typing on state values, which is
why MissionLoader validates strictly and fails loudly (FR-E8). That
trade-off is deliberate and recorded as ADR-4.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Parameters -- what a mission declares as its state
# ---------------------------------------------------------------------------


class ParameterType(str, Enum):
    NUMBER = "number"
    ENUM = "enum"
    BOOL = "bool"
    TEXT = "text"


class DynamicsKind(str, Enum):
    """How a parameter changes as mission time advances.

    A closed set on purpose. Each kind is a small pure function in
    core/dynamics.py, separately tested. Adding a kind is additive --
    nothing existing changes (FR-E10) -- and is the ONLY situation in which
    a new domain needs code rather than a new mission file.
    """

    HOLD = "hold"                              # unchanged until commanded
    LINEAR_DRAIN = "linear_drain"              # decreases at a rate, with a floor
    LINEAR_FILL = "linear_fill"                # increases at a rate, with a ceiling
    RATE_TOWARD_TARGET = "rate_toward_target"  # moves toward a commanded target
    STEPPED = "stepped"                        # changes only on command
    SCRIPTED = "scripted"                      # follows mission-time keyframes


class Keyframe(BaseModel):
    """One scheduled value change for a `scripted` parameter."""

    at: float = Field(description="Mission time in seconds")
    value: Any


class Dynamics(BaseModel):
    """Per-kind configuration. Which fields are required depends on `kind`;
    the cross-field check below is what turns a half-specified dynamics
    block into a loud load-time failure instead of a surprise at runtime."""

    kind: DynamicsKind
    rate_per_hour: float | None = None     # linear_drain / linear_fill
    rate_per_minute: float | None = None   # rate_toward_target
    floor: float | None = None             # linear_drain
    ceiling: float | None = None           # linear_fill
    keyframes: list[Keyframe] = Field(default_factory=list)  # scripted

    @model_validator(mode="after")
    def _check_required_fields_for_kind(self) -> Dynamics:
        kind = self.kind
        if kind in (DynamicsKind.LINEAR_DRAIN, DynamicsKind.LINEAR_FILL):
            if self.rate_per_hour is None:
                raise ValueError(f"dynamics kind '{kind.value}' requires 'rate_per_hour'")
        elif kind is DynamicsKind.RATE_TOWARD_TARGET:
            if self.rate_per_minute is None:
                raise ValueError("dynamics kind 'rate_toward_target' requires 'rate_per_minute'")
        elif kind is DynamicsKind.SCRIPTED:
            if not self.keyframes:
                raise ValueError("dynamics kind 'scripted' requires at least one keyframe")
        return self


class Parameter(BaseModel):
    """One declared piece of mission state.

    `visible_to_persona` is the knowledge boundary (FR-D6) and is enforced
    structurally, not by asking the model nicely: MissionStateEngine
    .snapshot(for_persona=True) omits these entirely, so the counterpart
    cannot report state it should not have even under an adversarial
    prompt. That is the main defence of perceived fairness -- a counterpart
    who knows the trainee's unspoken intent feels like cheating and the
    trainee stops trusting the simulation.
    """

    id: str
    type: ParameterType
    initial: Any
    label: str | None = None          # Hebrew name shown to model and UI; defaults to id
    unit: str | None = None
    min: float | None = None          # number only
    max: float | None = None          # number only
    values: list[str] | None = None   # enum only
    dynamics: Dynamics | None = None  # omitted means constant
    visible_to_persona: bool = True

    # Spoken form of each enum value, for when the id and the word the
    # counterpart should SAY differ. Enum ids stay machine-friendly ASCII
    # (`idle`, `tracking`) because they appear in conditions and effects,
    # but an operator on a Hebrew net must not say "idle" -- observed live,
    # and it breaks immersion instantly. Maps value -> spoken label.
    value_labels: dict[str, str] = Field(default_factory=dict)

    # Extra spoken forms the TRAINEE might use when ordering a change,
    # beyond the one label the counterpart says. Hebrew verb and noun
    # forms differ ("סורק" vs "סריקה"), and refusing a legal order over
    # word form would read as the simulator being obtuse.
    value_aliases: dict[str, list[str]] = Field(default_factory=dict)

    def spoken(self, value: Any) -> str:
        """How this value should be said aloud."""
        return self.value_labels.get(str(value), str(value))

    def resolve_spoken(self, text: str) -> str | None:
        """Map a spoken word back to its machine value, or None.

        Checks the canonical label first, then any aliases.
        """
        needle = text.strip().lower()
        for machine_value, spoken in self.value_labels.items():
            if spoken.strip().lower() == needle:
                return machine_value
        for machine_value, aliases in self.value_aliases.items():
            if any(a.strip().lower() == needle for a in aliases):
                return machine_value
        return None

    @property
    def display_name(self) -> str:
        return self.label or self.id

    @model_validator(mode="after")
    def _check_type_consistency(self) -> Parameter:
        if self.type is ParameterType.ENUM:
            if not self.values:
                raise ValueError(f"parameter '{self.id}': type 'enum' requires 'values'")
            if self.initial not in self.values:
                raise ValueError(
                    f"parameter '{self.id}': initial {self.initial!r} is not one of {self.values}"
                )
        else:
            if self.values is not None:
                raise ValueError(f"parameter '{self.id}': 'values' is only valid for type 'enum'")

        if self.type is ParameterType.NUMBER:
            if not isinstance(self.initial, (int, float)) or isinstance(self.initial, bool):
                raise ValueError(f"parameter '{self.id}': type 'number' requires a numeric initial")
            if self.min is not None and self.initial < self.min:
                raise ValueError(f"parameter '{self.id}': initial {self.initial} is below min {self.min}")
            if self.max is not None and self.initial > self.max:
                raise ValueError(f"parameter '{self.id}': initial {self.initial} is above max {self.max}")
        else:
            if self.min is not None or self.max is not None:
                raise ValueError(f"parameter '{self.id}': 'min'/'max' are only valid for type 'number'")

        if self.type is ParameterType.BOOL and not isinstance(self.initial, bool):
            raise ValueError(f"parameter '{self.id}': type 'bool' requires a true/false initial")

        return self


class DerivedValue(BaseModel):
    """A value computed from parameters on every read, never stored.

    This is what makes FR-D2 structural rather than aspirational: endurance
    cannot contradict fuel, because endurance has no independent existence
    to drift out of sync. `expr` is evaluated by core/derived.py's
    restricted evaluator -- NEVER Python eval(), since mission files are
    the natural "import a mission someone sent me" path and eval() there
    would be a code-execution hole.
    """

    id: str
    expr: str
    unit: str | None = None
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.id


# ---------------------------------------------------------------------------
# State changes
# ---------------------------------------------------------------------------


class StateCommand(BaseModel):
    """A requested change to one parameter, validated against that
    parameter's own declaration before anything mutates."""

    parameter_id: str
    value: Any
    source: Literal["tool", "trigger", "dynamics"] = "tool"


class EffectKind(str, Enum):
    VALUE_CHANGED = "value_changed"
    LIMIT_REACHED = "limit_reached"      # hit a declared floor/ceiling/min/max
    TARGET_REACHED = "target_reached"    # rate_toward_target arrived


class Effect(BaseModel):
    """Something that happened to state. Returned by the engine so callers
    (triggers, logging, the UI) can react without re-deriving it."""

    kind: EffectKind
    parameter_id: str
    old_value: Any = None
    new_value: Any = None
    detail: str | None = None


class CommandResult(BaseModel):
    """Outcome of applying a StateCommand.

    A rejected command mutates NOTHING (FR-D3) -- `accepted=False` means
    state is exactly as it was. `reason_code` is safe to speak in
    character; it never leaks internals.
    """

    accepted: bool
    parameter_id: str
    reason_code: str | None = None
    message: str | None = None
    effects: list[Effect] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Mission definition -- the authored file (docs/specs/mission-format.md)
# ---------------------------------------------------------------------------


class Setting(BaseModel):
    """Purpose and place -- the WHERE and WHY of a scenario (FR-E4)."""

    purpose: str
    place: str
    trainee_role: str
    briefing: str | None = None       # shown to the TRAINEE, not the model

    # The situation as the COUNTERPART understands it, in the author's own
    # words. Distinct from `briefing` on purpose: the trainee's briefing
    # may contain intent and priorities the operator must not know, and
    # feeding it to the model would quietly breach the knowledge boundary.
    situation: str | None = None

    # Free-text instructions appended last, so they can override anything
    # the generated prompt says. The escape hatch for a scenario that needs
    # something no field anticipates.
    extra_instructions: str | None = None


class PersonaTraits(BaseModel):
    """0.0-1.0 dials shaping what the counterpart volunteers versus waits to
    be asked. Not physics -- they feed the system prompt."""

    patience: float = Field(default=0.5, ge=0.0, le=1.0)
    deference: float = Field(default=0.5, ge=0.0, le=1.0)
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Persona(BaseModel):
    name: str
    role: str
    experience: str | None = None
    traits: PersonaTraits = Field(default_factory=PersonaTraits)

    # FREE-TEXT AUTHORING. The structured traits above cover the common
    # dials, but no fixed schema can anticipate every instruction a trainer
    # needs to give -- "he resents being second-guessed about sensor
    # settings", "he has flown this area before and says so".
    #
    # These go into the system prompt close to verbatim, so the author can
    # write guidance in their own words instead of being limited to the
    # fields someone thought of in advance. This is the main answer to
    # "the instructions may vary".
    speech_guide: str | None = None   # how he talks: phrasing, register, habits
    behaviour_guide: str | None = None  # how he acts: judgement, initiative, quirks
    background: str | None = None     # history he can draw on in conversation

    # Example transmissions, in his own voice. Worth more than any amount
    # of description: a model imitates a sample far more reliably than it
    # follows an adjective.
    speech_examples: list[str] = Field(default_factory=list)

    # Parameter ids the counterpart may read. Empty means "all visible
    # parameters". Anything outside this is filtered from its snapshot.
    knows: list[str] = Field(default_factory=list)
    does_not_know: list[str] = Field(default_factory=list)


class ToneProfile(BaseModel):
    """How a named tone sounds. `pace` multiplies delivery speed, so an
    urgent operator is both terser and faster -- the realism layer reads
    these (docs/specs/realism-engine.md §4)."""

    pace: float = Field(default=1.0, gt=0.0)
    terseness: float = Field(default=0.5, ge=0.0, le=1.0)
    filler_multiplier: float = Field(default=1.0, ge=0.0)


class ToneShift(BaseModel):
    when: str      # a trigger id
    to: str        # a tone name


class Tone(BaseModel):
    baseline: str = "neutral"
    shifts: list[ToneShift] = Field(default_factory=list)
    profiles: dict[str, ToneProfile] = Field(default_factory=dict)


class DelayRange(BaseModel):
    """Lognormal by default, deliberately.

    Real reply delays cluster short with a long tail. Uniform delay is the
    single biggest reason naive simulators read as network lag rather than
    as someone thinking (docs/specs/realism-engine.md §8).
    """

    min: int = Field(ge=0)
    max: int = Field(ge=0)
    distribution: Literal["lognormal", "uniform"] = "lognormal"

    @model_validator(mode="after")
    def _check_order(self) -> DelayRange:
        if self.max < self.min:
            raise ValueError(f"delay range max ({self.max}) is below min ({self.min})")
        return self


class GarbleBy(BaseModel):
    """Ties transmission degradation to live mission state (FR-A6): poor
    comms quality produces more garbling, without the model deciding it."""

    parameter: str
    probabilities: dict[str, float]


class RealismConfig(BaseModel):
    """Every human-imperfection number, all authored in the mission file so
    tuning never requires a code change (FR-E5). Expect several passes --
    realism is iterated, not specified."""

    response_delay_ms: DelayRange = Field(default_factory=lambda: DelayRange(min=600, max=2400))
    stall_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    stall_duration_ms: DelayRange = Field(default_factory=lambda: DelayRange(min=400, max=1600))
    filler_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    filler_sounds: list[str] = Field(default_factory=lambda: ["אה", "רגע"])
    self_correction_probability: float = Field(default=0.08, ge=0.0, le=1.0)
    interrupt_trainee_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    allow_barge_in: bool = True
    barge_in_grace_ms: int = Field(default=250, ge=0)
    garble_by: GarbleBy | None = None
    seed: int | None = None            # None = random per session

    # Delivery rate in characters per second. None uses the channel's
    # default (fast for text, real speaking rate for voice). Set it when a
    # persona should deliver noticeably faster or slower than normal.
    chars_per_second: float | None = Field(default=None, gt=0)


class BrevityTerm(BaseModel):
    term: str
    meaning: str


class ProcedureRule(BaseModel):
    """Procedure compliance is decided DETERMINISTICALLY from this config;
    only the wording of a challenge is the model's. A counterpart that
    challenges a correct call because the model "felt" it was wrong
    destroys trust in the simulator."""

    id: str
    requires_readback_for: list[str] = Field(default_factory=list)
    applies_to: str | None = None       # e.g. "first_transmission"
    on_violation: Literal["challenge", "accept_with_note", "ignore"] = "challenge"


class Callsigns(BaseModel):
    counterpart: str
    trainee: str


class ReportFormat(BaseModel):
    """An agreed message STRUCTURE, not just vocabulary.

    A brevity dictionary tells the counterpart which words exist; it does
    not tell him what a transmission is shaped like. Real nets have agreed
    formats -- who calls whom, in what order, which fields in which
    sequence -- and an operator who uses the right words in the wrong
    structure is still doing it wrong.

    `template` and `example` are authored free text, reproduced in the
    prompt close to verbatim. The example matters more than the template:
    a model matches a sample far more reliably than it parses a notation.
    """

    id: str
    when: str                        # when to use this format, in plain words
    template: str | None = None      # e.g. "<addressee>, <self>, <body>"
    example: str | None = None       # a real transmission in this format
    required_fields: list[str] = Field(default_factory=list)


class Procedure(BaseModel):
    callsigns: Callsigns
    brevity: list[BrevityTerm] = Field(default_factory=list)
    rules: list[ProcedureRule] = Field(default_factory=list)

    # The agreed message structure(s) for this net.
    report_formats: list[ReportFormat] = Field(default_factory=list)

    # Free-text comms doctrine, in the author's own words, appended after
    # the generated procedure section. For everything a schema cannot
    # capture: who initiates, how to handle a broken transmission, when to
    # authenticate, what never goes over the net in clear.
    comms_guide: str | None = None


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"              # survives barge-in; see ADR in spec


class TriggerEffect(BaseModel):
    parameter: str
    value: Any


class BaseTrigger(BaseModel):
    """Common shape for every trigger kind.

    `say_intent` is an INTENT, never a line of dialogue (FR-C7). It enters
    the conversation as labelled data so the counterpart phrases it in his
    own voice; a literal script would be read out identically every time
    and immediately feel canned.
    """

    id: str
    priority: Priority = Priority.NORMAL
    effects: list[TriggerEffect] = Field(default_factory=list)

    # Optional, because a checkpoint may exist purely to change the world
    # (weather turns, a target moves) with no transmission attached.
    say_intent: str | None = None

    # Author's notes for this checkpoint, in their own words. Added to the
    # cue so one moment can carry specific guidance -- "he is annoyed at
    # being asked again", "he should sound less certain here" -- without
    # needing a new schema field for every situation.
    instructions: str | None = None

    # A one-line label for the trainer's own reading: timelines get long,
    # and ids alone stop being readable.
    label: str | None = None


class TimelineTrigger(BaseTrigger):
    """A checkpoint at a set mission time.

    `when` makes a checkpoint CONDITIONAL: it fires at its time only if the
    condition also holds. Without that, a timeline is a fixed script that
    plays out identically no matter what the trainee did -- so a "target
    escapes" checkpoint would fire even after the trainee correctly tracked
    it, which teaches the opposite of the intended lesson.

    `after` chains checkpoints: this one is armed only once the named
    checkpoint has fired, so a sequence stays in order even when the
    earlier step was delayed by its own condition.
    """

    at_mission_seconds: float = Field(ge=0)
    when: str | None = None
    after: str | None = None


class ThresholdTrigger(BaseTrigger):
    when: str                 # restricted expression over state
    once: bool = True


class IdleTrigger(BaseTrigger):
    after_silence_seconds: float = Field(gt=0)
    max_fires: int = Field(default=3, ge=1)


class Triggers(BaseModel):
    timeline: list[TimelineTrigger] = Field(default_factory=list)
    thresholds: list[ThresholdTrigger] = Field(default_factory=list)
    idle: list[IdleTrigger] = Field(default_factory=list)

    def all_ids(self) -> list[str]:
        return [t.id for t in (*self.timeline, *self.thresholds, *self.idle)]


class PlanStep(BaseModel):
    """One briefed step of the mission plan, as the counterpart knows it
    BEFORE the simulation starts.

    WHY THIS IS SEPARATE FROM `triggers`: triggers are what the simulation
    DOES TO him -- events he discovers as they happen. A plan is what he
    was BRIEFED ON and already expects. A real operator takes off knowing
    the sortie profile: when to check in, when to be on station, what the
    reporting schedule is. Without this he behaves like someone who has
    never seen the mission before, which is the wrong training experience.

    The two work together: a plan step says "at H+5 I report on station",
    and a trigger makes it actually happen. Authors may use either alone --
    a plan step with no trigger is something he expects and must remember
    to do himself.
    """

    at: str | None = None            # free text: "H+5", "on station", "every 10 min"
    what: str                        # what happens or what he must do
    note: str | None = None          # author's aside for this step


class MissionPlan(BaseModel):
    """The mission plan the counterpart is briefed on up front.

    All free text deliberately: a plan is something a trainer writes in
    their own words, and forcing it into a schema would lose exactly the
    detail that makes a scenario specific.
    """

    overview: str | None = None      # the sortie in a few sentences
    steps: list[PlanStep] = Field(default_factory=list)

    # The agreed reporting schedule -- what he updates, and when. This is
    # the part that makes him proactive rather than purely reactive.
    reporting_schedule: str | None = None

    # What he has been told to expect, so a planned event is not a
    # surprise: "visibility is forecast to drop after midday".
    expected_events: list[str] = Field(default_factory=list)

    # Standing orders: what to do without being asked.
    standing_orders: list[str] = Field(default_factory=list)


class Limits(BaseModel):
    max_steps: int = Field(default=8, ge=1)
    session_max_minutes: int = Field(default=30, ge=1)


class Mission(BaseModel):
    """A complete authored scenario -- the product's real interface.

    The code is a generic engine; this is where a scenario actually lives.
    A new training scenario is a new file of this shape and nothing else
    (FR-E1/E7). Loaded and cross-validated by core/mission.py, which fails
    loudly rather than defaulting silently (FR-E8).
    """

    id: str
    version: int = 1
    language: str = "he"
    title: str

    setting: Setting
    persona: Persona
    tone: Tone = Field(default_factory=Tone)
    realism: RealismConfig = Field(default_factory=RealismConfig)
    parameters: list[Parameter]
    derived: list[DerivedValue] = Field(default_factory=list)
    procedure: Procedure

    # What the counterpart is BRIEFED on before the simulation starts,
    # as distinct from `triggers`, which is what happens TO him during it.
    plan: MissionPlan = Field(default_factory=MissionPlan)

    triggers: Triggers = Field(default_factory=Triggers)
    limits: Limits = Field(default_factory=Limits)

    def parameter(self, parameter_id: str) -> Parameter | None:
        return next((p for p in self.parameters if p.id == parameter_id), None)

    def parameter_ids(self) -> list[str]:
        return [p.id for p in self.parameters]

    def derived_ids(self) -> list[str]:
        return [d.id for d in self.derived]
