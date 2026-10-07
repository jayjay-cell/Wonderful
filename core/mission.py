"""Mission state: initial facts and per-exercise configuration.

Pure apart from reading YAML and Markdown.

WHAT THIS IS NOT ANY MORE. The previous version declared mutable
parameters with physical dynamics -- fuel draining, altitude climbing
toward a commanded target -- because the product was understood as a
flight simulator. It is not. The recording is immutable, so there is
nothing to simulate: there are authored INITIAL FACTS, and the timeline
updates them as the exercise unfolds.

So `parameters` with `dynamics` is gone, and with it the idea that a
trainee's instruction changes the world.

THREE AUTHORED SOURCES, and this is the third:

  context/*.md      reusable professional knowledge, shared by exercises
  timeline .xlsx    what happens during the recording
  missions/*.yaml   THIS FILE -- who, where, and how this exercise behaves

Validation fails loudly. An exercise that loaded with a dangling reference
would run missing a beat the author believed was there.

Fields are mostly optional on purpose: exercises differ in shape, and
requiring a fuel figure for an exercise where fuel is irrelevant would
invite a fabricated number that looks like a real fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from core.conversation_state import DEFAULT_REESTABLISH_SILENCE


class MissionError(Exception):
    """A mission file could not be loaded. Names the field path."""

    def __init__(self, code: str, message: str, path: str | None = None,
                 source: str | Path | None = None) -> None:
        """Carry the offending field's path alongside the message, so a bad mission file says where."""
        where = f" at {path}" if path else ""
        origin = f" in {Path(source).name}" if source else ""
        super().__init__(f"[{code}]{origin}{where}: {message}")
        self.code = code
        self.path = path


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class Callsigns(BaseModel):
    """Who is on the net. Mission DATA, not engine constants -- the brief
    is explicit that these vary by exercise."""

    operator: str                       # e.g. גלוק
    trainee: str                        # e.g. מדבקה
    controller: str | None = None       # e.g. משנה — named in context, not simulated


class Setting(BaseModel):
    """Where and why, plus what preceded the recording."""

    purpose: str = ""
    area: str = ""
    background: str = ""
    before_recording: str = ""          # what happened before the video starts

    # Shown to the TRAINEE only. Kept separate from crew.prior_briefing
    # because it may carry the intelligence intent Glok must not know.
    trainee_briefing: str = ""


class Crew(BaseModel):
    """The crew, and which member the trainee actually talks to."""

    operator_name: str = ""
    role: str = ""
    experience: str = ""
    squadron: str = ""
    composition: str = ""               # free text: "three-person crew"
    gender: str = ""                    # for voice and grammatical agreement

    # What the crew was told at the squadron before the sortie. Glok knows
    # this from the first word -- they do not pretend ignorance -- but it
    # must not contain unrevealed timeline events.
    prior_briefing: str = ""


class Platform(BaseModel):
    """Aircraft and sensor, as capabilities rather than as state."""

    aircraft: str = ""
    sensor: str = ""
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class Behaviour(BaseModel):
    """How this crew behaves. Independently configurable per exercise.

    0.0-1.0 with documented meaning at the ends, since an undocumented
    magic number is a setting nobody can tune:

      initiative  0 waits to be spoken to · 1 volunteers and chases gaps
      challenge   0 accepts every instruction · 1 argues its professional view
      verbosity   0 minimum words · 1 explains reasoning
      patience    0 audibly impatient on repetition · 1 unbothered
    """

    initiative: float = Field(default=0.5, ge=0.0, le=1.0)
    challenge: float = Field(default=0.5, ge=0.0, le=1.0)
    verbosity: float = Field(default=0.4, ge=0.0, le=1.0)
    patience: float = Field(default=0.6, ge=0.0, le=1.0)

    style: str = ""                     # free text, the most useful field

    # Grace period before a proactive crew asks for a skipped briefing.
    # INITIAL TUNING DEFAULT, not a real-world rule.
    briefing_request_after: float = Field(default=90.0, ge=0.0)


class HandoverPolicy(BaseModel):
    """Crew rotation behaviour.

    Timing is authored in the TIMELINE, not here: a recording can start
    partway through a shift, so a fixed four-hour schedule would be wrong
    for most exercises. What the crew SAYS at the rotation is the handover
    event's own `operator_information`.

    ACTUAL BEHAVIOUR ACROSS A HANDOVER: facts and agreements carry
    straight over, because one exercise keeps one timeline and one
    ledger. There is no separate incoming-crew memory, so settings for
    inheritance or re-briefing would have described something the engine
    does not do -- they were declared, documented, and read nowhere, and
    are deliberately absent rather than silently ignored.
    """

    busy_reply: str = ""                # if called mid-handover
    allow_urgent: bool = True           # authored urgent events may still come


class ImpossibleRequests(BaseModel):
    """How to decline something the recording cannot do.

    The trainee may ask to zoom, slew or change altitude. None of that can
    happen. The reply must stay in role: never mention a video, never
    invent a mechanical failure, never claim the action was taken.

    `explanations` maps a topic to an AUTHOR-SUPPLIED in-character reason.
    `fallback` is used when nothing specific is authored -- honest rather
    than inventive.
    """

    explanations: dict[str, str] = Field(default_factory=dict)
    fallback: str = ""


class Reporting(BaseModel):
    """Baseline reporting expectations, beyond per-event policy."""

    instructions: str = ""


class PrivateNotes(BaseModel):
    """Author/debrief material. NEVER reaches the operator.

    Its own section with an explicit flag so the exclusion is structural:
    a reader of the mission file can see at a glance that this is hidden,
    and the prompt builder has one obvious thing to skip.
    """

    visible_to_operator: bool = False
    solution: str = ""
    debrief_points: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _must_stay_hidden(self) -> "PrivateNotes":
        """Force private notes invisible to the operator, whatever the file says."""
        if self.visible_to_operator:
            raise ValueError(
                "private.visible_to_operator must be false; this section exists "
                "to hold material the operator must never see"
            )
        return self


class Mission(BaseModel):
    """One authored exercise."""

    id: str
    title: str
    version: int = 1
    language: str = "he"
    duration_seconds: float | None = None   # falls back to the timeline's end

    callsigns: Callsigns
    setting: Setting = Field(default_factory=Setting)
    crew: Crew = Field(default_factory=Crew)
    platform: Platform = Field(default_factory=Platform)
    behaviour: Behaviour = Field(default_factory=Behaviour)
    handover: HandoverPolicy = Field(default_factory=HandoverPolicy)
    impossible_requests: ImpossibleRequests = Field(default_factory=ImpossibleRequests)
    reporting: Reporting = Field(default_factory=Reporting)
    private: PrivateNotes = Field(default_factory=PrivateNotes)

    # Initial readings: free-form, all optional. A dict rather than a typed
    # schema because exercises differ in what matters, and a fixed
    # UAV-shaped schema would force irrelevant or invented values.
    initial_facts: dict[str, Any] = Field(default_factory=dict)

    # Which readings Glok cannot see. Everything else in initial_facts and
    # the revealed timeline is fair game.
    hidden_facts: list[str] = Field(default_factory=list)

    context_files: list[str] = Field(default_factory=list)
    timeline_file: str = ""

    reestablish_silence: float = Field(default=DEFAULT_REESTABLISH_SILENCE, ge=0.0)

    def operator_facts(self) -> dict[str, Any]:
        """Initial facts Glok may see."""
        hidden = set(self.hidden_facts)
        return {k: v for k, v in self.initial_facts.items() if k not in hidden}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mission(path: str | Path) -> Mission:
    """Read and validate a mission YAML. Raises MissionError, naming the field."""
    path = Path(path)
    if not path.exists():
        raise MissionError("FILE_NOT_FOUND", f"no such mission file: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise MissionError("INVALID_YAML", f"could not parse YAML: {err}",
                           source=path) from err
    if not isinstance(raw, dict):
        raise MissionError("INVALID_STRUCTURE",
                           "mission file must be a mapping at the top level",
                           source=path)
    return load_mission_dict(raw, source=path)


def load_mission_dict(raw: dict[str, Any], source: str | Path | None = None) -> Mission:
    """Validate an already-parsed mission. Separate so tests need no temp files."""
    try:
        mission = Mission.model_validate(raw)
    except ValidationError as err:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in err.errors()
        )
        raise MissionError("INVALID_FIELD", problems, source=source) from err

    _check_references(mission, source)
    return mission


def _check_references(mission: Mission, source: str | Path | None) -> None:
    """Cross-field checks Pydantic cannot do alone."""
    unknown = [k for k in mission.hidden_facts if k not in mission.initial_facts]
    if unknown:
        import difflib
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, list(mission.initial_facts), n=1)
            hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise MissionError(
            "UNKNOWN_FACT",
            f"hidden_facts names {', '.join(hints)}, which is not in initial_facts",
            path="hidden_facts", source=source,
        )

    if mission.duration_seconds is not None and mission.duration_seconds <= 0:
        raise MissionError("INVALID_DURATION",
                           "duration_seconds must be positive",
                           path="duration_seconds", source=source)


def load_context(paths: list[str], base: Path) -> str:
    """Concatenate global-context Markdown into one block.

    Shared professional knowledge, loaded per exercise rather than copied
    into each mission file -- so correcting a convention corrects it
    everywhere.
    """
    parts: list[str] = []
    for name in paths:
        candidate = (base / name) if not Path(name).is_absolute() else Path(name)
        if not candidate.exists():
            raise MissionError("CONTEXT_NOT_FOUND",
                               f"context file not found: {name}",
                               path="context_files")
        parts.append(candidate.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(p for p in parts if p)
