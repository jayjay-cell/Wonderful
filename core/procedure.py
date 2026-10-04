"""Comms procedure checking -- deterministic, from the mission's own rules.

Pure: no LLM, no network, no I/O.

WHY THIS IS CODE AND NOT A PROMPT INSTRUCTION: whether a transmission was
procedurally correct is decided here; only the WORDING of a challenge is
the model's. A counterpart that challenges a CORRECT call because the
model "felt" it was wrong destroys trust in the simulator faster than
almost anything else -- the trainee concludes the simulation is arbitrary
and stops taking the assessment seriously.

The inverse matters too: a counterpart that accepts a sloppy call teaches
the trainee that sloppiness is fine.

So the decision is a lookup against the mission's authored dictionary, and
the model is told not to override it.

DELIBERATE LIMITATION: this is keyword-and-pattern matching over the
mission's declared brevity terms and callsigns. It does not understand
Hebrew grammar, and it cannot judge whether a readback was semantically
correct -- only whether the value appears to have been read back at all.
That is honest for an MVP and avoids the worse failure of a confident
wrong verdict. Step 8's officer validation is where the rules get
sharpened against real procedure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from core.models import Mission

ViolationKind = Literal["missing_callsign", "missing_readback", "unknown_brevity"]


@dataclass(frozen=True)
class Violation:
    rule_id: str
    kind: ViolationKind
    detail: str


@dataclass(frozen=True)
class ProcedureReport:
    compliant: bool
    violations: list[Violation] = field(default_factory=list)
    should_challenge: bool = False


def check_transmission(
    mission: Mission,
    transmission: str,
    is_first_transmission: bool = False,
    awaiting_readback_for: list[str] | None = None,
) -> ProcedureReport:
    """Assess one trainee transmission against the mission's procedure.

    `awaiting_readback_for` lists parameter ids the counterpart has just
    reported and which the mission requires the trainee to read back. The
    session layer tracks that; this function stays pure.
    """
    procedure = mission.procedure
    violations: list[Violation] = []

    text = (transmission or "").strip()
    if not text:
        return ProcedureReport(compliant=True)

    normalized = _normalize(text)

    for rule in procedure.rules:
        if rule.on_violation == "ignore":
            continue

        # -- callsign ----------------------------------------------------
        if rule.applies_to == "first_transmission":
            # Only assessable on the opening transmission; applying it to
            # every message would flag normal mid-conversation traffic,
            # which no real net requires.
            if is_first_transmission:
                expected = {procedure.callsigns.counterpart, procedure.callsigns.trainee}
                if not any(_contains_phrase(normalized, c) for c in expected):
                    violations.append(Violation(
                        rule_id=rule.id,
                        kind="missing_callsign",
                        detail="no callsign in the opening transmission",
                    ))

        # -- readback ----------------------------------------------------
        if rule.requires_readback_for and awaiting_readback_for:
            for parameter_id in rule.requires_readback_for:
                if parameter_id not in awaiting_readback_for:
                    continue
                if not _looks_like_readback(normalized, mission, parameter_id):
                    parameter = mission.parameter(parameter_id)
                    name = parameter.display_name if parameter else parameter_id
                    violations.append(Violation(
                        rule_id=rule.id,
                        kind="missing_readback",
                        detail=f"no readback of {name}",
                    ))

    should_challenge = any(
        _rule_for(mission, v.rule_id) == "challenge" for v in violations
    )
    return ProcedureReport(
        compliant=not violations,
        violations=violations,
        should_challenge=should_challenge,
    )


def brevity_terms(mission: Mission) -> dict[str, str]:
    """The mission's brevity dictionary, for the system prompt.

    Supplied to the model so it USES the right terms, while compliance
    checking stays here.
    """
    return {term.term: term.meaning for term in mission.procedure.brevity}


# -- internals -------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace and punctuation.

    Hebrew has no case, so lowering is a no-op there and matters only for
    a Latin-script mission -- the engine is language-agnostic (FR-E9), so
    this handles both.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s֐-׿]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Whether a multi-word phrase such as a callsign appears.

    Substring matching on the normalized form, since a callsign like
    "נחשון 3" contains a space and a digit and will not survive naive
    token comparison.
    """
    return _normalize(phrase) in normalized_text


def _looks_like_readback(normalized_text: str, mission: Mission, parameter_id: str) -> bool:
    """Whether the transmission appears to read back a parameter.

    Accepts either the parameter's name plus a number, or a bare number
    when the counterpart has just reported that parameter -- a real
    readback is often just "two zero thousand", with the context implied.

    Deliberately lenient: a false "you did not read that back" challenge
    is more damaging to trust than a missed one, because the trainee knows
    they did read it back and concludes the simulator is broken.
    """
    parameter = mission.parameter(parameter_id)
    has_number = bool(re.search(r"\d", normalized_text))

    if parameter is not None:
        name = _normalize(parameter.display_name)
        if name and name in normalized_text:
            return True
        if parameter.unit and _normalize(parameter.unit) in normalized_text:
            return True

    return has_number


def _rule_for(mission: Mission, rule_id: str) -> str:
    rule = next((r for r in mission.procedure.rules if r.id == rule_id), None)
    return rule.on_violation if rule else "challenge"
