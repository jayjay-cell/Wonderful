"""Mission file loading and cross-field validation.

Pure apart from reading the YAML file itself.

THE RULE HERE IS: FAIL LOUDLY (FR-E8). An invalid mission file never loads
with silent defaults. It raises with the field path, the problem, and -- so
a typo takes seconds rather than a debugging session -- a suggestion where
one is obvious.

Why this is strict rather than forgiving: a mission that quietly ignored a
mistyped trigger id would run a scenario missing a safety-critical report,
the trainee would be assessed on it, and nobody would know the trigger was
never armed. Silence is the dangerous failure mode here, not noise.

Pydantic handles per-field validation in core/models.py. This module does
what Pydantic cannot: checks that references BETWEEN sections resolve --
derived expressions naming real parameters, tone shifts naming real
triggers, persona.knows naming real parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from core.derived import ExpressionError, referenced_names, validate_expression
from core.models import Mission, ParameterType


class MissionError(Exception):
    """A mission file could not be loaded.

    `code` is a stable machine-readable reason (the table in
    docs/specs/mission-format.md §12); `path` locates the offending field.
    """

    def __init__(self, code: str, message: str, path: str | None = None,
                 source: str | Path | None = None) -> None:
        location = f" at {path}" if path else ""
        origin = f" in {source}" if source else ""
        super().__init__(f"[{code}]{origin}{location}: {message}")
        self.code = code
        self.path = path
        self.source = str(source) if source else None


def load_mission(path: str | Path) -> Mission:
    """Load, parse and fully validate a mission file.

    Raises MissionError for anything wrong -- never returns a partially
    valid mission.
    """
    path = Path(path)
    if not path.exists():
        raise MissionError("FILE_NOT_FOUND", f"no such mission file: {path}", source=path)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise MissionError("INVALID_YAML", f"could not parse YAML: {err}", source=path) from err

    if raw is None:
        raise MissionError("EMPTY_FILE", "mission file is empty", source=path)
    if not isinstance(raw, dict):
        raise MissionError(
            "INVALID_STRUCTURE",
            f"mission file must be a mapping at the top level, got {type(raw).__name__}",
            source=path,
        )

    return load_mission_dict(raw, source=path)


def load_mission_dict(raw: dict[str, Any], source: str | Path | None = None) -> Mission:
    """Validate an already-parsed mission mapping. Separated from file
    reading so tests can build missions inline with no temp files."""
    try:
        mission = Mission.model_validate(raw)
    except ValidationError as err:
        raise MissionError("INVALID_FIELD", _format_pydantic_error(err), source=source) from err

    _validate_cross_references(mission, source)
    return mission


def _format_pydantic_error(err: ValidationError) -> str:
    """Turn Pydantic's structured errors into one readable line per problem,
    with the field path spelled out as it appears in the YAML."""
    lines = []
    for e in err.errors():
        location = ".".join(str(p) for p in e["loc"]) or "<root>"
        lines.append(f"{location}: {e['msg']}")
    return "; ".join(lines)


def _validate_cross_references(mission: Mission, source: str | Path | None) -> None:
    """Checks spanning more than one section -- the ones Pydantic cannot do
    because they need the whole document."""

    parameter_ids = mission.parameter_ids()
    derived_ids = mission.derived_ids()

    # -- duplicate ids --------------------------------------------------
    _check_duplicates(parameter_ids, "parameters", source)
    _check_duplicates(derived_ids, "derived", source)
    _check_duplicates(mission.triggers.all_ids(), "triggers", source)

    overlap = set(parameter_ids) & set(derived_ids)
    if overlap:
        raise MissionError(
            "DUPLICATE_ID",
            f"id(s) used by both a parameter and a derived value: {sorted(overlap)}",
            source=source,
        )

    known_names = set(parameter_ids) | set(derived_ids)

    # -- derived expressions --------------------------------------------
    for spec in mission.derived:
        try:
            validate_expression(spec.expr, allowed_names=known_names)
        except ExpressionError as err:
            raise MissionError(
                _expression_error_code(err),
                str(err) + _suggest_for_expression(err, known_names),
                path=f"derived.{spec.id}.expr", source=source,
            ) from err

        if spec.id in referenced_names(spec.expr):
            raise MissionError(
                "CIRCULAR_REFERENCE",
                f"derived value {spec.id!r} refers to itself",
                path=f"derived.{spec.id}.expr", source=source,
            )

    _check_derived_cycles(mission, source)

    # -- threshold conditions -------------------------------------------
    for trigger in mission.triggers.thresholds:
        try:
            validate_expression(trigger.when, allowed_names=known_names)
        except ExpressionError as err:
            raise MissionError(
                _expression_error_code(err),
                str(err) + _suggest_for_expression(err, known_names),
                path=f"triggers.thresholds.{trigger.id}.when", source=source,
            ) from err

    # -- enum spoken labels ---------------------------------------------
    for parameter in mission.parameters:
        for key in (*parameter.value_labels, *parameter.value_aliases):
            if key not in (parameter.values or []):
                raise MissionError(
                    "UNKNOWN_VALUE_LABEL",
                    f"value_labels on {parameter.id!r} has an entry for {key!r}, "
                    f"which is not one of {parameter.values}"
                    + _suggest(key, list(parameter.values or [])),
                    path=f"parameters.{parameter.id}.value_labels", source=source,
                )

    # -- timeline checkpoint conditions and chaining ---------------------
    timeline_ids = {t.id for t in mission.triggers.timeline}
    for trigger in mission.triggers.timeline:
        if trigger.when:
            try:
                validate_expression(trigger.when, allowed_names=known_names)
            except ExpressionError as err:
                raise MissionError(
                    _expression_error_code(err),
                    str(err) + _suggest_for_expression(err, known_names),
                    path=f"triggers.timeline.{trigger.id}.when", source=source,
                ) from err

        if trigger.after:
            if trigger.after == trigger.id:
                raise MissionError(
                    "CIRCULAR_REFERENCE",
                    f"checkpoint {trigger.id!r} waits for itself",
                    path=f"triggers.timeline.{trigger.id}.after", source=source,
                )
            if trigger.after not in timeline_ids:
                raise MissionError(
                    "UNKNOWN_TRIGGER_REF",
                    f"checkpoint {trigger.id!r} waits for {trigger.after!r}, which is "
                    f"not a timeline checkpoint"
                    + _suggest(trigger.after, sorted(timeline_ids)),
                    path=f"triggers.timeline.{trigger.id}.after", source=source,
                )

    _check_checkpoint_chain_cycles(mission, source)

    # A checkpoint with neither a transmission nor a world change does
    # nothing at all. Almost certainly an unfinished edit, and silence
    # here would mean a scenario quietly missing a beat the author
    # believed was there.
    for group_name, group in (
        ("timeline", mission.triggers.timeline),
        ("thresholds", mission.triggers.thresholds),
        ("idle", mission.triggers.idle),
    ):
        for trigger in group:
            if not trigger.say_intent and not trigger.effects:
                raise MissionError(
                    "EMPTY_TRIGGER",
                    f"{group_name} entry {trigger.id!r} has neither 'say_intent' nor "
                    f"'effects', so it would do nothing when it fires",
                    path=f"triggers.{group_name}.{trigger.id}", source=source,
                )

    # -- trigger effects ------------------------------------------------
    for group_name, group in (
        ("timeline", mission.triggers.timeline),
        ("thresholds", mission.triggers.thresholds),
        ("idle", mission.triggers.idle),
    ):
        for trigger in group:
            for effect in trigger.effects:
                param = mission.parameter(effect.parameter)
                if param is None:
                    raise MissionError(
                        "UNKNOWN_PARAMETER_REF",
                        f"effect targets unknown parameter {effect.parameter!r}"
                        + _suggest(effect.parameter, parameter_ids),
                        path=f"triggers.{group_name}.{trigger.id}.effects", source=source,
                    )
                if param.type is ParameterType.ENUM and effect.value not in (param.values or []):
                    raise MissionError(
                        "INVALID_VALUE",
                        f"effect sets {param.id!r} to {effect.value!r}, "
                        f"not one of {param.values}",
                        path=f"triggers.{group_name}.{trigger.id}.effects", source=source,
                    )

    # -- persona knowledge boundary -------------------------------------
    for field_name in ("knows", "does_not_know"):
        for ref in getattr(mission.persona, field_name):
            if ref not in known_names:
                raise MissionError(
                    "UNKNOWN_PARAMETER_REF",
                    f"persona.{field_name} names {ref!r}, which is not a declared "
                    f"parameter or derived value" + _suggest(ref, sorted(known_names)),
                    path=f"persona.{field_name}", source=source,
                )

    # -- tone shifts ----------------------------------------------------
    trigger_ids = set(mission.triggers.all_ids())
    for shift in mission.tone.shifts:
        if shift.when not in trigger_ids:
            raise MissionError(
                "UNKNOWN_TRIGGER_REF",
                f"tone shift condition {shift.when!r} is not a declared trigger id"
                + _suggest(shift.when, sorted(trigger_ids)),
                path="tone.shifts", source=source,
            )
        # A tone with no profile is legal -- it still reaches the prompt as
        # a named manner of speaking -- but a typo'd name would otherwise
        # be invisible, so require either a profile or the baseline.
        if shift.to not in mission.tone.profiles and shift.to != mission.tone.baseline:
            raise MissionError(
                "UNKNOWN_TONE",
                f"tone shift targets {shift.to!r}, which has no entry in tone.profiles"
                + _suggest(shift.to, sorted(mission.tone.profiles)),
                path="tone.shifts", source=source,
            )

    # -- garbling source ------------------------------------------------
    garble = mission.realism.garble_by
    if garble is not None:
        param = mission.parameter(garble.parameter)
        if param is None:
            raise MissionError(
                "INVALID_GARBLE_SOURCE",
                f"realism.garble_by.parameter {garble.parameter!r} is not a declared parameter"
                + _suggest(garble.parameter, parameter_ids),
                path="realism.garble_by.parameter", source=source,
            )
        if param.type is not ParameterType.ENUM:
            raise MissionError(
                "INVALID_GARBLE_SOURCE",
                f"realism.garble_by.parameter {garble.parameter!r} must be an enum "
                f"parameter, but is {param.type.value}",
                path="realism.garble_by.parameter", source=source,
            )
        missing = set(param.values or []) - set(garble.probabilities)
        if missing:
            raise MissionError(
                "INVALID_GARBLE_SOURCE",
                f"realism.garble_by.probabilities is missing entries for "
                f"{sorted(missing)}; every value of {param.id!r} needs one",
                path="realism.garble_by.probabilities", source=source,
            )

    # -- procedure ------------------------------------------------------
    for rule in mission.procedure.rules:
        for ref in rule.requires_readback_for:
            if ref not in known_names:
                raise MissionError(
                    "UNKNOWN_PARAMETER_REF",
                    f"procedure rule {rule.id!r} requires readback for {ref!r}, "
                    f"which is not a declared parameter"
                    + _suggest(ref, sorted(known_names)),
                    path=f"procedure.rules.{rule.id}.requires_readback_for", source=source,
                )


def _check_duplicates(ids: list[str], section: str, source: str | Path | None) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for i in ids:
        if i in seen:
            duplicates.add(i)
        seen.add(i)
    if duplicates:
        raise MissionError(
            "DUPLICATE_ID",
            f"duplicate id(s) in {section}: {sorted(duplicates)}",
            path=section, source=source,
        )


def _check_derived_cycles(mission: Mission, source: str | Path | None) -> None:
    """Detect cycles among derived values at LOAD time.

    MissionStateEngine would also raise on a cycle, but by then the mission
    is already running; catching it here keeps the failure where the author
    can act on it.
    """
    derived_ids = {d.id for d in mission.derived}
    dependencies = {d.id: referenced_names(d.expr) & derived_ids for d in mission.derived}

    resolved: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = [i for i, deps in remaining.items() if not (deps - resolved)]
        if not ready:
            raise MissionError(
                "CIRCULAR_REFERENCE",
                f"circular dependency among derived values: {sorted(remaining)}",
                path="derived", source=source,
            )
        for i in ready:
            resolved.add(i)
            del remaining[i]


def _expression_error_code(err: ExpressionError) -> str:
    """Distinguish a dangerous expression from a mistyped reference.

    Worth separating: 'unsafe' means someone wrote something that could
    execute code and the author should look hard at where the file came
    from; 'unknown reference' is an ordinary typo.
    """
    message = str(err)
    if "disallowed" in message or "unknown function" in message or "not allowed" in message:
        return "UNSAFE_EXPRESSION"
    if "syntax error" in message:
        return "INVALID_EXPRESSION"
    return "UNKNOWN_PARAMETER_REF"


def _suggest_for_expression(err: ExpressionError, known_names: set[str]) -> str:
    """Add a 'did you mean' hint for an unknown reference inside an
    expression.

    Expressions are where typos are MOST likely -- a parameter id is typed
    by hand there rather than picked from a list -- so the hint matters
    more here than anywhere else.
    """
    import re

    match = re.search(r"unknown reference '([^']+)'", str(err))
    if not match:
        return ""
    return _suggest(match.group(1), sorted(known_names))


def _check_checkpoint_chain_cycles(mission: Mission, source: str | Path | None) -> None:
    """Detect a cycle in `after` chaining at LOAD time.

    A cycle means those checkpoints could never fire. Caught here rather
    than at runtime, where the symptom would be a scenario that silently
    skips part of its timeline -- the trainee would be assessed on a beat
    that never happened.
    """
    waits_for = {
        t.id: t.after for t in mission.triggers.timeline if t.after
    }
    for start in waits_for:
        seen = {start}
        current = waits_for.get(start)
        while current is not None:
            if current in seen:
                raise MissionError(
                    "CIRCULAR_REFERENCE",
                    f"checkpoints wait on each other in a cycle: {sorted(seen)}",
                    path="triggers.timeline", source=source,
                )
            seen.add(current)
            current = waits_for.get(current)


def _suggest(given: str, candidates: list[str]) -> str:
    """A 'did you mean' hint for a near-miss id.

    Cheap to compute and disproportionately useful: most mission-file
    errors in practice are a typo or a renamed parameter, and naming the
    likely intended id turns a hunt into a glance.
    """
    import difflib

    close = difflib.get_close_matches(given, candidates, n=1, cutoff=0.6)
    return f". Did you mean {close[0]!r}?" if close else ""
