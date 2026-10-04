"""The mission state engine -- the correctness keystone of the simulator.

Pure: no LLM, no network, no I/O, no wall clock. Mission time is always
passed in, never read, which is what lets a 30-minute session be tested in
milliseconds (NFR-3).

EVERY NUMBER THE COUNTERPART SAYS ORIGINATES HERE (FR-D1). The model's
only legitimate relationship to a quantity is to relay it. That is not
enforced by asking the model nicely in a prompt -- it is enforced by the
tools returning structured data and never prose, and by a test that
extracts numeric literals from the counterpart's speech and asserts each
traces to a tool result in that turn.

Why this matters more than it might seem: a hallucinated fuel figure does
not merely look bad, it teaches a wrong habit. That is worse than no
training at all, which is why state correctness gets its own requirement
group (FR-D1..D7) and why this module is deliberately boring.

Two structural guarantees worth naming:

  * Derived values are recomputed on every read and never stored, so
    endurance CANNOT contradict fuel (FR-D2). It has no independent
    existence to drift out of sync.
  * snapshot(for_persona=True) omits parameters the mission hides, so the
    counterpart cannot report state it should not have (FR-D6). The
    knowledge boundary is an access restriction, not a request.
"""

from __future__ import annotations

from typing import Any, Mapping

from core.derived import ExpressionError, evaluate, referenced_names
from core.dynamics import apply_dynamics, kind_requires_target
from core.models import (
    CommandResult,
    Effect,
    EffectKind,
    Mission,
    Parameter,
    ParameterType,
    StateCommand,
)


class MissionStateEngine:
    """Holds and advances one session's mission state.

    State is an untyped id -> value map plus the mission's parameter
    DECLARATIONS; behaviour comes from the declarations, not from Python
    fields (ADR-4). This is what lets a naval or ground scenario work with
    no code change (FR-E1/E7).
    """

    def __init__(self, mission: Mission) -> None:
        self._mission = mission
        self._values: dict[str, Any] = {p.id: p.initial for p in mission.parameters}

        # Commanded destinations for rate_toward_target parameters. Held
        # separately from values because "where it is" and "where it was
        # told to go" are genuinely different facts -- an aircraft
        # mid-climb is at neither.
        self._targets: dict[str, Any] = {}

        self._mission_seconds: float = 0.0
        self._derived_order = self._resolve_derived_order()

    # -- time -------------------------------------------------------------

    @property
    def mission_seconds(self) -> float:
        return self._mission_seconds

    def advance_to(self, mission_seconds: float) -> list[Effect]:
        """Advance state to an absolute mission time.

        Monotonic and idempotent (FR-D4): advancing to a time already
        passed raises rather than silently rewinding, and advancing to the
        current time is a no-op. Time running backwards means a caller bug;
        absorbing it quietly would hide that while corrupting the fuel
        figures a trainee is being taught to trust.

        Advancing in one jump must equal advancing in many small steps,
        which is why `scripted` resolves from absolute mission time rather
        than accumulating.
        """
        if mission_seconds < self._mission_seconds:
            raise ValueError(
                f"advance_to({mission_seconds}) would rewind mission time from "
                f"{self._mission_seconds}; mission time is monotonic"
            )

        elapsed = mission_seconds - self._mission_seconds
        if elapsed == 0:
            return []

        effects: list[Effect] = []
        for param in self._mission.parameters:
            if param.dynamics is None:
                continue

            old = self._values[param.id]
            new = apply_dynamics(
                old,
                elapsed_seconds=elapsed,
                param=param,
                target=self._targets.get(param.id),
                mission_seconds=mission_seconds,
            )
            if new == old:
                continue

            self._values[param.id] = new
            effects.append(Effect(
                kind=EffectKind.VALUE_CHANGED,
                parameter_id=param.id,
                old_value=old,
                new_value=new,
                detail="dynamics",
            ))
            effects.extend(self._limit_effects(param, new))

            # A rate_toward_target parameter that has arrived is no longer
            # moving; clearing the target stops it being re-reported as
            # in-transit on every later advance.
            if param.id in self._targets and new == self._targets[param.id]:
                del self._targets[param.id]
                effects.append(Effect(
                    kind=EffectKind.TARGET_REACHED,
                    parameter_id=param.id,
                    new_value=new,
                    detail="commanded target reached",
                ))

        self._mission_seconds = mission_seconds
        return effects

    def _limit_effects(self, param: Parameter, value: Any) -> list[Effect]:
        """Emit a LIMIT_REACHED effect when a declared bound is hit, so a
        trigger can react to "fuel is empty" without polling for it."""
        if not isinstance(value, (int, float)):
            return []
        effects = []
        floor = param.dynamics.floor if (param.dynamics and param.dynamics.floor is not None) else param.min
        ceiling = param.dynamics.ceiling if (param.dynamics and param.dynamics.ceiling is not None) else param.max
        if floor is not None and value <= floor:
            effects.append(Effect(
                kind=EffectKind.LIMIT_REACHED, parameter_id=param.id,
                new_value=value, detail=f"at floor {floor}",
            ))
        if ceiling is not None and value >= ceiling:
            effects.append(Effect(
                kind=EffectKind.LIMIT_REACHED, parameter_id=param.id,
                new_value=value, detail=f"at ceiling {ceiling}",
            ))
        return effects

    # -- reading ----------------------------------------------------------

    def snapshot(self, for_persona: bool = False) -> dict[str, Any]:
        """Current parameter values plus freshly computed derived values.

        `for_persona=True` omits parameters declared invisible and any
        derived value that depends on one (FR-D6). Filtering the derived
        values too is essential: leaving `endurance_min` visible while
        hiding `fuel_lb` would leak the hidden value through arithmetic.
        """
        visible_ids = self._visible_parameter_ids() if for_persona else set(self._values)
        values: dict[str, Any] = {k: v for k, v in self._values.items() if k in visible_ids}

        # Derived values are computed against FULL state so a hidden input
        # still produces a correct number, then filtered by whether all
        # their inputs are visible.
        full = dict(self._values)
        for derived_id in self._derived_order:
            spec = next(d for d in self._mission.derived if d.id == derived_id)
            try:
                computed = evaluate(spec.expr, full)
            except ExpressionError:
                # A mission that loaded cleanly can still hit a runtime
                # arithmetic case (division by a drained-to-zero value).
                # None is honest -- "unavailable" -- and a tool reports it
                # as a limitation rather than inventing a figure.
                computed = None
            full[derived_id] = computed

            if not for_persona or referenced_names(spec.expr) <= visible_ids:
                values[derived_id] = computed

        return values

    def value(self, parameter_id: str) -> Any:
        """One current value. Derived ids are supported via snapshot()."""
        if parameter_id in self._values:
            return self._values[parameter_id]
        if parameter_id in self._mission.derived_ids():
            return self.snapshot().get(parameter_id)
        raise KeyError(f"unknown parameter or derived value: {parameter_id!r}")

    def target_of(self, parameter_id: str) -> Any:
        """The commanded destination of an in-transit parameter, if any."""
        return self._targets.get(parameter_id)

    def _visible_parameter_ids(self) -> set[str]:
        """Parameter ids the counterpart may read.

        Three gates, most restrictive winning: `visible_to_persona: false`
        on the parameter, an explicit `does_not_know` list, and -- if
        `knows` is non-empty -- membership of it. An empty `knows` means
        "everything not otherwise hidden", so authors need not enumerate
        every parameter just to hide one.
        """
        persona = self._mission.persona
        visible = {p.id for p in self._mission.parameters if p.visible_to_persona}
        if persona.knows:
            visible &= set(persona.knows)
        visible -= set(persona.does_not_know)
        return visible

    # -- writing ----------------------------------------------------------

    def apply(self, command: StateCommand) -> CommandResult:
        """Validate a command against the parameter's own declaration, then
        apply it.

        A REJECTED COMMAND MUTATES NOTHING (FR-D3). Validation happens
        entirely before any assignment, so there is no partial-application
        path and no need to roll back.

        `reason_code` plus `message` are safe for the counterpart to speak
        in character; neither carries internals (FR-G3).
        """
        param = self._mission.parameter(command.parameter_id)
        if param is None:
            return CommandResult(
                accepted=False,
                parameter_id=command.parameter_id,
                reason_code="UNKNOWN_PARAMETER",
                message=f"There is no parameter called {command.parameter_id!r} in this mission.",
            )

        # Coerce before validating: a stringly-typed "18000" from a tool
        # call must not be refused as if the order were illegal.
        value = self.coerce_value(param, command.value)

        error = self._validate_value(param, value)
        if error is not None:
            reason_code, message = error
            return CommandResult(
                accepted=False, parameter_id=param.id,
                reason_code=reason_code, message=message,
            )
        command = command.model_copy(update={"value": value})

        # A rate_toward_target parameter takes TIME to change: the command
        # sets a destination and advance_to() moves toward it. Snapping
        # instantly would erase the delay that makes mission management a
        # skill worth training.
        if param.dynamics is not None and kind_requires_target(param.dynamics.kind):
            self._targets[param.id] = command.value
            return CommandResult(
                accepted=True, parameter_id=param.id,
                reason_code="TARGET_SET",
                message=f"{param.display_name} is moving to {command.value}.",
                effects=[Effect(
                    kind=EffectKind.VALUE_CHANGED, parameter_id=param.id,
                    old_value=self._values[param.id], new_value=command.value,
                    detail="target set; in transit",
                )],
            )

        old = self._values[param.id]
        self._values[param.id] = command.value
        effects = [Effect(
            kind=EffectKind.VALUE_CHANGED, parameter_id=param.id,
            old_value=old, new_value=command.value, detail=command.source,
        )]
        effects.extend(self._limit_effects(param, command.value))
        return CommandResult(accepted=True, parameter_id=param.id, effects=effects)

    @staticmethod
    def coerce_value(param: Parameter, value: Any) -> Any:
        """Coerce a loosely-typed incoming value to the parameter's type.

        WHY THIS EXISTS: an LLM routinely sends "18000" instead of 18000,
        or "true" instead of True -- JSON tool arguments are stringly
        typed in practice regardless of the declared schema. Without
        coercion, a perfectly legal order is rejected, and the counterpart
        then TRUTHFULLY reports a refusal that should never have happened.

        Observed live: "climb to 18,000" (limit 25,000) produced "negative,
        unable" because the value arrived as a string. That is worse than a
        crash -- it silently teaches the trainee that a legal instruction
        was impossible, which is exactly the perceived-unfairness failure
        the simulation cannot afford.

        Strict about what it accepts: a genuinely non-numeric string still
        fails validation rather than becoming 0.
        """
        if param.type is ParameterType.NUMBER and isinstance(value, str):
            text = value.strip().replace(",", "")   # "18,000" -> "18000"
            try:
                number = float(text)
            except ValueError:
                return value                        # let validation reject it
            return int(number) if number.is_integer() else number

        if param.type is ParameterType.BOOL and isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "on"}:
                return True
            if lowered in {"false", "no", "0", "off"}:
                return False
            return value

        if param.type is ParameterType.ENUM and isinstance(value, str):
            text = value.strip()
            # Case-insensitive match against declared values, so "IDLE"
            # resolves to "idle" rather than being refused.
            for allowed in param.values or []:
                if allowed.lower() == text.lower():
                    return allowed
            # A SPOKEN label resolves back to its machine value. The
            # counterpart is told to say "סריקה" rather than "scanning",
            # so when the trainee orders it in Hebrew he naturally passes
            # the Hebrew word back -- and without this the engine refuses a
            # perfectly legal order. Observed live: "עבור למצב סריקה"
            # produced "שלילי, לא יכול" and listed the English ids aloud.
            resolved = param.resolve_spoken(text)
            if resolved is not None:
                return resolved
            return value

        return value

    def _validate_value(self, param: Parameter, value: Any) -> tuple[str, str] | None:
        """Returns (reason_code, safe_message) if invalid, else None."""
        if param.type is ParameterType.ENUM:
            if value not in (param.values or []):
                # Offer the SPOKEN forms: reciting machine ids aloud on a
                # Hebrew net ("idle, tracking, scanning") breaks character.
                options = [param.spoken(v) for v in (param.values or [])]
                return ("INVALID_VALUE",
                        f"{param.display_name} cannot be set to {value!r}; "
                        f"valid settings are {', '.join(options)}.")
            return None

        if param.type is ParameterType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return ("INVALID_VALUE", f"{param.display_name} requires a number.")
            if param.min is not None and value < param.min:
                return ("OUT_OF_RANGE",
                        f"{param.display_name} cannot go below {param.min}"
                        f"{' ' + param.unit if param.unit else ''}.")
            if param.max is not None and value > param.max:
                return ("OUT_OF_RANGE",
                        f"{param.display_name} cannot exceed {param.max}"
                        f"{' ' + param.unit if param.unit else ''}.")
            return None

        if param.type is ParameterType.BOOL and not isinstance(value, bool):
            return ("INVALID_VALUE", f"{param.display_name} requires true or false.")

        return None

    # -- derived ordering -------------------------------------------------

    def _resolve_derived_order(self) -> list[str]:
        """Order derived values so a dependency is computed before its
        dependents, allowing one derived value to build on another.

        A cycle is a mission-authoring error. It is reported here rather
        than allowed to recurse, but the loader also checks it so the
        failure surfaces at load time (FR-E8).
        """
        derived_ids = {d.id for d in self._mission.derived}
        dependencies = {
            d.id: referenced_names(d.expr) & derived_ids
            for d in self._mission.derived
        }

        ordered: list[str] = []
        remaining = dict(dependencies)
        while remaining:
            ready = [i for i, deps in remaining.items() if not (deps - set(ordered))]
            if not ready:
                raise ValueError(
                    f"circular dependency among derived values: {sorted(remaining)}"
                )
            for i in sorted(ready):
                ordered.append(i)
                del remaining[i]
        return ordered

    # -- introspection ----------------------------------------------------

    def describe_for_prompt(self, for_persona: bool = True) -> list[dict[str, Any]]:
        """Human-readable state for the system prompt: label, value, unit.

        Lives here rather than in agent/prompts.py so there is exactly one
        place that decides what the counterpart can see, and the knowledge
        boundary cannot drift between the tools and the prompt.
        """
        snapshot = self.snapshot(for_persona=for_persona)
        def _display(value: Any) -> Any:
            """Round floats for DISPLAY only; the engine keeps full
            precision. Without this a panel shows 179.84942222222202,
            which no instrument would read -- and a counterpart handed the
            long form will read the whole thing aloud."""
            return round(value, 1) if isinstance(value, float) else value

        rows: list[dict[str, Any]] = []
        for param in self._mission.parameters:
            if param.id not in snapshot:
                continue
            value = snapshot[param.id]
            row = {"name": param.display_name, "id": param.id,
                   "value": _display(value)}
            spoken = param.spoken(value)
            if spoken != str(value):
                row["say_as"] = spoken
            if param.unit:
                row["unit"] = param.unit
            if (target := self._targets.get(param.id)) is not None:
                row["in_transit_to"] = target
            rows.append(row)
        for spec in self._mission.derived:
            if spec.id in snapshot:
                row = {"name": spec.display_name, "id": spec.id,
                       "value": _display(snapshot[spec.id])}
                if spec.unit:
                    row["unit"] = spec.unit
                rows.append(row)
        return rows
