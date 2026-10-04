"""Dynamics tests -- one class per kind (FR-E10).

Table-driven and deliberately exhaustive on edge cases, because these six
tiny functions produce every number the trainee is taught to reason from.
A quietly wrong fuel burn is the kind of defect that survives for months
and teaches a wrong habit the whole time.
"""

from __future__ import annotations

import pytest

from core.dynamics import DynamicsKind, apply_dynamics, kind_requires_target
from core.models import Dynamics, Parameter


def _param(**kwargs) -> Parameter:
    defaults = {"id": "p", "type": "number", "initial": 100.0}
    return Parameter(**{**defaults, "type": kwargs.pop("type", "number"), **kwargs})


class TestHoldAndStepped:
    @pytest.mark.parametrize("kind", ["hold", "stepped"])
    def test_unchanged_by_time(self, kind) -> None:
        param = _param(dynamics=Dynamics(kind=kind))
        assert apply_dynamics(100.0, 86400.0, param) == 100.0


class TestLinearDrain:
    @pytest.mark.parametrize("elapsed,expected", [
        (0.0, 100.0),
        (1800.0, 70.0),      # half an hour at 60/hr
        (3600.0, 40.0),
        (7200.0, 0.0),       # clamped, not negative
    ])
    def test_drains_at_rate(self, elapsed, expected) -> None:
        param = _param(min=0, dynamics=Dynamics(kind="linear_drain",
                                                rate_per_hour=60.0, floor=0))
        assert apply_dynamics(100.0, elapsed, param) == pytest.approx(expected)

    def test_floor_prevents_negative(self) -> None:
        """Fuel below zero would propagate into every derived value that
        divides by it; clamping keeps 'empty' a mission condition rather
        than an arithmetic fault."""
        param = _param(dynamics=Dynamics(kind="linear_drain",
                                         rate_per_hour=60.0, floor=0))
        assert apply_dynamics(1.0, 36000.0, param) == 0.0

    def test_falls_back_to_param_min_when_no_floor(self) -> None:
        param = _param(min=10.0, dynamics=Dynamics(kind="linear_drain",
                                                   rate_per_hour=60.0))
        assert apply_dynamics(50.0, 36000.0, param) == 10.0


class TestLinearFill:
    def test_fills_at_rate(self) -> None:
        param = _param(dynamics=Dynamics(kind="linear_fill", rate_per_hour=30.0,
                                         ceiling=100))
        assert apply_dynamics(0.0, 3600.0, param) == pytest.approx(30.0)

    def test_ceiling_caps(self) -> None:
        param = _param(dynamics=Dynamics(kind="linear_fill", rate_per_hour=30.0,
                                         ceiling=50))
        assert apply_dynamics(0.0, 86400.0, param) == 50.0


class TestRateTowardTarget:
    def test_moves_toward_target(self) -> None:
        param = _param(dynamics=Dynamics(kind="rate_toward_target",
                                         rate_per_minute=500.0))
        assert apply_dynamics(1000.0, 60.0, param, target=5000.0) == pytest.approx(1500.0)

    def test_does_not_overshoot(self) -> None:
        """Overshooting would make the counterpart report an altitude that
        was never commanded, then drift back -- visibly wrong."""
        param = _param(dynamics=Dynamics(kind="rate_toward_target",
                                         rate_per_minute=500.0))
        assert apply_dynamics(1000.0, 3600.0, param, target=2000.0) == 2000.0

    def test_descends_as_well_as_climbs(self) -> None:
        param = _param(dynamics=Dynamics(kind="rate_toward_target",
                                         rate_per_minute=500.0))
        assert apply_dynamics(5000.0, 60.0, param, target=1000.0) == pytest.approx(4500.0)

    def test_no_target_means_no_movement(self) -> None:
        param = _param(dynamics=Dynamics(kind="rate_toward_target",
                                         rate_per_minute=500.0))
        assert apply_dynamics(1000.0, 600.0, param, target=None) == 1000.0

    def test_kind_requires_target(self) -> None:
        assert kind_requires_target(DynamicsKind.RATE_TOWARD_TARGET)
        assert not kind_requires_target(DynamicsKind.LINEAR_DRAIN)


class TestScripted:
    @pytest.fixture
    def param(self) -> Parameter:
        return Parameter(
            id="comms", type="enum", values=["good", "degraded", "poor"],
            initial="good",
            dynamics=Dynamics(kind="scripted", keyframes=[
                {"at": 300, "value": "degraded"},
                {"at": 900, "value": "poor"},
            ]),
        )

    @pytest.mark.parametrize("now,expected", [
        (0.0, "good"),
        (299.0, "good"),
        (300.0, "degraded"),
        (899.0, "degraded"),
        (900.0, "poor"),
        (9999.0, "poor"),
    ])
    def test_resolves_from_absolute_time(self, param, now, expected) -> None:
        assert apply_dynamics("good", now, param, mission_seconds=now) == expected

    def test_result_depends_only_on_absolute_time(self, param) -> None:
        """This is what makes advancing idempotent: one big jump and many
        small steps must agree (FR-D4)."""
        one_jump = apply_dynamics("good", 1000.0, param, mission_seconds=1000.0)
        stepwise = "good"
        for second in range(0, 1001, 50):
            stepwise = apply_dynamics(stepwise, 50.0, param, mission_seconds=float(second))
        assert one_jump == stepwise == "poor"

    def test_keyframes_out_of_order_still_resolve(self, param) -> None:
        """Authors will not always write keyframes in time order."""
        unordered = Parameter(
            id="x", type="enum", values=["a", "b", "c"], initial="a",
            dynamics=Dynamics(kind="scripted", keyframes=[
                {"at": 900, "value": "c"}, {"at": 300, "value": "b"},
            ]),
        )
        assert apply_dynamics("a", 500.0, unordered, mission_seconds=500.0) == "b"


class TestGeneralBehaviour:
    def test_no_dynamics_means_constant(self) -> None:
        assert apply_dynamics(42.0, 99999.0, _param()) == 42.0

    def test_negative_elapsed_rejected(self) -> None:
        """Silently absorbing negative time would hide the monotonicity
        violation the engine guards against."""
        param = _param(dynamics=Dynamics(kind="linear_drain", rate_per_hour=10.0))
        with pytest.raises(ValueError, match=">= 0"):
            apply_dynamics(100.0, -5.0, param)

    def test_declared_range_binds_regardless_of_kind(self) -> None:
        """A mission author's min/max must hold even if a kind's own clamp
        is absent or looser -- here linear_fill declares no ceiling, so
        only the parameter's max stops it."""
        param = Parameter(id="p", type="number", initial=60.0, min=50.0, max=80.0,
                          dynamics=Dynamics(kind="linear_fill", rate_per_hour=1000.0))
        assert apply_dynamics(60.0, 3600.0, param) == 80.0

    def test_every_kind_has_an_implementation(self) -> None:
        """Mirrors the import-time registry check, so a new enum member
        without a function is caught by the suite too."""
        for kind in DynamicsKind:
            param = Parameter(
                id="t", type="number", initial=1.0,
                dynamics=Dynamics(
                    kind=kind,
                    rate_per_hour=1.0, rate_per_minute=1.0,
                    keyframes=[{"at": 0, "value": 1.0}],
                ),
            )
            apply_dynamics(1.0, 1.0, param, target=1.0, mission_seconds=1.0)
