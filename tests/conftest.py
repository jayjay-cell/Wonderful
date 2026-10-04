"""Shared test fixtures and global test configuration.

OFFLINE BY DEFAULT: the whole suite runs with the network guard active
(NFR-2), so "the core path needs no network" is continuously verified
rather than assumed. A test that accidentally reaches the internet fails
loudly here instead of passing on a connected machine and failing in a
closed network.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Activate offline enforcement before any test imports a provider."""
    os.environ.setdefault("MASLUL_OFFLINE", "1")
    from providers import offline
    offline.install()


# -- mission fixtures ------------------------------------------------------


@pytest.fixture
def minimal_mission_dict() -> dict:
    """The smallest valid mission.

    Deliberately minimal so a test that cares about one field is not
    coupled to the whole reference mission -- and so required-versus-
    optional stays visible at a glance.
    """
    return {
        "id": "test_mission",
        "title": "Test mission",
        "language": "en",
        "setting": {
            "purpose": "testing",
            "place": "nowhere",
            "trainee_role": "tester",
        },
        "persona": {"name": "Counterpart", "role": "tester"},
        "parameters": [
            {"id": "fuel", "type": "number", "initial": 100.0, "unit": "lb", "min": 0,
             "dynamics": {"kind": "linear_drain", "rate_per_hour": 60.0, "floor": 0}},
        ],
        "procedure": {"callsigns": {"counterpart": "C", "trainee": "T"}},
    }


@pytest.fixture
def mission_with_everything() -> dict:
    """A mission exercising every section, for cross-validation tests."""
    return {
        "id": "full_mission",
        "title": "Full mission",
        "language": "en",
        "setting": {"purpose": "p", "place": "pl", "trainee_role": "r"},
        "persona": {
            "name": "Op",
            "role": "operator",
            "knows": ["fuel", "mode", "endurance_min"],
            "does_not_know": ["secret"],
        },
        "tone": {
            "baseline": "neutral",
            "shifts": [{"when": "low_fuel", "to": "urgent"}],
            "profiles": {"neutral": {"pace": 1.0}, "urgent": {"pace": 1.3}},
        },
        "realism": {
            "stall_probability": 0.2,
            "garble_by": {
                "parameter": "comms",
                "probabilities": {"good": 0.0, "bad": 0.4},
            },
            "seed": 1234,
        },
        "parameters": [
            {"id": "fuel", "type": "number", "initial": 100.0, "min": 0, "unit": "lb",
             "dynamics": {"kind": "linear_drain", "rate_per_hour": 60.0, "floor": 0}},
            {"id": "alt", "type": "number", "initial": 1000.0, "min": 0, "max": 10000,
             "dynamics": {"kind": "rate_toward_target", "rate_per_minute": 500.0}},
            {"id": "mode", "type": "enum", "values": ["idle", "active"], "initial": "idle",
             "dynamics": {"kind": "stepped"}},
            {"id": "comms", "type": "enum", "values": ["good", "bad"], "initial": "good",
             "dynamics": {"kind": "scripted", "keyframes": [{"at": 300, "value": "bad"}]}},
            {"id": "secret", "type": "text", "initial": "hidden truth",
             "visible_to_persona": False},
        ],
        "derived": [
            {"id": "endurance_min", "expr": "fuel / 60 * 60", "unit": "min"},
            {"id": "half_endurance", "expr": "endurance_min / 2", "unit": "min"},
        ],
        "procedure": {
            "callsigns": {"counterpart": "Op", "trainee": "HQ"},
            "brevity": [{"term": "ROGER", "meaning": "understood"}],
            "rules": [{"id": "rb", "requires_readback_for": ["alt"],
                       "on_violation": "challenge"}],
        },
        "triggers": {
            "timeline": [{"id": "t1", "at_mission_seconds": 120.0,
                          "say_intent": "report something",
                          "effects": [{"parameter": "mode", "value": "active"}]}],
            "thresholds": [{"id": "low_fuel", "when": "fuel <= 20",
                            "say_intent": "report low fuel", "priority": "critical"}],
            "idle": [{"id": "quiet", "after_silence_seconds": 45.0,
                      "say_intent": "check in", "priority": "low"}],
        },
    }


@pytest.fixture
def reference_mission_path() -> Path:
    """The real UAV mission, so tests catch breakage in the file you
    actually author against -- not only in synthetic fixtures."""
    path = PROJECT_ROOT / "missions" / "uav_operator_he.yaml"
    if not path.exists():
        pytest.skip("reference mission not present")
    return path
