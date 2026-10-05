"""Shared fixtures.

OFFLINE BY DEFAULT: the whole suite runs with the network guard active, so
"the domain layer needs no network" is continuously verified rather than
assumed. A test that accidentally reaches the internet fails here instead
of passing locally and failing in a closed network.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Install the offline guard before any test runs, so a stray network call fails loudly."""
    os.environ.setdefault("MASLUL_OFFLINE", "1")
    from providers import offline
    offline.install()


# -- the three authored sources -------------------------------------------


@pytest.fixture
def mission_dict() -> dict:
    """A minimal valid mission. Deliberately small, so a test that cares
    about one field is not coupled to the whole synthetic exercise."""
    return {
        "id": "t1",
        "title": "Test exercise",
        "language": "he",
        "duration_seconds": 1200,
        "callsigns": {"operator": "גלוק", "trainee": "מדבקה",
                      "controller": "משנה"},
        "setting": {"purpose": "test", "trainee_briefing": "TRAINEE_ONLY"},
        "crew": {"prior_briefing": "PRIOR_BRIEFING"},
        "behaviour": {"initiative": 0.6, "challenge": 0.5,
                      "briefing_request_after": 60.0},
        "impossible_requests": {
            "explanations": {"zoom": "ZOOM_REASON"},
            "fallback": "FALLBACK_REASON",
        },
        "initial_facts": {"ראות": "בינונית"},
        "private": {"solution": "PRIVATE_SOLUTION",
                    "debrief_points": ["PRIVATE_POINT"]},
        "timeline_file": "t1.csv",
    }


@pytest.fixture
def mission(mission_dict):
    """A loaded Mission built from the sample dict."""
    from core.mission import load_mission_dict
    return load_mission_dict(mission_dict)


@pytest.fixture
def timeline_rows() -> list[dict]:
    """Events covering every type and policy the engine supports."""
    from core.timeline import EventType, Priority, ReportingPolicy
    return [
        # Interval: current only inside its window.
        {"event_id": "vis", "start_time": 60.0, "end_time": 240.0,
         "event_type": EventType.INTERVAL,
         "description": "AUTHOR_VIS",
         "operator_information": "הראות ירדה",
         "state_updates": {"ראות": "ירודה"},
         "tags": ("conditions",),
         "reporting_policy": ReportingPolicy.REQUIRED},
        # Required point: volunteered.
        {"event_id": "veh1", "start_time": 180.0,
         "description": "AUTHOR_VEH1",
         "operator_information": "רכב ראשון עוצר",
         "tags": ("vehicle",), "entity_ids": ("veh1",),
         "reporting_policy": ReportingPolicy.REQUIRED},
        # Subscription point: silent unless agreed.
        {"event_id": "veh2", "start_time": 480.0,
         "description": "AUTHOR_VEH2",
         "operator_information": "רכב שני עוצר",
         "tags": ("vehicle",), "entity_ids": ("veh2",),
         "reporting_policy": ReportingPolicy.SUBSCRIPTION},
        # On-request point with an expiry, for stale handling.
        {"event_id": "veh3", "start_time": 600.0, "expires_at": 660.0,
         "description": "AUTHOR_VEH3",
         "operator_information": "רכב שלישי עובר",
         "tags": ("vehicle",), "entity_ids": ("veh3",),
         "reporting_policy": ReportingPolicy.ON_REQUEST},
        # Handover: blocks ordinary conversation.
        {"event_id": "ho", "start_time": 800.0, "end_time": 920.0,
         "event_type": EventType.HANDOVER,
         "description": "AUTHOR_HO"},
        # Urgent during handover: comes through anyway.
        {"event_id": "urgent", "start_time": 840.0,
         "description": "AUTHOR_URGENT",
         "operator_information": "תנועה חריגה",
         "priority": Priority.URGENT,
         "reporting_policy": ReportingPolicy.REQUIRED},
    ]


@pytest.fixture
def timeline(timeline_rows):
    """A built Timeline from the sample rows."""
    from core.timeline import build_timeline
    return build_timeline(timeline_rows)


@pytest.fixture
def clock():
    """A VirtualClock, so tests advance mission time without waiting."""
    from sim.clock import VirtualClock
    return VirtualClock()


@pytest.fixture
def exercise(mission, timeline, clock):
    """A prepared, started exercise with a scripted model."""
    from sim.exercise import Exercise
    from tests.fakes import ScriptedModel

    ex = Exercise(mission, timeline, ScriptedModel(replies=["רות."]),
                  context="GLOBAL_CONTEXT", session_id="s1", clock=clock)
    ex.mark_ready()
    clock.set(0.0)
    ex.start()
    return ex


@pytest.fixture
def ready_exercise(mission, timeline, clock):
    """Prepared but NOT started -- for lifecycle tests."""
    from sim.exercise import Exercise
    from tests.fakes import ScriptedModel

    return Exercise(mission, timeline, ScriptedModel(replies=["רות."]),
                    context="GLOBAL_CONTEXT", session_id="s2", clock=clock)


@pytest.fixture
def synthetic_paths():
    """The shipped synthetic exercise, so tests catch breakage in what the
    author actually copies -- not only in fixtures."""
    mission = ROOT / "missions" / "synthetic_he.yaml"
    timeline = ROOT / "timelines" / "synthetic_he.csv"
    if not mission.exists() or not timeline.exists():
        pytest.skip("synthetic exercise not present")
    return mission, timeline
