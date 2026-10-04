"""FastAPI app -- the HTTP boundary.

The UI talks to this over HTTP only and never imports the domain layer, so
the layering holds at the deployment boundary too.

LIFECYCLE IS EXPLICIT, which is the point of the endpoint set:

    POST /sessions              prepare (loads the three sources, t stays 0)
    POST /sessions/{id}/start   begin, alongside the trainer's video start
    POST /sessions/{id}/pause   freeze mission time
    POST /sessions/{id}/resume
    POST /sessions/{id}/end

Preparation takes time -- validating a timeline, connecting a voice socket
-- and if the clock ran during it, that setup would silently become
mission time and the exercise would already be out of step with the video
before the trainer pressed play.

ONE DOMAIN LOOP PER SESSION. The Session owns it; a voice socket attaches
a channel rather than starting its own. The previous version let the SSE
stream and the Gemini Live bridge each run a trigger loop against one
engine, so a once-only event could fire twice.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from api.schemas import (  # noqa: E402
    MessageRequest,
    SessionCreated,
    SharedContextRequest,
    StartSessionRequest,
)
from api.store import SqliteSessionStore  # noqa: E402
from core.lifecycle import ExerciseClock, LifecycleError  # noqa: E402
from core.mission import MissionError, load_context, load_mission  # noqa: E402
from core.timeline_import import TimelineError, load_timeline  # noqa: E402
from obs.logging import configure as configure_logging, get_logger  # noqa: E402
from providers.base import ProviderError, build_llm_provider  # noqa: E402
from sim.exercise import Exercise, Utterance  # noqa: E402
from sim.session import Session  # noqa: E402
from sim.turns import TurnRunner  # noqa: E402

configure_logging()
logger = get_logger("api.main")

app = FastAPI(title="Maslul — Hebrew conversation training counterpart")

app.add_middleware(
    CORSMiddleware,
    # Single-trainer local deployment. FLAGGED: tighten before this is
    # reachable beyond localhost.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent
MISSIONS = ROOT / "missions"
TIMELINES = ROOT / "timelines"
CONTEXT = ROOT / "context"

store = SqliteSessionStore()

# Live sessions, in memory. Documented prototype limitation: a process
# restart ends any running exercise.
_sessions: dict[str, "LiveSession"] = {}


class LiveSession:
    """A Session plus the queue its SSE stream drains.

    The queue exists because utterances are produced by the session's own
    loops -- a report can fire with no HTTP request in flight -- and
    consumed by whichever stream is open.
    """

    def __init__(self, session: Session | None, mission: Any) -> None:
        self.session = session
        self.mission = mission
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def emit(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)

    def on_utterance(self, utterance: Utterance) -> Any:
        return self.emit({
            "type": "utterance",
            "speaker": utterance.speaker,
            "text": utterance.text,
            "origin": utterance.origin,
            "event_id": utterance.event_id,
            "mission_seconds": round(utterance.at, 1),
        })


# -- error handling --------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """A generic message plus a request id; the real exception is logged
    server-side only, so no internal detail escapes."""
    request_id = uuid.uuid4().hex[:12]
    logger.exception("api.unhandled", request_id=request_id,
                     error_code=type(exc).__name__, status="error")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our side.",
                 "request_id": request_id},
    )


# -- discovery -------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/missions")
async def list_missions() -> dict[str, Any]:
    """Available exercises, with load errors surfaced.

    A mission that fails to load is reported WITH its error rather than
    vanishing from the list, so a typo is visible in the UI.
    """
    out = []
    for path in sorted(MISSIONS.glob("*.yaml")):
        try:
            mission = load_mission(path)
            out.append({
                "file": path.name, "id": mission.id, "title": mission.title,
                "language": mission.language,
                "trainee_callsign": mission.callsigns.trainee,
                "operator_callsign": mission.callsigns.operator,
                "trainee_briefing": mission.setting.trainee_briefing,
                "duration_seconds": mission.duration_seconds,
            })
        except MissionError as err:
            out.append({"file": path.name, "error": str(err)})
    return {"missions": out}


# -- lifecycle -------------------------------------------------------------


@app.post("/sessions", response_model=SessionCreated)
async def prepare_session(request: StartSessionRequest) -> SessionCreated:
    """Load the three sources and prepare. The clock does NOT start."""
    path = MISSIONS / request.mission_file
    # Resolve and contain: "../../etc/passwd" must not escape missions/.
    if not path.resolve().is_relative_to(MISSIONS.resolve()):
        raise HTTPException(status_code=400, detail="invalid mission file")

    try:
        mission = load_mission(path)
        if not mission.timeline_file:
            raise MissionError("NO_TIMELINE", "mission has no timeline_file")
        timeline = load_timeline(TIMELINES / mission.timeline_file)
        context = load_context(mission.context_files, CONTEXT)
    except (MissionError, TimelineError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    try:
        model = build_llm_provider().chat_model()
    except ProviderError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    # Server-generated and unguessable: a client can only resume an id the
    # server handed it.
    session_id = secrets.token_urlsafe(16)

    exercise = Exercise(
        mission, timeline, model, context=context,
        session_id=session_id, clock=ExerciseClock(),
    )
    turns = TurnRunner(exercise, model,
                       for_speech=request.channel == "gemini_live")

    live = LiveSession(None, mission)
    session = Session(exercise, turns, store=store, channel=request.channel,
                      on_utterance=live.on_utterance)
    live.session = session
    _sessions[session_id] = live

    store.create_session(
        session_id=session_id, mission_id=mission.id,
        mission_version=mission.version, mission_title=mission.title,
        channel=request.channel,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    await session.prepare()

    logger.info("session.prepared", session_id=session_id, mission_id=mission.id)
    return SessionCreated(
        session_id=session_id, mission_id=mission.id, title=mission.title,
        operator_callsign=mission.callsigns.operator,
        trainee_callsign=mission.callsigns.trainee,
        controller_callsign=mission.callsigns.controller,
        trainee_briefing=mission.setting.trainee_briefing,
        language=mission.language,
        duration_seconds=mission.duration_seconds or timeline.duration,
        phase=session.phase.value,
    )


def _live(session_id: str) -> LiveSession:
    live = _sessions.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")
    return live


@app.post("/sessions/{session_id}/start")
async def start_exercise(session_id: str) -> dict[str, Any]:
    """Begin the exercise. The trainer starts the video at this moment."""
    live = _live(session_id)
    try:
        await live.session.start()
    except LifecycleError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    await live.emit({"type": "started"})
    return {"phase": live.session.phase.value}


@app.post("/sessions/{session_id}/pause")
async def pause_exercise(session_id: str) -> dict[str, Any]:
    live = _live(session_id)
    try:
        await live.session.pause()
    except LifecycleError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    await live.emit({"type": "paused"})
    return {"phase": live.session.phase.value}


@app.post("/sessions/{session_id}/resume")
async def resume_exercise(session_id: str) -> dict[str, Any]:
    live = _live(session_id)
    try:
        await live.session.resume()
    except LifecycleError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    await live.emit({"type": "resumed"})
    return {"phase": live.session.phase.value}


@app.post("/sessions/{session_id}/end")
async def end_exercise(session_id: str) -> dict[str, Any]:
    live = _sessions.pop(session_id, None)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")
    await live.session.end()
    await live.emit({"type": "ended"})
    logger.info("session.ended", session_id=session_id)
    return {"phase": live.session.phase.value}


# -- conversation ----------------------------------------------------------


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
    live = _live(session_id)
    if not live.session.exercise.clock.is_running:
        raise HTTPException(status_code=409,
                            detail=f"exercise is {live.session.phase.value}")
    await live.session.trainee_says(request.text)
    return {"ok": True}


@app.post("/sessions/{session_id}/shared")
async def share_context(session_id: str,
                        request: SharedContextRequest) -> dict[str, Any]:
    """Record context the trainee passed to the crew."""
    live = _live(session_id)
    live.session.note_shared(request.fact)
    return {"ok": True}


@app.get("/sessions/{session_id}/stream")
async def stream(session_id: str) -> StreamingResponse:
    """SSE stream of utterances and lifecycle events.

    Does NOT start a domain loop -- the Session owns that, and starting
    one here is how the previous version ended up with two.
    """
    live = _live(session_id)

    async def source():
        try:
            yield _sse({"type": "connected", "session_id": session_id,
                        "phase": live.session.phase.value})
            while True:
                try:
                    event = await asyncio.wait_for(live.queue.get(), timeout=15.0)
                    yield _sse(event)
                except asyncio.TimeoutError:
                    # Keepalive: proxies and browsers drop an idle SSE
                    # connection, and an idle exercise is normal.
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        source(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}/state")
async def session_state(session_id: str) -> dict[str, Any]:
    """The TRAINER's view.

    Includes the private solution and what is still unrevealed: judging
    whether the crew answered honestly requires knowing what it could not
    see. Never sent to the model.
    """
    live = _live(session_id)
    exercise = live.session.exercise
    mission = exercise.mission

    return {
        "phase": exercise.phase.value,
        "mission_seconds": round(exercise.clock.now(), 1),
        "duration_seconds": exercise.clock.duration,
        "operator_facts": exercise.current_information(),
        "observations": exercise.revealed_observations(),
        "commitments": [
            {"commitment_id": c.commitment_id, "tags": list(c.tags),
             "entity_ids": list(c.entity_ids), "description": c.description}
            for c in exercise.ledger.active
        ],
        "reports_owed": [p.event.event_id for p in exercise.pending_reports()],
        "handover_active": exercise.handover_active() is not None,
        "contact_established": exercise.state.contact_established,
        "briefing_stage": exercise.state.briefing_stage.value,
        "shared_by_trainee": exercise.state.shared_by_trainee,
        # Trainer only.
        "private": {
            "solution": mission.private.solution,
            "debrief_points": mission.private.debrief_points,
        },
    }


# -- review ----------------------------------------------------------------


@app.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": [vars(s) for s in store.list_sessions()]}


@app.get("/sessions/{session_id}/review")
async def review(session_id: str) -> dict[str, Any]:
    """Everything a debrief needs, for any channel.

    Voice sessions previously returned an empty transcript -- neither
    voice path persisted anything -- which made them undebriefable.
    """
    record = store.session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such session")
    return {
        "session": record,
        "transcript": store.transcript(session_id),
        "revealed": store.revealed(session_id),
        "agreements": store.agreements(session_id),
    }


# -- voice -----------------------------------------------------------------


@app.websocket("/sessions/{session_id}/voice")
async def voice_socket(socket: WebSocket, session_id: str) -> None:
    """ElevenLabs cascade: STT -> shared domain layer -> TTS."""
    await socket.accept()
    live = _sessions.get(session_id)
    if live is None:
        await socket.send_text(json.dumps({"type": "error",
                                           "message": "no such session"}))
        await socket.close()
        return

    from api.voice import run_cascade
    try:
        await run_cascade(socket, live)
    except Exception as err:
        logger.exception("voice.failed", session_id=session_id,
                         error_code=type(err).__name__, status="error")


@app.websocket("/sessions/{session_id}/live")
async def live_voice_socket(socket: WebSocket, session_id: str) -> None:
    """Gemini Live: native Hebrew speech-to-speech.

    Shares the domain layer with every other channel; only audio
    transport differs. Its prosody and pacing genuinely differ from the
    cascade's -- that is a real difference, not one to claim away.
    """
    await socket.accept()
    live = _sessions.get(session_id)
    if live is None:
        await socket.send_text(json.dumps({"type": "error",
                                           "message": "no such session"}))
        await socket.close()
        return

    from api.live_voice import run_live
    try:
        await run_live(socket, live)
    except Exception as err:
        logger.exception("live_voice.failed", session_id=session_id,
                         error_code=type(err).__name__, status="error")


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
