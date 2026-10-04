"""FastAPI app -- the HTTP boundary.

The UI talks to this over HTTP only and never imports the agent, so the
layering holds at the deployment boundary too, not just in the source tree.

TRANSPORT: Server-Sent Events. A delivery plan is a timed sequence of
events pushed by the server -- typing indicator, chunk, pause, tone shift
-- which is exactly what SSE is for. A single JSON response would collapse
the realism layer back into one instant blob, and WebSockets would add
bidirectional machinery for a stream that only flows one way. Phase 2
replaces the SSE transport with a WebRTC data channel beside the audio;
the EVENT shapes stay the same, which is the point.

load_dotenv() runs before any module reads os.environ: uvicorn does not
load .env itself, so a key present in the file but unread is an easy and
confusing failure.
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
    ComposingRequest,
    MessageRequest,
    SessionCreated,
    StartSessionRequest,
)
from api.store import SqliteSessionStore  # noqa: E402
from core.mission import MissionError, load_mission  # noqa: E402
from obs.logging import configure as configure_logging, get_logger  # noqa: E402
from providers.base import (  # noqa: E402
    ProviderError,
    build_llm_provider,
    build_stt_provider,
    build_tts_provider,
)
from sim.runner import SessionRunner  # noqa: E402

configure_logging()
logger = get_logger("api.main")

app = FastAPI(title="Maslul — conversation training simulator")

app.add_middleware(
    CORSMiddleware,
    # Single-user local deployment. FLAGGED as a to-revisit item: this is
    # deliberately permissive for localhost development and must be
    # tightened before the app is reachable beyond this machine.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MISSIONS_DIR = Path(__file__).resolve().parent.parent / "missions"
store = SqliteSessionStore()

# Live sessions, in memory. Documented prototype limitation: a process
# restart ends any running session. The schema already permits rehydration
# from the last snapshot, so this is upgradeable without touching the
# layers above.
_runners: dict[str, "LiveSession"] = {}


class LiveSession:
    """A runner plus the queue its SSE stream drains.

    The queue exists because delivery events are produced by the runner's
    own task (a trigger may fire with no HTTP request in flight) and
    consumed by whichever stream is open. Without it, self-initiated
    utterances could only be delivered while the trainee happened to be
    waiting on a response.
    """

    def __init__(self, runner: SessionRunner, mission: Any) -> None:
        self.runner = runner
        self.mission = mission
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.tick_task: asyncio.Task | None = None


class QueueChannel:
    """DeliveryChannel that funnels events into a session's queue."""

    def __init__(self, live: LiveSession) -> None:
        self._live = live
        from delivery.text_channel import TextChannel
        self._text = TextChannel(self._put)

    async def _put(self, event: dict[str, Any]) -> None:
        await self._live.queue.put(event)

    async def on_event(self, event: Any) -> None:
        await self._text.on_event(event)


# -- error handling --------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """FR-G4: a generic message plus a request id to the client; the real
    exception is logged server-side only, so no internal detail escapes."""
    request_id = uuid.uuid4().hex[:12]
    logger.exception("api.unhandled", request_id=request_id,
                     error_code=type(exc).__name__, status="error")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our side.",
                 "request_id": request_id},
    )


# -- endpoints -------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "live_sessions": len(_runners)}


@app.get("/missions")
async def list_missions() -> dict[str, Any]:
    """Available missions, with load errors surfaced rather than hidden.

    A mission that fails to load is reported WITH its error, so a typo is
    visible in the UI instead of the file silently disappearing from the
    list.
    """
    missions = []
    for path in sorted(MISSIONS_DIR.glob("*.yaml")):
        try:
            mission = load_mission(path)
            missions.append({
                "file": path.name, "id": mission.id, "title": mission.title,
                "language": mission.language,
                "trainee_role": mission.setting.trainee_role,
                "briefing": mission.setting.briefing,
                "counterpart": mission.persona.name,
                "trainee_callsign": mission.procedure.callsigns.trainee,
            })
        except MissionError as err:
            missions.append({"file": path.name, "error": str(err)})
    return {"missions": missions}


@app.post("/sessions", response_model=SessionCreated)
async def start_session(request: StartSessionRequest) -> SessionCreated:
    path = MISSIONS_DIR / request.mission_file
    # Resolve and contain: a mission_file of "../../etc/passwd" must not
    # escape the missions directory.
    if not path.resolve().is_relative_to(MISSIONS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="invalid mission file")

    try:
        mission = load_mission(path)
    except MissionError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    try:
        model = build_llm_provider().chat_model()
    except ProviderError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    # Server-generated and unguessable: a client can only resume a session
    # id the server handed it.
    session_id = secrets.token_urlsafe(16)
    live = LiveSession(None, mission)  # type: ignore[arg-type]
    runner = SessionRunner(
        mission, model, QueueChannel(live), session_id=session_id,
        delivery_speed=request.delivery_speed, seed=request.seed,
    )
    live.runner = runner
    _runners[session_id] = live

    store.create_session(
        session_id=session_id, mission_id=mission.id,
        mission_version=mission.version, mission_title=mission.title,
        realism_seed=runner.seed,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info("session.created", session_id=session_id, mission_id=mission.id)
    return SessionCreated(
        session_id=session_id, mission_id=mission.id, title=mission.title,
        counterpart=mission.persona.name,
        trainee_callsign=mission.procedure.callsigns.trainee,
        counterpart_callsign=mission.procedure.callsigns.counterpart,
        briefing=mission.setting.briefing,
        language=mission.language,
    )


@app.get("/sessions/{session_id}/stream")
async def stream(session_id: str) -> StreamingResponse:
    """SSE stream of delivery events, plus the trigger tick loop.

    The tick loop runs HERE rather than at session creation so triggers
    only fire while someone is actually listening -- otherwise a session
    left open in a closed tab would keep generating model calls.
    """
    live = _runners.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")

    if live.tick_task is None or live.tick_task.done():
        live.tick_task = asyncio.create_task(_tick_loop(live))

    async def event_source():
        try:
            # An immediate event so the client knows the stream is open
            # rather than waiting for the first trigger.
            yield _sse({"type": "connected", "session_id": session_id})
            while True:
                try:
                    event = await asyncio.wait_for(live.queue.get(), timeout=15.0)
                    yield _sse(event)
                except asyncio.TimeoutError:
                    # A keepalive comment: proxies and browsers drop an
                    # idle SSE connection, and an idle session is normal
                    # here while the trainee is reading.
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: MessageRequest) -> dict[str, Any]:
    """A trainee transmission. Interrupts anything in flight."""
    live = _runners.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")

    runner = live.runner
    mission_seconds = runner.clock.now()
    store.add_trainee_message(session_id, request.text, mission_seconds)

    record = await runner.handle_trainee_message(request.text)
    if record is not None:
        _persist_utterance(session_id, runner, record)
    return {"ok": True}


@app.post("/sessions/{session_id}/composing")
async def set_composing(session_id: str, request: ComposingRequest) -> dict[str, Any]:
    """The trainee started or stopped typing.

    Feeds the composition-window guard: non-critical initiative waits
    while they are mid-sentence. In voice this same signal comes from VAD.
    """
    live = _runners.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")
    live.runner.set_composing(request.composing)
    return {"ok": True}


@app.post("/sessions/{session_id}/interrupt")
async def interrupt(session_id: str) -> dict[str, Any]:
    """Cut the counterpart off mid-transmission."""
    live = _runners.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")
    live.runner.interrupt()
    return {"ok": True}


@app.get("/sessions/{session_id}/state")
async def session_state(session_id: str) -> dict[str, Any]:
    """The trainer's view: FULL state, including what the counterpart
    cannot see. You need to know what he does not in order to judge
    whether his answers were honest."""
    live = _runners.get(session_id)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")

    runner = live.runner
    runner.session.engine.advance_to(runner.clock.now())
    visible = set(runner.session.engine.snapshot(for_persona=True))

    return {
        "mission_seconds": runner.clock.now(),
        # Surfaced so the console can show the trainer that mission time
        # is compressed -- otherwise a fast-moving clock looks like a bug.
        "mission_speed": getattr(runner.clock, "multiplier", 1.0),
        "readings": [
            {**row, "hidden_from_counterpart": row["id"] not in visible}
            for row in runner.session.engine.describe_for_prompt(for_persona=False)
        ],
        "fired": dict(runner.log.fired_counts),
        "tone": runner._current_tone(),
    }


@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str) -> dict[str, Any]:
    live = _runners.pop(session_id, None)
    if live is None:
        raise HTTPException(status_code=404, detail="no such session")
    live.runner.stop()
    if live.tick_task is not None:
        live.tick_task.cancel()
    store.end_session(session_id, "completed")
    logger.info("session.ended", session_id=session_id)
    return {"ok": True}


# -- voice ----------------------------------------------------------------


@app.websocket("/sessions/{session_id}/voice")
async def voice_socket(socket: WebSocket, session_id: str) -> None:
    """Browser audio in, counterpart audio out, over one socket.

    ONE socket for both directions, with binary frames for PCM and text
    frames for control. A separate control socket could desynchronize from
    the audio, which matters because an interrupt must be ordered relative
    to the audio around it.

    Attaching a voice socket REPLACES the session's delivery channel, so
    the counterpart speaks instead of streaming text. Everything upstream
    -- the agent, the tools, the mission state, the realism plan -- is
    unchanged; this is the seam the architecture was shaped around.
    """
    await socket.accept()

    live = _runners.get(session_id)
    if live is None:
        await socket.send_text(json.dumps({"type": "error",
                                           "message": "no such session"}))
        await socket.close()
        return

    from api.voice import VoiceBridge
    from delivery.voice_channel import VoiceChannel

    try:
        stt = build_stt_provider(live.mission)
        tts = build_tts_provider(live.mission)
    except ProviderError as err:
        await socket.send_text(json.dumps({"type": "error", "message": str(err)}))
        await socket.close()
        return

    bridge = VoiceBridge(socket, live, live.mission)

    # Swap text delivery for audio delivery. Kept so it can be restored:
    # a trainer may close the voice tab and carry on in text.
    previous_channel = live.runner.channel
    live.runner.channel = VoiceChannel(
        tts=tts, emit_audio=bridge.send_audio, emit_event=bridge.send_event,
    )

    # Voice pacing: the plan's speech durations must reflect real speaking
    # rate rather than the fast text rate.
    live.runner.for_voice = True

    if live.tick_task is None or live.tick_task.done():
        live.tick_task = asyncio.create_task(_tick_loop(live))

    logger.info("voice.connected", session_id=session_id)
    await socket.send_text(json.dumps({
        "type": "voice_ready",
        "sample_rate": 16000,
        "language": live.mission.language,
    }))

    try:
        await bridge.run(stt)
    finally:
        live.runner.channel = previous_channel
        live.runner.for_voice = False
        logger.info("voice.disconnected", session_id=session_id)


@app.websocket("/sessions/{session_id}/live")
async def live_voice_socket(socket: WebSocket, session_id: str) -> None:
    """Gemini Live: native Hebrew speech-to-speech.

    The alternative to /voice's cascade. Uses the GEMINI_API_KEY already
    configured, so it needs no second vendor -- which makes it the quickest
    way to answer "is Hebrew voice viable at all?" before committing to a
    paid STT/TTS stack.

    Measured on this project: ~1.3s to first Hebrew audio, and it DOES call
    read_state before quoting a figure, so the mission-state guarantee
    survives. What it gives up is the realism layer: Gemini Live owns its
    own pauses, so there are no controlled stalls (see
    providers/cloud/gemini_live.py).
    """
    await socket.accept()

    live = _runners.get(session_id)
    if live is None:
        await socket.send_text(json.dumps({"type": "error",
                                           "message": "no such session"}))
        await socket.close()
        return

    from agent.prompts import build_system_prompt
    from api.live_voice import LiveVoiceBridge
    from providers.cloud.gemini_live import GeminiLiveProvider

    try:
        provider = GeminiLiveProvider(language=live.mission.language)
        provider._require_key()
    except ProviderError as err:
        await socket.send_text(json.dumps({"type": "error", "message": str(err)}))
        await socket.close()
        return

    runner = live.runner
    runner.session.engine.advance_to(runner.clock.now())

    # The SAME prompt the text agent uses, minus the marker vocabulary.
    # Live controls its own prosody, and an earlier attempt to merely strip
    # the guillemets left the bare word behind -- it said "hesitate" aloud.
    prompt = build_system_prompt(live.mission, runner.session.engine,
                                 with_markers=False, for_speech=True)

    bridge = LiveVoiceBridge(socket, live, live.mission)
    logger.info("live_voice.connected", session_id=session_id)
    try:
        await bridge.run(provider, prompt)
    except Exception as err:
        logger.exception("live_voice.failed", session_id=session_id,
                         error_code=type(err).__name__, status="error")
    finally:
        logger.info("live_voice.disconnected", session_id=session_id)


# -- review ---------------------------------------------------------------


@app.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": [vars(s) for s in store.list_sessions()]}


@app.get("/sessions/{session_id}/review")
async def review(session_id: str) -> dict[str, Any]:
    """Everything a debrief needs, including what was WITHHELD.

    Suppressed firings are included deliberately: "why didn't he warn me
    about the fuel?" is unanswerable if the absence of an event left no
    trace.
    """
    record = store.session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such session")
    return {
        "session": record,
        "transcript": store.transcript(session_id),
        "firings": store.firings(session_id),
        "snapshots": store.snapshots(session_id),
    }


# -- internals ------------------------------------------------------------


async def _tick_loop(live: LiveSession) -> None:
    """Evaluate triggers while a stream is attached."""
    from sim.runner import TICK_SECONDS

    runner = live.runner
    limit = runner.mission.limits.session_max_minutes * 60
    try:
        while runner.clock.now() < limit:
            record = await runner.tick()
            if record is not None:
                _persist_utterance(runner.session_id, runner, record)
            _drain_suppressions(runner)
            await asyncio.sleep(TICK_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as err:
        # A tick-loop failure must not take the session down silently --
        # the trainee would see a counterpart that simply stopped
        # initiating, with no indication why.
        logger.exception("tick_loop.failed", session_id=runner.session_id,
                         error_code=type(err).__name__, status="error")


_persisted_suppressions: dict[str, int] = {}


def _drain_suppressions(runner: SessionRunner) -> None:
    """Persist newly recorded suppressions.

    Index-tracked rather than cleared from the runner's log, because the
    log is also the in-memory source for the live trainer view.
    """
    already = _persisted_suppressions.get(runner.session_id, 0)
    pending = runner.log.suppressions[already:]
    for mission_seconds, suppression in pending:
        store.add_trigger_firing(
            runner.session_id, suppression.trigger_id, mission_seconds,
            suppressed=True, reason=suppression.reason,
        )
    _persisted_suppressions[runner.session_id] = len(runner.log.suppressions)


def _persist_utterance(session_id: str, runner: SessionRunner, record: Any) -> None:
    store.add_utterance(
        session_id, text=record.text, origin=record.origin,
        mission_seconds=record.mission_seconds, status=record.status,
        plan_id=record.plan_id, turn_id=record.turn_id,
        trigger_id=record.trigger_id, planned_text=record.planned_text,
        delivered_segments=record.delivered_segments,
        total_segments=record.total_segments,
    )
    if record.trigger_id:
        store.add_trigger_firing(
            session_id, record.trigger_id, record.mission_seconds,
            suppressed=False, reason=None,
        )
    store.add_snapshot(
        session_id, record.mission_seconds,
        runner.session.engine.snapshot(for_persona=False),
        cause=f"trigger:{record.trigger_id}" if record.trigger_id else "turn",
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
