# Developer guide

Read this first. It is a map, not a tutorial — the code itself carries the
detail.

Maslul is a Hebrew conversation trainer. A trainee watches a **prerecorded
mission video in a completely separate player** and talks to a simulated
UAV operator about it. The recording never changes; nothing the trainee
says can alter the imagery. The only link between the video and this
program is an authored timeline and a clock the trainer starts by hand at
roughly the same moment.

---

## 1. Where execution starts

```
python run.py          →  run.py binds :8000, opens the browser
                          └─ api/main.py   the FastAPI app (the only server)
                             └─ ui/index.html served at /
```

`run.py` refuses to start if the port is busy, because a stale process
answering with old code is almost impossible to diagnose from the browser.

Two other entry points exist:

| Command | Purpose |
|---|---|
| `python -m sim.cli --mission missions/<m>.yaml` | Validate the three sources with no model and no API key |
| `python timelines/make_template.py` | Generate a formatted .xlsx timeline template |

---

## 2. Folder map

```
run.py                   entry point; port guard, then uvicorn
ui/
  index.html             the whole trainer console (one file, no build step)
  voice.js               browser microphone capture and audio playback
api/
  main.py                HTTP + SSE + WebSocket endpoints; owns live sessions
  schemas.py             request/response shapes
  store.py               SQLite: sessions, utterances, revealed events, agreements
  live_voice.py          Gemini Live TRANSPORT only (audio frames, barge-in)
sim/
  exercise.py            the running exercise: clock, revelation, owed reports
  session.py             lifecycle + the two loops + the one recording path
  recorder.py            the only writer to the store
  turns.py               runs one agent turn for the text channel
  clock.py               VirtualClock, for tests only
  cli.py                 model-free validation
core/                    PURE: no model, no network, no I/O, no framework
  timeline.py            what is revealed/current at time T
  timeline_import.py     .xlsx/.csv → normalized rows
  commitments.py         reporting agreements and their matching
  conversation_state.py  contact, briefing, addressing
  lifecycle.py           the phase machine and real mission time
  mission.py             mission file loading and validation
agent/
  prompts.py             composes the system prompt from revealed facts only
  loop.py                LangChain create_agent + middleware
tools/mission_tools.py   what the model may call: read facts, manage agreements
providers/               swappable model backends (cloud now, local later)
obs/logging.py           structured logging with a field allow-list
missions/*.yaml          per-exercise configuration
timelines/*.csv|xlsx     what happens during each recording
context/*.md             reusable professional doctrine, shared across exercises
```

`core/` purity is enforced by a test that walks imports. Without it the
layering decays quickly; with it the whole domain is testable with no API
key.

**One deliberate import cycle.** `providers/base.py` imports concrete
providers *inside* a function, while each provider imports `ProviderError`
from `base` at module level. That is not a runtime cycle — it exists so an
air-gapped install with no cloud SDK can still import `base` and select a
local model. Leave it.

---

## 3. The main objects and who owns them

```
api/main.py  _sessions: dict[session_id → LiveSession]        ← process-wide
  │
  └─ LiveSession            a Session + the asyncio.Queue its SSE stream drains
       └─ Session           OWNS the exercise, the turn runner, the recorder,
       │                    the two loop tasks and the speaking lock
       │    ├─ Exercise     clock, timeline position, owed reports, ledger,
       │    │               conversation state. All domain rules live here.
       │    ├─ TurnRunner   one agent turn (text channel only)
       │    └─ ExerciseRecorder   the ONLY thing that writes to the store
       │
       └─ LiveBridge        created per WebSocket, not per session.
                            Transport only; borrows the Session's methods.
```

Rules worth knowing before you change anything:

- **The `Session` creates the `Recorder`.** Nothing else writes to the
  store. A channel records an event; it never decides how it is stored.
- **`Exercise` holds all domain rules** — crew availability, staleness,
  claim/release of reports. Both channels ask it; neither reimplements it.
- **`LiveBridge` is per-socket and short-lived.** It takes report
  ownership while connected and hands it back on disconnect.
- **Release:** `POST /sessions/{id}/end` pops the session from `_sessions`
  and cancels both loop tasks. After that `/state` is a 404 by design;
  `/review` still works, because it reads the database.
- A process restart ends every running exercise. Documented prototype
  limitation.

---

## 4. The flows

### Prepare — `POST /sessions`
Loads the mission, timeline and context; builds `Exercise`, `TurnRunner`,
`Session`; calls `session.prepare()`. Phase becomes **READY** and
**mission time stays exactly 0.0**, however long preparation takes. No
loops are running yet.

### Start — `POST /sessions/{id}/start`
The trainer presses play on the external video at this same moment.
`exercise.start()` begins real mission time, then two tasks start:

```
_reveal_loop   every 0.25s: exercise.advance() → reveal due events,
               queue owed reports, persist. NEVER awaits the model, so a
               slow turn cannot delay the timeline.
_speak_loop    claims one owed report at a time and speaks it. Blocks for
               seconds; that is fine, because it does not keep time.
```

Splitting these is the central design decision: an authored event's moment
is a fact about the video playing beside you.

### A text message — `POST /sessions/{id}/messages`
```
Session.trainee_says(text)
  ├─ record_trainee(text)        → state, store, SSE
  ├─ TurnRunner.trainee_turn()   → prompt from REVEALED facts only, tools, model
  └─ _record("operator", result) → state, store, SSE
```
Takes the speaking lock, so a report in flight finishes first.

### Voice — `WS /sessions/{id}/live`
`LiveBridge` runs three loops: browser→model audio, model→browser audio,
and report injection. It handles only transport; everything else routes
into the shared `Session`/`Exercise`:

- Microphone audio is **dropped** unless `_accepting_input()` — running,
  and the crew is not mid-handover.
- Trainee and operator speech go through `session.record_trainee()` and
  `session.record_operator()`, so a voice transcript holds the same fields
  as a text one.
- Leaving RUNNING sends `flush_audio`, so buffered speech is discarded
  rather than resuming after a pause.

### Proactive reporting
```
_reveal_loop   event becomes due → exercise queues a PendingReport
consumer       exercise.claim_report()   ← ONE consumer only
               ├─ delivered  → complete_report()  (drop it, mark reported)
               └─ failed     → release_report()   (retry while relevant)
```
`exercise.reports_owner` decides the consumer: `"session"` normally,
`"gemini_live"` while a voice socket is connected. A claim does **not**
dequeue, so an abandoned delivery can be retried. Sending a cue to the
model is not delivery — only a completed turn with speech counts.

### Pause / Resume
`exercise.pause()` freezes mission time and bumps a generation counter, so
a model call already in flight cannot speak into a paused exercise. The
loops keep running but do nothing, since the clock is frozen.

### End
Explicit, or automatic once mission time passes the authored duration.
Cancels both loops, persists the final status, and refuses further turns.

---

## 5. Where state lives

**In memory, for the running exercise:**

| What | Where |
|---|---|
| Mission time and phase | `core/lifecycle.ExerciseClock` |
| Timeline position | `Exercise._revealed_through` |
| Owed reports | `Exercise._pending` + `_claimed` |
| Agreements | `Exercise.ledger` (`core/commitments`) |
| Contact, briefing, shared facts | `Exercise.state` (`core/conversation_state`) |

**On disk, in `maslul.db`** (SQLite, four tables, written only by
`ExerciseRecorder`):

| Table | Holds |
|---|---|
| `sessions` | one row per exercise: mission, channel, status, timings |
| `utterances` | the transcript, with origin, status and delivery fidelity |
| `revealed_events` | which events became known, and what became of each report |
| `agreements` | reporting agreements and their status changes |

`GET /sessions/{id}/review` reads these and is the debrief view. It works
after the session is gone from memory.

**Never stored:** undelivered model output. An abandoned turn is recorded
with empty text and a status, so a debrief cannot show words the trainee
never heard.

---

## 6. The three authored sources

| Source | Answers |
|---|---|
| `context/*.md` | How people on this net behave — shared across exercises |
| `timelines/*.csv\|xlsx` | What happens in this recording, and when |
| `missions/*.yaml` | Who is on the net, what the crew knows, how it behaves |

**The full timeline never enters a prompt.** `agent/prompts.py` reads only
`Exercise.revealed_observations()` and `current_information()`, so future
events, author descriptions and private notes cannot leak. See
`docs/AUTHORING.md` to write a new exercise.

⚠ Everything in `missions/`, `timelines/` and `context/` is **synthetic or
fictional** training material, apart from the callsigns גלוק / מדבקה /
משנה. None of it is validated professional doctrine.
