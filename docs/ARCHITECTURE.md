# Architecture — Maslul

**Status:** design, pre-implementation · **Last updated:** 2026-10-03

This document is the technical design: the layers, the seams that make the
flexibility and offline requirements possible, and the decisions behind them.

---

## 1. The three forces shaping this design

Every structural decision below traces to one of these:

1. **Nothing domain-specific in the code.** The mission file declares its own
   parameters. A naval or ground scenario must need zero Python changes.
2. **Nothing cloud-specific outside one folder.** The deployment target is
   air-gapped, so every external dependency sits behind a swap point from the
   first commit.
3. **Nothing important decided by the model.** Numbers, timing, and procedure
   compliance are computed in plain Python. The model narrates; it does not
   calculate. This is what makes the simulator trustworthy enough to train on.

Force 3 is inherited from `C:\dev\Wonderful Airport Investment Intelligence
Agent`, whose core principle this project keeps verbatim:

> The prompt shapes what the model is *inclined* to do. The tools enforce what
> *must* be true.

---

## 2. Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│  ui/            React — chat, typing indicator, barge-in    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP + SSE  (never imports the agent)
┌──────────────────────────┴──────────────────────────────────┐
│  api/           FastAPI boundary, session store, SSE        │
└──────────────────────────┬──────────────────────────────────┘
┌──────────────────────────┴──────────────────────────────────┐
│  sim/           session orchestrator · mission clock        │
│                 reactive turns + self-initiated turns       │
└───────┬──────────────────────────────────┬──────────────────┘
        │                                  │
┌───────┴────────────────┐    ┌────────────┴──────────────────┐
│  agent/   model loop   │    │  delivery/  pacing executor   │
│  prompts, state        │    │  plan · scheduler · channels  │
└───────┬────────────────┘    └────────────┬──────────────────┘
        │                                  │
┌───────┴────────────────┐                 │
│  tools/  @tool surface │                 │
└───────┬────────────────┘                 │
        │                                  │
┌───────┴──────────────────────────────────┴──────────────────┐
│  core/     PURE — no LLM, no network, no I/O                │
│  mission · parameters · dynamics · derived · state          │
│  realism · triggers · procedure                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  providers/   THE AIR-GAP SEAM                              │
│  base.py (protocols) │ cloud/ (dev) │ local/ (deployment)   │
└─────────────────────────────────────────────────────────────┘
```

**The rule that keeps this honest:** `core/` is pure. An automated test walks
imports and fails if anything in `core/` reaches a framework, an HTTP client, or
the filesystem. Without that test, the layering degrades within weeks.

---

## 3. Seam 1 — Mission-declared state (the flexibility seam)

### The problem with the obvious design

The natural approach is typed domain fields:

```python
class MissionState(BaseModel):     # ← WRONG for this product
    fuel_lb: float
    altitude_ft: float
    sensor_mode: str
```

This fails requirement FR-E2/E3/E7. Adding a parameter means editing the model,
the tools, and the prompt. A naval scenario with no fuel at all means a rewrite.

### The design

The engine holds an **untyped value map** plus the mission's **parameter
declarations**. Behaviour comes from the declaration, not from Python fields.

```python
# core/models.py
@dataclass(frozen=True)
class Parameter:
    id: str                        # "fuel_lb"
    label: str | None              # "דלק" — shown to the model and the UI
    type: Literal["number", "enum", "bool", "text"]
    initial: Any
    unit: str | None = None
    values: tuple[str, ...] | None = None    # enum only
    dynamics: Dynamics | None = None         # how it changes with time
    visible_to_persona: bool = True           # FR-D6 knowledge boundary
    min: float | None = None
    max: float | None = None

@dataclass(frozen=True)
class DerivedValue:
    id: str
    expr: str                      # "fuel_lb / burn_per_hour * 60"
    unit: str | None = None
```

### How a value changes over time

A small closed set of `dynamics` kinds, each a pure function of
`(current_value, elapsed_seconds, config, state)`:

| Kind | Behaviour | Example |
|---|---|---|
| `hold` | unchanged until commanded | altitude |
| `linear_drain` | decreases at a rate, with a floor | fuel |
| `linear_fill` | increases at a rate, with a ceiling | recording storage used |
| `rate_toward_target` | moves toward a commanded target at a max rate | climbing to a new altitude |
| `stepped` | changes only on command | sensor mode |
| `scripted` | follows mission-file keyframes | weather deteriorating on schedule |

Each is roughly 15 lines, separately tested. **A domain needing new physics adds
one kind** — the only case requiring code, and it is purely additive (FR-E10).

### Derived values are never stored

`endurance_min` is computed from `fuel_lb` on every read. This makes FR-D2
structural: endurance *cannot* contradict fuel, because it has no independent
existence. Expressions are evaluated by a restricted evaluator over parameter
references, numeric literals and arithmetic — **never Python `eval()`**, since
mission files are the natural "import a mission someone sent me" path.

### The engine

```python
# core/state.py — pure
class MissionStateEngine:
    def __init__(self, mission: Mission, clock: Clock) -> None: ...

    def snapshot(self, for_persona: bool = False) -> dict[str, Any]:
        """Current values plus derived values. for_persona=True filters out
        parameters the mission marks invisible (FR-D6)."""

    def advance_to(self, mission_seconds: float) -> list[Effect]:
        """Apply each parameter's dynamics up to this time. Monotonic and
        idempotent (FR-D4)."""

    def apply(self, command: StateCommand) -> CommandResult:
        """Validate against the parameter's declared type/range/values, then
        mutate. A rejected command mutates nothing (FR-D3)."""
```

Note `snapshot(for_persona=True)`: the knowledge boundary is enforced by the
counterpart being *unable to see* hidden state, rather than by a prompt asking
it not to peek.

---

## 4. Seam 2 — Provider interfaces (the air-gap seam)

Three protocols, each with a cloud implementation for development and a local
one for deployment:

```python
# providers/base.py
class LLMProvider(Protocol):
    name: str
    def chat_model(self) -> BaseChatModel: ...
    def is_unavailable_error(self, err: Exception) -> bool: ...

class STTProvider(Protocol):          # phase 2
    async def transcribe_stream(self, audio) -> AsyncIterator[Transcript]: ...

class TTSProvider(Protocol):          # phase 2
    async def synthesize_stream(self, text: str, voice: VoiceSettings) -> AsyncIterator[bytes]: ...
```

| | Cloud (development) | Local (deployment) |
|---|---|---|
| LLM | Anthropic | Ollama / vLLM / llama.cpp |
| STT | ElevenLabs Scribe | faster-whisper, ivrit.ai Hebrew model |
| TTS | ElevenLabs v3 | Piper / XTTS |

Selection is by environment variable (`LLM_PROVIDER=ollama`) — never a code
change (FR-F3). `is_unavailable_error` carries over from the airport project's
`providers.py`: fall back to the next provider on a *recognized* transient
error, and let a real bug propagate rather than disguising it as an outage.

### Three mechanisms that prove air-gap readiness before deployment

1. **Import guard test** (FR-F4) — fails if any module outside
   `providers/cloud/` imports a cloud SDK.
2. **`--offline` mode** (FR-F6) — installs a socket guard that raises on any
   outbound connection. The full test suite runs under it.
3. **Vendored wheels** (FR-F7) — `pip download` produces a local wheel
   directory so installation needs no package index.

Step 7 of the plan runs the whole system on a local model with networking
disabled. **The air-gap requirement is verified on your machine before anyone
carries this into a closed network.**

---

## 5. Seam 3 — The realism layer (the novel part)

Full detail in [specs/realism-engine.md](specs/realism-engine.md). The core
decision:

> **The model annotates intent. Deterministic code owns all timing.**

Both obvious approaches fail:

- *Model emits `<pause 2s>` tags* → untestable, drifts, token-wasteful, and the
  model cannot know wall-clock facts like how long the trainee has been silent.
- *A scheduler paces generically* → pauses land at meaningless points. A stall
  must come **before the number he is unsure of**, not at a random word.

So the model emits a **closed set of semantic markers, never durations**:

```
«hesitate» «correct» «filler» «breath» «urgent» «calm» «garble»
```

`core/realism.py` compiles markers + the mission's realism numbers + a seeded
RNG into a `DeliveryPlan`: segments with resolved millisecond timings,
interruptibility, and tone. `delivery/scheduler.py` executes it.

This buys three properties at once:

| Property | Why it matters |
|---|---|
| **Reproducible** — same markers + same seed → identical plan | Timing is unit-testable despite the model being non-deterministic (FR-A7) |
| **Channel-agnostic** — the plan contains no text or audio concepts | Voice is one new file, not a rewrite |
| **Bounded model role** — it never emits a number | Same principle as mission state: the model narrates, it does not compute (FR-A8) |

### One mechanism, two channels

| Event | Text channel | Voice channel (phase 2) |
|---|---|---|
| lead-in started | show typing indicator | hold TTS |
| speech segment | stream text chunk | push to TTS, stream audio |
| pause segment | keep indicator, send nothing | emit silence / breath |
| garbled segment | render degraded span | route through noise filter |
| tone shift | styling hint | change TTS voice settings mid-utterance |
| interrupted | stop, persist what was said | cancel TTS frames |

**Interruption ships in the text MVP.** Building it text-only is how the plan
shape gets validated *before* audio latency can hide a design flaw.

---

## 6. The tool surface

Tools are generic over the mission's declared parameters — there is no
`get_fuel` tool, because the engine does not know what fuel is.

| Tool | Purpose |
|---|---|
| `read_state` | current values of named parameters (persona-filtered) |
| `set_parameter` | commanded change, validated against the declaration |
| `check_procedure` | is this transmission procedurally compliant? |
| `report_condition` | structured summary of a mission-defined condition |

Following the airport project's convention, `@tool` is applied **directly to
the real function** — no separate wrapper layer. Every tool returns structured
data, never prose. Errors are typed:

```python
@dataclass(frozen=True)
class ToolError:
    code: Literal["INVALID_ARG", "OUT_OF_RANGE", "NOT_PERMITTED",
                  "UNKNOWN_PARAMETER", "NOT_VISIBLE_TO_PERSONA"]
    message: str      # safe, speakable in character, no internals
```

`check_procedure` matters for perceived fairness: *whether* a call was standard
is decided deterministically from the mission's dictionary; only the *phrasing*
of the challenge is the model's. An operator who challenges a correct call
because the model "felt" it was wrong destroys trust in the simulator.

---

## 7. Initiative

Full detail in [specs/agent-initiative.md](specs/agent-initiative.md).

One interface, three built-in kinds, each yielding an **intent rather than
words**:

```python
class Trigger(Protocol):
    id: str
    priority: Literal["low", "normal", "high", "critical"]
    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None: ...
```

A fourth kind later (for example "three missed readbacks → he gets impatient")
is a new class and nothing else changes (FR-C9).

Reactive and self-initiated turns go through the **same** delivery path, so
realism is uniform. They differ in: what enters history (initiative appends a
labelled *data* cue, so the model phrases it rather than parroting it), lead-in
delay, step budget, and whether trainee speech cancels them.

**Four guards against talking over the trainee:** one turn in flight at a time ·
lower-priority firings deferred, not dropped · suppression while the trainee is
composing (voice: while VAD hears them) · barge-in abandons at the next clause
boundary, **except** `critical`. A real operator shouting "BINGO FUEL" does not
stop because you started talking, and that asymmetry is pedagogically correct.

---

## 8. Persistence

SQLite behind a repository interface. Single trainer, single process — Redis or
Postgres would be unjustified here, and the interface keeps the upgrade additive.

| Table | Holds | Why it exists |
|---|---|---|
| `sessions` | mission id and version, realism seed, timing | Reproducibility (FR-H4) |
| `messages` | clean marker-free text, mission time, reactive vs initiated, trigger id | Transcript and review (FR-H1/H2) |
| `state_snapshots` | append-only state with the cause of each change | Answers "what did he know at 03:12?" (FR-D7) |
| `delivery_records` | planned vs actually delivered | Interruption audit (FR-H3) |
| `trigger_firings` | including **suppressed** firings and why | Answers "why did he not warn me?" (FR-C8) |

Snapshots plus transcripts are what make later scoring purely additive: scoring
reads history rather than re-simulating (FR-H5).

Live engine state is in memory; a process restart ends a session. Documented
prototype limitation — the schema already permits rehydration from the last
snapshot.

---

## 9. Verified environment

Checked on this machine, not assumed:

- `langchain` **1.4.0**, `langgraph` **1.2.11**, Python **3.12.10**
- **The airport project's `create_react_agent(prompt=...)` is deprecated here.**
  Verified by introspection, the current API is:
  `from langchain.agents import create_agent` with **`system_prompt=`**.
- Native middleware — signatures verified, so these are not hand-rolled:
  - `ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")` — the step
    limit with a **graceful stop**, better than the airport project's
    `GraphRecursionError` catch (FR-G1/G2)
  - `ToolErrorMiddleware` (FR-G3) · `ModelFallbackMiddleware`

Re-verify with `pip show langchain langgraph` at implementation time; these
packages move fast.

### Conventions kept from the airport project

Heavy "why" comments at file top · `@tool` on the real function, no wrapper
layer · provider fallback distinguishing transient from real errors ·
env-configurable step limit · server-generated unguessable session ids ·
documented store limitations · full history replayed every turn with no
summarization step (NFR-6) · system messages supplied fresh per call and never
persisted (FR-G8).

---

## 10. Architecture decision records

### ADR-1 — Text before voice
**Decision.** Build the text channel first; voice is phase 2.
**Why.** Realism, initiative and interruption are the hard parts and are far
cheaper to debug in text. Audio latency masks design flaws.
**Consequence.** The delivery plan must be channel-agnostic from day one, which
is a real constraint on its design, and interruption must ship in the MVP.

### ADR-2 — Cascade (STT → LLM → TTS), not speech-to-speech
**Decision.** Chain separate components for voice.
**Why.** A speech-to-speech model fuses understanding and generation into one
opaque loop, which would take timing control away from us — the one thing this
product cannot concede. The cascade also keeps a text checkpoint for transcripts
and is swappable per component, which the air-gap requirement demands.
**Consequence.** Higher latency to manage. Mitigated by streaming per segment,
and by real latency hiding inside the modeled hesitation delay.

### ADR-3 — Timing outside the model
**Decision.** The model emits semantic markers; deterministic seeded code
computes all timing.
**Why.** Testability and reproducibility, and so pacing correlates with meaning.
**Consequence.** A marker vocabulary must be taught in the prompt and stripped
before persistence.

### ADR-4 — Mission-declared state rather than typed domain fields
**Decision.** Parameters are data, not Python fields.
**Why.** FR-E2/E3/E7 — flexibility is the product.
**Consequence.** Loses static typing on state values, so the mission loader must
validate strictly and fail loudly (FR-E8). Accepted deliberately.

### ADR-5 — Offline from the first commit
**Decision.** Provider interfaces, import guard, and offline mode in step 1.
**Why.** Retrofitting an air gap touches every layer; building it in costs one
small file per provider type.
**Consequence.** Slightly more scaffolding up front, and local-model quality
becomes a step-3 question rather than a late surprise.

### ADR-6 — SQLite, not Postgres
**Decision.** SQLite behind a repository interface.
**Why.** Single trainer, single process, and it must work air-gapped with no
service to install.
**Consequence.** Not multi-instance safe. Documented; the interface makes the
upgrade additive.

---

## 11. The risk architecture cannot solve

**Local models are substantially weaker at sustained roleplay than frontier
cloud models**, and Hebrew narrows the field further. A small local model may
break character, flatten the persona, or lose fluency.

No seam fixes this. The mitigations are: test a candidate local model on Hebrew
persona adherence at **step 3** (early enough to change which model you target),
keep the system prompt short and the tools strict so less is demanded of the
model, and document a minimum viable model size once measured.

This is the single most likely reason the project would fall short of its
believability goal, and it is a model-capability question rather than a design
question.
