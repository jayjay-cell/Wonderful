# Spec — Realism Engine

**Status:** design · **Requirements covered:** FR-A1 … FR-A9, FR-B1 … FR-B6

This is the novel part of the system and the one most likely to decide whether
the product feels real. It is specified before implementation deliberately.

---

## 1. The problem

A counterpart that replies instantly with clean, complete sentences is
immediately recognisable as software. Real people on a net hesitate, trail off,
say "אה" while checking an instrument, correct themselves, get cut off, cut you
off, and change tone when things go wrong.

The naive implementations both fail:

| Approach | Why it fails |
|---|---|
| **Model emits timing tags** (`<pause 2s>`) | Untestable — output is non-deterministic. Drifts: the model starts emitting 40 tags per turn. Wastes tokens. And it cannot know wall-clock facts like how long the trainee has been silent. |
| **A scheduler paces generically** | Pauses land at meaningless points. A hesitation must come **before the number he is unsure of** — not at a random word boundary. Uniformly random delay reads as network lag, not as thought. |

## 2. The decision

> **The model annotates intent. Deterministic seeded code owns all timing.**

The model emits a closed set of **semantic markers** and never a duration
(FR-A8). `core/realism.py` compiles markers + the mission's realism numbers + a
seeded RNG into a `DeliveryPlan` — a fully-resolved, millisecond-precise,
channel-agnostic score. `delivery/scheduler.py` executes it against a clock.

Three properties follow, and all three are load-bearing:

| Property | Consequence |
|---|---|
| **Reproducible** — same markers + same seed → identical plan | Timing is unit-testable even though the model is not (FR-A7) |
| **Channel-agnostic** — the plan names no text or audio concept | Voice becomes one new file instead of a rewrite |
| **Bounded model role** — never emits a number | The same discipline as mission state: narrate, do not compute |

---

## 3. Marker vocabulary

A closed set, taught in the system prompt. Closed deliberately: an open
vocabulary invites invention and drift.

| Marker | Meaning | Rendered as |
|---|---|---|
| `«hesitate»` | Searching for a word, or checking an instrument | A pause of mission-configured length |
| `«correct»` | A self-correction follows | Used as `wrong «correct» right` |
| `«filler»` | A filler sound | Drawn from `realism.filler_sounds` |
| `«breath»` | A natural clause boundary | A short pause; **the cheapest interruption point** |
| `«urgent»` | Tone shift — compress pacing, clip words | Applies from here to end of utterance |
| `«calm»` | Tone shift — release toward baseline | — |
| `«garble»` | This span was a degraded transmission | Visually degraded, or noise-filtered audio |

### Markers never persist

Stripped before the transcript is written and before history is replayed to the
model (FR-A9). Two reasons: transcripts must be clean prose for review, and
markers accumulating in replayed history would make the model imitate its own
marker density — a feedback loop that drifts within a few turns.

### What the model sees and produces

Given mission state, the counterpart might produce:

```
«filler» רגע... «hesitate» יש לי בסביבות ארבעים דקות «breath» אולי פחות
```

Note the hesitation lands **before the quantity** — exactly the point a real
operator checks a gauge. That correlation with meaning is the thing a generic
scheduler cannot produce, and the reason markers exist at all.

---

## 4. Data shapes

```python
# delivery/plan.py — pure dataclasses, no framework imports

SegmentKind = Literal["speech", "pause", "filler", "selfcorrect",
                      "garble", "tone_shift"]

@dataclass(frozen=True)
class DeliverySegment:
    index: int
    kind: SegmentKind
    text: str                       # "" for pause and tone_shift

    # Timing — fully resolved. The single source of truth for both channels.
    start_ms: int                   # offset from plan start
    duration_ms: int                # text: dwell time; voice: expected TTS duration

    # Interruption semantics
    interruptible: bool
    resume_policy: Literal["restart_segment", "skip_segment", "abandon_plan"]

    # Channel hints — advisory, never timing
    tone: ToneProfile | None
    garbled: bool = False

@dataclass(frozen=True)
class DeliveryPlan:
    plan_id: str
    session_id: str
    turn_id: str
    origin: Literal["reactive", "initiated"]
    lead_in_ms: int                 # the "thinking" delay BEFORE segment 0
    segments: tuple[DeliverySegment, ...]
    total_ms: int
    clean_text: str                 # marker-free, for persistence
    seed: int
    yield_on_trainee_speech: bool = True    # False for critical initiative
```

### Compilation

```python
# core/realism.py — pure
def compile_plan(
    marked_text: str,
    realism: RealismConfig,      # from the mission file
    tone: ToneProfile,
    state: Mapping[str, Any],     # for state-dependent garbling
    seed: int,
    origin: Literal["reactive", "initiated"],
) -> DeliveryPlan: ...
```

Pure: no clock, no network, no I/O. Called with a fixed seed in tests, it is a
plain function — which is what makes timing testable at all.

### Timing rules

| Aspect | Rule |
|---|---|
| Lead-in delay | Drawn from `response_delay_ms`, **lognormal** — real delays cluster short with a long tail. Uniform is the main reason naive simulators feel like lag (FR-A1). |
| Speech dwell | Proportional to text length, scaled by `tone.pace` |
| Stall duration | Drawn from `stall_duration_ms` |
| Garbling | Probability looked up from `garble_by` against live state (FR-A6) |
| Tone shift | Applies from its segment to the end of the utterance, and persists into later turns until released (FR-A5) |
| Interruptibility | `«breath»` boundaries are always interruptible; mid-number spans never are |

---

## 5. Execution

```python
# delivery/scheduler.py
class DeliveryChannel(Protocol):
    async def on_event(self, event: DeliveryEvent) -> None: ...

DeliveryEvent = (PlanStarted | LeadInStarted | SegmentStarted | SegmentEnded
                 | PlanInterrupted | PlanCompleted | PlanAbandoned)

async def execute(plan: DeliveryPlan, channel: DeliveryChannel,
                  clock: Clock, barge_in: asyncio.Event) -> DeliveryOutcome: ...
```

`DeliveryOutcome` records **what was actually delivered versus planned**, which
is what gets persisted — so an interrupted utterance is stored as what was
really said, not what was intended (FR-B3).

### One mechanism, two channels

| Event | Text channel | Voice channel (phase 2) |
|---|---|---|
| `LeadInStarted` | SSE `typing_start` | hold TTS; optional mic-click |
| `SegmentStarted(speech)` | SSE `chunk` | push text to TTS, stream audio |
| `SegmentStarted(pause)` | keep indicator, send nothing | emit silence or breath |
| `SegmentStarted(filler)` | SSE `chunk` with the filler | synthesise the filler sound |
| `SegmentStarted(garble)` | degraded styling | route through noise/dropout |
| `SegmentStarted(tone_shift)` | styling metadata | change TTS voice settings mid-utterance |
| `PlanInterrupted` | stop chunks, persist partial | cancel TTS and LLM frames |

**The seam:** the scheduler and the plan know nothing about either channel.
Phase 2 adds one file implementing `DeliveryChannel`, plus a processor setting
`barge_in` from VAD. Nothing in `core/` or `sim/` changes.

### The one known wrinkle

`duration_ms` is *chosen* dwell time in text but *expected* TTS duration in
voice, and TTS estimates are imperfect. Resolution: the voice channel reports
actual segment durations back, and the scheduler re-anchors subsequent
`start_ms` — drift correction, rather than letting the two channels diverge in
how they interpret a plan.

---

## 6. Interruption

Both directions, both in the text MVP.

### Trainee interrupts the counterpart

1. Text: the trainee sends mid-delivery. Voice: VAD detects speech past
   `barge_in_grace_ms`.
2. The `barge_in` event is set.
3. The scheduler stops at the **next interruptible boundary** — never mid-word
   (FR-B2).
4. `resume_policy` decides what happens to the current segment.
5. The partial delivery is persisted as what was actually said.
6. **Exception:** `yield_on_trainee_speech=False` (critical initiative) rides
   through (FR-B5).

### Counterpart interrupts the trainee

At `interrupt_trainee_probability`, while the trainee is composing and a
non-critical cue is pending, the counterpart starts anyway. Rate-limited so it
is a texture rather than a nuisance.

**Why interruption ships in the text MVP:** building it text-only is how the
plan shape gets validated before audio latency can hide a design flaw. It also
means the voice phase inherits working logic rather than debugging two new
things at once.

---

## 7. Testing

Full strategy in [../PLAN.md](../PLAN.md). What is specific here:

| Technique | What it buys |
|---|---|
| **Seeded determinism** | The model is non-deterministic; the realism layer is not. Fixed seed → golden-file comparison of the whole plan. The main reason timing lives outside the model. |
| **Distribution assertions** | Over 1000 samples, assert the stall rate is within tolerance of config — not that any single plan matches exactly. |
| **Virtual clock** | A 45-second idle trigger and a 30-minute session test run in milliseconds with exact ordering assertions. |
| **Forced-failure markers** | `__FORCE_GARBLE__`, `__FORCE_INTERRUPT__` and friends, gated behind `ENABLE_TEST_MARKERS`. Without them, garbling can only be exercised by waiting for a 15% probability to land — a flaky test. |

**Not automated:** whether it *feels* human. That is your step-4 judgement, and
no assertion substitutes for it.

---

## 8. Tuning

Every number lives in the mission file's `realism` block (FR-E5), so tuning is
editing YAML and rerunning — never a code change. Expect several passes.

Known failure modes to listen for:

| Symptom | Likely cause |
|---|---|
| Feels like lag, not thought | `response_delay_ms` max too high, or distribution left uniform |
| Feels twitchy | `stall_probability` too high; real operators are mostly fluent |
| Fillers feel cartoonish | `filler_probability` too high, or sounds not natural for the persona |
| Hesitation in odd places | The model is misplacing markers — a prompt problem, not a config one |
| Interruption feels rude | `interrupt_trainee_probability` too high, or too many cues marked critical |
