# Spec — Agent Initiative

**Status:** design · **Requirements covered:** FR-C1 … FR-C9

You flagged this as the area needing the most thought. It is specified in full
here because the hard part is not making the counterpart speak unprompted — it
is making him speak at the *right* moments and stay quiet at the others.

---

## 1. Why this is harder than it looks

A counterpart who only answers questions is passive and unrealistic. A real
operator volunteers things: a fuel state, a lost target, a radio problem.

But a counterpart who speaks whenever a condition fires is worse than passive —
he talks over the trainee, interrupts their thinking, and buries a critical
report in chatter. **Timing and restraint are the feature**, not the speaking.

Three failure modes to design against:

| Failure | How it feels |
|---|---|
| Talks over the trainee | Broken software |
| Buries a critical report among chatter | The trainee misses it, and blames the sim |
| Never volunteers anything | A search engine, not an operator |

---

## 2. One interface, three kinds

```python
# core/triggers.py — pure predicates, no I/O
class Trigger(Protocol):
    id: str
    priority: Literal["low", "normal", "high", "critical"]
    def evaluate(self, ctx: TriggerContext) -> TriggerFiring | None: ...

@dataclass(frozen=True)
class TriggerContext:
    state: Mapping[str, Any]            # current mission state + derived
    mission_seconds: float
    last_trainee_message_at: float | None
    last_counterpart_message_at: float | None
    fired_counts: Mapping[str, int]
    delivery_active: bool
    trainee_composing: bool             # text: typing · voice: VAD active

@dataclass(frozen=True)
class TriggerFiring:
    trigger_id: str
    priority: Priority
    say_intent: str                     # INTENT, not words
    effects: tuple[Effect, ...] = ()
```

| Kind | Fires when | Example |
|---|---|---|
| `TimelineTrigger` | Mission time reaches a point | At T+3:00 the target starts moving |
| `ThresholdTrigger` | A condition over state becomes true | `fuel_lb <= 45` |
| `IdleTrigger` | The trainee has been silent long enough | 45 seconds of no traffic |

**A fourth kind is additive** (FR-C9). For example a
`TraineeErrorPatternTrigger` — "three missed readbacks → he gets impatient" — is
a new class implementing the same protocol. The session loop does not change.

### `say_intent` is an intent, never a line

```yaml
say_intent: "לדווח בינגו ולבקש החלטה על חזרה"     # ✓ intent
say_intent: "מפקדה, נחשון 3, בינגו דלק"            # ✗ a script
```

The intent enters history as **labelled data**, so the counterpart phrases it in
his own voice, with his own tone and hesitations (FR-C7). A literal line would
be read out verbatim every time and immediately feel canned.

---

## 3. The session loop

```
every ~250 ms:
  1. clock.advance()                         → engine.advance_to(now)
  2. collect effects from dynamics
  3. evaluate every trigger against TriggerContext
  4. filter firings (see §4)
  5. if one survives → request an initiated turn
```

Pure evaluation, so the whole loop is testable against a virtual clock: a
45-second idle trigger and a 30-minute session both test in milliseconds.

---

## 4. The four guards against talking over the trainee

Applied in order. This is the core of the spec.

### Guard 1 — Single flight

One turn in flight per session, ever. Enforced server-side (a per-session lock)
*and* client-side (a message queue). Both, because either alone has a race.

The airport project's `ui/src/useMessageQueue.ts` and `api/session_lock.py`
already solve exactly this and are reused.

### Guard 2 — Deferred, not dropped

A firing arriving while a delivery is active and of **equal or lower** priority
is **queued, not discarded** (FR-C5).

This matters: a bingo-fuel call suppressed because he happened to be mid-sentence
must still happen a moment later. Dropping it would be a safety-relevant lie
about the simulation.

Higher-priority firings may interrupt a lower-priority delivery in progress.

### Guard 3 — Composition window

While `trainee_composing` is true, **non-critical** initiative is suppressed
(FR-C6).

The elegance here: text and voice feed the *same* predicate from different
sources — a typing indicator in text, VAD activity in voice. The rule is written
once.

### Guard 4 — Yield on speech

If the trainee starts transmitting during an initiated delivery, `barge_in`
fires and the plan abandons at the next clause boundary.

**Except `critical`**, which sets `yield_on_trainee_speech=False` and rides
through (FR-B5).

> A real operator shouting "BINGO FUEL" does not stop because you started
> talking. That asymmetry is pedagogically correct: the trainee should
> experience being talked over when something urgent is happening, because
> that is what happens.

*This one asymmetry is a judgement about training doctrine rather than
engineering — flagged for your confirmation.*

---

## 5. Priority semantics

| Priority | Interrupts active delivery | Suppressed while composing | Survives barge-in | Example |
|---|---|---|---|---|
| `low` | no | yes | no | Idle check-in |
| `normal` | no | yes | no | Comms degraded |
| `high` | lower-priority only | yes | no | Target moving |
| `critical` | yes | **no** | **yes** | Bingo fuel |

Lead-in delay also scales with priority: a critical report comes fast, an idle
check-in comes slowly. The counterpart's urgency is visible in his timing
before a word is understood.

---

## 6. Initiated vs reactive turns

Both run through the **same** delivery path, which is what keeps realism
uniform. They differ only here:

| | Reactive | Initiated |
|---|---|---|
| Triggered by | A trainee message | A `TriggerFiring` |
| Appended to history | The trainee's turn | A labelled **data** cue: `unprompted-initiative cue: {say_intent}` |
| Lead-in delay | Persona's normal delay | Scales with priority |
| Step budget | Full `max_steps` | Lower (≈3) — it is a report, not problem-solving |
| `yield_on_trainee_speech` | n/a | `True` except `critical` |

The cue is **data, not an instruction** — the same prompt-injection discipline
applied everywhere else. It also stops the model echoing the intent string.

---

## 7. Observability

Every firing is recorded, **including suppressed ones**, with the reason
(FR-C8):

```sql
trigger_firings(session_id, seq, trigger_id, mission_seconds,
                suppressed, suppression_reason)
```

This exists to answer the question you will actually ask: *"why didn't he warn
me about the fuel?"* Without recording suppressions, that is undebuggable — the
absence of an event leaves no trace.

---

## 8. Testing

| Test | Asserts |
|---|---|
| Each kind fires | Timeline at its time, threshold on its condition, idle after its silence |
| `once: true` | Never fires twice (FR-C4) |
| `max_fires` | Honoured for idle triggers |
| Deferral | A firing during delivery is queued and delivered afterwards (FR-C5) |
| Composition suppression | Non-critical suppressed while composing; critical not (FR-C6) |
| Priority interruption | High interrupts low; low never interrupts high |
| Critical barge-in immunity | Critical delivery completes despite trainee speech (FR-B5) |
| Suppression recorded | A suppressed firing appears with its reason (FR-C8) |
| Extensibility | A new trigger class works with no change to the session loop (FR-C9) |
| Intent not parroted | The output is not the literal `say_intent` string (FR-C7) |

All run against a **virtual clock**, so timing assertions are exact and fast.

---

## 9. Your step-5 review

The machine can verify that triggers fire correctly. Only you can answer:

- Does he raise the fuel situation at a sensible moment, or too early/late?
- When he interrupts you — justified, or rude?
- Does the silence prompt feel like a real operator checking in, or like nagging?
- Is the critical-overrides-barge-in behaviour right for your training doctrine?

Tuning is mission-file editing: priorities, `after_silence_seconds`,
`max_fires`, `interrupt_trainee_probability`.

---

## 10. Deliberately not in the MVP

| Not now | Why |
|---|---|
| Triggers reacting to trainee *content* (not just silence) | Needs the error-pattern class; wait until the three basic kinds are tuned |
| Multi-step initiated sequences (a scripted chain of reports) | A single intent per firing is enough to prove the mechanism |
| Trainee-adaptive trigger rates ("he goes quiet if you are overloaded") | Interesting, speculative; revisit after officer validation |
