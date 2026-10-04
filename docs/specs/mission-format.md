# Spec — Mission File Format

**Status:** design · **Requirements covered:** FR-E1 … FR-E10

> **This file format is the product's real interface.** The code is a generic
> engine; a mission file is where a scenario actually lives. If authoring one is
> painful, the flexibility is worthless — so this spec optimises for the author,
> not the parser.

---

## 1. Principles

1. **A new scenario is a new file and nothing else.** No Python changes.
2. **The engine knows no domain.** It has no concept of fuel or aircraft. It
   reads declarations.
3. **Fail loudly.** An invalid mission file never loads with silent defaults —
   it reports what is wrong and where (FR-E8). A mission that silently drops a
   trigger teaches the wrong lesson.
4. **Every realism and tone number is here**, so tuning never needs code.

---

## 2. Top-level structure

```yaml
id: uav_operator_basic         # unique, filename-safe
version: 1                     # bump on change; recorded in session data
language: he                   # engine is language-agnostic (FR-E9)
title: "ניהול משימת כטב\"ם — בסיסי"

setting:        # purpose and place — WHERE and WHY (FR-E4)
persona:        # WHO the counterpart is
tone:           # HOW they speak, and what changes it
realism:        # the human-imperfection numbers
parameters:     # the mission state (FR-E1)
derived:        # computed values
procedure:      # comms discipline (FR-E6)
triggers:       # when the counterpart speaks unprompted
limits:         # safety rails
```

---

## 3. `setting` — purpose and place

```yaml
setting:
  purpose: "אימון קצין מודיעין בניהול משימת איסוף"
  place: "חמ\"ל, קשר עם מפעיל כטב\"ם באוויר"
  trainee_role: "קצין מודיעין בחמ\"ל"

  briefing: |            # shown to the TRAINEE. never given to the model.
    כטב\"ם במשימת איסוף מעל אזור יעד.

  situation: |           # the situation as the COUNTERPART understands it
    אתה באוויר כבר כמה שעות במשימה שגרתית.

  extra_instructions: |  # free text, appended LAST — overrides everything
    אם משהו לא ברור — תשאל, אל תנחש.
```

`briefing` and `situation` are deliberately separate. The trainee's briefing
often contains intent and priorities the counterpart must not know; feeding it
to the model would quietly breach the knowledge boundary. A test asserts the
briefing never reaches the prompt.

`extra_instructions` is placed after every generated section, so it can
countermand them — "he never uses brevity codes, he is sloppy" beats the
procedure section's encouragement to use them.

## 4. `persona` — who the counterpart is

```yaml
persona:
  name: "נחשון 3"
  role: "מפעיל חיישן כטב\"ם"
  experience: "שנה וחצי על הדגם; מקצועי, לא יוצא דופן"

  traits:                    # 0.0–1.0, shape what he volunteers vs. waits to be asked
    patience: 0.6
    deference: 0.4           # low = pushes back when he disagrees
    verbosity: 0.3           # low = terse, net-appropriate
    confidence: 0.7

  # ---------------------------------------------------------------------
  # FREE-TEXT GUIDANCE — the answer to "the instructions may vary"
  #
  # The numeric traits above cover common dials, but no fixed schema
  # anticipates every instruction. These reach the prompt close to
  # verbatim, so write in your own words.
  # ---------------------------------------------------------------------
  background: |          # history he can draw on
    שנה וחצי על הדגם. מכיר את האזור.
  speech_guide: |        # HOW HE TALKS — the most useful field for voice
    קצר ועניני. פותח ב"מפקדה, נחשון 3".
  behaviour_guide: |     # how he acts, judges, takes initiative
    מבצע הוראות בלי להתווכח, אבל אומר אם משהו בעייתי.
  speech_examples:       # worth more than any description
    - "מפקדה, נחשון 3, דלק 140 ליברות."
    - "מפקדה, שלילי. הגובה מעל התקרה שלי."

  # THE KNOWLEDGE BOUNDARY (FR-D6) — enforced by the tool surface, not the
  # prompt. He literally cannot read what is not listed here.
  knows: [fuel_lb, alt_ft, sensor_mode, comms_quality, weather_vis_km]
  does_not_know: [trainee_intent, target_true_identity, mission_priority]
```

`knows` lists parameter ids. Anything not listed is filtered out of his state
snapshot, so he cannot report it even under an adversarial prompt. This is the
main defence of *perceived fairness*: a counterpart who knows your unspoken
intent feels like cheating and the trainee stops trusting the simulation.

## 5. `tone` — how he speaks, and what changes it

```yaml
tone:
  baseline: professional_clipped

  shifts:                    # condition ids come from `triggers`
    - { when: fuel_below_bingo, to: urgent }
    - { when: comms_degraded,   to: strained }
    - { when: target_lost,      to: apologetic }

  profiles:                  # optional: override how a tone sounds
    urgent:
      pace: 1.3              # multiplier on delivery speed
      terseness: 0.8
      filler_multiplier: 0.4 # stressed people use fewer fillers
```

## 6. `realism` — the human-imperfection numbers

Tuned by you, by hand, repeatedly. Every value here is read by
`core/realism.py`; none is hardcoded (FR-E5).

```yaml
realism:
  response_delay_ms: { min: 600, max: 2400, distribution: lognormal }
  # lognormal, not uniform: real reply delays cluster short with a long tail.
  # Uniform delays are the main reason naive simulators feel like lag.

  stall_probability: 0.18           # chance of a mid-utterance hesitation
  stall_duration_ms: { min: 400, max: 1600 }
  filler_probability: 0.22
  filler_sounds: ["אה", "רגע", "אהh"]    # per-language, per-persona
  self_correction_probability: 0.08
  interrupt_trainee_probability: 0.05

  allow_barge_in: true              # may the trainee cut him off
  barge_in_grace_ms: 250            # ignore very short noises

  garble_by:                        # ties realism to mission state
    parameter: comms_quality
    probabilities: { good: 0.0, degraded: 0.15, poor: 0.45 }

  seed: null                        # null = random; an integer = reproducible
```

## 7. `parameters` — the mission state (the flexible core)

```yaml
parameters:
  - id: fuel_lb
    label: "דלק"
    type: number
    initial: 180               # PLACEHOLDER — tune this
    unit: lb
    min: 0
    dynamics: { kind: linear_drain, rate_per_hour: 22, floor: 0 }

  - id: alt_ft
    label: "גובה"
    type: number
    initial: 12000             # PLACEHOLDER — tune this
    unit: ft
    min: 0
    max: 25000
    dynamics: { kind: rate_toward_target, rate_per_minute: 1500 }

  - id: sensor_mode
    label: "מצב חיישן"
    type: enum
    values: [idle, tracking, scanning]
    initial: idle
    dynamics: { kind: stepped }

  - id: comms_quality
    type: enum
    values: [good, degraded, poor]
    initial: good
    dynamics: { kind: scripted, keyframes: [{ at: 600, value: degraded }] }

  - id: weather_vis_km
    label: "ראות"
    type: number
    initial: 8
    unit: km
    dynamics: { kind: hold }

  - id: target_true_identity    # exists in the world, hidden from the persona
    type: text
    initial: "רכב אזרחי"
    visible_to_persona: false
```

### Parameter fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique; referenced by derived values, triggers and tools |
| `type` | yes | `number` · `enum` · `bool` · `text` |
| `initial` | yes | Starting value; validated against type and range |
| `label` | no | Human/Hebrew name shown to the model and UI; defaults to `id` |
| `unit` | no | Reported alongside the value |
| `min` / `max` | no | `number` only; commands outside the range are rejected |
| `values` | enum only | Allowed values |
| `dynamics` | no | How it changes with time; omitted means constant |
| `visible_to_persona` | no | Default `true`; `false` hides it from the counterpart |
| `value_labels` | enum only | What each value is **said as**. Ids stay ASCII (conditions reference them); without labels the counterpart says "idle" aloud on a Hebrew net |

### Dynamics kinds

| Kind | Config | Behaviour |
|---|---|---|
| `hold` | — | Unchanged until commanded |
| `linear_drain` | `rate_per_hour`, `floor` | Decreases steadily |
| `linear_fill` | `rate_per_hour`, `ceiling` | Increases steadily |
| `rate_toward_target` | `rate_per_minute` | Moves toward a commanded target at a max rate |
| `stepped` | — | Changes only on command |
| `scripted` | `keyframes: [{at, value}]` | Follows mission time |

**Adding a kind is additive** (FR-E10) — a new ~15-line pure function. The only
case needing code, and it breaks nothing existing.

## 8. `derived` — computed, never stored

```yaml
derived:
  - id: endurance_min
    expr: "fuel_lb / 22 * 60"
    unit: min
  - id: time_to_bingo_min
    expr: "(fuel_lb - 45) / 22 * 60"
    unit: min
```

Recomputed on every read, so they **cannot** contradict their inputs (FR-D2).

Expressions allow parameter references, numeric literals, `+ - * /`,
parentheses, and `min`/`max`. Evaluated by a restricted evaluator — **never
Python `eval()`**, because mission files are the natural "import a mission
someone sent me" path, and `eval()` there is a code-execution hole.

## 8b. `plan` — what the counterpart is BRIEFED on

**This is different from `triggers`.**

| | |
|---|---|
| `plan` | what he already knows and expects. He was briefed before takeoff. |
| `triggers` | what the simulation does to him during the run. |

Without a plan he reacts to every event as a surprise and never volunteers a
scheduled report — the wrong training experience, because a real operator takes
off knowing the sortie profile. The trainee should be managing someone who
knows their job, not someone discovering it.

All free text: write it the way you would brief a person.

```yaml
plan:
  overview: |
    משימת איסוף שגרתית מעל אזור היעד, כשעתיים וחצי.

  # What he updates and when. This is what makes him PROACTIVE.
  reporting_schedule: |
    דיווח מצב מלא כל 10 דקות — גובה, דלק, מצב חיישן.
    דיווח דלק מיידי כשמגיעים לבינגו — בלי לחכות שישאלו.

  steps:
    - at: "תחילת המשימה"
      what: "דיווח הגעה לאזור"
    - at: "בינגו דלק"
      what: "דיווח בינגו ובקשת החלטה"
      note: "דיווח שהוא חייב ליזום"

  expected_events:        # briefed, so NOT a surprise
    - "בתחזית: ירידה בראות אחר הצהריים"

  standing_orders:        # he acts on these unasked
    - "לא מזהה יעדים בוודאות — מתאר מה רואה בלבד"
```

Trigger timings are deliberately **not** given to him. He is briefed on the
plan, not on the simulation's script — an event you leave out of
`expected_events` should still surprise him. A test asserts this.

---

## 9. `procedure` — comms discipline

```yaml
procedure:
  callsigns: { counterpart: "נחשון 3", trainee: "מפקדה" }

  brevity:
    - { term: "רות",     meaning: "התקבל והובן" }
    - { term: "וילקו",   meaning: "אבצע" }
    - { term: "אמור שוב", meaning: "חזור על ההודעה" }
    - { term: "בינגו",   meaning: "דלק מינימלי לחזרה" }

  rules:
    - id: readback_altitude
      requires_readback_for: [alt_ft]
      on_violation: challenge        # challenge | accept_with_note | ignore
    - id: callsign_required
      applies_to: first_transmission
      on_violation: challenge
```

### `report_formats` — the message STRUCTURE

A brevity list tells him which *terms* exist. It does not tell him what a
transmission is *shaped* like — and without formats he invents his own. An
operator using the right words in the wrong structure is still doing it wrong.

```yaml
  report_formats:
    - id: status_report
      when: "דיווח מצב יזום או לפי בקשה"
      template: "<נמען>, <אני>, <גובה>, <דלק>, <מצב חיישן>"
      required_fields: ["גובה", "דלק", "מצב חיישן"]
      example: "מפקדה, נחשון 3, גובה 12,000, דלק 180, חיישן סורק."

  # Free-text net doctrine, overrides the generated guidance above
  comms_guide: |
    כל תשדורת מתחילה בנמען ואז בקריאה שלו.
    אם לא הבין — "אמור שוב", לא מנחש.
```

**`example` matters more than `template`.** A model matches a sample far more
reliably than it parses a notation, so write real transmissions in your exact
schema. Verified live: with these in place the counterpart followed all four
formats exactly.

Compliance is decided **deterministically** from this dictionary; only the
*wording* of a challenge is the model's. A counterpart that challenges a correct
call because the model "felt" it was wrong destroys trust in the simulator.

## 10. `triggers` — when he speaks unprompted

Three kinds, one interface. See
[agent-initiative.md](agent-initiative.md) for the suppression rules.

```yaml
triggers:
  timeline:
    - id: target_starts_moving
      at_mission_seconds: 180
      effects: [{ parameter: sensor_mode, value: tracking }]
      say_intent: "לדווח שהיעד החל בתנועה"
      priority: high

  thresholds:
    - id: fuel_below_bingo
      when: "fuel_lb <= 45"
      once: true
      say_intent: "לדווח בינגו ולבקש החלטה על חזרה"
      priority: critical          # critical survives barge-in

    - id: comms_degraded
      when: "comms_quality != good"
      once: false
      say_intent: "לדווח על קשר ירוד"
      priority: normal

  idle:
    - id: silence_check
      after_silence_seconds: 45
      max_fires: 3
      say_intent: "לשאול אם להמשיך בהקפה הנוכחית"
      priority: low
```

`say_intent` is an **intent, not a line of dialogue**. It enters the
conversation as labelled data so the counterpart phrases it himself in his own
voice — and does not parrot the string (FR-C7).

`when:` uses the same restricted evaluator as `derived`.

## 11. `limits`

```yaml
limits:
  max_steps: 8                 # overridden by AGENT_MAX_STEPS
  session_max_minutes: 30
```

---

## 12. Validation rules (all fail at load, FR-E8)

| Check | Error |
|---|---|
| Duplicate parameter or trigger id | `DUPLICATE_ID` |
| `derived`/`when` references an unknown parameter | `UNKNOWN_PARAMETER_REF` |
| `initial` violates its own type, range or enum values | `INVALID_INITIAL` |
| `dynamics.kind` unknown | `UNKNOWN_DYNAMICS_KIND` |
| Dynamics config missing a required field | `INCOMPLETE_DYNAMICS` |
| `tone.shifts.when` names no existing trigger | `UNKNOWN_TRIGGER_REF` |
| `persona.knows` names an unknown parameter | `UNKNOWN_PARAMETER_REF` |
| An expression uses a forbidden construct | `UNSAFE_EXPRESSION` |
| `garble_by.parameter` is not an enum parameter | `INVALID_GARBLE_SOURCE` |

Every error names the field path and the line. The goal is that a typo takes
seconds to find, not a debugging session.

---

## 13. Authoring workflow

1. Copy `missions/uav_operator_he.yaml` — the fully-commented reference.
2. Edit `setting`, `persona`, `parameters`.
3. `python -m sim.cli --mission missions/yours.yaml --validate` — fails loudly.
4. `python -m sim.cli --mission missions/yours.yaml --timeline 10` — print ten
   minutes of state and sanity-check the numbers. No LLM, no API key.
5. Run a session; tune `realism` by feel.

Steps 3 and 4 need no model and no network, so iterating on a mission is fast.
