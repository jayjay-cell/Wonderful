# Authoring an exercise

An exercise is **three files**. No code changes.

```
context/*.md              reusable professional knowledge, shared by all exercises
timelines/<name>.xlsx     what happens during the recording
missions/<name>.yaml      who, where, and how this crew behaves
```

Copy `missions/synthetic_he.yaml` and `timelines/synthetic_he.csv` as a
starting point. Everything in them marked `SYNTHETIC` is invented and
needs replacing.

---

## 1. Global context — `context/*.md`

Written once, used by every exercise. Correcting a convention here corrects
it everywhere, which is the reason it is not copied into each mission.

| File | Holds |
|---|---|
| `roles-and-authority.md` | kamak, operator, controller; who decides what |
| `communication.md` | addressing, transmission style, uncertainty |
| `briefing-norms.md` | what an opening briefing covers |
| `mission-workflow.md` | desired outcome over mechanics |
| `crew-handover.md` | rotation, availability, inheritance |
| `examples.md` | annotated good and poor exchanges |

These reach the model **verbatim**. Write them as you would brief a person.

Poor examples must be marked as negative, with what is wrong and the
preferred behaviour — they are more useful than the good ones, and a model
will imitate an unmarked bad example.

> Everything currently marked ⚠ **PLACEHOLDER** is a gap: real brevity
> vocabulary, call formats, and the actual workflow have not been supplied.
> Nothing in these files should be treated as validated doctrine until you
> replace them.

---

## 2. Timeline — Excel or CSV

Generate a formatted template:

```bash
python timelines/make_template.py
```

It opens with the columns, hover notes on each, and example rows to edit
over. Each column note explains what the field does, so the guidance is
where you are working rather than in a separate document.

### Columns

Only `event_id` and `start_time` are required.

| Column | Notes |
|---|---|
| `event_id` | Unique. Appears in logs and the debrief, so avoid renaming after a run. |
| `start_time` | From the **start of the recording**. `HH:MM:SS`, `MM:SS`, seconds, or an Excel time cell. |
| `end_time` | Required for `interval`; also sets a handover's duration. |
| `event_type` | `point` · `persistent` · `interval` · `handover` |
| `description` | **Author-facing only.** Never reaches the model. Put intent and the solution here. |
| `operator_information` | **The only column the model sees.** Write it as the operator would say it. |
| `state_updates` | `key=value, key=value` or JSON. Facts this event changes. |
| `tags` | Categories for agreements: `vehicle, arrival, person`. |
| `entity_ids` | A specific thing: `veh_1`. Lets "only that vehicle" narrow an agreement. |
| `reporting_policy` | `required` · `on_request` · `subscription` |
| `priority` | `normal` · `high` · `urgent` |
| `instructions` | In-character guidance for this moment. Shapes delivery, not content. |
| `expires_at` | After this, an undelivered report is dropped rather than announced as news. |

### Event types, and why the distinction matters

| Type | Meaning |
|---|---|
| `point` | An occurrence. A vehicle that stopped at 03:00 is **not** still stopping at 10:00 — it is recalled in the past tense. |
| `persistent` | True until a later event supersedes it. |
| `interval` | Current only inside `[start, end)`. Outside it the observation has expired and must not be described as current. |
| `handover` | Crew rotation. Blocks ordinary conversation for its duration. |

### Reporting policy

| Policy | Behaviour |
|---|---|
| `required` | Volunteered as soon as it is known. |
| `on_request` | Only if asked. |
| `subscription` | Only if a dialogue agreement covers it. |

The useful shape: make one vehicle `required` and the rest
`subscription`. A trainee who asks "tell me about every vehicle" then gets
all of them; one who does not gets the first only. That difference is the
exercise.

### Handover

Author the actual moment. **Do not** assume four hours after the recording
starts — a recording can begin partway through a shift.

```
handover | 00:14:00 | 00:16:00 | handover | Crew rotation | | | | | on_request | normal
```

An `urgent` event inside that window still comes through, if the mission's
`handover.allow_urgent` is true.

---

## 3. Mission state — `missions/<name>.yaml`

Required: `id`, `title`, `callsigns.operator`, `callsigns.trainee`.
Everything else is optional, deliberately — exercises differ in shape, and
requiring a fuel figure for an exercise where fuel is irrelevant invites an
invented number that looks like a real fact.

```yaml
id: my_exercise
title: "..."
language: he
duration_seconds: 1200          # omit to use the timeline's end

callsigns:
  operator: "גלוק"
  trainee: "מדבקה"
  controller: "משנה"            # named in context, not simulated

setting:
  purpose: "..."
  area: "..."
  background: "..."
  before_recording: "..."        # what happened before the video starts
  trainee_briefing: "..."        # shown to the TRAINEE only

crew:
  prior_briefing: "..."          # the crew knows this from the first word

platform:
  capabilities: ["..."]
  limitations: ["..."]

initial_facts:                   # free-form, all optional
  ראות: "בינונית"
hidden_facts: []                 # keys the operator cannot see

behaviour:
  initiative: 0.6                # 0 waits to be spoken to · 1 volunteers
  challenge: 0.5                 # 0 complies · 1 argues its view
  verbosity: 0.35                # 0 minimum words · 1 explains
  patience: 0.6                  # 0 impatient on repetition · 1 unbothered
  style: "..."                   # free text; the most useful field
  briefing_request_after: 90     # seconds before a proactive crew asks

reestablish_silence: 30          # formal addressing returns after this

handover:
  busy_reply: "..."              # if called mid-rotation
  allow_urgent: true             # authored urgent events still come through

impossible_requests:
  explanations:
    zoom: "..."                  # in-character reason, authored by you
  fallback: "..."

private:                         # NEVER reaches the operator
  visible_to_operator: false     # must stay false; validated at load
  solution: "..."
  debrief_points: ["..."]

context_files: [roles-and-authority.md, communication.md, ...]
timeline_file: my_exercise.xlsx
```

### Behaviour dials

Only values at the extremes produce a prompt instruction. A mid-range
number says nothing deliberately — describing every dial as "moderate"
fills the prompt with noise and dilutes the ones you actually set.

| Setting | ≤ 0.35 | ≥ 0.65 |
|---|---|---|
| `initiative` | answers only what is asked | volunteers, chases gaps, asks for a skipped briefing |
| `challenge` | complies, queries only real ambiguity | disagrees, says what it would do instead |
| `verbosity` | terse | explains reasoning |
| `patience` | audibly worn by repetition | unbothered |

`style` is free text and reaches the prompt last, so it can countermand
anything generated above it.

### Impossible requests

The recording cannot change. When the trainee asks to zoom or change
angle, the operator declines using **your** reason:

```yaml
impossible_requests:
  explanations:
    zoom: "אני על ההגדלה המקסימלית שיש לי כרגע בגזרה הזו."
  fallback: "זה לא משהו שאני יכול לעשות כרגע."
```

The operator never mentions a recording, never invents a malfunction, and
never claims the action was taken. Without an authored reason it falls back
to an honest "I cannot do that right now" — which is why `fallback` is
worth writing.

---

## Validate before running

```bash
python -m sim.cli --mission missions/my_exercise.yaml
python -m sim.cli --mission missions/my_exercise.yaml --events
python -m sim.cli --mission missions/my_exercise.yaml --walk 20
```

No model, no API key, no network.

`--walk` steps a virtual clock and shows what the operator learns and when,
with `REPORT` marking what it will volunteer:

```
01:30  REPORT  הראות ירדה, פחות חדה מקודם
03:00  REPORT  רכב מגיע מכיוון מערב ועוצר ליד המבנה
08:00          רכב שני מגיע ועוצר ליד הראשון
14:00          (handover)
15:00  REPORT  תנועה נוספת באזור המבנה
```

That second vehicle is silent because nothing agreed to it — which is the
behaviour to check before a trainee meets it.

Validation fails loudly and names the row or field. An exercise that loaded
with a silently-dropped event would run missing a beat you believed was
there, and the trainee would be assessed on it.

---

## Running it

```bash
python run.py          # http://localhost:8000
```

1. Pick the exercise and the channel
2. **הכן תרגיל** — prepares; mission time stays at 0
3. **התחל תרגיל** — start the exercise and the video at the same moment
4. **השהה** pauses the exercise; pause the video yourself too

The clock measures elapsed time since the recording began, in real time.
There is no speed control: the exercise runs beside a video in a separate
player, and accelerating one would desynchronise them.

Key in `.env`: `GEMINI_API_KEY`. Voice uses the same key — Gemini Live
needs no separate speech account.
