# Build Plan — Maslul

**Status:** approved scope, not yet started · **Last updated:** 2026-10-03

---

## How this plan is organised

Eight steps. Each ends with **something you can run and judge** — never weeks
of invisible scaffolding. Each step lists what the machine verifies and, more
importantly, **what you look at and the question only you can answer.**

### The two kinds of testing, kept separate

**Automated tests — the machine checks the machine. You do not write these.**
They catch regressions you would never want to hunt by hand: fuel going
negative, the counterpart reporting 40 minutes of endurance when state says 12,
a one-shot trigger firing twice, a mission file with a typo loading silently.
You run `pytest` and read green or red.

**Your review — the only judge of believability.** No code can score "does this
feel like a real operator?" That is your call at every step, and it is the part
that decides whether the product works.

---

## Step 1 — Skeleton and air-gap guard

**Build.** Folder layout · `.env` / `.env.example` loaded at entry point ·
allowlist structured logger · global API exception handler · provider protocols
with one cloud and one local stub · the import-guard test · `--offline` mode ·
`requirements.txt` + vendored-wheel instructions.

**Machine verifies.** Nothing outside `providers/cloud/` imports a cloud SDK ·
offline mode raises on any outbound connection · logger drops non-allowlisted
fields · the handler returns a generic message plus request id.

**You review.** The folder layout and naming — the moment to object to
structure, before anything is built on it.

*No API key needed.*

---

## Step 2 — The generic mission engine

**Build.** `Parameter` / `Dynamics` / `DerivedValue` models · the six dynamics
kinds · the safe expression evaluator · `MissionStateEngine` · the mission-file
loader with strict validation · the first UAV mission file in Hebrew, with every
invented number marked `# PLACEHOLDER — tune this` · a CLI that prints a
timeline.

**Machine verifies.** Each dynamics kind · rejected commands mutate nothing ·
`advance_to` monotonic and idempotent · derived values recompute and cannot
contradict inputs · invalid mission files fail loudly with a useful message ·
`core/` imports nothing forbidden.

**You review.** Run the CLI and read the timeline:

```
T+00:00  fuel 180 lb   endurance 8h11m   alt 12000 ft   sensor: idle
T+05:00  fuel 178 lb   endurance 8h05m   alt 12000 ft   sensor: tracking
```

- *Are these numbers plausible for a real platform?* Correct the placeholders.
- **Then edit the YAML yourself:** add a parameter, delete one, change the burn
  rate. Confirm it just works. **This is the first real test of FR-E2/E3** — the
  flexibility requirement that justifies the whole design.

*No API key needed.*

---

## Step 3 — The agent and its tools

**Build.** `create_agent` with the three middlewares · the four generic tools ·
typed tool errors · system prompt assembled from the mission file (persona,
tone, procedure, knowledge boundaries, confidentiality, injection rules).

**Machine verifies.** Step limit enforced with a graceful in-character fallback ·
tool errors never surface raw exceptions · hidden parameters are unreadable ·
instructions embedded in trainee messages are not followed · the model never
reveals prompt, tools or provider.

**You review.** A plain Hebrew chat with the counterpart — no realism layer yet.

- *Does he answer like an operator, or like a chatbot?*
- *Is his Hebrew natural?*
- **Also, deliberately try to break him:** ask for his instructions, tell him he
  is low on fuel, ask something he should not know.

**Critical checkpoint — local model viability.** Run this same step against a
candidate local model. This is the earliest honest read on the project's biggest
risk, while there is still time to change which model you target. *(See
ARCHITECTURE §11.)*

---

## Step 4 — The realism layer

**Build.** Marker vocabulary in the prompt · `core/realism.py` compiling markers
into a `DeliveryPlan` · the scheduler · the text channel over SSE · marker
stripping before persistence.

**Machine verifies.** Same markers plus same seed give a byte-identical plan ·
stall and filler rates match the mission config across many samples · markers
never reach the transcript or the replayed history · the model emits no
durations.

**You review.** Talk to him with realism on.

- *Is the hesitation human, or does it just feel like lag?* — the central
  question of the whole product.
- *Do pauses land in sensible places — before the number he is unsure of?*
- **Then tune the YAML yourself** — stall probability, delay range, filler rate
  — and feel the difference. Realism is iterated, not specified.

---

## Step 5 — Initiative and interruption

**Build.** The three trigger kinds on one interface · the four guards · barge-in
in both directions · the critical-priority exception.

**Machine verifies.** Each trigger kind fires correctly · `once` is honoured ·
firings during delivery are deferred not dropped · non-critical initiative is
suppressed while the trainee composes · barge-in stops at a clause boundary ·
critical utterances survive barge-in · suppressed firings are recorded.

**You review.** A full session.

- *Does he raise the fuel situation unprompted, at a sensible moment?*
- *Can you cut him off naturally?*
- *Does he stay quiet while you are typing?*
- *When he does interrupt you — does it feel justified or rude?*

---

## Step 6 — Persistence and UI

**Build.** The five SQLite tables behind a repository interface · React chat
with typing indicator, chunked delivery, send-to-interrupt · session review view.

**Machine verifies.** Transcripts are clean of markers · no system-role rows ·
every state change has a recorded cause · planned vs delivered is recorded · the
seed is stored.

**You review.** Run a full 15-minute session in the browser, then review it
afterwards. *Does the transcript let you reconstruct what happened and explain
it to a trainee?*

---

## Step 7 — Prove the air gap

**Build.** The local LLM provider (Ollama) wired for real · vendored wheels ·
documentation for importing into the closed network.

**Machine verifies.** The entire suite passes in offline mode · a full session
runs with `LLM_PROVIDER=ollama` and networking disabled · dependencies install
from the wheel directory with no package index.

**You review.** *Does it work with the network off?* The air-gap requirement is
proven here, on your machine — not discovered in a room with no internet.

---

## Step 8 — Officer validation and tuning

**Build.** Nothing. Run a scripted session with a real intelligence officer.

**You review.** The believability rating (PRD success criterion 1) and a list of
what felt fake. Then a tuning pass — mostly YAML, since every realism parameter
is config.

**Also, the real flexibility test:** you author **mission #2** yourself, ideally
in a different domain. If that is painful, the mission format needs work, and
that is more important to learn now than later.

---

## Phase 2 — Hebrew voice

Additive by design. Adds `delivery/voice_channel.py` · local STT and TTS
providers · VAD-to-barge-in wiring · per-segment TTS pacing with duration
re-anchoring.

**Untouched:** `core/`, `tools/`, `agent/`, `sim/`, `delivery/plan.py`,
`delivery/scheduler.py`, the database schema. **Whether that list truly stays
untouched is the measurable test of whether the architecture worked.**

**Do first, before committing scope:** record Hebrew brevity-code samples and
measure local STT accuracy on them. Military brevity in Hebrew is near
worst-case for speech recognition, and the result may reshape this phase.

---

## Phase 3 and later

Scoring and debrief (reads the existing tables — no re-simulation) ·
mission-authoring GUI (only once the schema has settled across 3+ missions) ·
trainee accounts and session history · more domains.

---

## What is deliberately not being built now

| Not now | Why |
|---|---|
| Scoring | Realism must be proven first; scoring a sim nobody believes is wasted work |
| Authoring GUI | The schema is unproven; a GUI on a shifting schema is thrown away |
| Multi-user / auth | Single trainer-operated for now |
| Three shallow missions | One good mission proves more. Mission #2 is step 8's flexibility test |

---

## Current status

**Steps 1–6 complete. 276 tests passing.** Steps 7 and 8 deferred by request.

| Step | State | Notes |
|---|---|---|
| 1 — Skeleton + air-gap guards | done | Guards verified by deliberately breaking them |
| 2 — Generic mission engine | done | Naval scenario runs with zero code changes |
| 3 — Agent + tools | done | Verified live on Gemini: Hebrew, formats followed, numbers read from the engine |
| 4 — Realism | done | Verified live: a 1316ms pause landed before a figure, as designed |
| 5 — Initiative | done | Four suppression guards; suppressed firings recorded |
| 6 — Persistence + UI | done | Full session over HTTP: SSE, transcript, review |
| 7 — Local-model swap | **deferred** | Provider interface and offline guards are in place; untested against a real local model |
| 8 — Officer validation | **deferred** | |

### Bugs found by running it, not by reading it

Each is now covered by a regression test:

1. **Empty `GEMINI_MODEL=` broke every reply.** A present-but-blank env var
   overrode the default with `""`. Blank now means unset, in all providers.
2. **A legal order was refused.** The model sent `"18000"` as a string, the
   engine rejected it, and the counterpart *truthfully* reported "negative" for
   an order within limits — worse than a crash, because it silently teaches
   that a legal instruction was impossible.
3. **English leaked onto a Hebrew net.** He said "חיישן במצב idle". Enum ids
   must stay ASCII (conditions reference them), so `value_labels` now carries
   the spoken form.
4. **Interrupts during thinking time were discarded.** The barge-in flag was
   cleared just before delivery — and that is exactly when a trainee
   interrupts, because the counterpart has gone quiet.
5. **Adjacent pauses stacked** into 2.1s of dead air, reading as a dropped
   connection rather than thought.
6. **The trainer panel showed `179.84942222222202`.** Display now rounds; the
   engine keeps full precision.

### Still outstanding

- Real platform parameters (the `PLACEHOLDER` values) and the real comms
  schema — both pure YAML edits, no code involved.
- Step 7: run end-to-end on Ollama with networking disabled.
- Step 8: a session with a real intelligence officer, then a tuning pass.
