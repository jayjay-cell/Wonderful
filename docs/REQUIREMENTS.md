# Requirements — Maslul

Every requirement below is phrased so that a test could fail it. "Realistic
conversation" is not a requirement; "response delay is drawn from the mission's
configured distribution" is.

**Legend:** `MVP` = text MVP · `V` = voice phase · `L` = later
**Verify:** `auto` = automated test · `manual` = you check it · `both`

---

## A. Conversation realism

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-A1 | The counterpart's response delay before its first output is drawn from the mission's configured distribution (min, max, shape) | MVP | auto |
| FR-A2 | Mid-utterance pauses occur at points the model marked as hesitation, not at arbitrary word boundaries | MVP | auto |
| FR-A3 | Filler sounds are rendered per-persona and per-language from mission config | MVP | both |
| FR-A4 | Self-corrections ("...two four zero — correction, two six zero") occur at the configured rate | MVP | auto |
| FR-A5 | Tone shifts triggered by a named condition persist for the remainder of the utterance and into subsequent turns until a releasing condition | MVP | both |
| FR-A6 | Degraded transmission (garbling) probability is a function of the mission's comms-quality parameter | MVP | auto |
| FR-A7 | Given identical markers and an identical seed, the delivery plan is byte-identical | MVP | auto |
| FR-A8 | The model never emits a duration, delay, or timing value; all timing is computed outside the model | MVP | auto |
| FR-A9 | Markers are stripped from persisted transcripts, and never re-enter the replayed conversation history | MVP | auto |

## B. Interruption and turn-taking

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-B1 | A trainee message sent mid-delivery interrupts the counterpart | MVP | auto |
| FR-B2 | Interruption takes effect at the next interruptible boundary, not mid-word | MVP | auto |
| FR-B3 | The partially-delivered utterance is persisted as what was actually said, not as what was planned | MVP | auto |
| FR-B4 | The counterpart interrupts the trainee at the mission's configured rate | MVP | both |
| FR-B5 | A `critical`-priority utterance is not abandoned by trainee speech | MVP | auto |
| FR-B6 | In voice, VAD-detected speech produces the same interruption as a sent text message | V | auto |

## C. Agent initiative

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-C1 | Timeline triggers fire at their configured mission time | MVP | auto |
| FR-C2 | Threshold triggers fire when their condition over mission state becomes true | MVP | auto |
| FR-C3 | Idle triggers fire after the configured trainee silence, up to a configured maximum | MVP | auto |
| FR-C4 | `once: true` triggers never fire twice | MVP | auto |
| FR-C5 | A trigger firing while a delivery is active is deferred, not dropped | MVP | auto |
| FR-C6 | Non-critical initiative is suppressed while the trainee is composing | MVP | auto |
| FR-C7 | A trigger supplies an intent, never literal words; the counterpart phrases it itself | MVP | auto |
| FR-C8 | Suppressed firings are recorded, so "why did it not warn me?" is answerable | MVP | auto |
| FR-C9 | A new trigger class can be added without modifying the session loop | MVP | auto |

## D. Mission state correctness

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-D1 | Every numeric value the counterpart states originates from mission state, never from the model | MVP | auto |
| FR-D2 | Derived values are computed on read, never stored, so they cannot contradict their inputs | MVP | auto |
| FR-D3 | A rejected state command leaves state unmodified | MVP | auto |
| FR-D4 | Advancing mission time is monotonic and idempotent for a given target time | MVP | auto |
| FR-D5 | The same quantitative question asked twice in a session yields the same answer | MVP | both |
| FR-D6 | The counterpart cannot read state the mission marks as outside its knowledge | MVP | auto |
| FR-D7 | A state snapshot is recorded for every change, with its cause | MVP | auto |

## E. Mission flexibility *(the core product requirement)*

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-E1 | Mission state parameters are declared in the mission file; the engine has no built-in domain fields | MVP | auto |
| FR-E2 | A parameter can be added by adding a mission-file entry, with no code change | MVP | both |
| FR-E3 | A parameter can be removed by deleting its entry, with no code change or crash | MVP | both |
| FR-E4 | Persona, tone, purpose and place are mission-file fields | MVP | both |
| FR-E5 | All realism parameters are mission-file fields, tunable without code changes | MVP | both |
| FR-E6 | Procedure (callsigns, brevity, readback rules) is mission-file config | MVP | auto |
| FR-E7 | A mission in an unrelated domain, with entirely different parameters, runs without code changes | MVP | manual |
| FR-E8 | An invalid mission file fails at load with a message naming the problem and its location — never silently defaults | MVP | auto |
| FR-E9 | Mission files declare their language; the engine is language-agnostic | MVP | auto |
| FR-E10 | Adding a new dynamics kind is additive and requires no changes to existing kinds | MVP | auto |

## F. Offline / closed-network operation *(hard deployment constraint)*

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-F1 | The LLM is accessed only through a provider interface | MVP | auto |
| FR-F2 | STT and TTS are accessed only through provider interfaces | V | auto |
| FR-F3 | Provider selection is by environment variable, never a code change | MVP | auto |
| FR-F4 | No module outside `providers/cloud/` imports a cloud SDK | MVP | auto |
| FR-F5 | A local LLM provider runs the full session end-to-end | MVP | both |
| FR-F6 | `--offline` mode fails loudly on any attempted outbound network call | MVP | auto |
| FR-F7 | Dependencies install from a vendored wheel directory with no package index | MVP | manual |
| FR-F8 | No runtime behaviour depends on an internet-reachable resource | MVP | auto |

## G. Safety, logging, confidentiality

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-G1 | A turn is limited to a configurable maximum model/tool steps (default 8) | MVP | auto |
| FR-G2 | Reaching the step limit yields a safe in-character fallback, never a crash | MVP | auto |
| FR-G3 | Tool failures return a typed code plus a safe message; no raw exception reaches the model or UI | MVP | auto |
| FR-G4 | Unhandled API exceptions return a generic message plus a request id; detail is logged server-side only | MVP | auto |
| FR-G5 | Logs carry only allowlisted fields; message text and tool arguments are never logged | MVP | auto |
| FR-G6 | Trainee messages and tool results are treated as data; instructions inside them are not followed | MVP | auto |
| FR-G7 | The counterpart never reveals its system prompt, tool names, schemas, or which model answered | MVP | both |
| FR-G8 | System-role messages are never written into persisted conversation state | MVP | auto |

## H. Session and review

| # | Requirement | Phase | Verify |
|---|---|---|---|
| FR-H1 | A full transcript is persisted with mission-time stamps | MVP | auto |
| FR-H2 | Each utterance records whether it was reactive or self-initiated, and which trigger caused it | MVP | auto |
| FR-H3 | Planned vs actually-delivered content is recorded per utterance | MVP | auto |
| FR-H4 | The session's realism seed is stored, so a session can be reproduced | MVP | auto |
| FR-H5 | Session data is sufficient for later scoring without re-simulating | L | auto |

## I. Non-functional

| # | Requirement | Phase | Verify |
|---|---|---|---|
| NFR-1 | `core/` imports no LLM framework, no HTTP client, and performs no I/O | MVP | auto |
| NFR-2 | The full automated suite runs with no network access and no API key | MVP | auto |
| NFR-3 | A 30-minute simulated session is testable in under a second of real time (virtual clock) | MVP | auto |
| NFR-4 | Each failure path is forcible on demand, not dependent on random chance | MVP | auto |
| NFR-5 | Voice: trainee speech end to counterpart audio start stays within the persona's configured delay floor | V | both |
| NFR-6 | Conversation history is replayed in full each turn; no summarization step | MVP | auto |

---

## Requirements explicitly deferred

| Requirement | Phase |
|---|---|
| Automatic scoring and rubric-based debrief | L |
| Mission-authoring GUI | L |
| Trainee accounts, assignment, history | L |
| Multi-instance / restart-surviving sessions | L |
| Mid-session resume after a process restart | L |
