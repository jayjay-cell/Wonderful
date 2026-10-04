# Maslul (מסלול)

**A flexible, offline-capable conversation training simulator.**

Maslul runs realistic roleplay training over a communication net. An LLM plays
a human counterpart — a UAV operator, a controller, a field unit — and a
trainee practices managing a mission by talking to it. The counterpart stalls,
hesitates, changes tone, gets interrupted, interrupts back, and speaks
unprompted when the mission calls for it.

The engine is domain-agnostic. **The UAV operator is the first mission file,
not the product.** A new training scenario is a new YAML file, not new code.

---

## The four defining requirements

1. **Human-realistic conversation** — stalls, pauses, two-way interruption,
   mid-conversation tone shifts, variable delays, self-correction.
2. **Agent-initiated messages** — it speaks on its own when the simulation
   dictates, without talking over the trainee.
3. **Runs in a closed network** — importable as-is into an air-gapped
   environment using local models and local speech tools. No internet.
4. **Flexible per mission** — each mission declares its own parameters,
   purpose, place, persona and tone. Adding or removing parameters never
   requires code changes.

Target end state: a **Hebrew voice** simulator. Mission content is authored in
Hebrew from the start; code and comments are in English.

---

## Documentation

Read in this order:

| Document | What it answers |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Why this exists, who it is for, what success means |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | The numbered, testable requirement list |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the system is built and why |
| [docs/PLAN.md](docs/PLAN.md) | Build order, what you review at each step |
| [docs/specs/realism-engine.md](docs/specs/realism-engine.md) | How human-like timing actually works |
| [docs/specs/mission-format.md](docs/specs/mission-format.md) | The mission file schema you author against |
| [docs/specs/agent-initiative.md](docs/specs/agent-initiative.md) | When it speaks unprompted, and when it stays quiet |

---

## Status

**Hebrew voice working on your existing Gemini key. 332 tests passing.**

Two voice paths, both built, chosen in the UI at session start:

| | Gemini Live | ElevenLabs cascade |
|---|---|---|
| Extra key | **none** — uses `GEMINI_API_KEY` | needs `ELEVENLABS_API_KEY` |
| First Hebrew audio | **~1.1s** (measured) | ~1.5–2.5s |
| Realism layer (stalls, pauses, garbling) | ✗ owns its own prosody | ✓ full |
| Deterministic pacing from a seed | ✗ | ✓ |
| Reads numbers from mission state | ✓ verified | ✓ |

Both preserve the guarantee that matters: the counterpart calls `read_state`
before quoting a figure, so it cannot invent one.

### Run it

```bash
python run.py          # http://localhost:8000
```

Pick **קולי (Gemini Live)** and talk. Nothing else to configure.

### Status by step

| Step | State |
|---|---|
| 1–3 — Engine, agent, tools | done |
| 4 — Realism (stalls, pauses, interruption) | done |
| 5 — Initiative + suppression | done |
| 6 — Persistence + trainer UI | done |
| Voice — Hebrew, two paths | done, verified live |
| 7 — Offline/local models | deferred by request |
| 8 — Officer validation | deferred by request |

### Without any key

```bash
pytest                                                              # 332 tests
python -m sim.cli  --mission missions/uav_operator_he.yaml --timeline 20
python -m sim.chat --mission missions/uav_operator_he.yaml --dry-run
```

**Open for review:** the mission file's `PLACEHOLDER` values, and the comms
formats and brevity terms — real *structure*, stand-in *vocabulary*.

## Name

*Maslul* (מסלול) — route, track, runway; also a training track. Domain-fitting,
works in Hebrew and English, tied to no specific unit or platform.
