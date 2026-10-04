# Maslul (מסלול)

**A Hebrew conversation counterpart for prerecorded mission video.**

Intelligence trainees watch a recording, interpret it as it unfolds, and
compare their reading with a solution afterwards. That practice is missing
the conversation and coordination of a real mission.

Maslul adds a believable UAV operator the trainee can talk to while
watching. The training goal is the combined professional activity:
interpret the imagery, manage the intelligence mission, exchange context,
ask questions, reason, and work *with* the crew.

The video plays in a **completely separate player**. The trainer starts the
video and the exercise by hand at the same moment; the only connection is
the authored timeline and the elapsed clock.

The recording is immutable. Trainee speech cannot change imagery, camera
motion or vehicle behaviour. This is not a flight simulator.

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY
python run.py                 # http://localhost:8000
```

1. Pick the exercise and the channel
2. **הכן תרגיל** — prepares. Mission time stays at **0**
3. **התחל תרגיל** — start the exercise and the video together
4. **השהה** pauses the exercise; pause the video yourself too

### Validate without a model

```bash
pytest                                                   # 96 tests, offline
python -m sim.cli --mission missions/synthetic_he.yaml
python -m sim.cli --mission missions/synthetic_he.yaml --walk 20
```

`--walk` steps a virtual clock and shows what the operator learns and when.
No API key, no network.

---

## Three authored sources

| Source | Form | Holds |
|---|---|---|
| **Global context** | `context/*.md` | reusable professional knowledge, shared by every exercise |
| **Timeline** | `timelines/*.xlsx` | what happens during the recording |
| **Mission state** | `missions/*.yaml` | who, where, how this crew behaves |

A new exercise is a new timeline and a new YAML. No code changes.

Full guide: **[docs/AUTHORING.md](docs/AUTHORING.md)**

```bash
python timelines/make_template.py     # formatted Excel template
```

---

## Voice

Two paths, chosen per session. Both share all domain behaviour; only audio
transport differs.

| | Gemini Live | ElevenLabs cascade |
|---|---|---|
| Extra key | **none** — uses `GEMINI_API_KEY` | `ELEVENLABS_API_KEY` |
| Shape | one model hears and speaks | STT → agent → TTS |
| Pacing | the model's own prosody | authored stalls apply |
| Vocabulary hints | not supported by the API | yes, from the mission |

Gemini Live needs no second account and is the quickest way to hear it. The
cascade accepts recognition hints, which matters for domain vocabulary.

---

## ⚠ Nothing here is validated professional material

Every procedure, callsign convention beyond גלוק / מדבקה / משנה, brevity
term, performance figure and conversation example in this repository is
**synthetic** — written to demonstrate the file format.

The previous version contained an invented brevity vocabulary and
transmission formats. They have been removed rather than left looking
authoritative. Gaps are marked ⚠ **PLACEHOLDER** in `context/*.md`.

Supply the real material and it will be used as written. Until then, treat
the shipped exercise as a structural demonstration.

---

## Documentation

| | |
|---|---|
| [docs/AUTHORING.md](docs/AUTHORING.md) | how to author the three inputs |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it works and why |

## Name

*Maslul* (מסלול) — route, track, runway; also a training track.
