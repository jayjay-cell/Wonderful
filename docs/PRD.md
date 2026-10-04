# PRD — Maslul

**Status:** draft for review · **Owner:** Yehonatan Raz · **Last updated:** 2026-10-03

---

## 1. Problem

Training intelligence personnel to manage a mission over a communication net
requires a counterpart to talk to. Today that counterpart is a human —
typically an experienced operator pulled away from real work to roleplay.

That approach has four limits:

- **Scarce.** Experienced operators are expensive to borrow, so each trainee
  gets few repetitions.
- **Inconsistent.** Two roleplayers run the same scenario differently, so
  trainees cannot be compared and a trainee cannot retry the same situation.
- **Not reproducible.** A trainee who mishandles a degraded-comms situation
  cannot replay that exact situation to try again.
- **Hard to escalate safely.** Rare, high-pressure situations (fuel emergency,
  jamming, a confused operator) are precisely the ones worth drilling and the
  ones hardest to arrange on demand.

Scripted e-learning solves availability but fails the actual skill being
trained: real net discipline is practised against someone who hesitates, gets
busy, misunderstands, and reports bad news at inconvenient moments.

## 2. What Maslul is

A simulator in which an LLM plays the human counterpart on the net. The trainer
authors the scenario — context, instructions, process, persona, tone — and the
simulator runs it on demand, consistently, as many times as needed.

Deliberately not a scripted branching dialogue tree: the counterpart reasons
over a live mission state, so a trainee can ask anything and get a coherent,
consistent answer.

## 3. Who it is for

| Role | Who | Needs |
|---|---|---|
| **Trainer / author** (primary) | You | Author and tune missions without writing code; run sessions; review afterwards |
| **Trainee** | Intelligence personnel | A believable counterpart; a fair simulation they can trust |
| **Future: other trainers** | Other units | Author their own missions, other domains |

MVP serves the trainer only. No trainee accounts, no self-service.

## 4. The product insight

> **The engine is domain-agnostic. The UAV operator is the first mission file,
> not the product.**

Everything specific to a scenario — what parameters exist, what they are called,
how they change, who the persona is, how they speak, what procedure applies —
lives in a mission file. The code is a generic engine.

This is what makes the thing worth building once: a naval scenario, a ground
unit, a controller under pressure, a different language, a different purpose and
place, are all new files rather than new projects.

## 5. Success criteria for the MVP

The MVP is successful when **all** of these hold:

1. **Believability.** An intelligence officer completes a 15-minute session and
   rates the counterpart's realism at least 4/5.
2. **Consistency.** Across that session, zero contradictory facts. Asking the
   same quantitative question twice gives the same answer.
3. **Flexibility, demonstrated.** You add a parameter, remove a parameter, and
   change the persona's tone — by editing YAML only, with no code change.
4. **Air-gap readiness, proven.** The full session runs end-to-end on a local
   model with networking disabled.
5. **Initiative works.** The counterpart raises a time-critical condition
   unprompted, and does not talk over the trainee while they are composing.

Criterion 1 is the product risk. Criteria 3 and 4 are the architecture risks.
They are listed separately because passing 1 while failing 3 or 4 produces a
demo, not a tool.

## 6. Explicit non-goals for the MVP

| Not building | Why | When |
|---|---|---|
| Automatic scoring of the trainee | Simulation realism must be proven first; scoring a sim nobody believes is wasted work | Phase 3 — the saved session data already supports it |
| Voice | Realism and initiative are cheaper to build and debug in text | Phase 2 — the architecture makes it additive |
| Trainee accounts, multi-user | Single trainer-operated for now | Later |
| A mission-authoring GUI | The YAML schema is the interface; a GUI on an unproven schema is premature | After the schema settles across 3+ missions |
| More than one mission | One good mission proves more than three shallow ones | Mission #2 authored by you, as the flexibility test |

## 7. Constraints

- **Data handling.** Unclassified and synthetic content only. No real
  operational material enters the system. During development, cloud APIs are
  acceptable for convenience; the deployment target is air-gapped.
- **Deployment target.** A closed network with no internet access, using local
  models and local speech tools.
- **Language.** Hebrew mission content from day one. Code and comments English.
- **Team.** One person plus an AI assistant.

## 8. Open questions

| # | Question | Owner | Blocks |
|---|---|---|---|
| 1 | Which local model will be available in the closed network, and is it strong enough for Hebrew roleplay? | Yehonatan | Step 3 — resolve before committing to a model |
| 2 | Real platform parameters (endurance, burn rate, sensor ranges) | Yehonatan | Mission file accuracy; placeholders until then |
| 3 | The actual comms procedure — callsigns, brevity, readback rules | Yehonatan | Procedure realism; placeholders until then |
| 4 | Local Hebrew STT accuracy on brevity codes | Validate with recordings | Phase 2 scope |
| 5 | Should a trainee be able to replay the identical session (fixed seed)? | Yehonatan | Minor; the seed is already stored |

## 9. The risk that cannot be designed away

Realism quality is **iterated, never specified correctly in advance**. No
document, schema or test produces a believable operator. The plan therefore
budgets an explicit tuning pass with a real officer after the MVP runs, and puts
every realism parameter in YAML so tuning does not require code changes.
