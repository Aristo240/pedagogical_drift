# Pedagogical Drift in Extended LLM Tutoring

> **Status: ongoing pilot.** Phase 1 (LLM-as-judge pedagogy grading) is in progress. Phase 2 (activation-axis drift on open-weights model) is scaffolded but not yet run.

## The question

In multi-turn tutoring conversations, does the tutor LLM's **pedagogical quality** drift across turns? And does that quality trajectory relate to **representational drift** along an assistant-direction in activation space, as documented by Lu et al. and discussed in Solmira & Shiller (2026)?

This project frames the "extended-conversation drift" phenomenon through a **learning and cognitive outcomes** lens specifically, by:

1. Operationalizing pedagogical quality on a small explicit rubric (scaffolding, error responsiveness, productive struggle, conceptual depth).
2. Generating tutoring conversations under two student personas — **engaged** and **offloading** — to test whether the tutor's quality drift is moderated by user-side cognitive engagement.
3. (Phase 2) Extracting activations from an open-weights model along an assistant-axis direction and testing whether quality and representation drift co-vary.

## Why this exists

Most discussions of LLM "drift" in extended dialogue focus on alignment-style outcomes (corrigibility, sycophancy, philosophical preferences). For **learning and cognitive outcomes**, the relevant question is narrower and arguably more practical: does the model remain a *good tutor* across a long pedagogical interaction, or does it slide toward shallower, less scaffolded responses?

This project is a small attempt to make that question measurable.

## Phase 1 — pedagogy trajectory (in progress, runnable today)

No GPU required. Uses closed-API LLMs only.

1. **Topics** (`topics.json`) — 20 educational topics across STEM, humanities, and language learning, each suitable for ~10-turn dialogue.
2. **Conversation generation** (`generate_conversations.py`) — for each topic and student persona, generates a multi-turn dialogue using a closed-API LLM as both student and tutor (Gemini 2.0 Flash, cheap, deterministic). Saved to `data/conversations/`.
3. **Pedagogy rubric** (`rubric_pedagogy.md`) — four markers of pedagogical quality, 1–5 Likert each.
4. **Pedagogy grader** (`grade_pedagogy.py`) — LLM-as-judge that scores each *tutor turn* against the rubric. Two judges from disjoint families for inter-rater agreement.
5. **Trajectory analysis** (`analyze_trajectory.py`) — per-turn quality means, by-condition comparison (engaged vs. offloading), simple linear-trend test.

## Phase 2 — assistant-axis drift (planned, requires GPU)

Requires an open-weights model and activation hooks. Designed for **Gemma-2-9B-it** running on Lambda Labs (1×A10 or 1×A100, ~$1.30–2/hr).

1. **Conversation regeneration** with Gemma-2-9B-it as the tutor (so we can extract its activations).
2. **Assistant-axis derivation** (`compute_axis.py`, scaffolded) — contrastive activations from base (`gemma-2-9b`) vs. aligned (`gemma-2-9b-it`) on a probe set, mean-difference direction at a chosen residual-stream layer.
3. **Per-turn projection** (`extract_activations.py`, scaffolded) — for each tutor turn, project the residual at the final user-turn token onto the axis. Trajectory across turns = drift.
4. **Co-movement analysis** — does quality trajectory (Phase 1 grader) correlate with axis trajectory (Phase 2 projection)?

`lambda_serve.py` is a placeholder for a Lambda Cloud orchestrator that launches a GPU instance, runs the activation pipeline, pulls results, and terminates.

## Setup

```bash
cd pedagogical_drift
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env   # fill in keys
```

## Pilot pipeline (Phase 1)

```bash
# 1. Generate a small pilot batch (6 conversations, ~$0.10, ~3 min)
python generate_conversations.py --n-per-condition 3 --turns 8 --out data/conversations/pilot.jsonl

# 2. Grade pedagogical quality of each tutor turn (2 judges, ~$0.05)
python grade_pedagogy.py --in data/conversations/pilot.jsonl \
    --judges anthropic:claude-haiku-4-5 gemini:gemini-2.0-flash \
    --out-dir results/pilot/

# 3. Plot per-turn quality trajectory and condition contrast
python analyze_trajectory.py --in-dir results/pilot/ --outdir results/pilot/
```

## Project layout

```
pedagogical_drift/
├── README.md                 # this file
├── PLAN.md                   # detailed two-phase plan + decisions log
├── rubric_pedagogy.md        # 4-marker pedagogy rubric
├── topics.json               # 20 tutoring topics
├── generate_conversations.py # phase 1: generate engaged + offloading dialogues
├── grade_pedagogy.py         # phase 1: LLM-as-judge for tutor turns
├── analyze_trajectory.py     # phase 1: trajectory plots + simple stats
├── compute_axis.py           # phase 2 SCAFFOLD: assistant-axis derivation
├── extract_activations.py    # phase 2 SCAFFOLD: per-turn projections
├── lambda_serve.py           # phase 2 SCAFFOLD: GPU orchestrator
├── data/
│   └── conversations/        # generated dialogues (jsonl)
├── results/
│   └── pilot/                # phase 1 outputs (generated)
├── requirements.txt
├── .env.template
└── .gitignore
```

## Limitations (be honest)

- **Pilot scope.** Phase 1 results today are on a small generated dataset (n ≤ 20 conversations).
- **Generated, not real, tutoring data.** Conversations are LLM-simulated student-tutor dialogues, not transcripts of human learners.
- **Phase 2 not yet run.** Activation extraction requires GPU; the code path is scaffolded but not validated end-to-end.
- **Single rubric.** "Pedagogical quality" is operationalized through one 4-marker rubric; alternative operationalizations exist.
- **No causal claims.** Even if quality and axis drift co-vary, we cannot infer that representational drift *causes* pedagogical degradation. Disentangling this needs intervention studies (activation patching, steering).
- **No literature review.** Adjacent work in HCI, education research on LLM tutoring (e.g., Khanmigo evaluations), and metacognitive drift may already cover parts of this. Pointers welcome.

## Citing the upstream work

The Phase 2 design borrows the assistant-axis construct from:

- Lu et al. (assistant-axis direction in activation space).
- Solmira, V. & Shiller, D. (2026). *Persona Drift in Extended AI Interaction*. (Preprint / blog post, March 2026.)

This project does not claim novelty over those — it explores one specific application (pedagogical conversations) with a different measurement focus.
