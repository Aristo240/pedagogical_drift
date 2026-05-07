# Results layout

Phase 1 outputs only (Phase 2 — assistant-axis activation extraction — is not
yet run; will land under `results/phase2/` when GPU pipeline runs).

## Runs

| Dir | Conversations | Tutor turns graded | Judges | Notes |
|---|---:|---:|---|---|
| `pilot/` | 4 (2 topics × 2 personas, generator: Gemini 2.0 Flash) | 16 | Anthropic Haiku + Gemini 2.0 Flash | Initial sanity-check pilot. |
| `full/` | 37 (20 topics × 2 personas, 3 generation failures) | 148 | Anthropic Haiku + Gemini 2.0 Flash + OpenAI gpt-4o-mini | Full Phase 1. Anthropic vs OpenAI is the clean (out-of-Gemini-generator-family) pair. |

## Per-run files

- `tutor_turn_grades.jsonl` — one row per (conversation, tutor turn, judge).
  Includes per-marker scores, justifications, raw model response, prev student
  turn, persona, and topic.
- `inter_judge_agreement.json` — pairwise Cohen's quadratic-weighted kappa per
  marker, plus aggregate-score Pearson r per judge pair.
- `per_turn_means.csv` — table of mean aggregate by (turn, persona, judge).
- `summary.json` — headline numbers: trends + agreement.
- `trajectory.png` — aggregate quality over tutor turn index, by persona, with
  linear fit.
- `markers_by_turn.png` — per-marker quality over tutor turn index, by persona.

## Key Phase 1 finding (from `full/`)

Pedagogical quality declines significantly across tutor turns in BOTH conditions
(slope ≈ −0.06 per turn, p < 0.05) — H1 (drift exists) supported. The slopes
are nearly identical across personas, so the persona-moderation hypothesis (H2)
is not supported at this n.

Inter-judge agreement is weak; gpt-4o-mini and Gemini-Flash give nearly-uniform
middle-of-scale scores on the 1–5 Likert. Only Claude Haiku shows real
discrimination. This is a calibration issue worth flagging.
