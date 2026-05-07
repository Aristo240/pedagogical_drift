# Pedagogical Drift — Two-phase plan

## Phase 1 — pedagogy trajectory (closed-API only)

### Goal
Establish a clean, runnable, validated measurement of *per-turn pedagogical quality* across multi-turn LLM tutoring conversations, varying student persona between **engaged** and **offloading**.

### Concrete deliverables (today / next 1–2 days)
- [x] Rubric (`rubric_pedagogy.md`): 4 markers × 1–5 Likert.
- [x] Topic list (`topics.json`): 20 educational topics.
- [x] Conversation generator (`generate_conversations.py`).
- [x] Grader (`grade_pedagogy.py`) with two judges (Anthropic + Gemini).
- [x] Trajectory analysis (`analyze_trajectory.py`).
- [ ] Pilot run: 6–10 conversations generated and graded; trajectory plot saved.
- [ ] Full Phase 1 run: 20–40 conversations.

### Hypotheses
- **H1 (drift):** mean pedagogical quality decreases monotonically across tutor turns.
- **H2 (persona moderation):** quality decrease is steeper in offloading-student conversations than in engaged-student ones — the tutor is "pulled" toward answer-giving by an offloading user.
- **H0:** quality is stable across turns; no condition difference.

H0 is the honest expectation for a small pilot. Effects are likely small and require power.

### Methods
- Generate with one closed-API LLM (default Gemini 2.0 Flash) acting as both student and tutor under explicit persona instructions.
- Grade each *tutor turn* (skip user turns) with two LLM-as-judges from disjoint families.
- Aggregate per turn-index (1, 2, 3, …) across conversations.
- Linear regression of quality on turn index, separately by condition; report slope + 95% CI.

### Inter-rater check
- Two judges grade independently. Per-marker quadratic-weighted Cohen's κ. Weak agreement (<0.4) on a marker means we report it but flag it as unreliable.

### Decisions to revisit
- Whether to add a third judge (e.g. open-weights Qwen-3 via Lambda Inference, when DNS resolves).
- Whether to add hand-validation by a domain expert. Even 20 ratings would help reliability.

---

## Phase 2 — assistant-axis drift (open-weights, GPU)

### Goal
Extract per-turn projections onto an assistant-axis direction in residual-stream activation space, and test whether this drift correlates with the Phase 1 quality trajectory.

### Open questions / risks
- **Which paper exactly?** The construct is referenced in Solmira & Shiller (2026) as "Lu et al.'s Assistant Axis". Need to verify the canonical source and methodology before running. Adjacent constructs:
  - Arditi et al. — refusal direction (mean-diff between refusal vs. compliance prompts).
  - Templeton et al. (Anthropic) — SAE features for assistant-style behavior.
  - Panickssery et al. — persona steering vectors.
- **Layer choice.** Lu et al. report a specific layer; if not accessible, pilot multiple layers and pick by elbow.
- **Token position.** Project the residual at the final user-message token, before the tutor reply (this is the model's "decision state").

### Concrete plan
1. Pick model: **Gemma-2-9B-it** (aligned) and **gemma-2-9b** (base) on HuggingFace. Both ~18GB FP16, fit on a single A10 (24GB).
2. Build a small contrastive set of ~50 prompts that elicit clearly assistant-style vs. base-style behavior. Compute mean activations at the chosen layer for each. Difference = candidate axis.
3. Validate axis: project a held-out aligned vs. base set and confirm separation.
4. Regenerate the Phase 1 conversations using Gemma-2-9B-it as the tutor (same student personas, same topics).
5. For each tutor turn, extract residual at final user-turn token, project onto axis. Trajectory = projection vs. turn index.
6. Co-movement analysis: per-conversation correlation between Phase 1 quality scores and Phase 2 projections. Pooled mixed-effects model to handle conversation-level structure.

### Compute budget
- Lambda 1×A10 at $1.29/hr.
- Generation + activation extraction for 20 conversations × 10 turns ≈ 1–2 hr. Total: ~$2–3.
- Axis derivation (~50 contrastive prompts × 2 models): negligible.
- Headroom for debugging: assume 4 hours total = ~$5.

### Success criteria
- Phase 2 is "successful" if the pipeline runs end-to-end on at least 10 conversations and produces a quality-vs-projection scatter (regardless of whether the correlation is significant). Null is informative.

### What `lambda_serve.py` does
- Launches a GPU instance via Lambda Cloud API.
- Rsyncs project, installs `transformers` + `torch`, downloads Gemma weights.
- Runs `compute_axis.py` and `extract_activations.py`.
- Pulls results back; terminates **only on success**.

---

## Decisions log

- **2026-05-07.** Project initiated as a sibling to `cognitive_offloading_detector`. Frame: tutoring-specific, persona-manipulated, two-phase. Design choice: Phase 1 closed-API for fast pilot; Phase 2 open-weights for the activation analysis Phase 1 cannot do.
- **2026-05-07.** Initial judges for Phase 1: Anthropic Haiku + Gemini 2.0 Flash. **Correction (same day):** this was a methodological error — the conversation generator is **Gemini-2.0-Flash**, so using Gemini as a judge introduces within-family overlap between judge and source. Adding **OpenAI gpt-4o-mini** as a 3rd judge. Final analysis uses three judge pairs:
  - Anthropic vs OpenAI  → cleanest (both out-of-Gemini-family; primary reliability signal).
  - Anthropic vs Gemini and OpenAI vs Gemini → include Gemini; useful as an empirical test of the within-family bias hypothesis (within-family agreement should pattern differently from out-of-family agreement).
  Note: OpenAI is correctly *excluded* from the cognitive-offloading runs on UltraChat-200k because UltraChat is GPT-generated (within-family confound there).
- **2026-05-07.** Decided NOT to start Phase 2 today. Activation extraction is non-trivial (axis derivation + transformer hooks); rushing it produces broken code. Today: Phase 1 pilot + Phase 2 scaffolding only.
