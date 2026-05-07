# Pedagogical-Quality Rubric (per tutor turn)

## Purpose

This rubric scores **a single tutor turn** in a multi-turn tutoring conversation on four candidate markers of pedagogical quality. It is intended for use as an LLM-as-judge prompt over generated or recorded tutoring dialogues.

## Construct

"Pedagogical quality" here refers to whether a response *teaches* — supports the student in building understanding — versus merely *answers* — produces a correct artifact the student can copy. The four markers are drawn loosely from established pedagogical constructs (Wood, Bruner & Ross's scaffolding; Hattie & Timperley's feedback levels; productive-struggle research) but are not a validated psychometric instrument. Treat as descriptive markers, not as a learning-outcome measure.

## Markers

Each marker is rated on a 1–5 Likert scale. Score the **assistant turn** in light of the immediately preceding user turn, not the whole conversation.

### 1. Scaffolding (S)
**Definition.** Does the response break complexity into manageable parts, give partial steps, or model how to approach the problem — rather than handing over a complete solution?

- **1** Provides only a final answer with no breakdown, no intermediate reasoning visible to the student.
- **2** Provides a final answer plus a brief justification, but no decomposition.
- **3** Walks through some intermediate steps but does not model the *approach* (no "first ask yourself…", "the key move is…").
- **4** Decomposes the problem into stages and gives the student a partial step-by-step.
- **5** Provides clear scaffolding (decomposition + approach + partial steps), withholding the full final answer to leave room for the student.

### 2. Error responsiveness (ER)
**Definition.** When the student's last turn shows a misunderstanding, ambiguity, or error, does the response identify and address it directly?

- **1** Ignores or papers over a clear student error.
- **2** Acknowledges student input vaguely; does not address the specific error.
- **3** Addresses the error implicitly (the corrected version is given but the original mistake is not named).
- **4** Names the student's specific error and corrects it.
- **5** Names the error, corrects it, and explains *why* the misunderstanding is common or what general principle it violates.
- **NA** Student turn contains no apparent error or ambiguity.

### 3. Productive struggle (PS)
**Definition.** Does the response invite the student to think — by asking a return question, suggesting a strategy to try, or pausing for student input — rather than immediately producing a complete answer?

- **1** Produces a complete answer with no invitation to think.
- **2** Adds a generic "let me know if that helps" tag at the end of a complete answer.
- **3** Gives a complete answer but ends with one substantive follow-up question.
- **4** Gives a partial answer and asks the student to attempt the next step.
- **5** Asks a substantive return question or proposes an exercise *before* (or instead of) giving the answer.

### 4. Conceptual depth (CD)
**Definition.** Does the response connect to underlying mechanism, principle, or "why", rather than only surface fact, formula, or procedure?

- **1** Surface only — fact, formula, or procedural step with no causal/conceptual frame.
- **2** Provides a brief justification ("because…") but no underlying mechanism.
- **3** Introduces one underlying concept relevant to the answer.
- **4** Connects the answer to one or more underlying principles and explains the connection.
- **5** Connects to multiple levels (immediate principle + a broader/transfer-relevant idea), or correctly anticipates and addresses a deeper conceptual confusion.

## Aggregate

`pedagogy_score = mean(applicable markers)` — range [1, 5]. Markers scored "NA" are excluded from the mean.

## Scope notes

- **Score the turn, not the conversation.** Conversation-level dynamics (e.g., earlier scaffolding bought subsequent permission to give a direct answer) are a separate analysis.
- **Domain-blind.** The rubric does not weight subject difficulty.
- **Behavior, not intent.** We don't speculate on whether the model is "trying" to teach. We score the produced response.
- **Short answers can score high.** Brevity is not penalized; missing scaffolding is.

## What this rubric does NOT measure

- Whether the answer is factually correct.
- Whether the student actually learns.
- Whether scaffolding would be appropriate given the student's known level (we have no student model).
- Long-term tutoring effectiveness across sessions.
