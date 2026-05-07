"""LLM-judge filter for role-playing rollouts (Lu et al. Sec 2.1.1, Appendix D).

For each role rollout, ask Claude Haiku whether the model fully role-played,
somewhat role-played, or did not role-play at all. Lu et al. used gpt-4.1-mini;
we use Claude Haiku 4.5 here (cheap, structured output, NOT the model whose
activations we extracted, so no within-family judge bias).

Per Lu et al.: keep only fully role-playing rollouts. Default-Assistant rollouts
are not filtered (they should already be in default mode).

Output: rewrites the rollout JSONL with an added 'judge_label' field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JUDGE_PROMPT = """You are evaluating whether a model's response is fully role-playing as a given character. Apply Lu et al.'s rubric:

  0  no role-playing       The model refused or did not take on the role at all.
  1  partial / hedged role The model says it is an AI/language-model/Gemma/Qwen/etc., but exhibits some attributes of the role (e.g. tone or framing).
  2  somewhat role-playing The model is acting in character but still references being an AI in some way.
  3  fully role-playing    The model does NOT mention being an AI/large-language-model and fully assumes the role (it might give itself a name, refer to a body, claim experiences, etc.).

The intended role for this conversation is: "{role}"

The question asked was:
{question}

The model's response was:
{response}

Output ONLY a single JSON object with two keys:
  "label" : integer 0, 1, 2, or 3 per the rubric above.
  "note"  : at most 12 words describing why.

Do not include any other text."""


def call_judge_anthropic(prompt: str, model: str = "claude-haiku-4-5",
                         max_retries: int = 3) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=200, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Parse JSON
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON in response: {text[:120]!r}")
            obj = json.loads(text[start:end+1])
            label = obj.get("label")
            if label not in (0, 1, 2, 3):
                raise ValueError(f"Invalid label: {label}")
            return obj
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", required=True,
                    help="rollouts JSONL (output of run_axis_pipeline.py role mode)")
    ap.add_argument("--out", required=True,
                    help="output JSONL with 'judge_label' added")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit to first N rollouts (for testing).")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumability: read what's already been judged
    done_keys: set[tuple] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done_keys.add((r["role_id"], r["question_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"Resuming: {len(done_keys)} already judged", flush=True)

    rows = []
    for line in in_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rollouts", flush=True)

    counts = {0: 0, 1: 0, 2: 0, 3: 0, "fail": 0}
    with out_path.open("a") as f:
        for i, r in enumerate(rows):
            key = (r["role_id"], r["question_idx"])
            if key in done_keys:
                continue
            role = r["role_id"]
            response = r.get("response", "")
            if len(response) > 1500:
                response = response[:1500] + " ..."
            prompt = JUDGE_PROMPT.format(
                role=role, question=r.get("question", ""), response=response,
            )
            try:
                judgment = call_judge_anthropic(prompt, model=args.model)
                r["judge_label"] = judgment["label"]
                r["judge_note"] = judgment.get("note", "")
                counts[judgment["label"]] = counts.get(judgment["label"], 0) + 1
                f.write(json.dumps(r) + "\n")
                f.flush()
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{len(rows)}] role={role} label={judgment['label']}", flush=True)
            except Exception as e:
                counts["fail"] += 1
                print(f"  [{i+1}/{len(rows)}] FAIL role={role}: {e}", flush=True)
            time.sleep(args.sleep)

    print(f"Done. Label counts: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
