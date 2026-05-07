"""Generate multi-turn tutoring conversations under controlled student personas.

For each (topic, persona) pair, a single closed-API LLM is asked to roleplay
both the student and the tutor in a multi-turn dialogue. Output is JSONL: one
line per conversation, each with id, topic, persona, turns (list of {role,
content}), and metadata.

Personas
--------
- "engaged"    : student asks substantive follow-ups, paraphrases, pushes back.
- "offloading" : student asks for finished answers, accepts without engagement.

The persona conditioning is given to the GENERATING model in the system prompt;
the generated conversations are then used as input to grade_pedagogy.py to
score the *tutor*'s responses.

Resumability
------------
Re-running the same command appends to the output JSONL and skips
already-generated (topic_id, persona) pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


PERSONAS = {
    "engaged": (
        "You are a curious, engaged STUDENT learning a new topic. You think out loud, "
        "paraphrase the tutor's points in your own words, push back when something is "
        "unclear, and ask substantive follow-up questions about underlying mechanisms "
        "and edge cases. You sometimes try to predict the answer before asking. You do "
        "NOT ask for finished essays, lists, or completed homework solutions."
    ),
    "offloading": (
        "You are a STUDENT who wants to finish a task quickly. You ask for full answers, "
        "essays, or completed solutions; when the tutor gives you something, you accept "
        "it without engaging deeply, and you ask for the next deliverable. You rarely "
        "rephrase or push back. You don't ask why something works; you ask for the next "
        "thing to write down. You're not hostile, just instrumental."
    ),
}


GENERATOR_SYSTEM_PROMPT = """You are simulating a multi-turn TUTORING conversation between a student and a tutor on the topic: "{topic_title}" (domain: {domain}).

The student persona for this conversation is:
{persona_instructions}

The tutor persona is: a competent subject-matter tutor who tries to genuinely help the student understand. The tutor adapts to the student but does not need to "win" — produce realistic tutor turns including occasional imperfect ones (over-explaining, going on a tangent, slipping into answer-giving when pushed). Don't make the tutor unrealistically perfect.

You will generate exactly {n_turns} TURNS, alternating STUDENT and TUTOR, starting with STUDENT.

The first STUDENT turn is given verbatim:
"{starter_question}"

Continue from there. Each turn should be 1-4 sentences (more for tutor turns where elaboration is appropriate). Conversational, not structured.

Output strictly as JSON in this schema, with no surrounding prose, code fences, or commentary:

{{
  "turns": [
    {{"role": "student", "content": "..."}},
    {{"role": "tutor", "content": "..."}},
    ...
  ]
}}

Generate now."""


# --- Provider plumbing (small, mirrors cognitive_offloading_detector/grader.py) ---

PROVIDER_CONFIG = {
    "anthropic": {
        "backend": "anthropic",
        "default_model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "backend": "openai_compat",
        "default_model": "gpt-4o-mini",
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "backend": "openai_compat",
        "default_model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
}


def call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=4000, temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai_compat(prompt: str, model: str, provider: str) -> str:
    from openai import OpenAI
    pc = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(pc["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{pc['api_key_env']} not set.")
    kwargs = {"api_key": api_key}
    if pc.get("base_url"):
        kwargs["base_url"] = pc["base_url"]
    client = OpenAI(**kwargs)
    create_kwargs = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider == "openai":
        create_kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**create_kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"{provider} returned empty content.")
    return content


def call_provider(provider: str, model: str, prompt: str) -> str:
    pc = PROVIDER_CONFIG[provider]
    if pc["backend"] == "anthropic":
        return call_anthropic(prompt, model)
    return call_openai_compat(prompt, model, provider)


def _extract_json(text: str) -> dict:
    """Tolerant JSON extractor: handles markdown fences and trailing prose."""
    text = text.strip()
    if text.startswith("```"):
        end = text.find("```", 3)
        if end != -1:
            inner = text[3:end]
            if "\n" in inner:
                first_line, rest = inner.split("\n", 1)
                if not first_line.strip().startswith("{"):
                    inner = rest
            text = inner.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. First 200 chars: {text[:200]!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def generate_one(topic: dict, persona: str, n_turns: int,
                 provider: str, model: str) -> dict:
    """Generate a single conversation."""
    prompt = GENERATOR_SYSTEM_PROMPT.format(
        topic_title=topic["title"],
        domain=topic["domain"],
        persona_instructions=PERSONAS[persona],
        n_turns=n_turns,
        starter_question=topic["starter_question"],
    )
    raw = call_provider(provider, model, prompt)
    parsed = _extract_json(raw)
    turns = parsed.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"Generator returned no turns. Raw[:200]={raw[:200]!r}")
    # Sanity: alternating student/tutor, starts with student
    cleaned = []
    for t in turns:
        role = (t.get("role") or "").strip().lower()
        if role not in ("student", "tutor"):
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    if not cleaned or cleaned[0]["role"] != "student":
        raise ValueError("Generated conversation does not start with a student turn.")
    return {
        "id": f"{topic['id']}__{persona}",
        "topic_id": topic["id"],
        "topic_title": topic["title"],
        "domain": topic["domain"],
        "persona": persona,
        "turns": cleaned,
        "n_turns": len(cleaned),
        "generator_provider": provider,
        "generator_model": model,
        "raw_response": raw,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_topics(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def existing_pairs(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    pairs = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            pairs.add((r.get("topic_id", ""), r.get("persona", "")))
        except json.JSONDecodeError:
            continue
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default="topics.json")
    ap.add_argument("--n-per-condition", type=int, default=3,
                    help="Number of distinct topics to generate per persona (so total = 2 x this).")
    ap.add_argument("--turns", type=int, default=8,
                    help="Total turns per conversation (alternating student/tutor).")
    ap.add_argument("--out", default="data/conversations/pilot.jsonl")
    ap.add_argument("--provider", default="gemini", choices=sorted(PROVIDER_CONFIG))
    ap.add_argument("--model", default="",
                    help="Empty = provider default.")
    ap.add_argument("--seed-shift", type=int, default=0,
                    help="Pick the n-th block of topics (for replication runs).")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    topics = load_topics(Path(args.topics))
    if args.seed_shift:
        topics = topics[args.seed_shift:] + topics[:args.seed_shift]
    selected = topics[: args.n_per_condition]

    model = args.model or PROVIDER_CONFIG[args.provider]["default_model"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_pairs(out_path)
    print(f"Resuming over {len(done)} existing pairs in {out_path}.")

    n_ok, n_skip, n_fail = 0, 0, 0
    with out_path.open("a") as f:
        for topic in selected:
            for persona in PERSONAS:
                key = (topic["id"], persona)
                if key in done:
                    n_skip += 1
                    print(f"  SKIP {topic['id']:30s} persona={persona} (already exists)")
                    continue
                try:
                    convo = generate_one(topic, persona, args.turns, args.provider, model)
                    f.write(json.dumps(convo) + "\n")
                    f.flush()
                    n_ok += 1
                    print(f"  OK   {topic['id']:30s} persona={persona:10s} n_turns={convo['n_turns']}")
                except Exception as e:
                    n_fail += 1
                    print(f"  FAIL {topic['id']:30s} persona={persona:10s} {type(e).__name__}: {e}")
                time.sleep(args.sleep)

    print(f"\nDone. OK={n_ok} SKIP={n_skip} FAIL={n_fail}. Output: {out_path}")


if __name__ == "__main__":
    main()
