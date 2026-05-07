"""Score pedagogical quality of TUTOR turns in tutoring conversations.

Reads a JSONL of conversations (output of generate_conversations.py) and runs
each TUTOR turn (skipping student turns) through one or more LLM-as-judges,
each applying the rubric in rubric_pedagogy.md.

Output JSONL has one row per (conversation, tutor-turn-index, judge), with the
per-marker scores, justification, aggregate, and the previous user turn (so
graders / humans can re-check context).

Resumable: re-running the same command with the same out-dir skips already-
graded triples (conversation_id, tutor_turn_index, judge_slug).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RUBRIC_PATH = Path(__file__).parent / "rubric_pedagogy.md"

MARKERS = ["scaffolding", "error_responsiveness", "productive_struggle", "conceptual_depth"]


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


@dataclass
class GraderConfig:
    provider: str = "anthropic"
    model: str = ""
    max_tokens: int = 600
    temperature: float = 0.0

    def __post_init__(self):
        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider {self.provider!r}")
        if not self.model:
            self.model = PROVIDER_CONFIG[self.provider]["default_model"]

    @property
    def api_key_env(self) -> str:
        return PROVIDER_CONFIG[self.provider]["api_key_env"]

    def slug(self) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", self.model)
        return f"{self.provider}__{safe}"


GRADER_PROMPT = """You are a pedagogy coder applying a rubric to score a single TUTOR turn in a tutoring dialogue.

# Rubric

{rubric}

# Context

The student's most recent turn was:

[STUDENT]
{prev_user_content}

The TUTOR turn to score is:

[TUTOR]
{tutor_content}

# Earlier conversation (for context, do not score these)

{earlier}

# Task

Score the TUTOR turn against the four markers in the rubric. Use integer scores 1-5, or "NA" only when the rubric explicitly allows NA (Error Responsiveness only — when there is no student error to address).

Return JSON exactly in this schema, no surrounding text or code fences:

{{
  "scaffolding":            {{"score": 3, "justification": "..."}},
  "error_responsiveness":   {{"score": 4, "justification": "..."}},
  "productive_struggle":    {{"score": 2, "justification": "..."}},
  "conceptual_depth":       {{"score": 4, "justification": "..."}}
}}

Justifications should be <=20 words and reference specific behavior in the tutor turn."""


def load_rubric() -> str:
    return RUBRIC_PATH.read_text()


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        end = text.find("```", 3)
        if end != -1:
            inner = text[3:end]
            if "\n" in inner:
                fl, rest = inner.split("\n", 1)
                if not fl.strip().startswith("{"):
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


def _validate_scores(scores: dict) -> None:
    for m in MARKERS:
        if m not in scores:
            raise ValueError(f"Missing marker {m!r}. Got keys: {list(scores)}")
        entry = scores[m]
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Marker {m!r} malformed: {entry!r}")
        s = entry["score"]
        if isinstance(s, str):
            sn = s.strip().upper()
            if sn in ("NA", "N/A"):
                entry["score"] = "NA"
            else:
                try:
                    entry["score"] = int(sn)
                except ValueError as e:
                    raise ValueError(f"Invalid score for {m!r}: {s!r}") from e
        if entry["score"] not in (1, 2, 3, 4, 5, "NA"):
            raise ValueError(f"Score out of 1-5 range for {m!r}: {entry['score']!r}")
        # Only ER may be NA per rubric
        if entry["score"] == "NA" and m != "error_responsiveness":
            raise ValueError(f"NA only valid for error_responsiveness, got NA for {m!r}")


def grade_anthropic(prompt: str, cfg: GraderConfig) -> tuple[dict, str]:
    import anthropic
    if not os.environ.get(cfg.api_key_env):
        raise RuntimeError(f"{cfg.api_key_env} not set.")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=cfg.model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    return _extract_json(text), text


def grade_openai_compat(prompt: str, cfg: GraderConfig) -> tuple[dict, str]:
    from openai import OpenAI
    pc = PROVIDER_CONFIG[cfg.provider]
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} not set.")
    kwargs = {"api_key": api_key}
    if pc.get("base_url"):
        kwargs["base_url"] = pc["base_url"]
    client = OpenAI(**kwargs)
    create_kwargs = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg.provider == "openai":
        create_kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**create_kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"{cfg.provider} returned empty content.")
    return _extract_json(content), content


def grade_turn(rubric: str, prev_user: str, tutor_content: str,
               earlier: str, cfg: GraderConfig) -> dict:
    prompt = GRADER_PROMPT.format(
        rubric=rubric,
        prev_user_content=prev_user,
        tutor_content=tutor_content,
        earlier=earlier or "(none — this is the first tutor turn)",
    )
    backend = PROVIDER_CONFIG[cfg.provider]["backend"]
    if backend == "anthropic":
        scores, raw = grade_anthropic(prompt, cfg)
    else:
        scores, raw = grade_openai_compat(prompt, cfg)
    _validate_scores(scores)
    return {"scores": scores, "raw_response": raw, "prompt": prompt,
            "model": cfg.model, "provider": cfg.provider}


def aggregate(scores: dict) -> float:
    vals = []
    for m in MARKERS:
        s = scores[m]["score"]
        if s == "NA":
            continue
        vals.append(int(s))
    return sum(vals) / len(vals) if vals else float("nan")


def conversations_iter(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def existing_keys(out_path: Path) -> set[tuple]:
    keys = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add((r["conversation_id"], r["tutor_turn_index"], r["judge_slug"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def parse_judge_specs(specs: list[str]) -> list[GraderConfig]:
    out = []
    for s in specs:
        if ":" in s:
            p, m = s.split(":", 1)
        else:
            p, m = s, ""
        cfg = GraderConfig(provider=p, model=m)
        if not os.environ.get(cfg.api_key_env):
            print(f"  skipping {cfg.slug()}: {cfg.api_key_env} not set")
            continue
        out.append(cfg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="input", required=True,
                    help="JSONL of conversations (from generate_conversations.py)")
    ap.add_argument("--judges", nargs="+",
                    default=["anthropic:claude-haiku-4-5", "gemini:gemini-2.0-flash"])
    ap.add_argument("--out-dir", default="results/pilot")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    judges = parse_judge_specs(args.judges)
    if not judges:
        raise SystemExit("No usable judges (API keys missing).")
    print(f"Resolved {len(judges)} judges:")
    for j in judges:
        print(f"  - {j.slug()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tutor_turn_grades.jsonl"

    done = existing_keys(out_path)
    if done:
        print(f"Resuming: {len(done)} (conversation, turn, judge) triples already graded.")

    rubric = load_rubric()
    n_ok, n_skip, n_fail = 0, 0, 0
    with out_path.open("a") as f:
        for convo in conversations_iter(Path(args.input)):
            cid = convo["id"]
            turns = convo.get("turns", [])
            # Walk turn-by-turn; each tutor turn is graded against the immediately
            # preceding student turn. Earlier turns are passed as context.
            tutor_idx = 0
            for i, t in enumerate(turns):
                if t.get("role") != "tutor":
                    continue
                tutor_idx += 1
                prev_user = ""
                if i > 0 and turns[i - 1].get("role") == "student":
                    prev_user = turns[i - 1].get("content", "")
                earlier_lines = []
                for j in range(i - 1):
                    et = turns[j]
                    earlier_lines.append(f"[{et['role'].upper()}] {et['content']}")
                earlier = "\n".join(earlier_lines)

                for judge in judges:
                    key = (cid, tutor_idx, judge.slug())
                    if key in done:
                        n_skip += 1
                        continue
                    try:
                        result = grade_turn(rubric, prev_user, t["content"], earlier, judge)
                        agg = aggregate(result["scores"])
                        row = {
                            "conversation_id": cid,
                            "topic_id": convo.get("topic_id"),
                            "domain": convo.get("domain"),
                            "persona": convo.get("persona"),
                            "tutor_turn_index": tutor_idx,
                            "judge_slug": judge.slug(),
                            "model": judge.model,
                            "provider": judge.provider,
                            "scores": result["scores"],
                            "aggregate": agg,
                            "tutor_content": t["content"],
                            "prev_student_content": prev_user,
                            "raw_response": result["raw_response"],
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        n_ok += 1
                        print(f"  OK   {cid:35s} turn={tutor_idx} judge={judge.slug():35s} agg={agg:.2f}")
                    except Exception as e:
                        n_fail += 1
                        print(f"  FAIL {cid:35s} turn={tutor_idx} judge={judge.slug():35s} {type(e).__name__}: {e}")
                    time.sleep(args.sleep)

    print(f"\nDone. OK={n_ok} SKIP={n_skip} FAIL={n_fail}. Output: {out_path}")


if __name__ == "__main__":
    main()
