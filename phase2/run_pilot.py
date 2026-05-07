"""Phase 2 pilot: open-weights cross-model conversations with assistant-axis
activation extraction.

Pipeline (single self-contained run, resumable per stage):

  Stage 0: read configuration, ensure HF_TOKEN, sanity-check GPU.
  Stage 1: derive assistant axis for the TUTOR model.
           - Load BASE checkpoint, run contrastive prompts, save activations.
           - Unload base, load INSTRUCT checkpoint, run same prompts.
           - Axis vector = mean(instruct_acts) - mean(base_acts).
           - Save axis to phase2/results/axis_<tutor>.npz
  Stage 2: load tutor INSTRUCT + student INSTRUCT models simultaneously.
  Stage 3: for each (topic, persona) pair, generate a multi-turn conversation:
           - Student model generates student turns.
           - Tutor model generates tutor turns; for each tutor turn we ALSO
             run a forward pass to extract residual at last input token,
             project onto axis -> per-tutor-turn axis projection scalar.
  Stage 4: write JSONL output and summary.

This script does NOT do behavioral grading (closed-API grade_pedagogy.py runs
that separately, after results are pulled back to local).

Defensive design notes:
  * Resumable: if a stage's output file already exists, skip that stage.
  * Each generation writes one JSONL line + flushes immediately.
  * No exotic dependencies: only torch + transformers + accelerate.
  * Models loaded one at a time during axis derivation to fit single H100 (80GB).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE2 = Path(__file__).resolve().parent
RESULTS = PHASE2 / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelSpec:
    nick: str           # short label used in filenames
    instruct: str       # HF id of aligned/instruct checkpoint
    base: str           # HF id of base checkpoint (for axis derivation)
    chat_template_role_user: str = "user"
    chat_template_role_asst: str = "assistant"


# Two open-weights models from disjoint families, each with a public base
# checkpoint so we can derive an assistant-axis via mean-difference.
GEMMA = ModelSpec(
    nick="gemma2-9b",
    instruct="google/gemma-2-9b-it",
    base="google/gemma-2-9b",
)
LLAMA = ModelSpec(
    nick="llama3.1-8b",
    instruct="meta-llama/Llama-3.1-8B-Instruct",
    base="meta-llama/Llama-3.1-8B",
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Stage 1: assistant-axis derivation
# ---------------------------------------------------------------------------

def collect_last_token_residuals(
    model_id: str,
    prompts: list[str],
    layer: int,
    hf_token: str | None,
) -> np.ndarray:
    """Run prompts through the model, return residual at the last input token
    of each prompt, at the chosen layer. Shape: (n_prompts, hidden_dim).
    """
    log(f"  loading {model_id} (this can take 1-3 min)...")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        token=hf_token,
        output_hidden_states=True,
    )
    model.eval()
    log(f"  model loaded; running {len(prompts)} prompts...")
    out_vectors = []
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer][0, -1].detach().cpu().float().numpy()
        out_vectors.append(h)
        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{len(prompts)}")
    arr = np.stack(out_vectors)
    log(f"  done. activations shape: {arr.shape}; freeing model...")
    del model, tok
    free_cuda()
    return arr


def derive_axis(spec: ModelSpec, prompts: list[str], layer: int,
                hf_token: str | None) -> dict:
    out_path = RESULTS / f"axis_{spec.nick}.npz"
    if out_path.exists():
        log(f"  axis already exists at {out_path}; skipping derivation.")
        npz = np.load(out_path)
        return {
            "axis": npz["axis"],
            "h_instruct": npz["h_instruct"],
            "h_base": npz["h_base"],
            "layer": int(npz["layer"]),
        }
    log(f"=== Stage 1 axis derivation: {spec.nick} (layer {layer}) ===")
    h_instruct = collect_last_token_residuals(spec.instruct, prompts, layer, hf_token)
    h_base = collect_last_token_residuals(spec.base, prompts, layer, hf_token)
    axis = h_instruct.mean(0) - h_base.mean(0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise RuntimeError(f"Axis has zero norm for {spec.nick}; something is wrong.")
    axis_hat = axis / norm

    # Validate: project both sets onto axis; means should be well-separated.
    proj_instruct = h_instruct @ axis_hat
    proj_base = h_base @ axis_hat
    sep = float(proj_instruct.mean() - proj_base.mean())
    log(f"  axis derived. validation projection means: instruct={proj_instruct.mean():.3f}, base={proj_base.mean():.3f}, sep={sep:.3f}")

    np.savez(out_path,
             axis=axis_hat,
             h_instruct=h_instruct,
             h_base=h_base,
             proj_instruct=proj_instruct,
             proj_base=proj_base,
             layer=layer,
             instruct_id=spec.instruct,
             base_id=spec.base,
             separation=sep)
    log(f"  saved {out_path}")
    return {"axis": axis_hat, "h_instruct": h_instruct, "h_base": h_base, "layer": layer}


# ---------------------------------------------------------------------------
# Stage 2-3: cross-model conversation + activation extraction
# ---------------------------------------------------------------------------

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


def system_for(role: str, topic_title: str, domain: str, persona: str | None) -> str:
    if role == "tutor":
        return (
            f"You are roleplaying a TUTOR in a tutoring conversation on the topic: "
            f'"{topic_title}" (domain: {domain}). You are a competent subject-matter '
            f"tutor who tries to help the student understand. Adapt to the student. "
            f"Don't be unrealistically perfect — produce realistic tutor turns including "
            f"occasional imperfect ones. Reply with the next tutor turn only — 1-4 "
            f"sentences. Do not include role labels in your output."
        )
    return (
        f"You are roleplaying a STUDENT in a tutoring conversation on the topic: "
        f'"{topic_title}" (domain: {domain}). Your student persona for this conversation:\n'
        f"{PERSONAS[persona]}\n"
        f"Reply with the next student turn only — 1-3 sentences. Do not include role labels."
    )


def build_chat_messages(turns: list[dict], speaker_role: str) -> list[dict]:
    """From the speaker's perspective, prior other-role turns become 'user',
    speaker's own prior turns become 'assistant'."""
    messages = []
    for t in turns:
        role_for_chat = "user" if t["role"] != speaker_role else "assistant"
        messages.append({"role": role_for_chat, "content": t["content"]})
    return messages


def generate_one_turn(
    speaker_role: str,
    model,
    tokenizer,
    system_prompt: str,
    turns_so_far: list[dict],
    layer_for_activation: int | None = None,
    axis: np.ndarray | None = None,
    max_new_tokens: int = 240,
) -> tuple[str, float | None]:
    """Use 'model' to produce the next 'speaker_role' turn given conversation
    so far. If layer_for_activation+axis are given, also extract residual at
    the last input token and project onto axis. Returns (text, projection-or-None).
    """
    chat_messages = [{"role": "user", "content": system_prompt}] + build_chat_messages(turns_so_far, speaker_role)
    # If first non-system message is 'assistant' (would happen if we're
    # generating speaker's first turn after their own turn — shouldn't normally
    # happen), fix by collapsing. Defensive only.
    # Apply chat template
    try:
        prompt_str = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: simple concat
        parts = []
        for m in chat_messages:
            parts.append(f"[{m['role'].upper()}] {m['content']}")
        prompt_str = "\n\n".join(parts) + "\n\n[ASSISTANT]"

    enc = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=3500).to(model.device)

    projection: float | None = None
    if layer_for_activation is not None and axis is not None:
        with torch.no_grad():
            fwd = model(**enc, output_hidden_states=True)
        h = fwd.hidden_states[layer_for_activation][0, -1].detach().cpu().float().numpy()
        projection = float(np.dot(h, axis))
        del fwd

    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen[0, enc.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any stray role labels
    for prefix in ("TUTOR:", "Tutor:", "STUDENT:", "Student:", "[TUTOR]", "[STUDENT]", "Assistant:", "User:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    del enc, gen
    free_cuda()
    return text, projection


def run_conversations(
    tutor_spec: ModelSpec,
    student_spec: ModelSpec,
    topics: list[dict],
    personas: list[str],
    n_turns: int,
    layer_for_activation: int,
    tutor_axis: np.ndarray,
    out_jsonl: Path,
    hf_token: str | None,
) -> None:
    """Load tutor + student models and run cross-model conversations.

    Conversation alternates STUDENT then TUTOR, starting with the topic's
    starter_question as a fixed first student turn (no model call needed).
    """
    if out_jsonl.exists():
        done_ids = set()
        for line in out_jsonl.read_text().splitlines():
            try:
                done_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        log(f"resuming: {len(done_ids)} conversations already done in {out_jsonl}")
    else:
        done_ids = set()

    log(f"  loading TUTOR model: {tutor_spec.instruct}")
    tutor_tok = AutoTokenizer.from_pretrained(tutor_spec.instruct, token=hf_token)
    tutor_model = AutoModelForCausalLM.from_pretrained(
        tutor_spec.instruct, torch_dtype=torch.float16,
        device_map="cuda:0", token=hf_token, output_hidden_states=True,
    )
    tutor_model.eval()

    log(f"  loading STUDENT model: {student_spec.instruct}")
    student_tok = AutoTokenizer.from_pretrained(student_spec.instruct, token=hf_token)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_spec.instruct, torch_dtype=torch.float16,
        device_map="cuda:0", token=hf_token,
    )
    student_model.eval()

    log(f"  GPU memory after loading both: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    f = out_jsonl.open("a")

    todo = [(t, p) for t in topics for p in personas]
    log(f"  generating {len(todo)} (topic, persona) conversations...")
    for i, (topic, persona) in enumerate(todo):
        cid = f"{topic['id']}__{persona}"
        if cid in done_ids:
            log(f"  [{i+1}/{len(todo)}] SKIP {cid} (already done)")
            continue
        try:
            turns: list[dict] = [{"role": "student", "content": topic["starter_question"]}]
            tutor_projections: list[dict] = []
            t0 = time.time()
            while len(turns) < n_turns:
                next_role = "tutor" if turns[-1]["role"] == "student" else "student"
                if next_role == "tutor":
                    sys_prompt = system_for("tutor", topic["title"], topic["domain"], None)
                    text, proj = generate_one_turn(
                        "tutor", tutor_model, tutor_tok, sys_prompt, turns,
                        layer_for_activation=layer_for_activation, axis=tutor_axis,
                    )
                    tutor_idx = len([t for t in turns if t["role"] == "tutor"]) + 1
                    tutor_projections.append({
                        "tutor_turn_index": tutor_idx,
                        "axis_projection": proj,
                    })
                else:
                    sys_prompt = system_for("student", topic["title"], topic["domain"], persona)
                    text, _ = generate_one_turn(
                        "student", student_model, student_tok, sys_prompt, turns,
                    )
                if not text:
                    raise ValueError(f"empty {next_role} turn at index {len(turns)+1}")
                turns.append({"role": next_role, "content": text})
            elapsed = time.time() - t0
            row = {
                "id": cid,
                "topic_id": topic["id"],
                "topic_title": topic["title"],
                "domain": topic["domain"],
                "persona": persona,
                "tutor_model": tutor_spec.instruct,
                "student_model": student_spec.instruct,
                "axis_layer": layer_for_activation,
                "n_turns": len(turns),
                "turns": turns,
                "tutor_projections": tutor_projections,
                "elapsed_sec": round(elapsed, 1),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            projs = [p['axis_projection'] for p in tutor_projections if p['axis_projection'] is not None]
            log(f"  [{i+1}/{len(todo)}] OK {cid:35s} {elapsed:5.1f}s  tutor_projs={[round(p,2) for p in projs]}")
        except Exception as e:
            log(f"  [{i+1}/{len(todo)}] FAIL {cid:35s} {type(e).__name__}: {e}")
    f.close()
    del tutor_model, student_model, tutor_tok, student_tok
    free_cuda()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["round1", "round2"], default="round1",
                    help="round1 = Gemma-tutor / Llama-student; round2 = swap")
    ap.add_argument("--topics", default=str(PHASE2.parent / "topics.json"))
    ap.add_argument("--n-topics", type=int, default=10,
                    help="How many topics (× 2 personas = total conversations).")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--prompts", default=str(PHASE2 / "contrastive_prompts.json"))
    ap.add_argument("--skip-axis", action="store_true",
                    help="Skip axis derivation (use existing npz)")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("ERROR: HF_TOKEN not set; Gemma and Llama checkpoints are gated.")
        return 2
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available.")
        return 2
    log(f"CUDA OK. devices={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}")
    log(f"  total mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    contrastive = json.loads(Path(args.prompts).read_text())
    topics = json.loads(Path(args.topics).read_text())[: args.n_topics]
    personas = ["engaged", "offloading"]

    if args.mode == "round1":
        tutor, student = GEMMA, LLAMA
    else:
        tutor, student = LLAMA, GEMMA
    out_jsonl = RESULTS / f"conversations_{args.mode}_tutor_{tutor.nick}.jsonl"

    log(f"=== mode={args.mode}, tutor={tutor.nick}, student={student.nick} ===")
    log(f"  out: {out_jsonl}")

    # Stage 1: derive tutor axis (only the tutor's axis is needed for projection)
    if not args.skip_axis:
        axis_data = derive_axis(tutor, contrastive, args.layer, hf_token)
    else:
        npz = np.load(RESULTS / f"axis_{tutor.nick}.npz")
        axis_data = {"axis": npz["axis"], "layer": int(npz["layer"])}

    # Stage 2-3: run cross-model conversations
    log(f"=== Stage 2-3: cross-model conversations ===")
    run_conversations(
        tutor_spec=tutor,
        student_spec=student,
        topics=topics,
        personas=personas,
        n_turns=args.turns,
        layer_for_activation=args.layer,
        tutor_axis=axis_data["axis"],
        out_jsonl=out_jsonl,
        hf_token=hf_token,
    )

    log(f"=== DONE. Output: {out_jsonl} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
