"""Lu et al. (2026) Assistant Axis derivation, scaled-down pilot.

Replaces the earlier base-vs-instruct contrast approach with the actual Lu et
al. methodology:

  Stage A: Generate role-playing rollouts.
    For each (role, extraction_question) pair, ask the instruct model with the
    role's system prompt. Capture the model's response AND the mean residual
    activation across response tokens at the chosen middle layer.

  Stage B: Generate default-Assistant rollouts.
    Same extraction questions, but with neutral system prompts ("You are a
    large language model", "Respond as yourself") OR no system prompt. Same
    activation extraction.

  Stage C: LLM judge filter (separate script: judge_rollouts.py).
    For each role-playing rollout, ask Claude Haiku whether the model is
    fully role-playing (drops "I am an AI"), somewhat role-playing, or no
    role-playing. Keep only fully role-playing.

  Stage D: Compute axis (compute_axis.py):
    role_vector_per_role = mean of fully-role-playing activations for that role
    default_assistant_vector = mean of default Assistant activations
    axis = default_assistant_vector - mean(all role_vectors)

We run on ONE instruct model per axis (Lu et al. did three: Gemma 2 27B,
Qwen 3 32B, Llama 3.3 70B). For our pilot we default to Gemma 2 27B since it
matches Lu et al.'s headline figure and fits an 80GB H100 in FP16 (54 GB).

Pilot scope vs Lu et al.:
  - 30 roles (Lu et al.: 275)
  - 1 system prompt per role (Lu et al.: 5)
  - 20 extraction questions (Lu et al.: 240)
  - => 600 role rollouts (Lu et al.: 1200/role x 275 roles = 330000)
  - 4 neutral default-Assistant prompts + 1 no-system-prompt = 100
    default Assistant rollouts (Lu et al.: 1200)

This is dramatically scaled down. Numbers will be noisy. Methodology is intact.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE2 = Path(__file__).resolve().parent.parent
LU = PHASE2 / "lu_axis"
RESULTS = LU / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


NEUTRAL_SYSTEM_PROMPTS = [
    "You are a large language model.",
    "Respond as yourself.",
    "You are a helpful AI assistant.",
    "You are an AI assistant. Answer as yourself.",
]
# Lu et al. also include "no system prompt" as one of the conditions.
NO_SYSTEM_PROMPT = None


@dataclass
class GenerationConfig:
    model_id: str = "google/gemma-2-27b-it"
    layer: int = 22  # middle of 46-layer Gemma 2 27B
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_prompt(tokenizer, system_prompt: str | None, question: str) -> str:
    """Apply chat template. Gemma's template doesn't support a 'system' role —
    we prepend the system content to the first user turn."""
    if system_prompt:
        user_content = f"{system_prompt}\n\n{question}"
    else:
        user_content = question
    msgs = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_with_response_activation(
    model, tokenizer, prompt: str, layer: int, max_new_tokens: int,
    temperature: float, top_p: float,
) -> tuple[str, np.ndarray]:
    """Generate continuation, then run a second forward pass on the FULL
    sequence (prompt + response) and extract the mean residual activation
    over RESPONSE tokens at the chosen layer.

    Two-pass design (Lu et al.-style):
      1. .generate(prompt) -> response tokens
      2. forward(prompt + response, output_hidden_states=True) -> hidden states
      3. mean over hidden_states[layer][0, response_token_indices, :]

    Returns (response_text, mean_activation_vector).
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    prompt_len = enc.input_ids.shape[1]

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_ids = gen[0]
    response_ids = full_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    # Strip stray role labels that some models emit
    for prefix in ("Assistant:", "User:", "[ASSISTANT]", "[USER]"):
        if response_text.startswith(prefix):
            response_text = response_text[len(prefix):].strip()

    if response_ids.shape[0] == 0:
        return response_text, np.zeros(model.config.hidden_size, dtype=np.float32)

    # Run a forward pass on the full sequence to get hidden states at the layer
    full_input = full_ids.unsqueeze(0)
    out = model(full_input, output_hidden_states=True)
    # hidden_states[layer] has shape (1, seq_len, hidden_dim)
    hs = out.hidden_states[layer][0]  # (seq_len, hidden_dim)
    response_h = hs[prompt_len:]  # (response_len, hidden_dim)
    mean_act = response_h.mean(dim=0).detach().cpu().float().numpy()
    return response_text, mean_act


def run_rollouts(model, tokenizer, cfg: GenerationConfig,
                 roles: list[dict], questions: list[str],
                 out_path: Path, mode: str) -> None:
    """Run rollouts for either 'role' or 'default' mode and append rows to
    out_path JSONL. Resumable: skips (role_id, question_idx) pairs already
    present.
    """
    seen: set[tuple] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add((r["role_id"], r["question_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    log(f"  {mode}: {len(seen)} rollouts already done; resuming if applicable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("a")

    if mode == "role":
        # roles is a list of {role, system_prompt}
        items = []
        for role_obj in roles:
            for q_idx, q in enumerate(questions):
                items.append((role_obj["role"], role_obj["system_prompt"], q_idx, q))
    elif mode == "default":
        # roles is list of system_prompts (or None)
        items = []
        for sp_idx, sp in enumerate(roles):
            role_id = f"default__{sp_idx}"
            for q_idx, q in enumerate(questions):
                items.append((role_id, sp, q_idx, q))
    else:
        raise ValueError(f"Unknown mode {mode}")

    n_total = len(items)
    log(f"  {mode}: {n_total} rollouts to do; sample one to estimate time...")

    n_ok, n_fail = 0, 0
    for i, (role_id, sp, q_idx, q) in enumerate(items):
        key = (role_id, q_idx)
        if key in seen:
            continue
        try:
            prompt = build_prompt(tokenizer, sp, q)
            response, act = generate_with_response_activation(
                model, tokenizer, prompt,
                layer=cfg.layer, max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature, top_p=cfg.top_p,
            )
            row = {
                "role_id": role_id,
                "question_idx": q_idx,
                "question": q,
                "system_prompt": sp,
                "prompt_full": prompt,
                "response": response,
                "activation": act.tolist(),
                "layer": cfg.layer,
                "model_id": cfg.model_id,
                "mode": mode,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_ok += 1
            if (n_ok % 25) == 0 or n_ok == 1:
                log(f"    [{i+1}/{n_total}] ok role={role_id} q_idx={q_idx} resp_len={len(response)}")
        except Exception as e:
            n_fail += 1
            log(f"    [{i+1}/{n_total}] FAIL role={role_id} q_idx={q_idx} {type(e).__name__}: {e}")
        free_cuda()
    f.close()
    log(f"  {mode}: done. OK={n_ok} FAIL={n_fail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-27b-it")
    ap.add_argument("--layer", type=int, default=22,
                    help="Middle residual stream layer. Gemma 2 27B has 46 layers.")
    ap.add_argument("--roles", default=str(LU / "roles.json"))
    ap.add_argument("--questions", default=str(LU / "extraction_questions.json"))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out-roles", default=str(RESULTS / "rollouts_roles.jsonl"))
    ap.add_argument("--out-default", default=str(RESULTS / "rollouts_default.jsonl"))
    ap.add_argument("--skip-roles", action="store_true")
    ap.add_argument("--skip-default", action="store_true")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log("ERROR: HF_TOKEN not set; Gemma is gated.")
        return 2
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available.")
        return 2

    log(f"CUDA OK. Device: {torch.cuda.get_device_name(0)}, "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    cfg = GenerationConfig(model_id=args.model, layer=args.layer,
                            max_new_tokens=args.max_new_tokens)

    log(f"Loading {args.model}... (this can take 2-4 min for 27B)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        token=hf_token,
        output_hidden_states=True,
    )
    model.eval()
    log(f"Model loaded. {torch.cuda.memory_allocated()/1e9:.1f} GB in use.")

    roles = json.loads(Path(args.roles).read_text())
    questions = json.loads(Path(args.questions).read_text())
    log(f"Roles: {len(roles)}. Extraction questions: {len(questions)}.")

    if not args.skip_roles:
        log(f"=== Stage A: role-playing rollouts ({len(roles)} roles x {len(questions)} questions = {len(roles)*len(questions)}) ===")
        run_rollouts(model, tokenizer, cfg, roles, questions, Path(args.out_roles), mode="role")

    if not args.skip_default:
        log(f"=== Stage B: default-Assistant rollouts ({(len(NEUTRAL_SYSTEM_PROMPTS)+1)} prompts x {len(questions)} questions) ===")
        all_default_prompts = list(NEUTRAL_SYSTEM_PROMPTS) + [NO_SYSTEM_PROMPT]
        run_rollouts(model, tokenizer, cfg, all_default_prompts, questions, Path(args.out_default), mode="default")

    log("DONE rollout generation. Next: run judge_rollouts.py to filter, then compute_axis.py to compute the axis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
