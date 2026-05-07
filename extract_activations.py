"""[PHASE 2 SCAFFOLD - not yet validated end-to-end]

Per-turn assistant-axis projection for tutoring conversations.

For each tutoring conversation (where the tutor is an open-weights model whose
activations we have access to), this script:

1. Loads the conversation turns.
2. For each tutor turn t:
   a. Builds the input prompt = system + earlier turns + the user turn that
      precedes turn t.
   b. Runs the prompt through the aligned model.
   c. Extracts the residual-stream activation at layer L, at the final input
      token (the model's "decision state" before generating turn t).
   d. Projects that activation onto the assistant-axis vector v (loaded from
      the .npz produced by compute_axis.py).
3. Saves a JSONL per conversation with per-turn projection scalars.

The output of this script is the input to a final analysis step that
correlates per-turn projections with per-turn pedagogy scores from
grade_pedagogy.py.

Usage (Phase 2 only, on a Lambda GPU instance)
----------------------------------------------
  python extract_activations.py \\
      --conversations data/conversations/phase2_gemma.jsonl \\
      --axis results/axis/assistant_axis.npz \\
      --model google/gemma-2-9b-it \\
      --layer 20 \\
      --out results/projections.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Imports gated until torch / transformers are installed.
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


def format_chat_prompt_for_gemma(turns_so_far: list[dict], next_role: str = "tutor") -> str:
    """Convert a list of {role: student|tutor, content} into a Gemma chat-template
    prompt that ends just before the assistant (tutor) is to speak.

    Gemma uses <start_of_turn>user / <start_of_turn>model markers. Map:
      student -> user
      tutor   -> model
    """
    parts = []
    for t in turns_so_far:
        gemma_role = "user" if t["role"] == "student" else "model"
        parts.append(f"<start_of_turn>{gemma_role}\n{t['content']}<end_of_turn>")
    # Open the next turn (tutor) so we extract activations at the boundary
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", required=True,
                    help="JSONL of conversations (Phase 2 generations from Gemma).")
    ap.add_argument("--axis", required=True,
                    help=".npz file produced by compute_axis.py")
    ap.add_argument("--model", default="google/gemma-2-9b-it",
                    help="Must match the aligned model used in compute_axis.py")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out", default="results/projections.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # =================================================================
    # NOT YET RUNNABLE. Outline of the planned implementation:
    # =================================================================
    raise NotImplementedError(
        "extract_activations.py is a Phase 2 scaffold and is not yet wired up.\n"
        "See module docstring for the planned approach. Run on a Lambda GPU\n"
        "instance after installing torch + transformers and after generating\n"
        "Phase 2 conversations with the same model whose activations are\n"
        "being extracted."
    )

    # Pseudocode for the eventual implementation:
    #
    # axis_data = np.load(args.axis)
    # axis = axis_data['axis']                      # shape: (d_model,)
    # axis_layer = int(axis_data['layer'])
    # if axis_layer != args.layer:
    #     print(f"Warning: --layer={args.layer} differs from axis layer {axis_layer}")
    #
    # tok = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model, torch_dtype=torch.float16, device_map=args.device,
    #     output_hidden_states=True,
    # )
    #
    # out_path = Path(args.out)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # with out_path.open("w") as fout:
    #     for line in Path(args.conversations).read_text().splitlines():
    #         convo = json.loads(line)
    #         turns = convo["turns"]
    #         tutor_idx = 0
    #         for i, t in enumerate(turns):
    #             if t["role"] != "tutor":
    #                 continue
    #             tutor_idx += 1
    #             prefix = turns[:i]
    #             prompt_str = format_chat_prompt_for_gemma(prefix)
    #             enc = tok(prompt_str, return_tensors="pt").to(args.device)
    #             with torch.no_grad():
    #                 out = model(**enc, output_hidden_states=True)
    #             h = out.hidden_states[args.layer][0, -1].detach().cpu().float().numpy()
    #             projection = float(np.dot(h, axis))
    #             fout.write(json.dumps({
    #                 "conversation_id": convo["id"],
    #                 "tutor_turn_index": tutor_idx,
    #                 "persona": convo.get("persona"),
    #                 "projection": projection,
    #                 "layer": args.layer,
    #                 "model": args.model,
    #             }) + "\n")
    #             fout.flush()


if __name__ == "__main__":
    main()
