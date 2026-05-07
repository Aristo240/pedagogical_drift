"""[PHASE 2 SCAFFOLD - not yet validated end-to-end]

Compute the assistant-axis direction in residual-stream activation space.

Approach (canonical mean-difference, à la Arditi et al. / Lu et al.):

1. For a contrastive prompt set, run the SAME inputs through the BASE model
   (non-aligned) and the ALIGNED model. Both must share architecture so
   activations are comparable.
2. At a chosen residual-stream layer L and a chosen token position (default:
   final input token), record activations h_aligned[i] and h_base[i] for
   each prompt i.
3. Axis vector v = mean_i(h_aligned[i]) - mean_i(h_base[i]).
   Normalize: v_hat = v / ||v||.
4. Validate on a held-out set: project h_aligned and h_base onto v_hat and
   confirm the means are well-separated.
5. Save v_hat to disk along with the layer index, model identifiers, and
   the contrastive set used.

Defaults
--------
- Aligned model:  google/gemma-2-9b-it
- Base model:     google/gemma-2-9b
- Layer:          mid-network residual (paper-specific; pilot multiple if
                  unsure and pick the one with cleanest separation)
- Token position: last token of the user prompt

Caveats
-------
- This script is NOT validated end-to-end. The methodology is described per
  the canonical recipe; specific hyperparameters (layer, token position,
  contrastive set composition) need to be tuned for Gemma-2-9B specifically.
- Lu et al.'s exact methodology may differ in details (e.g. multi-token
  averaging, SAE-feature decomposition). Verify against the paper before
  publishing any claims.
- Loading two 9B models simultaneously needs ~36 GB FP16 — won't fit on a
  24 GB A10. Either: (a) load one at a time and cache activations, or
  (b) use an A100 (40 GB).

Run on Lambda Cloud GPU instance via lambda_serve.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Imports gated until torch / transformers are installed (Phase 2 only).
# Uncomment when running on the GPU instance.
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


CONTRASTIVE_PROMPTS_DEFAULT = [
    # Designed to elicit clearly assistant-style vs. raw-completion behavior.
    # The prompt itself is identical for both models; the difference shows in
    # internal representations of "what kind of agent am I being".
    "Could you help me understand why the sky is blue?",
    "I'm trying to debug a Python error. Can you walk me through it?",
    "What's a polite way to decline a meeting?",
    "Explain the difference between accuracy and precision in measurement.",
    "Can you summarize this article for me?",
    "What should I cook for dinner if I have eggs, rice, and spinach?",
    "Translate this sentence to Spanish, please.",
    "I'd like to learn the basics of statistics. Where should I start?",
    "Can you check this code for bugs?",
    "How do I write a thank-you note?",
    "What are the key claims in the introduction of this paper?",
    "Help me brainstorm names for a new project.",
    "I'm preparing for a job interview. What should I focus on?",
    "Could you proofread this paragraph and suggest improvements?",
    "Explain quantum entanglement in plain language.",
    "What's a healthy breakfast for someone with diabetes?",
    "I need to write a polite reminder email — can you help?",
    "What's the time complexity of a hash table lookup?",
    "Can you give me feedback on my essay outline?",
    "How do I set up a virtual environment in Python?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aligned-model", default="google/gemma-2-9b-it")
    ap.add_argument("--base-model", default="google/gemma-2-9b")
    ap.add_argument("--layer", type=int, default=20,
                    help="Residual-stream layer (0-indexed). Tune per paper.")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON list of contrastive prompts. If unset, uses CONTRASTIVE_PROMPTS_DEFAULT.")
    ap.add_argument("--out", default="results/axis/assistant_axis.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # =================================================================
    # NOT YET RUNNABLE. Outline of the planned implementation:
    # =================================================================
    raise NotImplementedError(
        "compute_axis.py is a Phase 2 scaffold and is not yet wired up.\n"
        "See module docstring for the planned approach. Run on a Lambda\n"
        "GPU instance after installing torch + transformers."
    )

    # Pseudocode for the eventual implementation:
    #
    # prompts = json.loads(Path(args.prompts_file).read_text()) if args.prompts_file \
    #     else CONTRASTIVE_PROMPTS_DEFAULT
    #
    # def collect_activations(model_name):
    #     tok = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name, torch_dtype=torch.float16, device_map=args.device,
    #         output_hidden_states=True,
    #     )
    #     acts = []
    #     for p in prompts:
    #         enc = tok(p, return_tensors="pt").to(args.device)
    #         with torch.no_grad():
    #             out = model(**enc, output_hidden_states=True)
    #         h = out.hidden_states[args.layer][0, -1].detach().cpu().float().numpy()
    #         acts.append(h)
    #     del model
    #     torch.cuda.empty_cache()
    #     return np.stack(acts)
    #
    # h_aligned = collect_activations(args.aligned_model)
    # h_base    = collect_activations(args.base_model)
    # axis = h_aligned.mean(0) - h_base.mean(0)
    # axis_hat = axis / np.linalg.norm(axis)
    #
    # out_path = Path(args.out)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez(out_path, axis=axis_hat, h_aligned=h_aligned, h_base=h_base,
    #          layer=args.layer, aligned_model=args.aligned_model,
    #          base_model=args.base_model)
    # print(f"Saved axis to {out_path}")


if __name__ == "__main__":
    main()
