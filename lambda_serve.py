"""[PHASE 2 SCAFFOLD - placeholder]

Lambda Cloud orchestrator for the Phase 2 GPU run. Mirrors lambda_run.py from
the cognitive_offloading_detector project.

Plan (to be wired up when Phase 2 is started):
  1. Pick the cheapest available GPU instance with >= 24 GB VRAM (gpu_1x_a10
     at $1.29/hr is sufficient for a single Gemma-2-9B in FP16; for axis
     derivation we need both base+aligned in memory or sequentially, fine
     either way on a single A10).
  2. Launch via Lambda Cloud REST API at https://cloud.lambda.ai/api/v1.
  3. Wait for sshd, rsync the project directory.
  4. Install: torch, transformers, accelerate, sentencepiece (HF Gemma tokenizer
     dep), and copy .env containing HF_TOKEN.
  5. Run the Phase 2 pipeline:
       a. compute_axis.py        -> results/axis/assistant_axis.npz
       b. (optional) regenerate Phase 2 conversations with Gemma-2-9B-it
          locally, since vLLM and full Gemma generation can both be done
          on the same instance.
       c. extract_activations.py -> results/projections.jsonl
  6. rsync results/ back to local.
  7. Terminate ONLY on success.

Failure handling:
  Same as cognitive_offloading_detector/lambda_run.py: on any error, leave
  the instance running and print the instance-id, IP, ssh-key path, and the
  --terminate-only command for cleanup.

For now this file exists to flag the integration point and document the plan.
Actual orchestration code is in cognitive_offloading_detector/lambda_run.py
and can be parameterized to point at this project's directory and the
Phase 2 command instead of duplicating it here.
"""
from __future__ import annotations

import sys


def main() -> None:
    print("Phase 2 orchestrator (lambda_serve.py) is a scaffold/placeholder.\n"
          "See PLAN.md for the GPU run plan, and reuse\n"
          "  cognitive_offloading_detector/lambda_run.py\n"
          "with --remote-dir pedagogical_drift and an appropriate --command\n"
          "(e.g. 'python3 compute_axis.py && python3 extract_activations.py ...').\n",
          file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
