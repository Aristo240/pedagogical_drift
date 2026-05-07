"""Compute the Assistant Axis from rollouts (Lu et al. Sec 3.1).

Inputs:
  - rollouts_roles_judged.jsonl (output of judge_rollouts.py)
  - rollouts_default.jsonl

Method (Lu et al.'s contrast-vector method):
  1. For each role, compute the role vector = mean of activations over rollouts
     where the model was 'fully role-playing' (judge label == 3). Lu et al. also
     use 'somewhat role-playing' (label >= 2) as a separate set; we keep both
     options here as 'strict' (label==3) and 'lenient' (label>=2).
  2. Default Assistant vector = mean of activations across all default
     rollouts (no filter applied; these are by construction default-mode).
  3. Assistant Axis = default_assistant - mean(role vectors).
  4. Normalize: axis_hat = axis / ||axis||.
  5. Validate: project both default and role activations onto axis; the means
     should be well-separated.

Output:
  results/assistant_axis.npz
    fields: axis_strict, axis_lenient, default_mean, role_vectors,
            role_ids, model_id, layer, validation_separation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def compute_role_vectors(judged_rollouts: list[dict], min_label: int = 3) -> dict[str, np.ndarray]:
    """For each role_id, average activations over rollouts where judge_label >= min_label."""
    by_role: dict[str, list[np.ndarray]] = {}
    for r in judged_rollouts:
        if r.get("judge_label", -1) < min_label:
            continue
        act = r.get("activation")
        if not act:
            continue
        by_role.setdefault(r["role_id"], []).append(np.array(act, dtype=np.float32))
    role_vectors: dict[str, np.ndarray] = {}
    for role_id, acts in by_role.items():
        if len(acts) < 1:
            continue
        role_vectors[role_id] = np.mean(np.stack(acts), axis=0)
    return role_vectors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged-roles", required=True,
                    help="JSONL output from judge_rollouts.py")
    ap.add_argument("--default", required=True,
                    help="JSONL output of run_axis_pipeline.py default mode")
    ap.add_argument("--out", default="results/assistant_axis.npz")
    args = ap.parse_args()

    judged = load_jsonl(Path(args.judged_roles))
    default = load_jsonl(Path(args.default))
    print(f"Loaded {len(judged)} judged role rollouts, {len(default)} default rollouts.")

    if not default:
        raise SystemExit("No default rollouts available.")
    default_acts = np.stack([np.array(r["activation"], dtype=np.float32) for r in default
                             if r.get("activation")])
    default_mean = default_acts.mean(axis=0)
    print(f"Default Assistant: n={len(default_acts)}, hidden_dim={default_mean.shape[0]}")

    role_vecs_strict = compute_role_vectors(judged, min_label=3)  # fully role-playing only
    role_vecs_lenient = compute_role_vectors(judged, min_label=2)  # somewhat or fully

    print(f"Role vectors (strict, label==3): {len(role_vecs_strict)} roles")
    print(f"Role vectors (lenient, label>=2): {len(role_vecs_lenient)} roles")

    if not role_vecs_strict:
        print("WARNING: no rollouts passed the strict 'fully role-playing' filter. "
              "Falling back to lenient. Consider increasing the rollout count.")
        role_vecs_strict = role_vecs_lenient

    if not role_vecs_strict:
        raise SystemExit("No usable role vectors at any threshold.")

    role_ids_strict = sorted(role_vecs_strict.keys())
    role_matrix_strict = np.stack([role_vecs_strict[k] for k in role_ids_strict])
    mean_role_strict = role_matrix_strict.mean(axis=0)

    role_ids_lenient = sorted(role_vecs_lenient.keys()) if role_vecs_lenient else []
    role_matrix_lenient = np.stack([role_vecs_lenient[k] for k in role_ids_lenient]) if role_vecs_lenient else np.zeros_like(role_matrix_strict)
    mean_role_lenient = role_matrix_lenient.mean(axis=0) if role_vecs_lenient else mean_role_strict

    axis_strict = default_mean - mean_role_strict
    axis_strict_hat = axis_strict / (np.linalg.norm(axis_strict) + 1e-8)
    axis_lenient = default_mean - mean_role_lenient
    axis_lenient_hat = axis_lenient / (np.linalg.norm(axis_lenient) + 1e-8)

    # Validation: project both pools onto strict axis
    proj_default = default_acts @ axis_strict_hat
    proj_role_strict = role_matrix_strict @ axis_strict_hat
    sep = float(proj_default.mean() - proj_role_strict.mean())
    print(f"Validation projection (strict axis): "
          f"default mean={proj_default.mean():.3f}, "
          f"role-vec mean={proj_role_strict.mean():.3f}, "
          f"separation={sep:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        axis_strict=axis_strict_hat,
        axis_strict_unnorm=axis_strict,
        axis_lenient=axis_lenient_hat,
        axis_lenient_unnorm=axis_lenient,
        default_mean=default_mean,
        role_matrix_strict=role_matrix_strict,
        role_matrix_lenient=role_matrix_lenient,
        role_ids_strict=np.array(role_ids_strict),
        role_ids_lenient=np.array(role_ids_lenient),
        proj_default=proj_default,
        proj_role_strict=proj_role_strict,
        validation_separation=sep,
    )
    print(f"Saved: {out_path}")
    print(f"  hidden_dim: {axis_strict_hat.shape[0]}")
    print(f"  strict axis L2 norm (pre-normalize): {np.linalg.norm(axis_strict):.2f}")
    print(f"  separation default-vs-role on strict axis: {sep:.3f}")


if __name__ == "__main__":
    main()
