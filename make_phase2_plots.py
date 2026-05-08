"""Generate Phase 2 (Lu et al. axis) validation plots from assistant_axis.npz."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
NPZ = ROOT / "phase2" / "lu_axis" / "results" / "assistant_axis.npz"
JUDGED = ROOT / "phase2" / "lu_axis" / "results" / "rollouts_roles_judged.jsonl"
OUT = ROOT / "phase2" / "lu_axis" / "results"


def main() -> None:
    data = np.load(NPZ, allow_pickle=True)
    axis = data["axis_strict"]
    default_mean = data["default_mean"]
    role_matrix_strict = data["role_matrix_strict"]
    role_ids = list(data["role_ids_strict"])
    sep = float(data["validation_separation"])

    # Project all individual fully-role-playing rollouts onto axis.
    proj_individual = []
    role_for_individual = []
    rows = [json.loads(l) for l in JUDGED.read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get("judge_label") != 3:
            continue
        a = np.array(r["activation"], dtype=np.float32)
        if not np.all(np.isfinite(a)):
            continue
        proj_individual.append(float(np.dot(a, axis)))
        role_for_individual.append(r["role_id"])
    proj_individual = np.array(proj_individual)
    print(f"Projected {len(proj_individual)} individual fully-role-playing rollouts onto axis.")

    # Default Assistant: load activations from rollouts_default.jsonl if available
    default_jsonl = ROOT / "phase2" / "lu_axis" / "results" / "rollouts_default.jsonl"
    if default_jsonl.exists():
        proj_default_individual = []
        for line in default_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            a = np.array(r["activation"], dtype=np.float32)
            if not np.all(np.isfinite(a)):
                continue
            proj_default_individual.append(float(np.dot(a, axis)))
        proj_default_individual = np.array(proj_default_individual)
        print(f"Projected {len(proj_default_individual)} individual default rollouts onto axis.")
    else:
        proj_default_individual = data["proj_default"]

    # === Plot 1: histogram, default vs role-playing on the same axis ===
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(proj_individual, bins=40, alpha=0.7, color="#C44E52", label=f"role-playing (n={len(proj_individual)})", edgecolor="black")
    ax.hist(proj_default_individual, bins=40, alpha=0.7, color="#4C72B0", label=f"default Assistant (n={len(proj_default_individual)})", edgecolor="black")
    ax.axvline(proj_individual.mean(), linestyle="--", color="#C44E52", linewidth=1.5)
    ax.axvline(proj_default_individual.mean(), linestyle="--", color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("projection onto Assistant Axis (raw activation units, Gemma-2-27b-it layer 22)")
    ax.set_ylabel("rollouts")
    ax.set_title(f"Validation: default-Assistant rollouts project higher than role-playing rollouts\n"
                 f"means: default={proj_default_individual.mean():.0f}, role={proj_individual.mean():.0f}, "
                 f"separation={proj_default_individual.mean()-proj_individual.mean():.0f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "axis_validation_histogram.png", dpi=150)
    print(f"Saved: {OUT / 'axis_validation_histogram.png'}")

    # === Plot 2: per-role mean projection (which roles are most/least Assistant-like) ===
    role_proj_means = []
    role_labels_sorted = []
    for ri in role_ids:
        idx = [i for i, r in enumerate(role_for_individual) if r == ri]
        if idx:
            role_proj_means.append(np.mean([proj_individual[i] for i in idx]))
            role_labels_sorted.append(ri)
    sort_idx = np.argsort(role_proj_means)
    sorted_roles = [role_labels_sorted[i] for i in sort_idx]
    sorted_means = [role_proj_means[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(sorted_roles))
    bars = ax.barh(y, sorted_means, color="#55A868", edgecolor="black")
    ax.axvline(proj_default_individual.mean(), linestyle="--", color="#4C72B0", linewidth=2,
               label=f"default Assistant mean ({proj_default_individual.mean():.0f})")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_roles, fontsize=8)
    ax.set_xlabel("mean projection onto Assistant Axis")
    ax.set_title("Per-role projection on the Assistant Axis (Gemma-2-27b-it layer 22)\n"
                 "Lower (left) = more role-playing-like; Higher (right) = closer to default Assistant\n"
                 "Lu et al. predict: 'helpful-human' roles project higher than 'mystical/non-human' roles")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "axis_per_role_projection.png", dpi=150)
    print(f"Saved: {OUT / 'axis_per_role_projection.png'}")


if __name__ == "__main__":
    main()
