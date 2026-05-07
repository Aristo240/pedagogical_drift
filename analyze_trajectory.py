"""Trajectory analysis: per-turn pedagogical-quality means, by-condition contrast,
inter-judge agreement, and a linear-trend test on the aggregate score.

Inputs
------
A directory containing tutor_turn_grades.jsonl (output of grade_pedagogy.py).

Outputs (in the same directory)
-------------------------------
- per_turn_means.csv       : table of mean aggregate per (turn_index, persona, judge)
- inter_judge_agreement.json: per-marker quadratic-weighted Cohen's kappa per judge pair
- trajectory.png           : aggregate vs. turn_index, by persona, with linear fit
- markers_by_turn.png      : per-marker means over turn_index
- summary.json             : headline numbers
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

MARKERS = ["scaffolding", "error_responsiveness", "productive_struggle", "conceptual_depth"]


def _to_int(x):
    if x == "NA" or x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def load_grades(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "scores" not in r:
            continue
        row = {
            "conversation_id": r["conversation_id"],
            "topic_id": r.get("topic_id"),
            "domain": r.get("domain"),
            "persona": r.get("persona"),
            "turn": r["tutor_turn_index"],
            "judge": r["judge_slug"],
            "aggregate": r.get("aggregate"),
        }
        for m in MARKERS:
            row[m] = _to_int(r["scores"].get(m, {}).get("score"))
        rows.append(row)
    return pd.DataFrame(rows)


def linear_trend(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope of y on x, with t-test on slope. Returns slope, p, n."""
    if len(x) < 3 or np.std(x) == 0:
        return {"n": int(len(x)), "slope": None, "intercept": None, "p_value": None,
                "note": "insufficient data or zero variance in x"}
    # Manual OLS for one independent var (avoid scipy dep)
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    sxy = ((x - x_mean) * (y - y_mean)).sum()
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    yhat = intercept + slope * x
    resid = y - yhat
    s2 = (resid ** 2).sum() / (n - 2) if n > 2 else float("nan")
    se_slope = (s2 / sxx) ** 0.5 if sxx > 0 else float("nan")
    if se_slope == 0 or np.isnan(se_slope):
        p = None
    else:
        from math import erf, sqrt
        t_stat = slope / se_slope
        # 2-sided p from normal approx (n usually small; this is a rough p)
        p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return {"n": int(n), "slope": float(slope), "intercept": float(intercept),
            "se_slope": float(se_slope) if se_slope is not None else None,
            "p_value": float(p) if p is not None else None}


def inter_judge_agreement(df: pd.DataFrame) -> dict:
    judges = sorted(df["judge"].unique())
    summary: dict = {"judges": judges, "pairs": {}}
    if len(judges) < 2:
        summary["note"] = "fewer than 2 judges"
        return summary
    # pivot to one row per (conversation_id, turn) and column per (judge, marker)
    wide = df.pivot_table(index=["conversation_id", "turn"], columns="judge",
                           values=MARKERS + ["aggregate"], aggfunc="first")
    for j1, j2 in combinations(judges, 2):
        per_marker = {}
        for m in MARKERS:
            try:
                a = wide[(m, j1)].dropna()
                b = wide[(m, j2)].dropna()
            except KeyError:
                per_marker[m] = {"n": 0, "kappa_quadratic": None, "exact_agreement": None}
                continue
            common = wide[[(m, j1), (m, j2)]].dropna()
            n = len(common)
            if n < 5:
                per_marker[m] = {"n": int(n), "kappa_quadratic": None,
                                 "exact_agreement": None,
                                 "note": "fewer than 5 non-NA pairs"}
                continue
            ai = common[(m, j1)].astype(int)
            bi = common[(m, j2)].astype(int)
            if ai.nunique() < 2 or bi.nunique() < 2:
                per_marker[m] = {"n": int(n), "kappa_quadratic": None,
                                 "exact_agreement": float((ai == bi).mean()),
                                 "note": "degenerate"}
                continue
            per_marker[m] = {
                "n": int(n),
                "kappa_quadratic": float(cohen_kappa_score(ai, bi, weights="quadratic")),
                "exact_agreement": float((ai == bi).mean()),
            }
        # aggregate-score Pearson r
        agg_pair = wide[[("aggregate", j1), ("aggregate", j2)]].dropna()
        if len(agg_pair) >= 5:
            r = float(agg_pair.corr().iloc[0, 1])
        else:
            r = None
        summary["pairs"][f"{j1} vs {j2}"] = {
            "n_overlap": int(len(agg_pair)),
            "aggregate_pearson_r": r,
            "per_marker": per_marker,
        }
    return summary


def plot_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"engaged": "#2ca02c", "offloading": "#d62728"}
    for persona, sub in df.groupby("persona"):
        per_turn = sub.groupby("turn")["aggregate"].agg(["mean", "sem", "count"]).reset_index()
        ax.errorbar(per_turn["turn"], per_turn["mean"],
                    yerr=per_turn["sem"].fillna(0),
                    marker="o", capsize=3, label=f"{persona} (n_turns={int(per_turn['count'].sum())})",
                    color=palette.get(persona, "#1f77b4"))
        # overlay linear fit
        x = per_turn["turn"].astype(float).values
        y = per_turn["mean"].astype(float).values
        if len(x) >= 3 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, ls="--", alpha=0.5,
                    color=palette.get(persona, "#1f77b4"))
    ax.set_xlabel("tutor turn index (1 = first tutor reply)")
    ax.set_ylabel("aggregate pedagogy score (1-5)")
    ax.set_ylim(1, 5)
    ax.set_title("Pedagogical quality by tutor turn, by student persona")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_markers(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    palette = {"engaged": "#2ca02c", "offloading": "#d62728"}
    for ax, m in zip(axes.flat, MARKERS):
        for persona, sub in df.groupby("persona"):
            per_turn = sub.groupby("turn")[m].mean().reset_index()
            ax.plot(per_turn["turn"], per_turn[m], marker="o",
                    label=persona, color=palette.get(persona, "#1f77b4"))
        ax.set_title(m.replace("_", " "))
        ax.set_ylim(1, 5)
        ax.set_xlabel("tutor turn")
        ax.set_ylabel("score")
    axes[0, 0].legend()
    fig.suptitle("Per-marker pedagogy by turn, by persona")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", dest="indir", required=True,
                    help="Directory with tutor_turn_grades.jsonl")
    ap.add_argument("--outdir", default=None,
                    help="Output dir. Defaults to --in-dir.")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    grades_path = indir / "tutor_turn_grades.jsonl"
    if not grades_path.exists():
        raise SystemExit(f"Missing {grades_path}. Run grade_pedagogy.py first.")

    df = load_grades(grades_path)
    if df.empty:
        raise SystemExit(f"No usable rows in {grades_path}.")

    # Per-turn means table
    grouped = df.groupby(["turn", "persona", "judge"])["aggregate"].agg(["mean", "std", "count"]).reset_index()
    grouped.to_csv(outdir / "per_turn_means.csv", index=False)
    print("=== Per-turn aggregate means ===")
    print(grouped.to_string(index=False))

    # Linear trend on aggregate, per persona (averaged over judges so each
    # turn-conversation contributes one observation)
    avg_judges = df.groupby(["conversation_id", "turn", "persona"])["aggregate"].mean().reset_index()
    trends: dict = {}
    print("\n=== Linear trend (aggregate ~ turn) by persona ===")
    for persona, sub in avg_judges.groupby("persona"):
        x = sub["turn"].astype(float).values
        y = sub["aggregate"].astype(float).values
        trend = linear_trend(x, y)
        trends[persona] = trend
        slope = trend.get("slope")
        p = trend.get("p_value")
        print(f"  {persona:12s}: slope={slope:+.3f} per turn  p={p}  n={trend['n']}")

    # Inter-judge agreement
    agreement = inter_judge_agreement(df)
    (outdir / "inter_judge_agreement.json").write_text(json.dumps(agreement, indent=2))
    print("\n=== Inter-judge agreement ===")
    print(json.dumps(agreement, indent=2))

    # Plots (use per-turn averaged-over-judges for cleanliness)
    plot_trajectory(avg_judges.assign(persona=avg_judges["persona"]), outdir / "trajectory.png")
    df_avg_markers = df.groupby(["conversation_id", "turn", "persona"])[MARKERS].mean().reset_index()
    plot_markers(df_avg_markers, outdir / "markers_by_turn.png")

    summary = {
        "n_grades": int(len(df)),
        "n_conversations": int(df["conversation_id"].nunique()),
        "n_tutor_turns": int(df.groupby(["conversation_id", "turn"]).ngroups),
        "n_judges": int(df["judge"].nunique()),
        "trends": trends,
        "agreement": agreement,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {outdir/'per_turn_means.csv'}")
    print(f"Saved: {outdir/'inter_judge_agreement.json'}")
    print(f"Saved: {outdir/'trajectory.png'}")
    print(f"Saved: {outdir/'markers_by_turn.png'}")
    print(f"Saved: {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
