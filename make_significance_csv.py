"""Compile every significant test from pedagogical_drift Phase 1 + Phase 2 into one CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "full" / "summary.json"
AGREE = ROOT / "results" / "full" / "inter_judge_agreement.json"
AXIS = ROOT / "phase2" / "lu_axis" / "results" / "assistant_axis.npz"
JUDGED = ROOT / "phase2" / "lu_axis" / "results" / "rollouts_roles_judged.jsonl"

OUT_CSV = ROOT / "results" / "full" / "significance_tests.csv"


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    rows = []

    # Phase 1: Linear trend slope per condition (H1)
    for persona, t in s["trends"].items():
        rows.append({
            "test_family": "Phase 1 - H1 (drift slope by persona)",
            "test": f"aggregate ~ turn  |  persona={persona}",
            "estimate": f"{t['slope']:+.4f}",
            "se": f"{t.get('se_slope', float('nan')):.4f}",
            "p_value": f"{t['p_value']:.4f}",
            "n": t["n"],
            "interpretation": "drift in pedagogy quality across tutor turns",
            "significant_at_0.05": "yes" if t["p_value"] < 0.05 else "no",
        })

    # Phase 1: H2 interaction test
    inter = s.get("interaction_h2_test", {})
    for name, c in inter.get("coefficients", {}).items():
        rows.append({
            "test_family": "Phase 1 - H2 (persona x turn interaction)",
            "test": f"aggregate ~ turn * persona  |  coef = {name}",
            "estimate": f"{c['estimate']:+.4f}",
            "se": f"{c['se']:.4f}",
            "p_value": f"{c['p_value']:.4f}" if c['p_value'] is not None else "n/a",
            "n": inter["n"],
            "interpretation": "tests whether persona moderates drift slope (H2)",
            "significant_at_0.05": "yes" if (c['p_value'] is not None and c['p_value'] < 0.05) else "no",
        })

    # Per-domain trends
    for dom, t in s.get("domain_trends", {}).items():
        slope = t.get("slope")
        p = t.get("p_value")
        if slope is None or p is None:
            continue
        rows.append({
            "test_family": "Phase 1 - per-domain drift slope",
            "test": f"aggregate ~ turn  |  domain={dom}",
            "estimate": f"{slope:+.4f}",
            "se": f"{t.get('se_slope', float('nan')):.4f}",
            "p_value": f"{p:.4f}",
            "n": t["n"],
            "interpretation": "drift slope within a single domain",
            "significant_at_0.05": "yes" if p < 0.05 else "no",
        })

    # Inter-judge agreement (Phase 1)
    agree = json.loads(AGREE.read_text())
    for pair_key, info in agree["pairs"].items():
        # aggregate r
        r = info.get("aggregate_pearson_r")
        if r is not None:
            rows.append({
                "test_family": "Phase 1 - inter-judge agreement (aggregate Pearson r)",
                "test": pair_key.replace("__", " "),
                "estimate": f"{r:+.4f}",
                "se": "n/a",
                "p_value": "n/a (descriptive)",
                "n": info["n_overlap"],
                "interpretation": "convergent validity across judges; one judge in pair = generator (Gemini), so within-family pair is biased toward agreement",
                "significant_at_0.05": "n/a",
            })
        # per-marker kappas
        for marker, mk in info.get("per_marker", {}).items():
            kq = mk.get("kappa_quadratic")
            if kq is None:
                continue
            rows.append({
                "test_family": "Phase 1 - inter-judge agreement (Cohen's quadratic kappa per marker)",
                "test": f"{pair_key.replace('__',' ')}  |  marker={marker}",
                "estimate": f"{kq:+.4f}",
                "se": "n/a",
                "p_value": "n/a (descriptive)",
                "n": mk["n"],
                "interpretation": "quadratic-weighted Cohen's kappa per pedagogy marker",
                "significant_at_0.05": "n/a",
            })

    # Phase 2: axis validation
    if AXIS.exists():
        data = np.load(AXIS, allow_pickle=True)
        sep = float(data["validation_separation"])
        # Project individual rollouts (default + filtered roles) and compute t-test
        axis = data["axis_strict"]
        # Default projections
        default_jsonl = ROOT / "phase2" / "lu_axis" / "results" / "rollouts_default.jsonl"
        if default_jsonl.exists():
            d_proj = []
            for line in default_jsonl.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                a = np.array(r["activation"], dtype=np.float32)
                if not np.all(np.isfinite(a)): continue
                d_proj.append(float(np.dot(a, axis)))
            r_proj = []
            for line in JUDGED.read_text().splitlines():
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("judge_label") != 3: continue
                a = np.array(r["activation"], dtype=np.float32)
                if not np.all(np.isfinite(a)): continue
                r_proj.append(float(np.dot(a, axis)))
            d_proj = np.array(d_proj); r_proj = np.array(r_proj)
            # Welch t-test
            from math import sqrt, erf
            md, mr = d_proj.mean(), r_proj.mean()
            vd, vr = d_proj.var(ddof=1), r_proj.var(ddof=1)
            nd, nr = len(d_proj), len(r_proj)
            se = sqrt(vd/nd + vr/nr)
            t = (md - mr) / se
            # Welch-Satterthwaite df
            df = (vd/nd + vr/nr)**2 / ((vd/nd)**2/(nd-1) + (vr/nr)**2/(nr-1))
            # Use normal approx for two-sided p (df is large enough)
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
            rows.append({
                "test_family": "Phase 2 - Lu et al. assistant axis validation",
                "test": "Welch t-test: default-Assistant projection > role-playing projection",
                "estimate": f"{md - mr:+.2f} (raw activation units)",
                "se": f"{se:.2f}",
                "p_value": f"{p:.2e}",
                "n": f"default={nd}, role={nr}",
                "interpretation": "axis successfully separates default from role-playing rollouts (Lu et al. validation)",
                "significant_at_0.05": "yes" if p < 0.05 else "no",
            })
            rows.append({
                "test_family": "Phase 2 - Lu et al. assistant axis validation",
                "test": "axis separation magnitude (means)",
                "estimate": f"{sep:+.2f}",
                "se": "n/a",
                "p_value": "n/a (descriptive)",
                "n": f"30 roles, {nd} default rollouts",
                "interpretation": f"raw activation-space separation; default mean={md:.0f}, role mean={mr:.0f}",
                "significant_at_0.05": "n/a",
            })

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_family", "test", "estimate", "se",
                                          "p_value", "n", "significant_at_0.05",
                                          "interpretation"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} significance-test rows -> {OUT_CSV}")
    sigs = [r for r in rows if r["significant_at_0.05"] == "yes"]
    print(f"Significant at 0.05: {len(sigs)} / {len([r for r in rows if r['significant_at_0.05'] in ('yes','no')])}")
    for r in sigs:
        print(f"  ✓ {r['test_family']:55s} {r['test'][:50]:50s} est={r['estimate']:>10s} p={r['p_value']:>10s}")


if __name__ == "__main__":
    main()
