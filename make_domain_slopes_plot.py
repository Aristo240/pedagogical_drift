"""Clean horizontal bar chart of per-domain drift slopes with significance markers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "full" / "summary.json"
OUT = ROOT / "results" / "full" / "domain_slopes_bar.png"


s = json.loads(SUMMARY.read_text())
trends = s["domain_trends"]

records = []
for dom, info in trends.items():
    slope = info.get("slope")
    p = info.get("p_value")
    n = info.get("n")
    if slope is None or p is None:
        continue
    records.append((dom, slope, p, n))

records.sort(key=lambda x: x[1])  # by slope, low to high
domains = [r[0] for r in records]
slopes = np.array([r[1] for r in records])
ps = np.array([r[2] for r in records])
ns = [r[3] for r in records]

fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(domains))
colors = ["#C44E52" if p < 0.05 else "#888888" if p < 0.10 else "#cccccc" for p in ps]
bars = ax.barh(y, slopes, color=colors, edgecolor="black")

# Add significance stars
for i, (slope, p, n) in enumerate(zip(slopes, ps, ns)):
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.10 else ""
    label = f"  {slope:+.3f} (p={p:.3f}, n={n}) {star}"
    align = "left" if slope >= 0 else "right"
    x_pos = slope + (0.005 if slope >= 0 else -0.005)
    ax.text(x_pos, i, label, va="center", ha=align, fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels(domains, fontsize=9)
ax.set_xlabel("drift slope (change in pedagogy score per tutor turn)")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Per-domain pedagogical drift slope (Phase 1, n=37 conversations)\n"
             "Red = p < 0.05;  grey = p ≥ 0.10")
ax.set_xlim(-0.20, 0.10)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved: {OUT}")
