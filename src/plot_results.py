"""Builds results/no_signal.png from results/metrics.json.

Run:  python src/plot_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))

names = ["Stratified\ndummy baseline", "Logistic\nregression", "Random forest\n(tuned)"]
values = [m["baseline_dummy_cv_auc"], m["logistic_regression_cv_auc"], m["random_forest_cv_auc"]]
colors = ["#94a3b8", "#0e1a2b", "#14b8a6"]

fig, ax = plt.subplots(figsize=(7.5, 4.4))
bars = ax.bar(names, values, color=colors, width=0.6)
ax.axhline(0.5, ls="--", color="grey", lw=1)
ax.text(2.35, 0.505, "chance (AUC = 0.5)", ha="right", fontsize=9, color="grey")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylim(0.4, 0.6)
ax.set_ylabel("5-fold CV ROC AUC")
ax.set_title("Fixing the plan_type bug doesn't fix the model —\nthe dataset itself carries almost no signal")
plt.tight_layout()
plt.savefig(ROOT / "results" / "no_signal.png", dpi=150, bbox_inches="tight")
print("Saved results/no_signal.png")
