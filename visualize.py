"""
Visualize AG News ablation results from best_loss.json files.
Usage: python visualize.py
"""
import json
import pathlib
import collections
import matplotlib.pyplot as plt
import numpy as np

LOGDIR = pathlib.Path("logs_ablation_ag_news")
OUT    = pathlib.Path("results_ag_news.png")

# ── Load all results ──────────────────────────────────────────────────────────
records = []
for f in LOGDIR.rglob("best_loss.json"):
    data = json.loads(f.read_text())
    records.append({
        "method":    data["cfg"]["covariance_mode"],
        "noise_std": data["cfg"]["noise_std"],
        "test_01":   data["test_01"],
        "valid_01":  data["valid_01"],
        "train_01":  data["train_01"],
        "seed":      data["cfg"]["seed"],
    })

# ── Aggregate: mean ± std over seeds ─────────────────────────────────────────
groups = collections.defaultdict(list)
for r in records:
    groups[(r["method"], r["noise_std"])].append(r["test_01"])

methods      = sorted({r["method"] for r in records})
noise_levels = sorted({r["noise_std"] for r in records if r["noise_std"] is not None})
baseline_key = ("isotropic", None)
baseline     = np.mean(groups[baseline_key]) if baseline_key in groups else None

# ── Plot 1 : test error vs sigma (one curve per method) ──────────────────────
colors = {"bineta": "#e74c3c", "isotropic": "#3498db",
          "sq_grads": "#2ecc71", "inv_sq_grads": "#f39c12"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AG News — Ablation sur les méthodes de bruit (TF-IDF + MLP)", fontsize=13)

ax = axes[0]
for method in methods:
    xs, ys, errs = [], [], []
    for sigma in noise_levels:
        vals = groups.get((method, sigma), [])
        if vals:
            xs.append(sigma)
            ys.append(np.mean(vals))
            errs.append(np.std(vals))
    if xs:
        color = colors.get(method, "gray")
        ax.errorbar(range(len(xs)), ys, yerr=errs, marker="o", label=method,
                    color=color, capsize=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{s:.4g}" for s in xs], rotation=30, ha="right")

if baseline is not None:
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
               label=f"baseline (no noise) = {baseline:.4f}")

ax.set_xlabel("noise_std (sigma)")
ax.set_ylabel("Test 0-1 error (lower = better)")
ax.set_title("Erreur test vs sigma")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 2 : best test error per method (bar chart) ──────────────────────────
ax2 = axes[1]
best_per_method = {}
for method in methods:
    all_vals = []
    for sigma in noise_levels:
        all_vals.extend(groups.get((method, sigma), []))
    if all_vals:
        best_per_method[method] = (np.min(all_vals), np.mean(all_vals))

bar_methods = list(best_per_method.keys())
bar_means   = [best_per_method[m][1] for m in bar_methods]
bar_bests   = [best_per_method[m][0] for m in bar_methods]
bar_colors  = [colors.get(m, "gray") for m in bar_methods]

x = np.arange(len(bar_methods))
bars = ax2.bar(x, bar_means, color=bar_colors, alpha=0.7, label="mean over seeds & sigmas")
ax2.scatter(x, bar_bests, color=bar_colors, zorder=5, s=60, marker="D", label="best single run")

if baseline is not None:
    ax2.axhline(baseline, color="black", linestyle="--", linewidth=1.2,
                label=f"baseline = {baseline:.4f}")

ax2.set_xticks(x)
ax2.set_xticklabels(bar_methods, rotation=15)
ax2.set_ylabel("Test 0-1 error")
ax2.set_title("Meilleure erreur par méthode")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Graphique sauvegardé : {OUT}")
plt.show()
