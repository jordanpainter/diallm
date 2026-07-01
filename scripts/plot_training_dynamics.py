"""
Four training-dynamics figures for the DiaLLM paper.

Figures produced (saved to diallm-rl/figures/):
  1. fig1_dialect_density_bar.pdf   — grouped bar: dialect_gen_mean vs chosen baseline, broad thread
  2. fig2_dialect_vs_cosine.pdf     — scatter: dialect density vs cosine, narrow thread, coloured by variety
  3. fig3_dpo_margin_steps.pdf      — DPO reward margin over training steps, by variety
  4. fig4_reward_heatmap.pdf        — heatmap: final total reward, method × variety-model

Usage:
    pip install pandas matplotlib seaborn
    python scripts/plot_training_dynamics.py
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
FINALS_CSV = os.path.join(ROOT_DIR, "wandb_finals.csv")
TRENDS_CSV = os.path.join(ROOT_DIR, "wandb_trends.csv")
OUT_DIR    = os.path.join(ROOT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

MODEL_LABELS = {
    "gemma": "Gemma 3-4B",
    "llama": "Llama 3.1-8B",
    "qwen":  "Qwen3-8B",
}
VARIETY_LABELS = {
    "aus":  "en-AU",
    "brit": "en-UK",
    "ind":  "en-IN",
}
VARIETY_COLOURS = {
    "aus":  "#2196F3",   # blue
    "brit": "#4CAF50",   # green
    "ind":  "#FF5722",   # orange-red
}
METHOD_COLOURS = {
    "grpo": "#7B61FF",
    "gspo": "#00BCD4",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def extract_variety(run_name: str) -> str:
    """Pull 'aus' / 'brit' / 'ind' from a narrow run name, else 'all'."""
    for v in ("aus", "brit", "ind"):
        if v in run_name.lower():
            return v
    return "all"

def extract_model(run_name: str) -> str:
    """Pull 'gemma' / 'llama' / 'qwen' from run name."""
    for m in ("gemma", "llama", "qwen"):
        if m in run_name.lower():
            return m
    return run_name

# ── load data ─────────────────────────────────────────────────────────────────
finals = pd.read_csv(FINALS_CSV)
trends = pd.read_csv(TRENDS_CSV)

# Derived columns
finals["variety"]   = finals["run"].apply(extract_variety)
finals["model_key"] = finals["run"].apply(extract_model)

trends["variety"]   = trends["run"].apply(extract_variety)
trends["model_key"] = trends["run"].apply(extract_model)

# Short column aliases
D_GEN   = "train/train/reward_raw/dialect_gen_mean"
D_CHO   = "train/train/reward_raw/dialect_chosen_mean"
COS     = "train/train/reward_raw/cosine_mean"
TOT     = "train/train/reward_total/mean"
MARGINS = "train/rewards/margins"


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Grouped bar: dialect_gen_mean vs chosen baseline (broad thread)
# ══════════════════════════════════════════════════════════════════════════════

broad_rl = finals[
    finals["algorithm"].isin(["grpo", "gspo"]) &
    (finals["variety"] == "all")
].copy()

models   = ["gemma", "llama", "qwen"]
methods  = ["grpo", "gspo"]
n_models = len(models)

fig, ax = plt.subplots(figsize=(9, 4.5))

bar_width = 0.18
group_gap = 0.15
positions  = np.arange(n_models) * (len(methods) * bar_width + group_gap)

for i, method in enumerate(methods):
    sub = broad_rl[broad_rl["algorithm"] == method].set_index("model_key")
    gen_vals = [sub.loc[m, D_GEN]  if m in sub.index else np.nan for m in models]
    x_pos    = positions + i * bar_width

    bars = ax.bar(x_pos, gen_vals, bar_width,
                  label=method.upper(),
                  color=METHOD_COLOURS[method],
                  alpha=0.88, edgecolor="white", linewidth=0.5)

# Chosen baseline per model (same for all methods — taken from GSPO row)
gspo_sub = broad_rl[broad_rl["algorithm"] == "gspo"].set_index("model_key")
cho_vals = [gspo_sub.loc[m, D_CHO] if m in gspo_sub.index else np.nan for m in models]

# Draw baseline as horizontal line segments spanning each model group
for k, (cho, base_x) in enumerate(zip(cho_vals, positions)):
    span_left  = base_x - bar_width * 0.3
    span_right = base_x + len(methods) * bar_width + bar_width * 0.1
    ax.hlines(cho, span_left, span_right,
              colors="#E53935", linestyles="--", linewidth=1.8)

# Axis labels & ticks
tick_centres = positions + (len(methods) - 1) * bar_width / 2
ax.set_xticks(tick_centres)
ax.set_xticklabels([MODEL_LABELS[m] for m in models])
ax.set_ylabel("Dialect feature density (raw)")
ax.set_title("Fig 1 — Broad-thread dialect feature density:\ngenerated outputs vs. SFT chosen reference")

# Legend
method_patches = [mpatches.Patch(color=METHOD_COLOURS[m], alpha=0.88, label=m.upper())
                  for m in methods]
ref_line = plt.Line2D([0], [0], color="#E53935", linestyle="--", linewidth=1.8,
                      label="SFT chosen reference")
ax.legend(handles=method_patches + [ref_line], loc="lower right")
ax.set_ylim(0, 3.6)
ax.yaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
out1 = os.path.join(OUT_DIR, "fig1_dialect_density_bar.pdf")
fig.savefig(out1, bbox_inches="tight")
fig.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Scatter: dialect density vs cosine similarity, narrow thread
# ══════════════════════════════════════════════════════════════════════════════

narrow_rl = finals[
    finals["algorithm"].isin(["grpo", "gspo"]) &
    (finals["variety"] != "all")
].copy()

fig, ax = plt.subplots(figsize=(7, 5))

for variety, grp in narrow_rl.groupby("variety"):
    colour = VARIETY_COLOURS[variety]
    label  = VARIETY_LABELS[variety]
    # Differentiate GRPO vs GSPO by marker
    for method, mgrp in grp.groupby("algorithm"):
        marker = "o" if method == "gspo" else "^"
        ax.scatter(
            mgrp[D_GEN], mgrp[COS],
            c=colour, marker=marker,
            s=90, alpha=0.85, edgecolors="white", linewidth=0.5,
            label=f"{label} ({method.upper()})" if variety == list(narrow_rl["variety"].unique())[0] else "_nolegend_"
        )

# Annotate with model initials
for _, row in narrow_rl.iterrows():
    initial = row["model_key"][0].upper()  # G / L / Q
    ax.annotate(initial, (row[D_GEN], row[COS]),
                fontsize=7, ha="center", va="center",
                color="black", alpha=0.7)

# Quadrant guide lines
ax.axvline(x=2.68, color="gray", linestyle=":", linewidth=1, alpha=0.6,
           label="Chosen baseline (dialect)")
ax.axhline(y=0.6,  color="gray", linestyle="--", linewidth=1, alpha=0.6,
           label="Cosine 0.6 threshold")

ax.set_xlabel("Generated dialect feature density")
ax.set_ylabel("Cosine similarity (meaning preservation)")
ax.set_title("Fig 2 — Narrow-thread: dialect density vs. meaning preservation\n(by variety; G=Gemma, L=Llama, Q=Qwen; ○=GSPO △=GRPO)")

# Custom legend: colour per variety + marker per method
variety_patches = [mpatches.Patch(color=VARIETY_COLOURS[v], label=VARIETY_LABELS[v])
                   for v in ["aus", "brit", "ind"]]
gspo_marker = plt.Line2D([0], [0], marker="o", color="gray", linestyle="None",
                          markersize=8, label="GSPO")
grpo_marker = plt.Line2D([0], [0], marker="^", color="gray", linestyle="None",
                          markersize=8, label="GRPO")
ax.legend(handles=variety_patches + [gspo_marker, grpo_marker], loc="upper left",
          fontsize=9)
ax.yaxis.grid(True, linestyle=":", alpha=0.4)
ax.xaxis.grid(True, linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
out2 = os.path.join(OUT_DIR, "fig2_dialect_vs_cosine.pdf")
fig.savefig(out2, bbox_inches="tight")
fig.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — DPO reward margin over training steps, by variety
# ══════════════════════════════════════════════════════════════════════════════

dpo_trends = trends[trends["algorithm"] == "dpo"].copy()
# Keep only rows with non-null margins
dpo_trends = dpo_trends[dpo_trends[MARGINS].notna()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
variety_order = ["aus", "brit", "ind"]

for ax, variety in zip(axes, variety_order):
    sub = dpo_trends[dpo_trends["variety"] == variety]
    for model_key, mgrp in sub.groupby("model_key"):
        mgrp_sorted = mgrp.sort_values("step")
        colour = {"gemma": "#9C27B0", "llama": "#F57C00", "qwen": "#0097A7"}.get(model_key, "gray")
        ax.plot(mgrp_sorted["step"], mgrp_sorted[MARGINS],
                label=MODEL_LABELS.get(model_key, model_key),
                color=colour, linewidth=1.8, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(VARIETY_LABELS[variety])
    ax.set_xlabel("Training step")
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Reward margin (chosen − rejected)")
axes[0].legend(fontsize=9)

fig.suptitle("Fig 3 — DPO reward margin over training steps by variety", y=1.02)
fig.tight_layout()
out3 = os.path.join(OUT_DIR, "fig3_dpo_margin_steps.pdf")
fig.savefig(out3, bbox_inches="tight")
fig.savefig(out3.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out3}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Heatmap: final total reward, method × variety-model
# ══════════════════════════════════════════════════════════════════════════════

narrow_rl2 = finals[
    finals["algorithm"].isin(["grpo", "gspo"]) &
    (finals["variety"] != "all")
].copy()

narrow_rl2["col_label"] = (
    narrow_rl2["variety"].map(VARIETY_LABELS) + "\n" +
    narrow_rl2["model_key"].map(MODEL_LABELS)
)

pivot = narrow_rl2.pivot_table(
    index="algorithm", columns="col_label",
    values=TOT, aggfunc="mean"
)
pivot.index = pivot.index.str.upper()

# Reorder columns: variety groups (aus, brit, ind) × model (gemma, llama, qwen)
ordered_cols = []
for v in ["en-AU", "en-UK", "en-IN"]:
    for m in ["Gemma 3-4B", "Llama 3.1-8B", "Qwen3-8B"]:
        col = f"{v}\n{m}"
        if col in pivot.columns:
            ordered_cols.append(col)
pivot = pivot[[c for c in ordered_cols if c in pivot.columns]]

fig, ax = plt.subplots(figsize=(11, 2.8))
sns.heatmap(
    pivot,
    ax=ax,
    cmap="RdYlGn",
    center=0,
    annot=True, fmt=".2f",
    annot_kws={"size": 9},
    linewidths=0.5,
    cbar_kws={"label": "Final total reward", "shrink": 0.8},
)
ax.set_xlabel("")
ax.set_ylabel("Method")
ax.set_title("Fig 4 — Final total reward at convergence: narrow thread (GRPO vs GSPO)")
ax.tick_params(axis="x", labelsize=8)

fig.tight_layout()
out4 = os.path.join(OUT_DIR, "fig4_reward_heatmap.pdf")
fig.savefig(out4, bbox_inches="tight")
fig.savefig(out4.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out4}")
plt.close(fig)

print("\nAll figures saved to", OUT_DIR)
