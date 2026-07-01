"""
Pull compact summaries from all 36 DiaLLM alignment runs.
Outputs two CSVs:
  - wandb_finals.csv  : final-step values for key metrics, one row per run
  - wandb_trends.csv  : values sampled every ~500 steps for trend analysis

Usage:
    pip install wandb pandas
    python scripts/wandb_summary.py
"""

import wandb
import pandas as pd

ENTITY = "jordanpainter"
PROJECTS = [
    "gspo-all-new",
    "gspo-narrow-new",
    "grpo-all",
    "grpo-narrow",
    "dpo-all",
    "dpo-narrow",
]

# Metrics for GSPO / GRPO runs
ONLINE_METRICS = [
    "train/train/reward_total/mean",
    "train/train/reward_total/std",
    "train/train/reward_raw/dialect_gen_mean",
    "train/train/reward_raw/dialect_chosen_mean",
    "train/train/reward_raw/cosine_mean",
    "train/train/reward_raw/comet_mean",
    "train/train/reward_norm/dialect_mean",
    "train/train/reward_norm/cosine_mean",
    "train/train/reward_norm/comet_mean",
    "train/rewards/reward_fn/mean",
    "train/rewards/reward_fn/std",
]

# Metrics for DPO runs
DPO_METRICS = [
    "train/rewards/chosen",
    "train/rewards/rejected",
    "train/rewards/margins",
    "train/rewards/accuracies",
    "train/loss",
    "train/logps/rejected",
]

SAMPLE_STEP = 500

api = wandb.Api()
finals_rows = []
trends_rows = []

for project in PROJECTS:
    is_dpo = "dpo" in project
    metrics = DPO_METRICS if is_dpo else ONLINE_METRICS
    print(f"Fetching {project} ({'DPO' if is_dpo else 'online RL'})...")

    try:
        runs = api.runs(f"{ENTITY}/{project}")
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    for run in runs:
        name = run.name
        config = run.config
        summary = run.summary._json_dict

        base = {
            "project": project,
            "run": name,
            "model": config.get("model_name_or_path", config.get("model", "")).split("/")[-1],
            "algorithm": "dpo" if is_dpo else config.get("algorithm", project.split("-")[0]),
            "dialect": config.get("target_dialect", config.get("dialect", "all")),
            "final_step": summary.get("_step", None),
        }

        # Finals
        row = dict(base)
        for m in metrics:
            row[m] = summary.get(m, None)
        finals_rows.append(row)

        # Trends — sample history at coarse resolution
        try:
            hist = run.history(keys=metrics, samples=200)
            if not hist.empty and "_step" in hist.columns:
                max_step = int(hist["_step"].max())
                for cp in range(0, max_step + 1, SAMPLE_STEP):
                    closest = hist.iloc[(hist["_step"] - cp).abs().argsort()[:1]]
                    if closest.empty:
                        continue
                    t_row = dict(base)
                    t_row["step"] = int(closest["_step"].values[0])
                    for m in metrics:
                        t_row[m] = closest[m].values[0] if m in closest.columns else None
                    trends_rows.append(t_row)
        except Exception as e:
            print(f"  Trend fetch failed for {name}: {e}")

finals_df = pd.DataFrame(finals_rows)
trends_df = pd.DataFrame(trends_rows)

finals_df.to_csv("wandb_finals.csv", index=False)
trends_df.to_csv("wandb_trends.csv", index=False)

print(f"\nDone.")
print(f"  {len(finals_rows)} runs  -> wandb_finals.csv")
print(f"  {len(trends_rows)} rows  -> wandb_trends.csv")
print("\nFinals preview:")
print(finals_df.to_string(index=False))
