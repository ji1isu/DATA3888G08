"""
LightGBM Post-hoc Analysis
DATA3888 Group 8

Reads pre-generated outputs from lgbm_model.py and produces:

Analysis 1 — Stock Clustering
  Cluster the 112 stocks by their volatility/liquidity profile
  Evaluate QLIKE per cluster to identify which stock types are harder to forecast

Analysis 2 — Volatility Regime Detection
  Flag each session as high/low volatility or stressed (spread spike)
  Compare QLIKE across regimes to see where the model struggles

Outputs (lgbm_outputs/):
  cluster_stock_profiles.csv    — per-stock cluster assignments + profile
  cluster_qlike_summary.csv     — mean QLIKE per cluster
  regime_qlike_summary.csv      — mean QLIKE per regime
  plot_stock_clusters.png       — scatter: spread vs vol coloured by cluster
  plot_cluster_qlike.png        — boxplot: QLIKE distribution per cluster
  plot_regime_qlike.png         — boxplot: QLIKE distribution per regime
  plot_regime_spread_vol.png    — scatter: spread vs vol coloured by regime
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV    = r"d:\USYD\DATA3888\group_asm\optiver_aggregated.csv"
EVAL_CSV     = r"d:\USYD\DATA3888\group_asm\DATA3888G08\lgbm_outputs\lgbm_eval_results.csv"
PER_STOCK    = r"d:\USYD\DATA3888\group_asm\DATA3888G08\lgbm_outputs\lgbm_per_stock.csv"
OUTPUT_DIR   = r"d:\USYD\DATA3888\group_asm\DATA3888G08\lgbm_outputs"

N_CLUSTERS   = 4     # number of stock clusters
N_TRAIN      = 16    # training buckets
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data ...")
eval_df     = pd.read_csv(EVAL_CSV)
per_stock   = pd.read_csv(PER_STOCK)

print("Loading aggregated data (train buckets only) ...")
agg = pd.read_csv(INPUT_CSV)
train_raw = agg[agg["time_bucket"] <= N_TRAIN].copy()
print(f"  Eval sessions : {len(eval_df):,}")
print(f"  Stocks        : {per_stock['stock_id'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis 1 — Stock Clustering
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Analysis 1: Stock Clustering --")

# Build a profile for each stock using bucket 1-16 statistics across all sessions
stock_profiles = (
    train_raw
    .groupby("stock_id")
    .agg(
        mean_vol      = ("volatility",        "mean"),
        std_vol       = ("volatility",        "std"),
        mean_spread   = ("BidAskSpread_mean",  "mean"),
        std_spread    = ("BidAskSpread_mean",  "std"),
        mean_wap_std  = ("WAP_mean",           "std"),
    )
    .reset_index()
)

# Standardise before clustering
profile_features = ["mean_vol", "std_vol", "mean_spread", "std_spread", "mean_wap_std"]
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(stock_profiles[profile_features])

# K-means clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
stock_profiles["cluster"] = kmeans.fit_predict(X_scaled)

# Sort clusters by mean_spread so cluster 0 = most liquid, N-1 = most illiquid
cluster_order = (
    stock_profiles.groupby("cluster")["mean_spread"]
    .mean()
    .sort_values()
    .reset_index()
    .reset_index()
    .rename(columns={"index": "new_cluster", "cluster": "old_cluster"})
)
remap = dict(zip(cluster_order["old_cluster"], cluster_order["new_cluster"]))
stock_profiles["cluster"] = stock_profiles["cluster"].map(remap)

# Attach cluster to eval results
eval_df = eval_df.merge(stock_profiles[["stock_id", "cluster"]], on="stock_id", how="left")

# Cluster summary
cluster_summary = (
    eval_df.groupby("cluster")
    .agg(
        n_stocks     = ("stock_id",   "nunique"),
        n_sessions   = ("time_id",    "count"),
        mean_QLIKE   = ("QLIKE",      "mean"),
        median_QLIKE = ("QLIKE",      "median"),
        mean_MSE     = ("MSE",        "mean"),
    )
    .reset_index()
)

# Attach mean profile stats per cluster
profile_cluster = (
    stock_profiles.groupby("cluster")[profile_features]
    .mean()
    .reset_index()
)
cluster_summary = cluster_summary.merge(profile_cluster, on="cluster")
cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_qlike_summary.csv"), index=False)

stock_profiles = stock_profiles.merge(
    per_stock[["stock_id", "mean_QLIKE", "mean_MSE"]], on="stock_id", how="left"
)
stock_profiles.to_csv(os.path.join(OUTPUT_DIR, "cluster_stock_profiles.csv"), index=False)

print("  Cluster summary:")
print(cluster_summary[["cluster", "n_stocks", "mean_QLIKE", "mean_spread"]].to_string(index=False))

# ── Plot 1: Scatter — spread vs vol coloured by cluster ──────────────────────

colors = cm.tab10(np.linspace(0, 0.4, N_CLUSTERS))
fig, ax = plt.subplots(figsize=(8, 5))
for c in sorted(stock_profiles["cluster"].unique()):
    sub = stock_profiles[stock_profiles["cluster"] == c]
    ax.scatter(sub["mean_spread"], sub["mean_vol"],
               label=f"Cluster {c} (n={len(sub)})",
               color=colors[c], s=60, alpha=0.8)
ax.set_xlabel("Mean Bid-Ask Spread (liquidity proxy)")
ax.set_ylabel("Mean Volatility")
ax.set_title("Stock Clusters by Liquidity and Volatility Profile")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_stock_clusters.png"), dpi=150)
plt.close()
print("  Saved: plot_stock_clusters.png")

# ── Plot 2: Boxplot — QLIKE per cluster ──────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
cluster_data = [eval_df[eval_df["cluster"] == c]["QLIKE"].dropna().values
                for c in sorted(eval_df["cluster"].unique())]
bp = ax.boxplot(cluster_data, patch_artist=True, showfliers=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels([f"Cluster {c}\n(n={cluster_summary.iloc[c]['n_stocks']:.0f} stocks)"
                    for c in sorted(eval_df["cluster"].unique())])
ax.set_ylabel("QLIKE (higher = worse)")
ax.set_title("Forecast Error (QLIKE) by Stock Cluster\nCluster 0 = most liquid, higher = more illiquid")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_cluster_qlike.png"), dpi=150)
plt.close()
print("  Saved: plot_cluster_qlike.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis 2 — Volatility Regime Detection
# ══════════════════════════════════════════════════════════════════════════════

print("\n-- Analysis 2: Volatility Regime Detection --")

# Compute per-session regime features from buckets 1-16
base = (
    train_raw
    .groupby(["stock_id", "time_id"])
    .agg(mean_vol=("volatility", "mean"), mean_spread=("BidAskSpread_mean", "mean"))
    .reset_index()
)
early = (
    train_raw[train_raw["time_bucket"] <= 4]
    .groupby(["stock_id", "time_id"])["BidAskSpread_mean"]
    .mean()
    .reset_index()
    .rename(columns={"BidAskSpread_mean": "spread_early"})
)
late = (
    train_raw[train_raw["time_bucket"] >= 13]
    .groupby(["stock_id", "time_id"])["BidAskSpread_mean"]
    .mean()
    .reset_index()
    .rename(columns={"BidAskSpread_mean": "spread_late"})
)
session_stats = base.merge(early, on=["stock_id", "time_id"]).merge(late, on=["stock_id", "time_id"])

# Regime thresholds (based on median across all sessions)
vol_thresh    = session_stats["mean_vol"].median()
spread_thresh = session_stats["mean_spread"].median()
spike_ratio   = session_stats["spread_late"] / session_stats["spread_early"].replace(0, np.nan)

def assign_regime(row, spike):
    if spike > 1.5:
        return "stressed"        # spread spiked in second half → liquidity deteriorating
    elif row["mean_vol"] >= vol_thresh and row["mean_spread"] >= spread_thresh:
        return "high-vol illiquid"
    elif row["mean_vol"] >= vol_thresh and row["mean_spread"] < spread_thresh:
        return "high-vol liquid"
    elif row["mean_vol"] < vol_thresh and row["mean_spread"] >= spread_thresh:
        return "low-vol illiquid"
    else:
        return "low-vol liquid"

session_stats["regime"] = [
    assign_regime(row, sp)
    for (_, row), sp in zip(session_stats.iterrows(), spike_ratio.values)
]

# Attach regime to eval results
eval_df = eval_df.merge(
    session_stats[["stock_id", "time_id", "regime", "mean_vol"]],
    on=["stock_id", "time_id"], how="left"
)

# Regime summary
regime_summary = (
    eval_df.groupby("regime")
    .agg(
        n_sessions   = ("time_id",     "count"),
        mean_QLIKE   = ("QLIKE",       "mean"),
        median_QLIKE = ("QLIKE",       "median"),
        mean_MSE     = ("MSE",         "mean"),
        mean_vol     = ("mean_vol",    "mean"),
        mean_spread  = ("mean_spread", "mean"),
    )
    .reset_index()
    .sort_values("mean_QLIKE")
)
regime_summary.to_csv(os.path.join(OUTPUT_DIR, "regime_qlike_summary.csv"), index=False)

print("  Regime summary:")
print(regime_summary[["regime", "n_sessions", "mean_QLIKE", "mean_spread", "mean_vol"]].to_string(index=False))

# ── Plot 3: Boxplot — QLIKE per regime ───────────────────────────────────────

regime_order  = regime_summary["regime"].tolist()
regime_colors = {"low-vol liquid": "#2196F3", "high-vol liquid": "#FF9800",
                 "low-vol illiquid": "#9C27B0", "high-vol illiquid": "#F44336",
                 "stressed": "#212121"}

fig, ax = plt.subplots(figsize=(10, 5))
regime_data = [eval_df[eval_df["regime"] == r]["QLIKE"].dropna().values
               for r in regime_order]
bp = ax.boxplot(regime_data, patch_artist=True, showfliers=False)
for patch, regime in zip(bp["boxes"], regime_order):
    patch.set_facecolor(regime_colors.get(regime, "grey"))
    patch.set_alpha(0.7)
ax.set_xticklabels(
    [f"{r}\n(n={regime_summary[regime_summary['regime']==r]['n_sessions'].values[0]:,})"
     for r in regime_order],
    fontsize=8
)
ax.set_ylabel("QLIKE (higher = worse)")
ax.set_title("Forecast Error by Volatility Regime")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_regime_qlike.png"), dpi=150)
plt.close()
print("  Saved: plot_regime_qlike.png")

# ── Plot 4: Scatter — spread vs vol coloured by regime ───────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
for regime, grp in eval_df.groupby("regime"):
    sample = grp.sample(min(2000, len(grp)), random_state=RANDOM_STATE)
    ax.scatter(sample["mean_spread"], sample["mean_vol"],
               label=regime, color=regime_colors.get(regime, "grey"),
               s=5, alpha=0.3)
ax.set_xlabel("Mean Bid-Ask Spread")
ax.set_ylabel("Mean Volatility (buckets 1-16)")
ax.set_title("Session Regimes: Spread vs Volatility")
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_regime_spread_vol.png"), dpi=150)
plt.close()
print("  Saved: plot_regime_spread_vol.png")

print("\nDone. All outputs saved to lgbm_outputs/")
