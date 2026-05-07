"""
LightGBM Volatility Forecasting — All Stocks
DATA3888 Group 8

Input  : optiver_aggregated.csv  (stock_id, time_id, time_bucket, WAP_mean, BidAskSpread_mean, volatility)
Setup  : buckets 1-16 → features,  buckets 17-20 → target (mean volatility)
Output : lgbm_outputs/
    lgbm_eval_results.csv     — per (stock_id, time_id) predictions + QLIKE + MSE
    lgbm_per_stock.csv        — mean QLIKE/MSE per stock  (cross-reference with liquidity)
    lgbm_per_timeid.csv       — mean QLIKE/MSE per time_id
    lgbm_feature_importance.csv
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV  = r"d:\USYD\DATA3888\group_asm\optiver_aggregated.csv"
OUTPUT_DIR = r"d:\USYD\DATA3888\group_asm\DATA3888G08\lgbm_outputs"

N_TRAIN      = 16
N_VAL        = 4
N_FOLDS      = 5      # cross-validation folds across time_ids
RANDOM_STATE = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def qlike(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    pred   = np.maximum(pred,   1e-10)
    actual = np.maximum(actual, 1e-10)
    return np.log(pred ** 2) + actual ** 2 / pred ** 2

def engineer_features(grp: pd.DataFrame) -> pd.Series:
    """Aggregate buckets 1-16 into a flat feature vector per (stock_id, time_id)."""
    vol    = grp["volatility"].values
    spread = grp["BidAskSpread_mean"].values
    wap    = grp["WAP_mean"].values

    return pd.Series({
        # volatility
        "vol_mean":        vol.mean(),
        "vol_std":         vol.std(),
        "vol_first":       vol[0],
        "vol_last":        vol[-1],
        "vol_trend":       vol[-1] - vol[0],
        "vol_max":         vol.max(),
        "vol_min":         vol.min(),
        "vol_first3_mean": vol[:3].mean(),
        "vol_last3_mean":  vol[-3:].mean(),
        "vol_mid_mean":    vol[6:10].mean(),
        # bid-ask spread
        "spread_mean":     spread.mean(),
        "spread_std":      spread.std(),
        "spread_last":     spread[-1],
        "spread_max":      spread.max(),
        "spread_trend":    spread[-1] - spread[0],
        # WAP
        "wap_mean":        wap.mean(),
        "wap_std":         wap.std(),
        "wap_trend":       wap[-1] - wap[0],
        # ratio features
        "spread_vol_ratio": spread.mean() / max(vol.mean(), 1e-10),
    })


print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["stock_id", "time_id", "time_bucket"]).reset_index(drop=True)

print(f"  Total rows: {len(df):,}")
print(f"  Stocks: {df['stock_id'].nunique()}")
print(f"  Time IDs: {df['time_id'].nunique()}")

# Filter to complete sessions only (must have all 20 buckets)
bucket_counts = df.groupby(["stock_id", "time_id"])["time_bucket"].count()
complete      = bucket_counts[bucket_counts == N_TRAIN + N_VAL].index
df = df.set_index(["stock_id", "time_id"]).loc[complete].reset_index()
print(f"  Complete sessions: {len(complete):,}")

train_raw = df[df["time_bucket"] <= N_TRAIN]
val_raw   = df[df["time_bucket"] >  N_TRAIN]

# ── Feature engineering ───────────────────────────────────────────────────────

print("Engineering features ...")
features = (
    train_raw
    .groupby(["stock_id", "time_id"])[["volatility", "BidAskSpread_mean", "WAP_mean"]]
    .apply(engineer_features)
    .reset_index()
)

# Target: mean volatility over validation buckets 17-20
targets = (
    val_raw
    .groupby(["stock_id", "time_id"])["volatility"]
    .mean()
    .reset_index()
    .rename(columns={"volatility": "target_vol"})
)

# Per-session mean spread (used in output for liquidity cross-reference)
session_spread = (
    train_raw
    .groupby(["stock_id", "time_id"])["BidAskSpread_mean"]
    .mean()
    .reset_index()
    .rename(columns={"BidAskSpread_mean": "mean_spread"})
)

stock_global = (
    train_raw
    .groupby("stock_id")
    .agg(
        stock_mean_vol   =("volatility",        "mean"),
        stock_std_vol    =("volatility",        "std"),
        stock_mean_spread=("BidAskSpread_mean", "mean"),
    )
    .reset_index()
)

data = features.merge(targets, on=["stock_id", "time_id"])
data = data.merge(session_spread, on=["stock_id", "time_id"])
data = data.merge(stock_global, on="stock_id")

print(f"  Dataset shape: {data.shape}")

# ── Cross-validation across time_ids ─────────────────────────────────────────
# Each session is independent — the real train/test split is buckets 1-16 (features)
# vs buckets 17-20 (target). We use k-fold CV across time_ids so every session
# gets an out-of-sample prediction from a model that never saw it during training.

FEATURE_COLS = [c for c in data.columns
                if c not in ["stock_id", "time_id", "target_vol", "mean_spread", "fold"]]

time_ids    = np.array(sorted(data["time_id"].unique()))
fold_labels = np.arange(len(time_ids)) % N_FOLDS  # assign fold 0-4 round-robin
tid_to_fold = dict(zip(time_ids, fold_labels))
data["fold"] = data["time_id"].map(tid_to_fold)
    
print(f"\n{N_FOLDS}-fold cross-validation across {len(time_ids):,} time_ids")
print(f"  Total sessions: {len(data):,}  ({data['stock_id'].nunique()} stocks each fold)")

all_preds = []
feat_imps  = []

for fold in range(N_FOLDS):
    train_data = data[data["fold"] != fold]
    val_data   = data[data["fold"] == fold]

    X_tr, y_tr = train_data[FEATURE_COLS], train_data["target_vol"]
    X_val       = val_data[FEATURE_COLS]

    model = lgb.LGBMRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(X_tr, y_tr)

    fold_result = val_data[["stock_id", "time_id", "mean_spread", "target_vol"]].copy()
    fold_result["pred_vol"] = model.predict(X_val)
    all_preds.append(fold_result)

    feat_imps.append(model.feature_importances_)
    print(f"  Fold {fold+1}/{N_FOLDS} done — {len(val_data):,} sessions predicted")

# Combine all out-of-sample predictions
results = pd.concat(all_preds, ignore_index=True)
results["QLIKE"] = qlike(results["pred_vol"].values, results["target_vol"].values)
results["MSE"]   = (results["pred_vol"] - results["target_vol"]) ** 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full eval results — every session, every stock
eval_out = results[["stock_id", "time_id", "mean_spread", "pred_vol", "target_vol", "QLIKE", "MSE"]]
eval_out.to_csv(os.path.join(OUTPUT_DIR, "lgbm_eval_results.csv"), index=False)
print("\nSaved: lgbm_eval_results.csv")

# Per-stock summary
per_stock = (
    results.groupby("stock_id")
    .agg(
        mean_QLIKE   = ("QLIKE",       "mean"),
        median_QLIKE = ("QLIKE",       "median"),
        mean_MSE     = ("MSE",         "mean"),
        mean_spread  = ("mean_spread", "mean"),
        n_sessions   = ("time_id",     "count"),
    )
    .reset_index()
    .sort_values("mean_QLIKE")
)
per_stock.to_csv(os.path.join(OUTPUT_DIR, "lgbm_per_stock.csv"), index=False)
print("Saved: lgbm_per_stock.csv")

# Per-time_id summary
per_tid = (
    results.groupby("time_id")
    .agg(mean_QLIKE=("QLIKE", "mean"), mean_MSE=("MSE", "mean"))
    .reset_index()
)
per_tid.to_csv(os.path.join(OUTPUT_DIR, "lgbm_per_timeid.csv"), index=False)
print("Saved: lgbm_per_timeid.csv")

# Feature importance averaged across folds
feat_imp = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": np.mean(feat_imps, axis=0)})
    .sort_values("importance", ascending=False)
)
feat_imp.to_csv(os.path.join(OUTPUT_DIR, "lgbm_feature_importance.csv"), index=False)
print("Saved: lgbm_feature_importance.csv")


print("\n-- Evaluation Summary (all sessions, out-of-sample) --")
print(f"  Total sessions : {len(results):,}")
print(f"  Median QLIKE   : {results['QLIKE'].median():.4f}")
print(f"  Mean   QLIKE   : {results['QLIKE'].mean():.4f}")
print(f"  Median MSE     : {results['MSE'].median():.2e}")
print(f"  Mean   MSE     : {results['MSE'].mean():.2e}")

print("\n-- Best 5 stocks (lowest mean QLIKE) --")
print(per_stock.head().to_string(index=False))

print("\n-- Worst 5 stocks (highest mean QLIKE) --")
print(per_stock.tail().to_string(index=False))

print("\n-- Feature importance (top 10) --")
print(feat_imp.head(10).to_string(index=False))
                                                    