"""
M4 — Performance Hyperparameter Tuning
DATA3888 Group 8

The tutor asked to improve predictive performance via hyperparameter tuning.
This script runs a grid search over the parameters that most directly affect
forecast quality (QLIKE and MSE), rather than just model order selection.

What we tune and why:
─────────────────────────────────────────────────────────────────────────────
1. Error distribution  (Normal vs Student-t vs Skewed-t)
   WHY: Our pred-vs-actual plots show systematic over-prediction. Normal-GARCH
   inflates variance estimates to compensate for fat tails it can't model.
   Student-t models the heavy tails directly, which should reduce this bias.

2. Forecast horizon / sim_steps  (4, 8, 16, 30)
   WHY: We validate against 4 buckets but simulate 30 steps. Longer horizons
   accumulate variance and inflate predictions. Matching horizon to the val
   window length is a natural candidate for reducing over-prediction.

3. RV simulation filter bounds  (floor, ceiling)
   WHY: The current floor=1e-4 and ceiling=0.05 are hardcoded. The ceiling
   discards high-vol simulations, biasing the mean RV downward for volatile
   time_ids; the floor discards low-vol simulations, biasing it upward for
   quiet time_ids. Tuning these directly affects QLIKE/MSE.

4. Solver tolerance (ftol)
   WHY: Tighter convergence (1e-9 vs 1e-7) can shift parameter estimates
   on short series (n=16), which flows through to forecast quality.

Outputs:
  perf_tuning_garch.csv       — full grid results for GARCH
  perf_tuning_egarchx.csv     — full grid results for EGARCH-X
  perf_tuning_summary.png     — heatmaps + bar charts of QLIKE/MSE by config
  best_garch_eval.csv         — eval results using best GARCH config
  best_egarchx_eval.csv       — eval results using best EGARCH-X config
  best_comparison.png         — boxplots comparing default vs best configs
─────────────────────────────────────────────────────────────────────────────
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from arch import arch_model
from arch.univariate.volatility import EGARCH, _common_names
from arch.univariate import ARX, Normal, StudentsT, SkewStudent
from arch.utility.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV  = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"
OUTPUT_DIR = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\m4_outputs"

STOCK_ID = 1
SCALE    = 1000.0
N_TRAIN  = 16
N_VAL    = 4
N_SIM    = 500      # reduced for tuning speed; use 1000 in final run

# ── Tuning grids ──────────────────────────────────────────────────────────────

# 1. Error distributions to try
DISTRIBUTIONS = ["normal", "t", "skewt"]

# 2. Forecast horizons (steps simulated per path)
SIM_STEPS_GRID = [4, 8, 16, 30]

# 3. RV filter bounds [floor, ceiling]
RV_FILTERS = [
    (1e-5, 0.10),   # wider: keep more paths
    (1e-4, 0.05),   # current default
    (1e-4, 0.02),   # tighter ceiling
    (1e-3, 0.05),   # higher floor
]

# 4. Solver tolerances
FTOL_GRID = [1e-7, 1e-9]

# How many time_ids to use in the tuning grid (speed vs accuracy tradeoff)
# Use all of them for the final best-config evaluation
TUNE_SAMPLE = 150

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EGARCH-X class (same as main script)
# ══════════════════════════════════════════════════════════════════════════════

class EGARCHX(EGARCH):
    """EGARCH(p,o,q)-X: adds a log bid-ask spread term to the variance equation."""

    def __init__(self, spread, p=1, o=1, q=1):
        super().__init__(p=p, o=o, q=q)
        self._spread     = np.asarray(spread, dtype=float)
        self._num_params = 1 + p + o + q + 1

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q) + ["delta[spread]"]

    def starting_values(self, resids):
        return np.append(super().starting_values(resids), 0.05)

    def bounds(self, resids):
        return super().bounds(resids) + [(-10.0, 10.0)]

    def constraints(self):
        A = np.zeros((1, self._num_params))
        b = np.zeros(1)
        return A, b

    def compute_variance(self, parameters, resids, sigma2, backcast, var_bounds):
        p, o, q     = self.p, self.o, self.q
        nobs        = len(resids)
        omega       = parameters[0]
        alphas      = parameters[1       : 1+p]
        gammas      = parameters[1+p     : 1+p+o]
        betas       = parameters[1+p+o   : 1+p+o+q]
        delta       = parameters[-1]
        spread      = self._spread
        norm_const  = np.sqrt(2.0 / np.pi)
        lnsigma2    = np.zeros(nobs)
        lnsigma2[0] = float(np.atleast_1d(backcast)[0])
        for t in range(1, nobs):
            v = omega
            for j in range(p):
                if t-1-j >= 0:
                    sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                    v += alphas[j] * (abs(resids[t-1-j] / sj) - norm_const)
            for j in range(o):
                if t-1-j >= 0:
                    sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                    v += gammas[j] * (resids[t-1-j] / sj)
            for j in range(q):
                if t-1-j >= 0:
                    v += betas[j] * lnsigma2[t-1-j]
            if t-1 < len(spread):
                v += delta * np.log(max(abs(spread[t-1]), 1e-10) ** 2)
            lnsigma2[t] = v
        sigma2[:] = np.clip(np.exp(lnsigma2), var_bounds[:, 0], var_bounds[:, 1])
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q    = self.p, self.o, self.q
        parameters = np.asarray(parameters, dtype=float)
        omega  = parameters[0]
        alphas = parameters[1       : 1+p]
        gammas = parameters[1+p     : 1+p+o]
        betas  = parameters[1+p+o   : 1+p+o+q]
        delta  = parameters[-1]
        nc     = np.sqrt(2.0 / np.pi)
        ss     = np.where(np.abs(self._spread) < 1e-10, 1e-10, np.abs(self._spread))
        spread_offset = delta * float(np.mean(np.log(ss ** 2)))
        beta_sum  = float(np.sum(betas))
        init_ls   = (omega + spread_offset) / (1.0 - beta_sum) \
                    if abs(beta_sum) < 1.0 else omega + spread_offset
        e   = rng(nobs + burn)
        ae  = np.abs(e)
        ls  = np.full(nobs + burn, init_ls)
        lag = max(p, o, q, 1)
        for t in range(lag, nobs + burn):
            v = omega + spread_offset
            for j in range(p): v += alphas[j] * (ae[t-1-j] - nc)
            for j in range(o): v += gammas[j] * e[t-1-j]
            for j in range(q): v += betas[j] * ls[t-1-j]
            ls[t] = v
        sig = np.sqrt(np.maximum(np.exp(ls[burn:]), 1e-16))
        return pd.DataFrame({"data": e[burn:] * sig, "volatility": sig, "errors": e[burn:]})


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def qlike(pred, actual):
    if pred <= 0 or actual <= 0 or not np.isfinite(pred):
        return np.nan
    return np.log(pred ** 2) + actual ** 2 / pred ** 2


def simulate_rv(res, vol_obj=None, n_sim=500, n_steps=30,
                scale=1000.0, rv_floor=1e-4, rv_ceil=0.05):
    """Simulate n_sim paths, return mean RV with configurable filter bounds."""
    last_sig2 = float(res.conditional_volatility[-1] ** 2)
    valid_rvs = []
    for _ in range(n_sim):
        try:
            if vol_obj is not None:
                vp  = res.params[vol_obj.parameter_names()].values
                sim = vol_obj.simulate(vp, nobs=n_steps,
                                       rng=np.random.standard_normal, burn=0)
            else:
                sim = res.model.simulate(res.params, nobs=n_steps,
                                         burn=0, initial_value=last_sig2)
            r  = sim["data"].values / scale
            rv = np.sqrt(np.sum(r ** 2))
            if np.isfinite(rv) and rv_floor <= rv <= rv_ceil:
                valid_rvs.append(rv)
        except Exception:
            continue
    return float(np.mean(valid_rvs)) if valid_rvs else np.nan


def evaluate_preds(preds_dict, vol_val_dict):
    records = []
    for tid, pred in preds_dict.items():
        if not np.isfinite(pred):
            continue
        for actual in vol_val_dict[tid]["RV"].values:
            q = qlike(pred, actual)
            if np.isfinite(q):
                records.append({
                    "time_id": tid, "pred_RV": pred, "actual": actual,
                    "QLIKE": q, "MSE": (actual - pred) ** 2,
                })
    return pd.DataFrame(records)


def get_dist_object(dist_name):
    """Return the arch distribution object for a given name string."""
    return {"normal": Normal(), "t": StudentsT(), "skewt": SkewStudent()}.get(dist_name, Normal())


def garch_is_stationary(res):
    try:
        return (res.params.get("alpha[1]", 1) + res.params.get("beta[1]", 1)) < 0.999
    except Exception:
        return False


def egarchx_passes_sanity_check(res):
    if res.convergence_flag != 0:
        return False
    try:
        return (abs(res.params.get("gamma[1]",      99)) < 5 and
                abs(res.params.get("delta[spread]", 99)) < 5 and
                abs(res.params.get("beta[1]",        1)) < 0.9999)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Load data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df     = pd.read_csv(INPUT_CSV)
stock1 = df[df["stock_id"] == STOCK_ID].copy()
stock1 = stock1.sort_values(["time_id", "time_bucket"]).reset_index(drop=True)

vol_train, vol_val = {}, {}
for tid in sorted(stock1["time_id"].unique()):
    buckets = stock1[stock1["time_id"] == tid].sort_values("time_bucket").reset_index(drop=True)
    if len(buckets) < N_TRAIN + N_VAL:
        continue
    vol_train[tid] = buckets.iloc[:N_TRAIN].copy()
    vol_val[tid]   = buckets.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()

time_IDs = list(vol_train.keys())
tune_IDs = time_IDs[:TUNE_SAMPLE]
print(f"  {len(time_IDs)} total time_ids, using {len(tune_IDs)} for tuning grid")


# ══════════════════════════════════════════════════════════════════════════════
#  GARCH Performance Tuning Grid
#
#  For each combination of (distribution, sim_steps, rv_filter, ftol):
#    1. Fit GARCH(1,1) on tune_IDs
#    2. Simulate forecasts
#    3. Compute median QLIKE and median MSE
#  This tells us which config actually minimises loss, not just BIC.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── GARCH Performance Tuning ─────────────────────────────────────────")

# We tune distribution and ftol together (they affect the fit),
# then tune sim_steps and rv_filter (they affect forecast generation).
# Full factorial would be 3×4×4×2 = 96 combos — we run dist×ftol first
# to find the best fit config, then tune the forecast params on top.

# Phase A: tune distribution + ftol (fit quality)
print("Phase A: tuning distribution and solver tolerance ...")
garch_fit_results = []

for dist, ftol in itertools.product(DISTRIBUTIONS, FTOL_GRID):
    models  = {}
    n_conv  = 0
    for tid in tune_IDs:
        rv_s = vol_train[tid]["RV"].values
        try:
            am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                             vol="GARCH", p=1, q=1, dist=dist)
            res = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 1000, "ftol": ftol})
            if res.convergence_flag == 0 and garch_is_stationary(res):
                models[tid] = res
                n_conv += 1
        except Exception:
            continue

    # Quick forecast eval with default sim params
    preds = {}
    for tid, res in models.items():
        pred = simulate_rv(res, n_sim=N_SIM, n_steps=N_VAL,
                           scale=SCALE, rv_floor=1e-4, rv_ceil=0.05)
        preds[tid] = pred

    eval_df = evaluate_preds(preds, {k: vol_val[k] for k in preds if k in vol_val})
    if eval_df.empty:
        continue
    per_tid = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()

    garch_fit_results.append({
        "dist": dist, "ftol": ftol,
        "n_converged":  n_conv,
        "median_QLIKE": per_tid["QLIKE"].median(),
        "mean_QLIKE":   per_tid["QLIKE"].mean(),
        "median_MSE":   per_tid["MSE"].median(),
        "mean_MSE":     per_tid["MSE"].mean(),
    })
    print(f"  dist={dist:<8}  ftol={ftol:.0e}  "
          f"n={n_conv:3d}  median QLIKE={per_tid['QLIKE'].median():.4f}  "
          f"median MSE={per_tid['MSE'].median():.2e}")

garch_fit_df = pd.DataFrame(garch_fit_results).sort_values("median_QLIKE")
best_dist_g  = garch_fit_df.iloc[0]["dist"]
best_ftol_g  = garch_fit_df.iloc[0]["ftol"]
print(f"\n  → Best fit config: dist={best_dist_g}, ftol={best_ftol_g:.0e}")

# Phase B: tune sim_steps + rv_filter using best fit config
print("\nPhase B: tuning forecast horizon and RV filter bounds ...")
garch_forecast_results = []

# Re-fit models with best config on tune_IDs
best_garch_models = {}
for tid in tune_IDs:
    rv_s = vol_train[tid]["RV"].values
    try:
        am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                         vol="GARCH", p=1, q=1, dist=best_dist_g)
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": best_ftol_g})
        if res.convergence_flag == 0 and garch_is_stationary(res):
            best_garch_models[tid] = res
    except Exception:
        continue

for n_steps, (rv_floor, rv_ceil) in itertools.product(SIM_STEPS_GRID, RV_FILTERS):
    preds = {}
    for tid, res in best_garch_models.items():
        pred = simulate_rv(res, n_sim=N_SIM, n_steps=n_steps,
                           scale=SCALE, rv_floor=rv_floor, rv_ceil=rv_ceil)
        preds[tid] = pred

    eval_df = evaluate_preds(preds, {k: vol_val[k] for k in preds if k in vol_val})
    if eval_df.empty:
        continue
    per_tid = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()

    garch_forecast_results.append({
        "dist": best_dist_g, "ftol": best_ftol_g,
        "sim_steps": n_steps,
        "rv_floor": rv_floor, "rv_ceil": rv_ceil,
        "filter_label": f"[{rv_floor:.0e}, {rv_ceil}]",
        "n_forecasts":  len(per_tid),
        "median_QLIKE": per_tid["QLIKE"].median(),
        "mean_QLIKE":   per_tid["QLIKE"].mean(),
        "median_MSE":   per_tid["MSE"].median(),
        "mean_MSE":     per_tid["MSE"].mean(),
    })
    print(f"  steps={n_steps:2d}  filter=[{rv_floor:.0e},{rv_ceil}]  "
          f"median QLIKE={per_tid['QLIKE'].median():.4f}  "
          f"median MSE={per_tid['MSE'].median():.2e}")

garch_forecast_df = pd.DataFrame(garch_forecast_results).sort_values("median_QLIKE")

# Combine all GARCH tuning results
all_garch_tuning = pd.concat([garch_fit_df, garch_forecast_df], ignore_index=True)
all_garch_tuning.to_csv(os.path.join(OUTPUT_DIR, "perf_tuning_garch.csv"), index=False)
print("  Saved: perf_tuning_garch.csv")

best_garch_row  = garch_forecast_df.iloc[0]
best_steps_g    = int(best_garch_row["sim_steps"])
best_floor_g    = float(best_garch_row["rv_floor"])
best_ceil_g     = float(best_garch_row["rv_ceil"])
print(f"\n  → Best forecast config: steps={best_steps_g}, "
      f"filter=[{best_floor_g:.0e}, {best_ceil_g}]")


# ══════════════════════════════════════════════════════════════════════════════
#  EGARCH-X Performance Tuning Grid (same structure)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── EGARCH-X Performance Tuning ──────────────────────────────────────")

print("Phase A: tuning distribution and solver tolerance ...")
egarchx_fit_results = []

for dist, ftol in itertools.product(DISTRIBUTIONS, FTOL_GRID):
    models = {}
    n_conv = 0
    dist_obj = get_dist_object(dist)
    for tid in tune_IDs:
        rv_s = vol_train[tid]["RV"].values
        sp_s = vol_train[tid]["BidAskSpread_mean"].values
        try:
            vol_obj = EGARCHX(sp_s * SCALE, p=1, o=1, q=1)
            am      = ARX(rv_s * SCALE, lags=1)
            am.volatility   = vol_obj
            am.distribution = type(dist_obj)()   # fresh instance per fit
            res = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 1000, "ftol": ftol})
            if egarchx_passes_sanity_check(res):
                models[tid] = (res, vol_obj)
                n_conv += 1
        except Exception:
            continue

    preds = {}
    for tid, (res, vol_obj) in models.items():
        pred = simulate_rv(res, vol_obj=vol_obj, n_sim=N_SIM, n_steps=N_VAL,
                           scale=SCALE, rv_floor=1e-4, rv_ceil=0.05)
        preds[tid] = pred

    eval_df = evaluate_preds(preds, {k: vol_val[k] for k in preds if k in vol_val})
    if eval_df.empty:
        continue
    per_tid = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()

    egarchx_fit_results.append({
        "dist": dist, "ftol": ftol,
        "n_converged":  n_conv,
        "median_QLIKE": per_tid["QLIKE"].median(),
        "mean_QLIKE":   per_tid["QLIKE"].mean(),
        "median_MSE":   per_tid["MSE"].median(),
        "mean_MSE":     per_tid["MSE"].mean(),
    })
    print(f"  dist={dist:<8}  ftol={ftol:.0e}  "
          f"n={n_conv:3d}  median QLIKE={per_tid['QLIKE'].median():.4f}  "
          f"median MSE={per_tid['MSE'].median():.2e}")

egarchx_fit_df = pd.DataFrame(egarchx_fit_results).sort_values("median_QLIKE")
best_dist_ex   = egarchx_fit_df.iloc[0]["dist"]
best_ftol_ex   = egarchx_fit_df.iloc[0]["ftol"]
print(f"\n  → Best fit config: dist={best_dist_ex}, ftol={best_ftol_ex:.0e}")

print("\nPhase B: tuning forecast horizon and RV filter bounds ...")
egarchx_forecast_results = []

best_egarchx_models = {}
for tid in tune_IDs:
    rv_s = vol_train[tid]["RV"].values
    sp_s = vol_train[tid]["BidAskSpread_mean"].values
    try:
        vol_obj  = EGARCHX(sp_s * SCALE, p=1, o=1, q=1)
        am       = ARX(rv_s * SCALE, lags=1)
        am.volatility   = vol_obj
        am.distribution = get_dist_object(best_dist_ex).__class__()
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": best_ftol_ex})
        if egarchx_passes_sanity_check(res):
            best_egarchx_models[tid] = (res, vol_obj)
    except Exception:
        continue

for n_steps, (rv_floor, rv_ceil) in itertools.product(SIM_STEPS_GRID, RV_FILTERS):
    preds = {}
    for tid, (res, vol_obj) in best_egarchx_models.items():
        pred = simulate_rv(res, vol_obj=vol_obj, n_sim=N_SIM, n_steps=n_steps,
                           scale=SCALE, rv_floor=rv_floor, rv_ceil=rv_ceil)
        preds[tid] = pred

    eval_df = evaluate_preds(preds, {k: vol_val[k] for k in preds if k in vol_val})
    if eval_df.empty:
        continue
    per_tid = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()

    egarchx_forecast_results.append({
        "dist": best_dist_ex, "ftol": best_ftol_ex,
        "sim_steps": n_steps,
        "rv_floor": rv_floor, "rv_ceil": rv_ceil,
        "filter_label": f"[{rv_floor:.0e}, {rv_ceil}]",
        "n_forecasts":  len(per_tid),
        "median_QLIKE": per_tid["QLIKE"].median(),
        "mean_QLIKE":   per_tid["QLIKE"].mean(),
        "median_MSE":   per_tid["MSE"].median(),
        "mean_MSE":     per_tid["MSE"].mean(),
    })
    print(f"  steps={n_steps:2d}  filter=[{rv_floor:.0e},{rv_ceil}]  "
          f"median QLIKE={per_tid['QLIKE'].median():.4f}  "
          f"median MSE={per_tid['MSE'].median():.2e}")

egarchx_forecast_df = pd.DataFrame(egarchx_forecast_results).sort_values("median_QLIKE")
all_egarchx_tuning  = pd.concat([egarchx_fit_df, egarchx_forecast_df], ignore_index=True)
all_egarchx_tuning.to_csv(os.path.join(OUTPUT_DIR, "perf_tuning_egarchx.csv"), index=False)
print("  Saved: perf_tuning_egarchx.csv")

best_egarchx_row = egarchx_forecast_df.iloc[0]
best_steps_ex    = int(best_egarchx_row["sim_steps"])
best_floor_ex    = float(best_egarchx_row["rv_floor"])
best_ceil_ex     = float(best_egarchx_row["rv_ceil"])
print(f"\n  → Best forecast config: steps={best_steps_ex}, "
      f"filter=[{best_floor_ex:.0e}, {best_ceil_ex}]")


# ══════════════════════════════════════════════════════════════════════════════
#  Full evaluation with best configs (all time_ids)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Full Evaluation: Default vs Best Config ──────────────────────────")

# Default config (from main script)
DEFAULT = dict(dist="normal", ftol=1e-7, sim_steps=30,
               rv_floor=1e-4, rv_ceil=0.05)

def run_full_garch(dist, ftol, sim_steps, rv_floor, rv_ceil, label=""):
    models, preds = {}, {}
    for tid in time_IDs:
        rv_s = vol_train[tid]["RV"].values
        try:
            am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                             vol="GARCH", p=1, q=1, dist=dist)
            res = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 1000, "ftol": ftol})
            if res.convergence_flag == 0 and garch_is_stationary(res):
                models[tid] = res
        except Exception:
            continue
    for tid, res in models.items():
        pred = simulate_rv(res, n_sim=N_SIM, n_steps=sim_steps,
                           scale=SCALE, rv_floor=rv_floor, rv_ceil=rv_ceil)
        preds[tid] = pred
    eval_df = evaluate_preds(preds, vol_val)
    print(f"  GARCH {label}: n={len(eval_df.groupby('time_id'))}  "
          f"median QLIKE={eval_df.groupby('time_id')['QLIKE'].mean().median():.4f}  "
          f"median MSE={eval_df.groupby('time_id')['MSE'].mean().median():.2e}")
    return eval_df

def run_full_egarchx(dist, ftol, sim_steps, rv_floor, rv_ceil, label=""):
    models, preds = {}, {}
    dist_cls = get_dist_object(dist).__class__
    for tid in time_IDs:
        rv_s = vol_train[tid]["RV"].values
        sp_s = vol_train[tid]["BidAskSpread_mean"].values
        try:
            vol_obj = EGARCHX(sp_s * SCALE, p=1, o=1, q=1)
            am      = ARX(rv_s * SCALE, lags=1)
            am.volatility   = vol_obj
            am.distribution = dist_cls()
            res = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 1000, "ftol": ftol})
            if egarchx_passes_sanity_check(res):
                models[tid] = (res, vol_obj)
        except Exception:
            continue
    for tid, (res, vol_obj) in models.items():
        pred = simulate_rv(res, vol_obj=vol_obj, n_sim=N_SIM, n_steps=sim_steps,
                           scale=SCALE, rv_floor=rv_floor, rv_ceil=rv_ceil)
        preds[tid] = pred
    eval_df = evaluate_preds(preds, vol_val)
    print(f"  EGARCHX {label}: n={len(eval_df.groupby('time_id'))}  "
          f"median QLIKE={eval_df.groupby('time_id')['QLIKE'].mean().median():.4f}  "
          f"median MSE={eval_df.groupby('time_id')['MSE'].mean().median():.2e}")
    return eval_df

print("Running default config ...")
garch_default_eval   = run_full_garch(**DEFAULT,  label="[default]")
egarchx_default_eval = run_full_egarchx(**DEFAULT, label="[default]")

print("\nRunning best tuned config ...")
garch_best_eval   = run_full_garch(
    dist=best_dist_g, ftol=best_ftol_g,
    sim_steps=best_steps_g, rv_floor=best_floor_g, rv_ceil=best_ceil_g,
    label="[tuned]"
)
egarchx_best_eval = run_full_egarchx(
    dist=best_dist_ex, ftol=best_ftol_ex,
    sim_steps=best_steps_ex, rv_floor=best_floor_ex, rv_ceil=best_ceil_ex,
    label="[tuned]"
)

garch_best_eval.to_csv(os.path.join(OUTPUT_DIR, "best_garch_eval.csv"), index=False)
egarchx_best_eval.to_csv(os.path.join(OUTPUT_DIR, "best_egarchx_eval.csv"), index=False)
print("  Saved: best_garch_eval.csv, best_egarchx_eval.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating plots ...")

# ── Plot 1: Tuning grid heatmaps ──────────────────────────────────────────────
# Show how QLIKE varies across sim_steps × rv_filter for the best dist/ftol

def make_heatmap_data(forecast_df, metric="median_QLIKE"):
    """Pivot forecast results into a heatmap-friendly matrix."""
    pivot = forecast_df.pivot_table(
        index="sim_steps", columns="filter_label", values=metric, aggfunc="mean"
    )
    return pivot

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hyperparameter Tuning — Median QLIKE and MSE\n"
             "(more negative QLIKE = better; lower MSE = better)", fontsize=13)

for row, (fdf, model_label) in enumerate([
    (garch_forecast_df,   "GARCH(1,1)"),
    (egarchx_forecast_df, "EGARCH-X"),
]):
    if fdf.empty:
        continue
    for col, metric in enumerate(["median_QLIKE", "median_MSE"]):
        ax = axes[row][col]
        try:
            pivot = make_heatmap_data(fdf, metric)
            im = ax.imshow(pivot.values, aspect="auto",
                           cmap="RdYlGn" if "QLIKE" in metric else "RdYlGn_r")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_xlabel("RV filter [floor, ceiling]", fontsize=9)
            ax.set_ylabel("Sim steps", fontsize=9)
            ax.set_title(f"{metric.replace('_', ' ').title()} — {model_label}", fontsize=10)
            plt.colorbar(im, ax=ax)
            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        fmt = f"{val:.3f}" if "QLIKE" in metric else f"{val:.2e}"
                        ax.text(j, i, fmt, ha="center", va="center",
                                fontsize=7, color="black")
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n({e})", ha="center", va="center",
                    transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "perf_tuning_heatmaps.png"), dpi=150)
plt.close()
print("  Saved: perf_tuning_heatmaps.png")

# ── Plot 2: Distribution comparison bar chart ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Error Distribution Comparison — Median QLIKE by Distribution\n"
             "(evaluated with default sim_steps and filter bounds)", fontsize=12)

for ax, (fdf, model_label) in zip(axes, [
    (garch_fit_df,   "GARCH(1,1)"),
    (egarchx_fit_df, "EGARCH-X"),
]):
    if fdf.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue
    colors = {"normal": "#4C72B0", "t": "#DD8452", "skewt": "#55A868"}
    for _, row_data in fdf.iterrows():
        ax.bar(f"{row_data['dist']}\nftol={row_data['ftol']:.0e}",
               row_data["median_QLIKE"],
               color=colors.get(row_data["dist"], "grey"), alpha=0.8, edgecolor="white")
    ax.set_title(model_label, fontsize=11)
    ax.set_ylabel("Median QLIKE (more negative = better)")
    ax.set_xlabel("Distribution + solver tolerance")
    # Mark the best
    best_val = fdf["median_QLIKE"].min()
    ax.axhline(best_val, color="red", linestyle="--", linewidth=1,
               label=f"Best: {best_val:.4f}")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "perf_tuning_distributions.png"), dpi=150)
plt.close()
print("  Saved: perf_tuning_distributions.png")

# ── Plot 3: Default vs Best boxplot comparison ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Default vs Tuned Config — Stock {STOCK_ID}\n"
             f"Tuned GARCH: dist={best_dist_g}, steps={best_steps_g}, "
             f"filter=[{best_floor_g:.0e},{best_ceil_g}]\n"
             f"Tuned EGARCH-X: dist={best_dist_ex}, steps={best_steps_ex}, "
             f"filter=[{best_floor_ex:.0e},{best_ceil_ex}]",
             fontsize=10)

for row, (default_eval, best_eval, model_label, color_default, color_best) in enumerate([
    (garch_default_eval,   garch_best_eval,   "GARCH(1,1)",      "#4C72B0", "#1A4480"),
    (egarchx_default_eval, egarchx_best_eval, "EGARCH(1,1,1)-X", "#DD8452", "#8B2500"),
]):
    for col, metric in enumerate(["QLIKE", "MSE"]):
        ax = axes[row][col]
        if default_eval.empty or best_eval.empty:
            continue

        default_per_tid = default_eval.groupby("time_id")[metric].mean().dropna()
        best_per_tid    = best_eval.groupby("time_id")[metric].mean().dropna()

        bp = ax.boxplot(
            [default_per_tid.values, best_per_tid.values],
            patch_artist=True, vert=False, widths=0.5,
            labels=["Default", "Tuned"],
        )
        bp["boxes"][0].set_facecolor(color_default)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(color_best)
        bp["boxes"][1].set_alpha(0.7)

        med_default = default_per_tid.median()
        med_best    = best_per_tid.median()
        improvement = (med_best - med_default) / abs(med_default) * 100

        ax.axvline(med_default, color=color_default, linestyle="--",
                   linewidth=1.2, label=f"Default median: {med_default:.4f}")
        ax.axvline(med_best,    color=color_best,    linestyle="--",
                   linewidth=1.2, label=f"Tuned median:   {med_best:.4f}")

        direction = "↑ worse" if (metric == "QLIKE" and improvement > 0) or \
                                  (metric == "MSE"   and improvement > 0) else "↓ better"
        ax.set_title(f"{metric} — {model_label}\n"
                     f"Change: {improvement:+.1f}%  {direction}", fontsize=10)
        ax.set_xlabel(f"{metric} loss per time_id")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_comparison.png"), dpi=150)
plt.close()
print("  Saved: best_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def summarise(eval_df):
    if eval_df.empty:
        return {"median_QLIKE": np.nan, "median_MSE": np.nan, "n": 0}
    pt = eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
    return {"median_QLIKE": pt["QLIKE"].median(),
            "median_MSE":   pt["MSE"].median(),
            "n": len(pt)}

rows = [
    ("GARCH   default", DEFAULT["dist"], DEFAULT["ftol"], DEFAULT["sim_steps"],
     DEFAULT["rv_floor"], DEFAULT["rv_ceil"], summarise(garch_default_eval)),
    ("GARCH   tuned",   best_dist_g, best_ftol_g, best_steps_g,
     best_floor_g, best_ceil_g, summarise(garch_best_eval)),
    ("EGARCH-X default", DEFAULT["dist"], DEFAULT["ftol"], DEFAULT["sim_steps"],
     DEFAULT["rv_floor"], DEFAULT["rv_ceil"], summarise(egarchx_default_eval)),
    ("EGARCH-X tuned",   best_dist_ex, best_ftol_ex, best_steps_ex,
     best_floor_ex, best_ceil_ex, summarise(egarchx_best_eval)),
]

print("\n" + "="*90)
print(f"  {'Model':<20} {'dist':<8} {'ftol':<8} {'steps':<7} "
      f"{'filter':<16} {'med QLIKE':<12} {'med MSE':<12} {'n'}")
print("="*90)
for name, dist, ftol, steps, fl, fc, s in rows:
    print(f"  {name:<20} {dist:<8} {ftol:<8.0e} {steps:<7d} "
          f"[{fl:.0e},{fc}]{'':<6} {s['median_QLIKE']:<12.4f} "
          f"{s['median_MSE']:<12.2e} {s['n']}")
print("="*90)

print(f"\n✓ Tuning complete. Outputs saved to: {OUTPUT_DIR}")
print()
print("  perf_tuning_garch.csv")
print("  perf_tuning_egarchx.csv")
print("  perf_tuning_heatmaps.png")
print("  perf_tuning_distributions.png")
print("  best_garch_eval.csv")
print("  best_egarchx_eval.csv")
print("  best_comparison.png")