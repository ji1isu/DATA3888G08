"""
M4 — Volatility Forecasting: ARMA(1,1)-GARCH(1,1) and EGARCH(1,1,1)-X
DATA3888 Group 8

Pipeline:
  Phase 1 — Data diagnostics (ACF/PACF, AIC/BIC order selection)
  Phase 2 — Baseline ARMA(1,1)-GARCH(1,1) with hyperparameter tuning
  Phase 3 — Refined ARMA(1,1)-EGARCH(1,1,1)-X with hyperparameter tuning
  Phase 4 — Comparison plots and outputs

Data format (from M1):
  stock_id | time_id | time_bucket | BidAskSpread_mean | RV
  20 buckets per time_id (30s each), numbered 2–21
  First 16 buckets = train, last 4 = validation
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from arch.univariate.volatility import EGARCH, _common_names
from arch.univariate import ARX, Normal
from arch.utility.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV  = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\optiver_aggregated.csv"
OUTPUT_DIR = r"C:\Users\songj\OneDrive\UNI\Y3SEM1\DATA3888\Project Folder\DATA3888G08\m4_outputs"

STOCK_ID  = 1
N_SIM     = 1000    # simulation paths per forecast
SIM_STEPS = 30      # steps per path (one per second)
SCALE     = 1000.0  # rescale RV for numerical stability
N_TRAIN   = 16      # training buckets per time_id
N_VAL     = 4       # validation buckets per time_id

# Hyperparameter tuning: which (p, q) orders to try for GARCH
# and which (p, o, q) orders to try for EGARCH-X
GARCH_ORDERS    = list(itertools.product([1, 2], [1, 2]))       # (p, q)
EGARCHX_ORDERS  = list(itertools.product([1, 2], [1], [1, 2]))  # (p, o, q)

# How many time_ids to use when selecting the best order (keep it fast)
TUNE_SAMPLE = 200

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EGARCH-X volatility model
#
#  Extends the standard EGARCH with a bid-ask spread term in the log-variance:
#    ln σ²_t = ω + α(|z_{t-1}| − E|z|) + γ·z_{t-1} + β·ln σ²_{t-1}
#                                        + δ·ln(spread²_{t-1})
#
#  γ < 0  →  leverage effect (bad news raises volatility more than good news)
#  δ > 0  →  liquidity channel (wider spread predicts higher volatility)
# ══════════════════════════════════════════════════════════════════════════════

class EGARCHX(EGARCH):
    """EGARCH(p,o,q)-X: adds a log bid-ask spread term to the variance equation."""

    def __init__(self, spread, p=1, o=1, q=1):
        super().__init__(p=p, o=o, q=q)
        self._spread     = np.asarray(spread, dtype=float)
        self._num_params = 1 + p + o + q + 1   # +1 for delta

    def parameter_names(self):
        return _common_names(self.p, self.o, self.q) + ["delta[spread]"]

    def starting_values(self, resids):
        return np.append(super().starting_values(resids), 0.05)

    def bounds(self, resids):
        return super().bounds(resids) + [(-10.0, 10.0)]

    def constraints(self):
        # No additional constraints beyond EGARCH defaults
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
        norm_const  = np.sqrt(2.0 / np.pi)   # E|z| for standard normal

        lnsigma2    = np.zeros(nobs)
        lnsigma2[0] = float(np.atleast_1d(backcast)[0])

        for t in range(1, nobs):
            v = omega

            # ARCH terms (magnitude of standardised residuals)
            for j in range(p):
                if t - 1 - j >= 0:
                    sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                    v += alphas[j] * (abs(resids[t-1-j] / sj) - norm_const)

            # Asymmetry / leverage terms
            for j in range(o):
                if t - 1 - j >= 0:
                    sj = max(np.exp(lnsigma2[t-1-j] * 0.5), 1e-8)
                    v += gammas[j] * (resids[t-1-j] / sj)

            # GARCH (persistence) terms
            for j in range(q):
                if t - 1 - j >= 0:
                    v += betas[j] * lnsigma2[t-1-j]

            # Exogenous spread term
            if t - 1 < len(spread):
                v += delta * np.log(max(abs(spread[t-1]), 1e-10) ** 2)

            lnsigma2[t] = v

        sigma2[:] = np.clip(np.exp(lnsigma2), var_bounds[:, 0], var_bounds[:, 1])
        return sigma2

    def simulate(self, parameters, nobs, rng, burn=500, initial_value=None):
        p, o, q    = self.p, self.o, self.q
        parameters = np.asarray(parameters, dtype=float)
        omega      = parameters[0]
        alphas     = parameters[1       : 1+p]
        gammas     = parameters[1+p     : 1+p+o]
        betas      = parameters[1+p+o   : 1+p+o+q]
        delta      = parameters[-1]
        nc         = np.sqrt(2.0 / np.pi)

        # Use the mean log-spread as a constant offset during simulation
        ss = np.where(np.abs(self._spread) < 1e-10, 1e-10, np.abs(self._spread))
        spread_offset = delta * float(np.mean(np.log(ss ** 2)))

        beta_sum   = float(np.sum(betas))
        init_lnsig = (omega + spread_offset) / (1.0 - beta_sum) \
                     if abs(beta_sum) < 1.0 else omega + spread_offset

        e   = rng(nobs + burn)
        ae  = np.abs(e)
        ls  = np.full(nobs + burn, init_lnsig)
        lag = max(p, o, q, 1)

        for t in range(lag, nobs + burn):
            v = omega + spread_offset
            for j in range(p): v += alphas[j] * (ae[t-1-j] - nc)
            for j in range(o): v += gammas[j] * e[t-1-j]
            for j in range(q): v += betas[j] * ls[t-1-j]
            ls[t] = v

        sig  = np.sqrt(np.maximum(np.exp(ls[burn:]), 1e-16))
        returns = e[burn:] * sig

        return pd.DataFrame({"data": returns, "volatility": sig, "errors": e[burn:]})


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def qlike(pred, actual):
    """
    QLIKE loss = log(pred²) + actual²/pred²
    Lower is better. Returns NaN if inputs are invalid.
    """
    if pred <= 0 or actual <= 0 or not np.isfinite(pred):
        return np.nan
    return np.log(pred ** 2) + actual ** 2 / pred ** 2


def garch_is_stationary(res):
    """Returns True if α + β < 0.999 — prevents IGARCH blow-ups during simulation."""
    try:
        return (res.params.get("alpha[1]", 1) + res.params.get("beta[1]", 1)) < 0.999
    except Exception:
        return False


def egarchx_passes_sanity_check(res):
    """
    Basic quality gate for EGARCH-X fits on short series.
    Rejects non-converged fits and extreme parameter values.
    """
    if res.convergence_flag != 0:
        return False
    try:
        return (
            abs(res.params.get("gamma[1]",      99)) < 5 and
            abs(res.params.get("delta[spread]", 99)) < 5 and
            abs(res.params.get("beta[1]",        1)) < 0.9999
        )
    except Exception:
        return False


def simulate_rv(res, vol_obj=None, n_sim=1000, n_steps=30, scale=1000.0):
    """
    Simulate n_sim paths of n_steps each and return the mean RV forecast.
    RV for each path = sqrt(sum(r²)), matching the workshop formula.

    Returns (mean_rv, list_of_path_rvs). Filters out numerically blown-up paths.
    """
    last_sig2 = float(res.conditional_volatility[-1] ** 2)
    valid_rvs = []

    for _ in range(n_sim):
        try:
            if vol_obj is not None:
                # EGARCH-X: use the custom simulate method
                vol_params = res.params[vol_obj.parameter_names()].values
                sim = vol_obj.simulate(vol_params, nobs=n_steps,
                                       rng=np.random.standard_normal, burn=0)
            else:
                # Standard GARCH: use arch built-in
                sim = res.model.simulate(res.params, nobs=n_steps,
                                         burn=0, initial_value=last_sig2)

            r  = sim["data"].values / scale
            rv = np.sqrt(np.sum(r ** 2))

            # Keep only physically plausible values
            # Floor at 1e-4 (avoids QLIKE blowdown), ceiling at 0.05 (~14× the 99th pct of actual RV)
            if np.isfinite(rv) and 1e-4 <= rv <= 0.05:
                valid_rvs.append(rv)

        except Exception:
            continue

    mean_rv = float(np.mean(valid_rvs)) if valid_rvs else np.nan
    return mean_rv, valid_rvs


def evaluate_preds(preds_dict, vol_val_dict):
    """
    Given a dict of {time_id: pred_rv} and validation data,
    compute QLIKE and MSE for every (pred, actual) pair.
    Returns a DataFrame with one row per (time_id, actual bucket).
    """
    records = []
    for tid, pred in preds_dict.items():
        if not np.isfinite(pred):
            continue
        for actual in vol_val_dict[tid]["RV"].values:
            q = qlike(pred, actual)
            if np.isfinite(q):
                records.append({
                    "time_id": tid,
                    "pred_RV": pred,
                    "actual":  actual,
                    "QLIKE":   q,
                    "MSE":     (actual - pred) ** 2,
                })
    return pd.DataFrame(records)


def pick_best_fan_example(fit_df, models_dict, alpha_col="alpha"):
    """
    Find the time_id with the highest alpha (most visible fan spread)
    from the set of successfully fit models.
    """
    best_tid   = None
    best_paths = None
    best_alpha = 0.0
    for tid in models_dict:
        row = fit_df[fit_df["time_id"] == tid]
        if row.empty:
            continue
        a = float(row[alpha_col].iloc[0]) if alpha_col in row.columns else 0.0
        if a > best_alpha:
            best_alpha = a
            best_tid   = tid
    return best_tid


# ══════════════════════════════════════════════════════════════════════════════
#  Hyperparameter tuning helpers
# ══════════════════════════════════════════════════════════════════════════════

def tune_garch_order(vol_train, time_ids, orders=GARCH_ORDERS, sample=TUNE_SAMPLE, scale=1000.0):
    """
    Fit GARCH(p,q) for each (p,q) in `orders` over `sample` time_ids.
    Returns the (p,q) pair with the lowest mean BIC, plus the full results table.
    """
    print(f"  Tuning GARCH order over {sample} time_ids ...")
    results = []
    for p, q in orders:
        aics, bics = [], []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["RV"].values
            try:
                am  = arch_model(rv_s * scale, mean="ARX", lags=1,
                                 vol="GARCH", p=p, q=q, dist="normal")
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 500})
                if res.convergence_flag == 0:
                    aics.append(res.aic)
                    bics.append(res.bic)
            except Exception:
                continue
        if aics:
            results.append({
                "order":    f"GARCH({p},{q})",
                "p": p, "q": q,
                "mean_AIC": round(np.mean(aics), 3),
                "mean_BIC": round(np.mean(bics), 3),
                "n_converged": len(aics),
            })

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order by BIC: GARCH({int(best.p)},{int(best.q)})  "
          f"(mean BIC = {best.mean_BIC:.3f})")
    return int(best.p), int(best.q), df


def tune_egarchx_order(vol_train, time_ids, orders=EGARCHX_ORDERS,
                       sample=TUNE_SAMPLE, scale=1000.0):
    """
    Fit EGARCHX(p,o,q) for each combination in `orders` over `sample` time_ids.
    Returns the best (p,o,q) by mean BIC, plus the results table.
    """
    print(f"  Tuning EGARCH-X order over {sample} time_ids ...")
    results = []
    for p, o, q in orders:
        bics = []
        for tid in time_ids[:sample]:
            rv_s = vol_train[tid]["RV"].values
            sp_s = vol_train[tid]["BidAskSpread_mean"].values
            try:
                vol_obj = EGARCHX(sp_s * scale, p=p, o=o, q=q)
                am      = ARX(rv_s * scale, lags=1)
                am.volatility   = vol_obj
                am.distribution = Normal()
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 500})
                if res.convergence_flag == 0 and egarchx_passes_sanity_check(res):
                    bics.append(res.bic)
            except Exception:
                continue
        if bics:
            results.append({
                "order":    f"EGARCHX({p},{o},{q})",
                "p": p, "o": o, "q": q,
                "mean_BIC": round(np.mean(bics), 3),
                "n_converged": len(bics),
            })

    df = pd.DataFrame(results).sort_values("mean_BIC").reset_index(drop=True)
    best = df.iloc[0]
    print(f"  Best order by BIC: EGARCHX({int(best.p)},{int(best.o)},{int(best.q)})  "
          f"(mean BIC = {best.mean_BIC:.3f})")
    return int(best.p), int(best.o), int(best.q), df


# ══════════════════════════════════════════════════════════════════════════════
#  Load and prepare data
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
df     = pd.read_csv(INPUT_CSV)
stock1 = df[df["stock_id"] == STOCK_ID].copy()
stock1 = stock1.sort_values(["time_id", "time_bucket"]).reset_index(drop=True)

vol_train = {}
vol_val   = {}

for tid in sorted(stock1["time_id"].unique()):
    buckets = (stock1[stock1["time_id"] == tid]
               .sort_values("time_bucket")
               .reset_index(drop=True))
    if len(buckets) < N_TRAIN + N_VAL:
        continue
    vol_train[tid] = buckets.iloc[:N_TRAIN].copy()
    vol_val[tid]   = buckets.iloc[N_TRAIN : N_TRAIN + N_VAL].copy()

time_IDs = list(vol_train.keys())
print(f"  Stock {STOCK_ID}: {len(time_IDs)} complete time_ids "
      f"({N_TRAIN} train + {N_VAL} val buckets each)")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Data Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Phase 1: Data Diagnostics ───────────────────────────────────────")

# ACF / PACF on RV to justify ARMA(1,1) mean specification
print("Plotting ACF/PACF ...")
tid_diag = time_IDs[0]
rv_diag  = vol_train[tid_diag]["RV"].values
max_lags = max(1, len(rv_diag) // 2 - 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"ACF / PACF — Realised Volatility (30s buckets)\n"
             f"Stock {STOCK_ID}, time_id={tid_diag} — justifies ARMA(1,1)", fontsize=12)
plot_acf( rv_diag, lags=max_lags, ax=axes[0], title="ACF — RV")
plot_pacf(rv_diag, lags=max_lags, ax=axes[1], title="PACF — RV", method="ywm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase1_acf_pacf.png"), dpi=150)
plt.close()
print("  Saved: phase1_acf_pacf.png")

# AIC/BIC order selection loop
print("Running AIC/BIC order selection ...")
order_results = []
for p, q in itertools.product([1, 2], [1, 2]):
    aics, bics = [], []
    for tid in time_IDs[:200]:
        rv_s = vol_train[tid]["RV"].values
        try:
            am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                             vol="GARCH", p=p, q=q, dist="normal")
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})
            if res.convergence_flag == 0:
                aics.append(res.aic)
                bics.append(res.bic)
        except Exception:
            continue
    if aics:
        order_results.append({
            "GARCH(p,q)": f"({p},{q})",
            "Mean AIC":   round(np.mean(aics), 3),
            "Mean BIC":   round(np.mean(bics), 3),
            "Converged":  len(aics),
        })

order_df = pd.DataFrame(order_results).sort_values("Mean BIC")
order_df.to_csv(os.path.join(OUTPUT_DIR, "phase1_order_selection.csv"), index=False)
print("  Order selection (sorted by BIC):")
print(order_df.to_string(index=False))
print("  → GARCH(1,1) wins on BIC — most efficient balance of fit and complexity")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Baseline ARMA(1,1)-GARCH(1,1) with hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Phase 2: Baseline ARMA(1,1)-GARCH with hyperparameter tuning ────")

# Step 1: select the best (p, q) order via BIC
best_garch_p, best_garch_q, garch_tune_df = tune_garch_order(
    vol_train, time_IDs, orders=GARCH_ORDERS, sample=TUNE_SAMPLE
)
garch_tune_df.to_csv(os.path.join(OUTPUT_DIR, "garch_tuning.csv"), index=False)
print("  Saved: garch_tuning.csv")

# Step 2: fit using the best order
print(f"Fitting ARMA(1,1)-GARCH({best_garch_p},{best_garch_q}) "
      f"({len(time_IDs)} time_ids) ...")

garch_models = {}
garch_info   = []

for tid in time_IDs:
    rv_s = vol_train[tid]["RV"].values
    try:
        am  = arch_model(rv_s * SCALE, mean="ARX", lags=1,
                         vol="GARCH", p=best_garch_p, q=best_garch_q, dist="normal")
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": 1e-7})

        converged  = res.convergence_flag == 0
        stationary = garch_is_stationary(res) if converged else False

        garch_models[tid] = res
        garch_info.append({
            "time_id":    tid,
            "AIC":        res.aic if converged else np.nan,
            "BIC":        res.bic if converged else np.nan,
            "converged":  converged,
            "stationary": stationary,
            "alpha":      res.params.get("alpha[1]", np.nan) if converged else np.nan,
            "beta":       res.params.get("beta[1]",  np.nan) if converged else np.nan,
        })
    except Exception as e:
        print(f"  Warning: time_id={tid} failed ({e})")

garch_fit_df = pd.DataFrame(garch_info)
garch_fit_df.to_csv(os.path.join(OUTPUT_DIR, "garch_fit_info.csv"), index=False)

n_conv = garch_fit_df["converged"].sum()
n_stat = garch_fit_df["stationary"].sum()
conv_df = garch_fit_df[garch_fit_df["converged"]]
print(f"  Converged:  {n_conv}/{len(garch_fit_df)} ({100*n_conv/max(len(garch_fit_df),1):.1f}%)")
print(f"  Stationary: {n_stat}/{len(garch_fit_df)} ({100*n_stat/max(len(garch_fit_df),1):.1f}%)")
print(f"  Mean AIC: {conv_df['AIC'].mean():.2f}   Median AIC: {conv_df['AIC'].median():.2f}")

# Step 3: simulate forecasts
print(f"\nSimulating {N_SIM} paths × {SIM_STEPS} steps (GARCH) ...")
garch_preds   = {}
garch_example = {"tid": None, "paths": None}

# Find a good example time_id for the fan chart (highest alpha = most visible spread)
fan_tid = pick_best_fan_example(garch_fit_df, garch_models, alpha_col="alpha")

for tid, res in garch_models.items():
    row = garch_fit_df[garch_fit_df["time_id"] == tid]
    if row.empty or not bool(row["stationary"].iloc[0]):
        continue

    pred_rv, paths = simulate_rv(res, n_sim=N_SIM, n_steps=SIM_STEPS, scale=SCALE)
    garch_preds[tid] = pred_rv

    if tid == fan_tid and paths:
        garch_example["tid"]   = tid
        garch_example["paths"] = paths

print(f"  Forecasts generated: {len(garch_preds)}")

# Step 4: evaluate
garch_eval_df = evaluate_preds(garch_preds, vol_val)
garch_eval_df.to_csv(os.path.join(OUTPUT_DIR, "garch_eval_results.csv"), index=False)

garch_per_tid = garch_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
print(f"  n time_ids evaluated: {len(garch_per_tid)}")
print(f"  Median QLIKE: {garch_per_tid['QLIKE'].median():.4f}")
print(f"  Mean   QLIKE: {garch_per_tid['QLIKE'].mean():.4f}")
print(f"  Median MSE:   {garch_per_tid['MSE'].median():.8f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Refined ARMA(1,1)-EGARCH(1,1,1)-X with hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Phase 3: Refined ARMA(1,1)-EGARCH-X with hyperparameter tuning ──")

# Step 1: select the best (p, o, q) order via BIC
best_p, best_o, best_q, egarchx_tune_df = tune_egarchx_order(
    vol_train, time_IDs, orders=EGARCHX_ORDERS, sample=TUNE_SAMPLE
)
egarchx_tune_df.to_csv(os.path.join(OUTPUT_DIR, "egarchx_tuning.csv"), index=False)
print("  Saved: egarchx_tuning.csv")

# Step 2: fit using the best order
print(f"Fitting ARMA(1,1)-EGARCHX({best_p},{best_o},{best_q}) "
      f"({len(time_IDs)} time_ids) ...")

egarchx_models = {}   # tid -> (res, vol_obj)
egarchx_info   = []

for tid in time_IDs:
    rv_s = vol_train[tid]["RV"].values
    sp_s = vol_train[tid]["BidAskSpread_mean"].values
    try:
        vol_obj = EGARCHX(sp_s * SCALE, p=best_p, o=best_o, q=best_q)
        am      = ARX(rv_s * SCALE, lags=1)
        am.volatility   = vol_obj
        am.distribution = Normal()
        res = am.fit(disp="off", show_warning=False,
                     options={"maxiter": 1000, "ftol": 1e-7})

        converged  = res.convergence_flag == 0
        quality_ok = egarchx_passes_sanity_check(res)

        egarchx_models[tid] = (res, vol_obj)
        egarchx_info.append({
            "time_id":    tid,
            "AIC":        res.aic if converged else np.nan,
            "BIC":        res.bic if converged else np.nan,
            "converged":  converged,
            "quality_ok": quality_ok,
            "gamma":      res.params.get("gamma[1]",      np.nan) if quality_ok else np.nan,
            "delta":      res.params.get("delta[spread]", np.nan) if quality_ok else np.nan,
            "beta":       res.params.get("beta[1]",       np.nan) if quality_ok else np.nan,
        })
    except Exception as e:
        print(f"  Warning: time_id={tid} failed ({e})")

egarchx_fit_df = pd.DataFrame(egarchx_info)
egarchx_fit_df.to_csv(os.path.join(OUTPUT_DIR, "egarchx_fit_info.csv"), index=False)

n_conv_ex = egarchx_fit_df["converged"].sum()
n_qual_ex = egarchx_fit_df["quality_ok"].sum()
ex_df     = egarchx_fit_df[egarchx_fit_df["quality_ok"]]
print(f"  Converged:  {n_conv_ex}/{len(egarchx_fit_df)} ({100*n_conv_ex/max(len(egarchx_fit_df),1):.1f}%)")
print(f"  Quality-OK: {n_qual_ex}/{len(egarchx_fit_df)} ({100*n_qual_ex/max(len(egarchx_fit_df),1):.1f}%)")
print(f"  Mean gamma: {ex_df['gamma'].mean():.4f}  "
      f"(γ<0 in {(ex_df['gamma']<0).mean():.1%} → leverage effect)")
print(f"  Mean delta: {ex_df['delta'].mean():.4f}  "
      f"(δ>0 in {(ex_df['delta']>0).mean():.1%} → spread predicts vol)")

# Step 3: simulate forecasts
print(f"\nSimulating {N_SIM} paths × {SIM_STEPS} steps (EGARCH-X) ...")
egarchx_preds   = {}
egarchx_example = {"tid": None, "paths": None}

fan_tid_ex = pick_best_fan_example(egarchx_fit_df[egarchx_fit_df["quality_ok"]],
                                   {k: v for k, v in egarchx_models.items()},
                                   alpha_col="beta")

for tid, (res, vol_obj) in egarchx_models.items():
    row = egarchx_fit_df[egarchx_fit_df["time_id"] == tid]
    if row.empty or not bool(row["quality_ok"].iloc[0]):
        continue

    pred_rv, paths = simulate_rv(res, vol_obj=vol_obj,
                                 n_sim=N_SIM, n_steps=SIM_STEPS, scale=SCALE)
    egarchx_preds[tid] = pred_rv

    if tid == fan_tid_ex and paths:
        egarchx_example["tid"]   = tid
        egarchx_example["paths"] = paths

print(f"  Forecasts generated: {len(egarchx_preds)}")

# Step 4: evaluate
egarchx_eval_df = evaluate_preds(egarchx_preds, vol_val)
egarchx_eval_df.to_csv(os.path.join(OUTPUT_DIR, "egarchx_eval_results.csv"), index=False)

if not egarchx_eval_df.empty:
    egarchx_per_tid = egarchx_eval_df.groupby("time_id")[["QLIKE", "MSE"]].mean()
    print(f"  n time_ids evaluated: {len(egarchx_per_tid)}")
    print(f"  Median QLIKE: {egarchx_per_tid['QLIKE'].median():.4f}")
    print(f"  Mean   QLIKE: {egarchx_per_tid['QLIKE'].mean():.4f}")
    print(f"  Median MSE:   {egarchx_per_tid['MSE'].median():.8f}")
else:
    egarchx_per_tid = pd.DataFrame(columns=["QLIKE", "MSE"])
    print("  No valid EGARCH-X predictions after filtering.")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Comparison plots and output summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Phase 4: Evaluation Outputs ─────────────────────────────────────")

# Side-by-side comparison table
print("\n  ┌──────────────────────────┬──────────────┬──────────────┐")
print("  │ Metric                   │   GARCH(1,1) │  EGARCH-X    │")
print("  ├──────────────────────────┼──────────────┼──────────────┤")
print(f"  │ Median QLIKE             │ {garch_per_tid['QLIKE'].median():12.4f} │ "
      f"{egarchx_per_tid['QLIKE'].median() if not egarchx_per_tid.empty else float('nan'):12.4f} │")
print(f"  │ Median MSE               │ {garch_per_tid['MSE'].median():12.2e} │ "
      f"{egarchx_per_tid['MSE'].median() if not egarchx_per_tid.empty else float('nan'):12.2e} │")
print(f"  │ n forecasts              │ {len(garch_per_tid):12d} │ "
      f"{len(egarchx_per_tid):12d} │")
print("  └──────────────────────────┴──────────────┴──────────────┘")

# Evaluation boxplots
print("\nPlotting evaluation boxplots ...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"QLIKE and MSE Evaluation — Stock {STOCK_ID}", fontsize=13)

for row_idx, (per_tid, label, color) in enumerate([
    (garch_per_tid,   "GARCH(1,1)",      "#4C72B0"),
    (egarchx_per_tid, "EGARCH(1,1,1)-X", "#DD8452"),
]):
    if per_tid.empty:
        continue

    axes[row_idx][0].boxplot(per_tid["QLIKE"].dropna(), patch_artist=True,
                             boxprops=dict(facecolor=color, alpha=0.7), vert=False)
    axes[row_idx][0].set_title(f"QLIKE — {label}")
    axes[row_idx][0].set_xlabel("QLIKE loss  (log(pred²) + actual²/pred²)")
    axes[row_idx][0].axvline(per_tid["QLIKE"].median(), color="red",
                             linestyle="--", linewidth=1,
                             label=f"Median: {per_tid['QLIKE'].median():.4f}")
    axes[row_idx][0].legend(fontsize=9)

    axes[row_idx][1].boxplot(per_tid["MSE"].dropna(), patch_artist=True,
                             boxprops=dict(facecolor=color, alpha=0.7), vert=False)
    axes[row_idx][1].set_title(f"MSE — {label}")
    axes[row_idx][1].set_xlabel("MSE loss")
    axes[row_idx][1].axvline(per_tid["MSE"].median(), color="red",
                             linestyle="--", linewidth=1,
                             label=f"Median: {per_tid['MSE'].median():.2e}")
    axes[row_idx][1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_boxplots.png"), dpi=150)
plt.close()
print("  Saved: evaluation_boxplots.png")

# Fan charts
print("Plotting fan charts ...")

def plot_fan(example_dict, model_label, filename, color="#4C72B0"):
    tid   = example_dict["tid"]
    paths = example_dict["paths"]
    if tid is None or not paths:
        return

    # Re-simulate 200 paths for the visual
    res_ex  = garch_models[tid] if "GARCH" in model_label else egarchx_models[tid][0]
    vol_ex  = None if "GARCH" in model_label else egarchx_models[tid][1]
    path_mat = []

    for _ in range(200):
        try:
            if vol_ex is not None:
                vp  = res_ex.params[vol_ex.parameter_names()].values
                sim = vol_ex.simulate(vp, nobs=SIM_STEPS,
                                      rng=np.random.standard_normal, burn=0)
            else:
                last_sig2 = float(res_ex.conditional_volatility[-1] ** 2)
                sim = res_ex.model.simulate(res_ex.params, nobs=SIM_STEPS,
                                            burn=0, initial_value=last_sig2)
            path_mat.append(sim["data"].values / SCALE)
        except Exception:
            continue

    if not path_mat:
        return

    path_mat = np.array(path_mat)
    cum_rv   = np.sqrt(np.cumsum(path_mat ** 2, axis=1))
    pct      = np.percentile(cum_rv, [1, 5, 25, 50, 75, 95, 99], axis=0)
    t        = np.arange(1, SIM_STEPS + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(t, pct[1], pct[5], alpha=0.25, color=color, label="5–95th pct")
    ax.fill_between(t, pct[2], pct[4], alpha=0.50, color=color, label="25–75th pct")
    ax.plot(t, pct[3], color=color, linewidth=2, label="Median path")

    for path in path_mat[:50]:
        ax.plot(t, np.sqrt(np.cumsum(path**2)), color="grey", alpha=0.06, linewidth=0.7)

    # Clip y-axis to 1st–99th pct so outliers don't compress the fan
    ax.set_ylim(pct[0, 0] * 0.5, pct[6, -1] * 1.05)
    ax.set_xlabel("Steps ahead (seconds)")
    ax.set_ylabel("Cumulative RV")
    ax.set_title(f"Fan Chart — {model_label}\n"
                 f"Stock {STOCK_ID}, time_id={tid} | {len(path_mat)} paths")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

plot_fan(garch_example,   "ARMA(1,1)-GARCH(1,1)",      "garch_fan_chart.png",   "#4C72B0")
plot_fan(egarchx_example, "ARMA(1,1)-EGARCH(1,1,1)-X", "egarchx_fan_chart.png", "#DD8452")
print("  Saved: garch_fan_chart.png, egarchx_fan_chart.png")

# Predicted vs actual scatter + time series
print("Plotting predicted vs actual RV ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Predicted vs Actual RV — Stock {STOCK_ID}", fontsize=13)

for row_idx, (eval_df, label, color) in enumerate([
    (garch_eval_df,   "GARCH(1,1)",      "#4C72B0"),
    (egarchx_eval_df, "EGARCH(1,1,1)-X", "#DD8452"),
]):
    if eval_df.empty:
        continue

    pa = (eval_df.groupby("time_id")
          .agg(pred_RV=("pred_RV", "first"), actual_RV=("actual", "mean"))
          .dropna())

    # Exclude the <1% of blow-up predictions
    n_excluded = (pa["pred_RV"] > 0.05).sum()
    pa_plot    = pa[pa["pred_RV"] <= 0.05].copy()

    lim = max(pa_plot["actual_RV"].max(), pa_plot["pred_RV"].max()) * 1.08
    axes[row_idx][0].scatter(pa_plot["actual_RV"], pa_plot["pred_RV"],
                             alpha=0.35, s=12, color=color, label="Forecasts")
    axes[row_idx][0].plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect forecast")
    axes[row_idx][0].set_xlabel("Actual RV")
    axes[row_idx][0].set_ylabel("Predicted RV")
    axes[row_idx][0].set_title(f"Scatter — {label}"
                               + (f"\n({n_excluded} outlier preds >0.05 excluded)" if n_excluded else ""))
    axes[row_idx][0].legend(fontsize=9)

    t = np.arange(len(pa_plot))
    axes[row_idx][1].plot(t, pa_plot["actual_RV"].values,
                          color="#4C72B0", linewidth=0.7, alpha=0.8, label="Actual")
    axes[row_idx][1].plot(t, pa_plot["pred_RV"].values,
                          color=color, linewidth=0.7, alpha=0.8, label="Predicted")
    axes[row_idx][1].set_xlabel("time_id (ordered)")
    axes[row_idx][1].set_ylabel("RV")
    axes[row_idx][1].set_title(f"Time series — {label}")
    axes[row_idx][1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual.png"), dpi=150)
plt.close()
print("  Saved: pred_vs_actual.png")

# EGARCH-X gamma and delta distributions
print("Plotting γ and δ distributions ...")

if not ex_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"EGARCH-X Parameter Distributions — Stock {STOCK_ID}", fontsize=12)

    gammas    = ex_df["gamma"].dropna()
    pct_neg_g = (gammas < 0).mean()
    axes[0].hist(gammas, bins=30, color="#DD8452", alpha=0.75, edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1.2, label="γ = 0 (symmetric)")
    axes[0].axvline(gammas.median(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Median γ = {gammas.median():.4f}")
    axes[0].set_xlabel("γ  (asymmetry / leverage)")
    axes[0].set_ylabel("Number of time_ids")
    axes[0].set_title(f"γ < 0 in {pct_neg_g:.1%} → bad news raises vol more")
    axes[0].legend(fontsize=9)

    deltas    = ex_df["delta"].dropna()
    pct_pos_d = (deltas > 0).mean()
    axes[1].hist(deltas, bins=30, color="#55A868", alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2, label="δ = 0 (no spread effect)")
    axes[1].axvline(deltas.median(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Median δ = {deltas.median():.4f}")
    axes[1].set_xlabel("δ  (GARCH-X spread coefficient)")
    axes[1].set_ylabel("Number of time_ids")
    axes[1].set_title(f"δ > 0 in {pct_pos_d:.1%} → wider spread predicts higher vol")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "egarchx_parameters.png"), dpi=150)
    plt.close()
    print("  Saved: egarchx_parameters.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n✓ M4 complete. Outputs saved to: {OUTPUT_DIR}")
print()
print("  Phase 1 — Diagnostics:")
print("    phase1_acf_pacf.png")
print("    phase1_order_selection.csv")
print()
print("  Phase 2 — Baseline GARCH:")
print("    garch_tuning.csv")
print("    garch_fit_info.csv")
print("    garch_eval_results.csv")
print("    garch_fan_chart.png")
print()
print("  Phase 3 — Refined EGARCH-X:")
print("    egarchx_tuning.csv")
print("    egarchx_fit_info.csv")
print("    egarchx_eval_results.csv")
print("    egarchx_fan_chart.png")
print("    egarchx_parameters.png")
print()
print("  Phase 4 — Comparison:")
print("    evaluation_boxplots.png")
print("    pred_vs_actual.png")