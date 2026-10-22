"""
Bengaluru air pollution (PM2.5) - end-to-end time series project
Run:  python analysis.py
Outputs: figures/*.png, tables/*.csv, results.json
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tslib import *

warnings.filterwarnings("ignore")
T0 = time.time()
import sys
CITY = sys.argv[1]; DATA = sys.argv[2]; OUT = sys.argv[3]; CITY_T = CITY.title()
FIG, TAB = f"{OUT}/figures", f"{OUT}/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
R = {}  # results collected for the report

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "legend.frameon": False})
C = {"obs": "#3b3b3b", "blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a", "red": "#c53030",
     "purple": "#6b46c1", "teal": "#0f766e", "grey": "#9ca3af", "gold": "#b7791f"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def tick(s):
    print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)


# =============================================================================
# 1. DATA
# =============================================================================
raw = pd.read_csv(DATA, parse_dates=["date"]).set_index("date").asfreq("D")
n_raw = len(raw)
df = raw.dropna(subset=["pm2_5"]).copy()  # first 3 days have no data at all
assert df.index.is_monotonic_increasing and (df.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
s = df["pm2_5"]
y = s.values.astype(float)
t = np.arange(len(y))
n = len(y)
train_n = int(round(0.8 * n))
test_n = n - train_n
ytr, ttr = y[:train_n], t[:train_n]
R["data"] = {"rows_raw": n_raw, "rows_used": n, "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
             "missing_dropped": int(n_raw - n), "train_n": train_n, "test_n": test_n,
             "train_start": str(s.index[0].date()), "train_end": str(s.index[train_n - 1].date()),
             "test_start": str(s.index[train_n].date()), "test_end": str(s.index[-1].date())}
print(R["data"])
desc = s.describe()
R["desc"] = {k: float(v) for k, v in desc.items()}
R["desc"].update({"skew": float(stats.skew(y)), "kurt": float(stats.kurtosis(y)), "cv": float(y.std() / y.mean())})
R["exceed"] = {"who_15": float((y > 15).mean() * 100), "naaqs_60": float((y > 60).mean() * 100),
               "gt_35": float((y > 35).mean() * 100)}
split_date = s.index[train_n]

# =============================================================================
# 2. EDA
# =============================================================================
tick("EDA")
pol = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"]
lab = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2", "ozone": "O3",
       "carbon_monoxide": "CO", "sulphur_dioxide": "SO2"}
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax[0].plot(s.index, s, color=C["grey"], lw=0.7, label="Daily PM2.5")
ax[0].plot(s.index, s.rolling(30, center=True).mean(), color=C["blue"], lw=1.8, label="30-day centred mean")
ax[0].axvspan(split_date, s.index[-1], color=C["orange"], alpha=0.12, label="Test set (held out)")
ax[0].axhline(15, color=C["green"], ls="--", lw=0.9); ax[0].text(s.index[5], 16, "WHO 24-h guideline (15)", color=C["green"], fontsize=8)
ax[0].axhline(60, color=C["red"], ls="--", lw=0.9); ax[0].text(s.index[5], 61, "India NAAQS 24-h (60)", color=C["red"], fontsize=8)
ax[0].set_ylabel("µg/m³"); ax[0].set_title(f"{CITY_T} daily PM2.5"); ax[0].legend(loc="upper right", ncol=3)
zs = df[pol].apply(lambda c: (c - c.mean()) / c.std()).rolling(30, center=True).mean()
for c_, col in zip(pol, [C["blue"], C["teal"], C["orange"], C["purple"], C["red"], C["gold"]]):
    ax[1].plot(zs.index, zs[c_], label=lab[c_], lw=1.3, color=col)
ax[1].set_ylabel("z-score (30-day mean)"); ax[1].set_title("All pollutants share one annual rhythm"); ax[1].legend(ncol=6, loc="upper right")
save(fig, "fig01_overview")

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
mon = pd.DataFrame({"v": y, "m": s.index.month})
ax[0, 0].boxplot([mon[mon.m == m]["v"] for m in range(1, 13)], tick_labels=list("JFMAMJJASOND"), showfliers=False,
                 patch_artist=True, boxprops=dict(facecolor="#bee3f8"), medianprops=dict(color=C["blue"]))
ax[0, 0].set_title("Monthly distribution: monsoon trough (Jun-Sep)"); ax[0, 0].set_ylabel("µg/m³")
doy = pd.Series(y, index=s.index)
for yr, col in zip(sorted(set(s.index.year))[:-1][-3:], [C["blue"], C["orange"], C["green"]]):
    d_ = doy[doy.index.year == yr].rolling(7, center=True).mean()
    ax[0, 1].plot(d_.index.dayofyear, d_.values, label=str(yr), color=col, lw=1.5)
d_ = doy[doy.index.year == s.index.year.max()]
ax[0, 1].plot(d_.index.dayofyear, d_.rolling(7, center=True).mean().values, label=f"{s.index.year.max()} (partial)", color=C["red"], lw=1.8)
ax[0, 1].set_title("Same seasonal shape every year (7-day mean)"); ax[0, 1].set_xlabel("Day of year"); ax[0, 1].legend(ncol=2)
ax[1, 0].hist(y, bins=40, color="#90cdf4", edgecolor="white"); ax[1, 0].set_title(f"Distribution (skew={stats.skew(y):.2f})"); ax[1, 0].set_xlabel("µg/m³")
dw = pd.DataFrame({"v": y, "d": s.index.dayofweek})
ax[1, 1].boxplot([dw[dw.d == d]["v"] for d in range(7)], tick_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], showfliers=False,
                 patch_artist=True, boxprops=dict(facecolor="#fefcbf"), medianprops=dict(color=C["gold"]))
kw = stats.kruskal(*[dw[dw.d == d]["v"] for d in range(7)])
ax[1, 1].set_title(f"No weekday effect (Kruskal-Wallis p={kw.pvalue:.2f})")
save(fig, "fig02_eda")
R["eda"] = {"monthly_mean": {int(k): float(v) for k, v in s.groupby(s.index.month).mean().items()},
            "yearly_mean": {int(k): float(v) for k, v in s.groupby(s.index.year).mean().items()},
            "kw_dow_p": float(kw.pvalue),
            "corr_pm25": {c_: float(df[c_].corr(df["pm2_5"])) for c_ in df.columns if c_ != "pm2_5"}}

cm = df.corr()
fig, ax = plt.subplots(figsize=(7.5, 6.3))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha="right")
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns); ax.grid(False)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(cm.values[i, j]) > 0.6 else "black")
plt.colorbar(im, shrink=0.8); ax.set_title("Correlation between daily series")
save(fig, "fig03_correlation")

# =============================================================================
# 3. DECOMPOSITION (full sample, descriptive)
# =============================================================================
tick("Decomposition")
dec = classical_decompose(s, "additive", 365)
decm = classical_decompose(s, "multiplicative", 365)
fs_a, ft_a = strength(dec, "additive")
fs_m, ft_m = strength(decm, "multiplicative")
fig, ax = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
for a_, c_, col in zip(ax, ["observed", "trend", "seasonal", "remainder"], [C["obs"], C["blue"], C["orange"], C["grey"]]):
    a_.plot(dec.index, dec[c_], color=col, lw=1.0 if c_ != "trend" else 2)
    a_.set_ylabel(c_.capitalize())
ax[0].set_title(f"Classical additive decomposition, period = 365 days   (seasonal strength {fs_a:.2f}, trend strength {ft_a:.2f})")
ax[3].axhline(0, color="k", lw=0.6)
save(fig, "fig04_decomposition")
rem = dec["remainder"].dropna()
sea = dec["seasonal"].reindex(rem.index)
hetero_add = float(np.corrcoef(np.abs(rem), sea)[0, 1])
rem_m = decm["remainder"].dropna()
R["decomp"] = {"fs_add": fs_a, "ft_add": ft_a, "fs_mult": fs_m, "ft_mult": ft_m,
               "seasonal_range": [float(dec["seasonal"].min()), float(dec["seasonal"].max())],
               "trend_range": [float(dec["trend"].min()), float(dec["trend"].max())],
               "corr_abs_remainder_vs_seasonal": hetero_add, "remainder_sd": float(rem.std())}
# variance stabilisation check on the raw series
rm, rs = s.rolling(30).mean().dropna(), s.rolling(30).std().dropna()
R["decomp"]["corr_rollmean_rollsd"] = float(np.corrcoef(rm, rs.reindex(rm.index))[0, 1])

f_, p_ = periodogram(y)
top = np.argsort(p_)[::-1][:8]
R["periodogram_top"] = [{"period_days": float(1 / f_[i]), "power": float(p_[i])} for i in top[:6]]
wk_idx = np.argmin(np.abs(f_ - 1 / 7))
R["periodogram_weekly_rank"] = int((p_ > p_[wk_idx]).sum() + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(f_[f_ < 0.06], p_[f_ < 0.06], color=C["blue"]); ax[0].set_xlabel("Frequency (cycles/day)"); ax[0].set_title("Periodogram: low frequencies")
for i in top[:3]:
    ax[0].annotate(f"{1 / f_[i]:.0f} d", (f_[i], p_[i]), textcoords="offset points", xytext=(6, 2), fontsize=8)
ax[1].plot(1 / f_[f_ > 1 / 60], p_[f_ > 1 / 60], color=C["blue"], lw=0.8); ax[1].set_xscale("log"); ax[1].set_xlabel("Period (days)")
ax[1].axvline(7, color=C["red"], ls="--", lw=0.9); ax[1].text(7.3, ax[1].get_ylim()[1] * 0.8, "7 d", color=C["red"]); ax[1].set_title("Periodogram: periods below 60 days")
save(fig, "fig05_periodogram")

# =============================================================================
# 4. STATIONARITY (training sample only - no test leakage)
# =============================================================================
tick("Stationarity tests")
Xh = np.column_stack([np.ones(train_n), fourier_terms(ttr, 4)])
bh = np.linalg.lstsq(Xh, ytr, rcond=None)[0]
deseas = ytr - Xh @ bh
series_sets = {
    "PM2.5 (level)": ytr,
    "log PM2.5": np.log(ytr),
    "Δ PM2.5 (d=1)": np.diff(ytr),
    "Δ365 PM2.5 (D=1)": ytr[365:] - ytr[:-365],
    "Δ365 then Δ1 (D=1,d=1)": np.diff(ytr[365:] - ytr[:-365]),
    "Deseasonalised (harmonic K=4 residual)": deseas,
}
rows = []
for nm, x in series_sets.items():
    a_c, a_ct = adf_test(x, "c"), adf_test(x, "ct")
    k_c, k_ct = kpss_test(x, "c"), kpss_test(x, "ct")
    rows.append({"series": nm, "n": len(x),
                 "ADF(c) stat": a_c["stat"], "ADF(c) p": a_c["pvalue"], "ADF lags": a_c["lags"], "ADF(c) crit5%": a_c["crit"]["5%"],
                 "ADF(ct) stat": a_ct["stat"], "ADF(ct) p": a_ct["pvalue"],
                 "KPSS(c) stat": k_c["stat"], "KPSS(c) p": k_c["pvalue"],
                 "KPSS(ct) stat": k_ct["stat"], "KPSS(ct) p": k_ct["pvalue"],
                 "ADF verdict": "stationary" if a_c["reject_5pct"] else "unit root",
                 "KPSS verdict": "non-stationary" if k_c["reject_5pct"] else "stationary"})
stat_tab = pd.DataFrame(rows)
stat_tab.to_csv(f"{TAB}/stationarity_tests.csv", index=False)
print(stat_tab.round(3).to_string())
R["stationarity"] = stat_tab.round(4).to_dict("records")
R["adf_crit_raw"] = adf_test(ytr, "c")["crit"]

fig, ax = plt.subplots(4, 1, figsize=(11, 9))
plot_sets = [("PM2.5 (level)", ytr), ("Δ PM2.5 (d=1)", np.diff(ytr)), ("Δ365 PM2.5 (D=1)", ytr[365:] - ytr[:-365]),
             ("Deseasonalised (harmonic K=4 residual)", deseas)]
for a_, (nm, x) in zip(ax, plot_sets):
    a_.plot(x, color=C["grey"], lw=0.7)
    a_.plot(pd.Series(x).rolling(60).mean(), color=C["blue"], lw=1.6, label="60-day mean")
    a_.plot(pd.Series(x).rolling(60).std(), color=C["orange"], lw=1.4, label="60-day std")
    r_ = stat_tab[stat_tab.series == nm].iloc[0]
    a_.set_title(f"{nm}   |   ADF p={r_['ADF(c) p']:.4f} ({r_['ADF verdict']})   KPSS stat={r_['KPSS(c) stat']:.2f} p={r_['KPSS(c) p']:.2f} ({r_['KPSS verdict']})", fontsize=9.5)
ax[0].legend(ncol=2, loc="upper right")
save(fig, "fig06_stationarity")

fig, ax = plt.subplots(4, 2, figsize=(11, 10))
combos = [("PM2.5 (level)", ytr, 400, 40), ("Δ PM2.5 (d=1)", np.diff(ytr), 40, 40),
          ("Δ365 PM2.5 (D=1)", ytr[365:] - ytr[:-365], 40, 40), ("Deseasonalised residual", deseas, 40, 40)]
for i, (nm, x, la, lp) in enumerate(combos):
    a_ = acf(x, la); p__ = pacf(x, lp); cb = 1.96 / np.sqrt(len(x))
    ax[i, 0].bar(range(la + 1), a_, width=1.0 if la > 100 else 0.6, color=C["blue"]); ax[i, 0].axhspan(-cb, cb, color=C["grey"], alpha=0.3)
    ax[i, 0].set_title(f"ACF - {nm}"); ax[i, 0].set_ylim(-1, 1.05)
    ax[i, 1].bar(range(lp + 1), p__, width=0.6, color=C["orange"]); ax[i, 1].axhspan(-cb, cb, color=C["grey"], alpha=0.3)
    ax[i, 1].set_title(f"PACF - {nm}"); ax[i, 1].set_ylim(-1, 1.05)
save(fig, "fig07_acf_pacf")
R["acf"] = {"raw_lag1": float(acf(ytr, 1)[1]), "raw_lag7": float(acf(ytr, 7)[7]), "raw_lag365": float(acf(ytr, 365)[365]),
            "deseas_lag1": float(acf(deseas, 1)[1]), "deseas_pacf1": float(pacf(deseas, 3)[1]), "deseas_pacf2": float(pacf(deseas, 3)[2]),
            "deseas_pacf3": float(pacf(deseas, 3)[3]), "diff_lag1": float(acf(np.diff(ytr), 1)[1])}

# =============================================================================
# 5. MODEL IDENTIFICATION on TRAIN (AIC grid)
# =============================================================================
tick("Model selection grids")
PQ = [(p, q) for p in range(4) for q in range(3)]


def grid(make, orders):
    out = []
    for o in orders:
        try:
            m = make(*o).fit(ytr, ttr)
            out.append({"spec": m.name, "order": o, "k_params": len(m.params) + 1, "loglik": m.loglik, "AIC": m.aic, "BIC": m.bic, "sigma": np.sqrt(m.sigma2)})
        except Exception as ex:
            print("fail", o, ex)
    return pd.DataFrame(out).sort_values("AIC").reset_index(drop=True)


# Harmonic regression + ARMA errors: choose K, p, q together
KS = [1, 2, 3, 4, 5, 6, 8]
g_dhr = grid(lambda K, p, q: RegArima(p, q, K=K), [(K, p, q) for K in KS for (p, q) in PQ])
g_dhr.to_csv(f"{TAB}/grid_harmonic_arma.csv", index=False)
best_dhr = g_dhr.iloc[0]["order"]
best_dhr_bic = g_dhr.sort_values("BIC").iloc[0]["order"]
tick(f"harmonic grid done, best (K,p,q)={best_dhr}")
g_dhr_log = grid(lambda K, p, q: RegArima(p, q, K=K, log=True), [(K, p, q) for K in KS for (p, q) in PQ])
g_dhr_log.to_csv(f"{TAB}/grid_harmonic_arma_log.csv", index=False)
best_dhr_log = g_dhr_log.iloc[0]["order"]
tick(f"log harmonic grid done, best={best_dhr_log}")
# does a linear trend help?
m_tr = RegArima(best_dhr[1], best_dhr[2], K=best_dhr[0], trend=True).fit(ytr, ttr)
m_notr = RegArima(best_dhr[1], best_dhr[2], K=best_dhr[0]).fit(ytr, ttr)
tr_row = m_tr.param_table().iloc[1]
R["trend_check"] = {"aic_with": m_tr.aic, "aic_without": m_notr.aic, "coef": float(tr_row["coef"]), "z": float(tr_row["z"])}

g_ar0 = grid(lambda p, q: RegArima(p, q), PQ)
g_ar1 = grid(lambda p, q: RegArima(p, q, diff_lag=1), PQ)
g_sar = grid(lambda p, q: RegArima(p, q, diff_lag=365), PQ)
for nm_, g_ in [("arima_d0", g_ar0), ("arima_d1", g_ar1), ("sarima365", g_sar)]:
    g_.to_csv(f"{TAB}/grid_{nm_}.csv", index=False)
best_ar0, best_ar1, best_sar = g_ar0.iloc[0]["order"], g_ar1.iloc[0]["order"], g_sar.iloc[0]["order"]
R["selected"] = {"harmonic": [int(v) for v in best_dhr], "harmonic_log": [int(v) for v in best_dhr_log],
                 "arima_d0": [int(v) for v in best_ar0], "arima_d1": [int(v) for v in best_ar1], "sarima": [int(v) for v in best_sar]}
R["grid_top"] = {"harmonic": g_dhr.head(5)[["spec", "AIC", "BIC"]].round(1).to_dict("records"),
                 "harmonic_log": g_dhr_log.head(3)[["spec", "AIC", "BIC"]].round(1).to_dict("records"),
                 "arima_d0": g_ar0.head(3)[["spec", "AIC", "BIC"]].round(1).to_dict("records"),
                 "arima_d1": g_ar1.head(3)[["spec", "AIC", "BIC"]].round(1).to_dict("records"),
                 "sarima": g_sar.head(3)[["spec", "AIC", "BIC"]].round(1).to_dict("records")}
print("selected:", R["selected"])
# AIC vs K plot
fig, ax = plt.subplots(figsize=(6.5, 4))
kb = g_dhr.groupby(g_dhr["order"].apply(lambda o: o[0]))["AIC"].min()
kbl = g_dhr_log.groupby(g_dhr_log["order"].apply(lambda o: o[0]))["AIC"].min()
ax.plot(kb.index, kb.values - kb.min(), "o-", color=C["blue"], label="raw scale")
ax.plot(kbl.index, kbl.values - kbl.min(), "s-", color=C["orange"], label="log scale (own baseline)")
ax.set_xlabel("Number of Fourier harmonic pairs K"); ax.set_ylabel("ΔAIC (min over p,q)"); ax.set_title("Choosing K for the annual cycle"); ax.legend()
save(fig, "fig08_aic_vs_K")

# =============================================================================
# 6. MODEL SET
# =============================================================================
K_, p_, q_ = best_dhr
Kl, pl_, ql_ = best_dhr_log
# sensitivity: wider ARMA search (p<=4, q<=4) at the selected K
g_ext = grid(lambda p, q: RegArima(p, q, K=K_), [(p, q) for p in range(5) for q in range(5)])
g_ext.to_csv(f"{TAB}/grid_harmonic_arma_extended.csv", index=False)
best_ext = tuple(g_ext.iloc[0]["order"])
m_ext = RegArima(best_ext[0], best_ext[1], K=K_).fit(ytr, ttr)
R["extended_search"] = {"order": list(best_ext), "aic": float(m_ext.aic), "bic": float(m_ext.bic),
                        "aic_standard": float(g_dhr.iloc[0]["AIC"]), "bic_standard": float(g_dhr.iloc[0]["BIC"]),
                        "max_ma_root_modulus": float(np.abs(np.roots(np.r_[1, m_ext.theta])).max()) if m_ext.q else None,
                        "ma": [float(v) for v in m_ext.theta], "ar": [float(v) for v in m_ext.phi],
                        "lb20_p": float(ljung_box(m_ext.std_resid(), 20, best_ext[0] + best_ext[1])[1]),
                        "lb60_p": float(ljung_box(m_ext.std_resid(), 60, best_ext[0] + best_ext[1])[1])}
print("extended:", R["extended_search"])
FACT = {
    "Mean": lambda: Mean(),
    "Naive": lambda: Naive(),
    "Seasonal naive": lambda: SeasonalNaive(365),
    f"ARIMA({best_ar0[0]},0,{best_ar0[1]})+const": lambda: RegArima(best_ar0[0], best_ar0[1]),
    f"ARIMA({best_ar1[0]},1,{best_ar1[1]})": lambda: RegArima(best_ar1[0], best_ar1[1], diff_lag=1),
    "Holt-Winters (A,N,A) m=365": lambda: HoltWintersAdd(365),
    f"SARIMA({best_sar[0]},0,{best_sar[1]})(0,1,0)[365]": lambda: RegArima(best_sar[0], best_sar[1], diff_lag=365),
    f"Harmonic(K={K_}) + ARMA({p_},{q_})": lambda: RegArima(p_, q_, K=K_),
    f"Harmonic(K={Kl}) + ARMA({pl_},{ql_}) [log]": lambda: RegArima(pl_, ql_, K=Kl, log=True),
    "Gradient boosting (ML)": lambda: GBMLag(),
}
if best_ext != (p_, q_):
    FACT[f"Harmonic(K={K_}) + ARMA({best_ext[0]},{best_ext[1]}) [extended search]"] = lambda: RegArima(best_ext[0], best_ext[1], K=K_)
if tuple(best_dhr_bic) != tuple(best_dhr):
    Kb, pb, qb = best_dhr_bic
    FACT[f"Harmonic(K={Kb}) + ARMA({pb},{qb}) [BIC choice]"] = lambda: RegArima(pb, qb, K=Kb)
R["selected"]["harmonic_bic"] = [int(v) for v in best_dhr_bic]
NAMES = list(FACT)
fitted = {}
for nm, f in FACT.items():
    fitted[nm] = f().fit(ytr, ttr)
    tick(f"fitted {nm}")

scale_mase = float(np.mean(np.abs(np.diff(ytr))))
sd_tr = ytr.std()

# =============================================================================
# 7. IN-SAMPLE EVALUATION (one-step-ahead fitted values on the training data)
# =============================================================================
tick("In-sample evaluation")
ins_rows, oos1_rows, fixed_rows = [], [], []
one_step_full = {}
for nm in NAMES:
    yh = fitted[nm].one_step(y, t)  # uses each day's own past only, parameters fixed at train estimates
    one_step_full[nm] = yh
    mi = metrics(ytr, yh[:train_n], scale_mase, ytr.mean())
    mo = metrics(y[train_n:], yh[train_n:], scale_mase, ytr.mean())
    ins_rows.append({"model": nm, **mi})
    oos1_rows.append({"model": nm, **mo})
ins_tab = pd.DataFrame(ins_rows).set_index("model")
oos1_tab = pd.DataFrame(oos1_rows).set_index("model")
ins_tab.round(3).to_csv(f"{TAB}/insample_metrics.csv")
oos1_tab.round(3).to_csv(f"{TAB}/oos_onestep_metrics.csv")
print(ins_tab.round(2).to_string())
print(oos1_tab.round(2).to_string())

# =============================================================================
# 8. OUT-OF-SAMPLE, fixed origin: forecast the entire test period at once
# =============================================================================
tick("Fixed-origin forecasts")
fc_fixed = {}
for nm in NAMES:
    fc_fixed[nm] = fitted[nm].forecast(ytr, ttr, test_n)
    mf = metrics(y[train_n:], fc_fixed[nm]["mean"], scale_mase, ytr.mean())
    lo, hi = fc_fixed[nm]["lo95"], fc_fixed[nm]["hi95"]
    mf["cov95%"] = float(np.mean((y[train_n:] >= lo) & (y[train_n:] <= hi)) * 100) if not np.isnan(lo).any() else np.nan
    loh, hih = fc_fixed[nm].get("lo95h", lo), fc_fixed[nm].get("hi95h", hi)
    mf["cov95% seasonal-sd"] = float(np.mean((y[train_n:] >= loh) & (y[train_n:] <= hih)) * 100) if not np.isnan(loh).any() else np.nan
    fixed_rows.append({"model": nm, **mf})
fixed_tab = pd.DataFrame(fixed_rows).set_index("model")
fixed_tab.round(3).to_csv(f"{TAB}/oos_fixed_origin_metrics.csv")
print(fixed_tab.round(2).to_string())

# =============================================================================
# 9. Walk-forward (rolling origin) evaluation, horizons 1..30
# =============================================================================
tick("Walk-forward evaluation (this takes a while)")
WF = {}
for nm in NAMES:
    WF[nm] = walk_forward(FACT[nm], y, t, train_n, horizon=30, step=7, refit_every=8)
    tick(f"  walk-forward {nm}")
wf_rows = []
for nm in NAMES:
    d_ = WF[nm]
    d_["err"] = d_["actual"] - d_["fc"]
    row = {"model": nm}
    for h in (1, 7, 14, 30):
        e_ = d_[d_.h == h]["err"]
        row[f"RMSE h={h}"] = float(np.sqrt((e_ ** 2).mean()))
        row[f"MAE h={h}"] = float(e_.abs().mean())
    row["RMSE avg 1-30"] = float(np.sqrt((d_["err"] ** 2).mean()))
    row["MAE avg 1-30"] = float(d_["err"].abs().mean())
    ok = ~d_["lo95"].isna()
    row["cov95% avg"] = float(((d_["actual"] >= d_["lo95"]) & (d_["actual"] <= d_["hi95"]))[ok].mean() * 100) if ok.any() else np.nan
    row["cov95% avg seasonal-sd"] = float(((d_["actual"] >= d_["lo95h"]) & (d_["actual"] <= d_["hi95h"]))[ok].mean() * 100) if ok.any() else np.nan
    row["n_origins"] = int(d_.origin.nunique())
    wf_rows.append(row)
wf_tab = pd.DataFrame(wf_rows).set_index("model")
wf_tab.round(3).to_csv(f"{TAB}/oos_walkforward_metrics.csv")
print(wf_tab.round(2).to_string())
byh = {nm: WF[nm].groupby("h")["err"].apply(lambda e: np.sqrt((e ** 2).mean())) for nm in NAMES}
byh_mae = {nm: WF[nm].groupby("h")["err"].apply(lambda e: e.abs().mean()) for nm in NAMES}
pd.DataFrame(byh).round(3).to_csv(f"{TAB}/oos_walkforward_rmse_by_horizon.csv")

# pick best interval-capable model by walk-forward RMSE
cand = [nm for nm in NAMES if not np.isnan(wf_tab.loc[nm, "cov95% avg"])]
cand = [nm for nm in cand if nm.startswith("Harmonic")]
mn_ = wf_tab.loc[cand, "RMSE avg 1-30"].min()
pool = [nm for nm in cand if wf_tab.loc[nm, "RMSE avg 1-30"] <= 1.02 * mn_ and nm.startswith("Harmonic")]   # selection restricted to the harmonic (annual-cycle) family
assert pool   # statistically tied top group (within 2%)
npar = {nm: len(getattr(fitted[nm], "params", [0] * 999)) for nm in pool}
best_nm = min(pool, key=lambda nm: (npar[nm], wf_tab.loc[nm, "RMSE avg 1-30"]))   # parsimony rule
R["tie_pool"] = {nm: {"rmse": float(wf_tab.loc[nm, "RMSE avg 1-30"]), "n_params": npar[nm]} for nm in pool}
best_any = wf_tab["RMSE avg 1-30"].idxmin()
R["best_model"] = best_nm
cov_c, cov_h = wf_tab.loc[best_nm, "cov95% avg"], wf_tab.loc[best_nm, "cov95% avg seasonal-sd"]
SFX = "h" if abs(cov_h - 95) < abs(cov_c - 95) else ""
R["pi_type"] = "seasonal-sd" if SFX else "constant-sd"
R["pi_cov"] = {"constant_sd_wf": float(cov_c), "seasonal_sd_wf": float(cov_h),
               "constant_sd_fixed": float(fixed_tab.loc[best_nm, "cov95%"]), "seasonal_sd_fixed": float(fixed_tab.loc[best_nm, "cov95% seasonal-sd"])}
R["best_any"] = best_any
print("BEST (with intervals):", best_nm, "| best overall:", best_any)

# =============================================================================
# 10. Forecast comparison tests (Diebold-Mariano)
# =============================================================================
tick("Diebold-Mariano tests")
e1 = {nm: y[train_n:] - one_step_full[nm][train_n:] for nm in NAMES}
dm_rows = []
for nm in NAMES:
    if nm == best_nm:
        continue
    dm1, p1 = dm_test(e1[best_nm], e1[nm], h=1)
    a_, b_ = WF[best_nm], WF[nm]
    ea = a_[a_.h == 7].set_index("origin")["err"]; eb = b_[b_.h == 7].set_index("origin")["err"]
    dm7, p7 = dm_test(ea.values, eb.reindex(ea.index).values, h=1)
    ea = a_[a_.h == 30].set_index("origin")["err"]; eb = b_[b_.h == 30].set_index("origin")["err"]
    dm30, p30 = dm_test(ea.values, eb.reindex(ea.index).values, h=1)
    dm_rows.append({"vs": nm, "DM 1-step": dm1, "p 1-step": p1, "DM h=7": dm7, "p h=7": p7, "DM h=30": dm30, "p h=30": p30})
dm_tab = pd.DataFrame(dm_rows).set_index("vs")
dm_tab.round(4).to_csv(f"{TAB}/diebold_mariano_vs_best.csv")
print(dm_tab.round(3).to_string())

# =============================================================================
# 11. Residual diagnostics of the best model (in-sample)
# =============================================================================
tick("Residual diagnostics")
bm = fitted[best_nm]
z_res = bm.std_resid()
ddof = bm.p + bm.q
diag = {"model": best_nm, "sigma": float(np.sqrt(bm.sigma2)), "aic": float(bm.aic), "bic": float(bm.bic), "N": int(bm.N)}
for L in (10, 20, 30, 60):
    Q, pv = ljung_box(z_res, L, ddof)
    diag[f"LB{L}_Q"], diag[f"LB{L}_p"] = float(Q), float(pv)
jb, jbp, sk, ku = jarque_bera(z_res)
lm, lmp = arch_lm(z_res, 5)
diag.update({"JB": float(jb), "JB_p": float(jbp), "skew": float(sk), "kurt": float(ku), "ARCH_LM": float(lm), "ARCH_p": float(lmp),
             "resid_mean": float(z_res.mean())})
# seasonal heteroscedasticity: residual sd by month
_r = bm.resid[bm.BURN:]; res_s = pd.Series(_r, index=s.index[:train_n][-len(_r):])
diag["resid_sd_by_month"] = {int(k): float(v) for k, v in res_s.groupby(res_s.index.month).std().items()}
R["diag"] = diag
R["diag_all"] = {}
for nm in ("Seasonal naive", f"SARIMA({best_sar[0]},0,{best_sar[1]})(0,1,0)[365]", f"Harmonic(K={K_}) + ARMA({p_},{q_})",
           f"Harmonic(K={Kl}) + ARMA({pl_},{ql_}) [log]"):
    m_ = fitted[nm]
    if hasattr(m_, "std_resid"):
        zr = m_.std_resid()
        R["diag_all"][nm] = {"LB20_p": float(ljung_box(zr, 20, m_.p + m_.q)[1]), "LB60_p": float(ljung_box(zr, 60, m_.p + m_.q)[1]),
                             "JB_p": float(jarque_bera(zr)[1]), "ARCH_p": float(arch_lm(zr, 5)[1])}
coef_tab = bm.param_table()
coef_tab.round(4).to_csv(f"{TAB}/best_model_coefficients.csv", index=False)
print(coef_tab.round(3).to_string()); print(diag)
R["coef"] = coef_tab.round(4).to_dict("records")

# =============================================================================
# 12. FINAL MODEL: refit on all data, forecast 90 days
# =============================================================================
tick("Final forecast")
H = 90
final = FACT[best_nm]().fit(y, t)
fc_final = final.forecast(y, t, H)
future_idx = pd.date_range(s.index[-1] + pd.Timedelta("1D"), periods=H)
final_df = pd.DataFrame({"mean": fc_final["mean"], "lo80": fc_final["lo80" + SFX], "hi80": fc_final["hi80" + SFX],
                         "lo95": fc_final["lo95" + SFX], "hi95": fc_final["hi95" + SFX]}, index=future_idx).round(2)
final_df.index.name = "date"
final_df.to_csv(f"{TAB}/final_forecast_90d.csv")
R["final"] = {"model": best_nm, "aic": float(final.aic), "from": str(future_idx[0].date()), "to": str(future_idx[-1].date()),
              "mean_fc": float(final_df["mean"].mean()),
              "monthly": {str(k): float(v) for k, v in final_df["mean"].groupby(final_df.index.to_period("M")).mean().items()},
              "params": final.param_table().round(4).to_dict("records")}
print(final_df.head(3), final_df.tail(3))

# =============================================================================
# 13. FIGURES for evaluation
# =============================================================================
tick("Plots")
# fig09 in-sample fit for best model (last 365 days of training)
yh_best = one_step_full[best_nm]
_hm = fitted[f"Harmonic(K={K_}) + ARMA({p_},{q_})"]   # harmonic model gives the deterministic cycle
Xdet = _hm._X(t[:train_n]) @ _hm.beta
det = np.exp(Xdet) if _hm.log else Xdet
lastN = slice(train_n - 365, train_n)
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax[0].plot(s.index[lastN], y[lastN], color=C["grey"], lw=1, label="Actual")
ax[0].plot(s.index[lastN], det[lastN], color=C["orange"], lw=2, label="Deterministic annual cycle (harmonics)")
ax[0].plot(s.index[lastN], yh_best[:train_n][lastN], color=C["blue"], lw=1.3, label="One-step-ahead fitted (cycle + ARMA)")
ax[0].legend(ncol=3, loc="upper right"); ax[0].set_ylabel("µg/m³"); ax[0].set_title(f"In-sample fit, last training year - {best_nm}")
ax[1].plot(s.index[lastN], (y[:train_n] - yh_best[:train_n])[lastN], color=C["red"], lw=0.9); ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_ylabel("Residual (µg/m³)"); ax[1].set_title("One-step-ahead errors")
save(fig, "fig09_insample_fit")

# fig10 residual diagnostics
fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
ax[0, 0].plot(z_res, color=C["grey"], lw=0.7); ax[0, 0].axhline(0, color="k", lw=0.6); ax[0, 0].set_title("Standardised residuals")
a_ = acf(z_res, 40); cb = 1.96 / np.sqrt(len(z_res))
ax[0, 1].bar(range(41), a_, width=0.6, color=C["blue"]); ax[0, 1].axhspan(-cb, cb, color=C["grey"], alpha=0.3); ax[0, 1].set_title("Residual ACF")
ax[0, 2].bar(range(1, 41), acf(z_res ** 2, 40)[1:], width=0.6, color=C["purple"]); ax[0, 2].axhspan(-cb, cb, color=C["grey"], alpha=0.3); ax[0, 2].set_title("ACF of squared residuals (ARCH check)")
ax[1, 0].hist(z_res, bins=40, density=True, color="#90cdf4", edgecolor="white")
xs_ = np.linspace(-4, 4, 200); ax[1, 0].plot(xs_, stats.norm.pdf(xs_), color=C["red"]); ax[1, 0].set_title(f"Histogram (skew {sk:.2f}, kurt {ku:.2f})")
stats.probplot(z_res, dist="norm", plot=ax[1, 1]); ax[1, 1].set_title("Normal Q-Q"); ax[1, 1].get_lines()[0].set_markersize(3)
lags_ = np.arange(1, 61); pv_ = [ljung_box(z_res, L, ddof)[1] for L in lags_]
ax[1, 2].scatter(lags_, pv_, s=10, color=C["blue"]); ax[1, 2].axhline(0.05, color=C["red"], ls="--"); ax[1, 2].set_ylim(0, 1); ax[1, 2].set_title("Ljung-Box p-values by lag"); ax[1, 2].set_xlabel("lag")
save(fig, "fig10_residual_diagnostics")

# fig11 out-of-sample fixed origin
test_idx = s.index[train_n:]
show = [nm for nm in NAMES if nm not in ("Mean",)]
_cols = [C["grey"], "#a0aec0", C["purple"], C["gold"], "#718096", C["green"], C["teal"], C["blue"], C["orange"], C["red"], "#805ad5", "#d53f8c", "#2c5282"]
palette = dict(zip(NAMES, _cols))
fig, ax = plt.subplots(2, 1, figsize=(11, 8.5))
ax[0].plot(s.index[train_n - 120:train_n], y[train_n - 120:train_n], color=C["obs"], lw=1)
ax[0].plot(test_idx, y[train_n:], color=C["obs"], lw=1.4, label="Actual")
_f11 = ["Naive", "Seasonal naive", "Holt-Winters (A,N,A) m=365", f"SARIMA({best_sar[0]},0,{best_sar[1]})(0,1,0)[365]",
        f"Harmonic(K={K_}) + ARMA({p_},{q_})", "Gradient boosting (ML)"]
if best_nm not in _f11:
    _f11.insert(4, best_nm)
for nm in _f11:
    ax[0].plot(test_idx, fc_fixed[nm]["mean"], lw=1.4, label=f"{nm}  (RMSE {fixed_tab.loc[nm, 'RMSE']:.1f})", color=palette[nm])
ax[0].axvline(split_date, color="k", ls=":"); ax[0].set_ylabel("µg/m³")
ax[0].set_title(f"Out-of-sample, fixed origin: one forecast for all {test_n} held-out days"); ax[0].legend(ncol=2, fontsize=8)
fb = fc_fixed[best_nm]
ax[1].plot(test_idx, y[train_n:], color=C["obs"], lw=1.2, label="Actual")
ax[1].plot(test_idx, fb["mean"], color=C["blue"], lw=1.8, label=best_nm)
ax[1].fill_between(test_idx, fb["lo95" + SFX], fb["hi95" + SFX], color=C["blue"], alpha=0.12, label="95% PI")
ax[1].fill_between(test_idx, fb["lo80" + SFX], fb["hi80" + SFX], color=C["blue"], alpha=0.22, label="80% PI")
ax[1].set_ylabel("µg/m³"); ax[1].legend(ncol=4, loc="upper right"); ax[1].set_title("Best model with prediction intervals")
save(fig, "fig11_oos_fixed_origin")

# fig12 walk-forward errors by horizon
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
for nm in NAMES:
    if nm == "Mean":
        continue
    lw_ = 2.4 if nm == best_nm else 1.2
    ax[0].plot(byh[nm].index, byh[nm].values, label=nm, color=palette[nm], lw=lw_)
ax[0].set_xlabel("Forecast horizon h (days)"); ax[0].set_ylabel("RMSE (µg/m³)"); ax[0].set_title("Walk-forward RMSE by horizon")
ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
order_ = wf_tab["RMSE avg 1-30"].sort_values()
ax[1].barh(order_.index, order_.values, color=[palette[k] for k in order_.index]); ax[1].invert_yaxis()
ax[1].set_xlabel("RMSE averaged over h=1..30"); ax[1].set_title("Overall ranking (rolling origin)")
save(fig, "fig12_walkforward_horizon")

# fig13 one-step-ahead OOS zoom
fig, ax = plt.subplots(figsize=(11, 4.2))
z_ = slice(train_n, train_n + 120)
ax.plot(s.index[z_], y[z_], color=C["obs"], lw=1.3, label="Actual")
for nm in ["Naive", best_nm]:
    ax.plot(s.index[z_], one_step_full[nm][z_], lw=1.4, color=palette[nm], label=f"{nm}  (1-step RMSE {oos1_tab.loc[nm, 'RMSE']:.2f})")
ax.set_ylabel("µg/m³"); ax.set_title("Out-of-sample one-step-ahead predictions (first 120 test days, parameters frozen at train estimates)"); ax.legend()
save(fig, "fig13_oos_onestep")

# fig14 final forecast
fig, ax = plt.subplots(figsize=(11, 4.8))
hist = slice(n - 300, n)
ax.plot(s.index[hist], y[hist], color=C["obs"], lw=1.1, label="Observed")
ax.plot(future_idx, fc_final["mean"], color=C["blue"], lw=2, label=f"Forecast ({best_nm})")
ax.fill_between(future_idx, fc_final["lo95" + SFX], fc_final["hi95" + SFX], color=C["blue"], alpha=0.12, label="95% PI")
ax.fill_between(future_idx, fc_final["lo80" + SFX], fc_final["hi80" + SFX], color=C["blue"], alpha=0.25, label="80% PI")
ax.axhline(60, color=C["red"], ls="--", lw=0.8); ax.text(future_idx[0], 61, "India NAAQS 24-h (60)", color=C["red"], fontsize=8)
ax.set_ylabel("µg/m³"); ax.set_title(f"90-day forecast, {future_idx[0].date()} to {future_idx[-1].date()} (model refit on all data)"); ax.legend(ncol=4, loc="upper left")
save(fig, "fig14_final_forecast")

# =============================================================================
# 14. Summary table + save
# =============================================================================
summ = pd.DataFrame({
    "In-sample RMSE": ins_tab["RMSE"], "OOS 1-step RMSE": oos1_tab["RMSE"], "OOS fixed-origin RMSE": fixed_tab["RMSE"],
    "OOS fixed-origin MAE": fixed_tab["MAE"], "OOS fixed-origin MASE": fixed_tab["MASE"], "OOS fixed-origin R2": fixed_tab["R2"],
    "Walk-fwd RMSE h=7": wf_tab["RMSE h=7"], "Walk-fwd RMSE h=30": wf_tab["RMSE h=30"], "Walk-fwd RMSE avg": wf_tab["RMSE avg 1-30"]})
summ["OOS/IS 1-step ratio"] = summ["OOS 1-step RMSE"] / summ["In-sample RMSE"]
summ.round(3).to_csv(f"{TAB}/summary_all_models.csv")
print(summ.round(2).to_string())
R["tables"] = {"insample": ins_tab.round(4).reset_index().to_dict("records"),
               "oos1": oos1_tab.round(4).reset_index().to_dict("records"),
               "fixed": fixed_tab.round(4).reset_index().to_dict("records"),
               "wf": wf_tab.round(4).reset_index().to_dict("records"),
               "dm": dm_tab.round(4).reset_index().to_dict("records"),
               "summary": summ.round(4).reset_index().to_dict("records")}
R["scale_mase"] = scale_mase
R["hw"] = {"alpha": float(fitted["Holt-Winters (A,N,A) m=365"].alpha), "gamma": float(fitted["Holt-Winters (A,N,A) m=365"].gamma)}
R["params_all"] = {nm: fitted[nm].param_table().round(4).to_dict("records") for nm in NAMES if hasattr(fitted[nm], "param_table")}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


with open(f"{OUT}/results.json", "w") as fh:
    json.dump(clean(R), fh, indent=1)
tick("DONE")
