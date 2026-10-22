import matplotlib.dates
import json, numpy as np, pandas as pd, tslib as T
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 170})
import os; os.makedirs("out/paper_figs", exist_ok=True)
CI = ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata", "delhi"]; O = "out/paper_figs"
COL = dict(zip(CI, ["#2b6cb0", "#2f855a", "#dd6b20", "#b7791f", "#6b46c1", "#c53030", "#111111"]))
ser = {c: pd.read_csv(f"data/analysis/{c}/air_quality_historical.csv", parse_dates=["date"]).set_index("date").pm2_5.dropna() for c in CI}
R = {c: json.load(open(f"out/{c}/results.json")) for c in CI}
lab = lambda c: c.title() + (" (stations)" if c == "delhi" else " (CAMS)")
# 1 series panels
fig, ax = plt.subplots(4, 2, figsize=(11, 10), sharex=False); ax = ax.ravel()
for a, c in zip(ax, CI):
    s = ser[c]; a.plot(s.index, s, color="#aaa", lw=.6); a.plot(s.index, s.rolling(30, center=True).mean(), color=COL[c], lw=1.6)
    a.axvspan(s.index[R[c]["data"]["train_n"]], s.index[-1], color="#f6ad55", alpha=.18); a.axhline(15, color="g", ls="--", lw=.7); a.axhline(60, color="r", ls="--", lw=.7); a.set_title(lab(c)); a.set_ylabel("µg/m³"); a.tick_params(labelsize=7)
ax[-1].axis("off"); ax[-1].text(0, .7, "Grey: daily PM2.5   Coloured: 30-day mean\nShaded: held-out test period (last 20%)\nGreen/red dashes: WHO 24-h guideline (15)\nand India NAAQS 24-h (60) µg/m³\nNote: y-axis scales differ by city", fontsize=9, va="top")
fig.tight_layout(); fig.savefig(f"{O}/f1_series.png"); plt.close(fig)
# 2 seasonal components (classical decomposition), by day of year, normalised view in absolute units
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); fs = []
for c in CI:
    d = T.classical_decompose(ser[c], "additive", 365); sea = d["seasonal"].dropna(); g = sea.groupby(sea.index.dayofyear).mean()
    ax[0].plot(g.index, g.values, color=COL[c], label=c.title(), ls="--" if c == "delhi" else "-"); fs.append(R[c]["decomp"]["fs_add"])
ax[0].set_xlabel("Day of year"); ax[0].set_ylabel("Seasonal component (µg/m³)"); ax[0].set_title("Estimated annual cycle"); ax[0].legend(ncol=2, fontsize=7)
ax[1].bar([c[:5].title() for c in CI], fs, color=[COL[c] for c in CI]); ax[1].set_ylim(0, 1); ax[1].set_title("Seasonal strength (1 = purely seasonal)")
fig.tight_layout(); fig.savefig(f"{O}/f2_seasonal.png"); plt.close(fig)
# 3 ACF of deseasonalised
fig, ax = plt.subplots(2, 4, figsize=(11, 5), sharey=True); ax = ax.ravel()
for a, c in zip(ax, CI):
    y = ser[c].values[:R[c]["data"]["train_n"]]; X = np.column_stack([np.ones(len(y)), T.fourier_terms(np.arange(len(y), dtype=float), 4)]); r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ac = T.acf(r, 40); a.bar(range(41), ac, color=COL[c], width=.6); cb = 1.96 / np.sqrt(len(r)); a.axhspan(-cb, cb, color="#ccc", alpha=.5); a.set_title(c.title()); a.set_ylim(-.3, 1)
ax[-1].axis("off"); fig.suptitle("Autocorrelation of the deseasonalised series (training sample, lags 0–40)", fontsize=10); fig.tight_layout(); fig.savefig(f"{O}/f3_acf.png"); plt.close(fig)
# 4 forecast accuracy relative to naive
om = pd.read_csv("out/compare/model_comparison.csv"); mods = ["Seasonal naive", "Holt-Winters", "SARIMA[365]", "ARIMA d=0", "Gradient boosting", "Selected model"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, col, ti in ((ax[0], "RMSE_1step", "One-step-ahead, test period"), (ax[1], "WF_avg", "Walk-forward, horizons 1–30 days")):
    pv = om.pivot(index="city", columns="model", values=col).reindex(CI); rel = pv[mods].div(pv["Naive"], axis=0); w = .13
    for i, m in enumerate(mods): a.bar(np.arange(7) + i * w, rel[m], w, label=m, color=["#a0aec0", "#ed8936", "#9f7aea", "#4299e1", "#48bb78", "#e53e3e"][i])
    a.axhline(1, color="k", lw=.8); a.set_xticks(np.arange(7) + 2.5 * w); a.set_xticklabels([c[:5].title() for c in CI]); a.set_title(ti + " (RMSE / naive RMSE)"); a.set_ylim(0, 1.5)
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=6, loc="lower center", fontsize=8); fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(f"{O}/f4_accuracy.png"); plt.close(fig)
# 5 RMSE by horizon of selected model, scaled by city mean
fig, ax = plt.subplots(figsize=(6.5, 4))
for c in CI:
    h = pd.read_csv(f"out/{c}/tables/oos_walkforward_rmse_by_horizon.csv", index_col=0); b = R[c]["best_model"]; ax.plot(h.index, h[b] / ser[c].mean(), color=COL[c], label=c.title(), ls="--" if c == "delhi" else "-")
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("RMSE / mean PM2.5"); ax.set_title("Walk-forward error of the selected model"); ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.15)); fig.tight_layout(); fig.savefig(f"{O}/f5_horizon.png"); plt.close(fig)
# 6 final forecasts
fig, ax = plt.subplots(4, 2, figsize=(11, 10)); ax = ax.ravel(); F = pd.read_csv("out/compare/forecast_90d_all.csv", parse_dates=["date"])
for a, c in zip(ax, CI):
    f = F[F.city == c]; s = ser[c].iloc[-240:]; a.plot(s.index, s, color="#666", lw=.8); a.fill_between(f.date, f.lo95, f.hi95, color="#bee3f8", alpha=.8); a.fill_between(f.date, f.lo80, f.hi80, color="#63b3ed", alpha=.8); a.plot(f.date, f["mean"], color="#c53030", lw=1.6); a.set_ylim(bottom=0); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y")); a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2)); a.set_title(lab(c)); a.tick_params(labelsize=7); a.set_ylabel("µg/m³")
ax[-1].axis("off"); ax[-1].text(0, .7, "Grey: last 240 observed days\nRed: 90-day forecast (selected model)\nDark / light band: 80% / 95% interval\nForecast origin differs by city (data end dates)", fontsize=9, va="top")
fig.tight_layout(); fig.savefig(f"{O}/f6_forecasts.png"); plt.close(fig)
print("ok")
