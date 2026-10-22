import json, numpy as np, pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tslib as T
import os; os.makedirs("out/compare", exist_ok=True)
CI = ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata"]; O = "out/compare"
S = {c: pd.read_csv(f"data/analysis/{c}/air_quality_historical.csv", parse_dates=["date"]).set_index("date").pm2_5.dropna() for c in CI}
dl = pd.read_csv("data/analysis/delhi/air_quality_historical.csv", parse_dates=["date"]).set_index("date").pm2_5
P = pd.DataFrame(S).dropna()
rows = []
for c in CI + ["delhi"]:
    s = P[c] if c != "delhi" else dl
    rows.append(dict(city=c, n=len(s), mean=s.mean(), median=s.median(), sd=s.std(), cv=s.std() / s.mean(), p95=s.quantile(.95), max=s.max(), skew=stats.skew(s),
                     pct_gt15=(s > 15).mean() * 100, pct_gt35=(s > 35).mean() * 100, pct_gt60=(s > 60).mean() * 100, pct_gt90=(s > 90).mean() * 100))
Dsc = pd.DataFrame(rows); Dsc.round(2).to_csv(f"{O}/descriptives.csv", index=False)
# tests across six CAMS cities on the common calendar
kw = stats.kruskal(*[P[c] for c in CI]); fr = stats.friedmanchisquare(*[P[c] for c in CI])
pw = []
for i, a in enumerate(CI):
    for b in CI[i + 1:]:
        w = stats.wilcoxon(P[a], P[b]); pw.append(dict(a=a, b=b, mean_diff=(P[a] - P[b]).mean(), wilcoxon_p=w.pvalue))
pw = pd.DataFrame(pw); m = len(pw); o = pw.wilcoxon_p.argsort().values; adj = np.empty(m); run = 0
for rank, i in enumerate(o):
    run = max(run, (m - rank) * pw.wilcoxon_p.values[i]); adj[i] = min(1, run)
pw["holm_p"] = adj; pw.round(4).to_csv(f"{O}/pairwise_wilcoxon_holm.csv", index=False)
# monthly profile and seasonal ratios
mm = pd.DataFrame({c: P[c].groupby(P.index.month).mean() for c in CI}); mm["delhi"] = dl.groupby(dl.index.month).mean(); mm.round(2).to_csv(f"{O}/monthly_means.csv")
sea = pd.DataFrame({c: dict(peak_month=int(mm[c].idxmax()), trough_month=int(mm[c].idxmin()), peak_to_trough=mm[c].max() / mm[c].min(),
                            winter_DJF=mm[c].loc[[12, 1, 2]].mean(), monsoon_JJAS=mm[c].loc[[6, 7, 8, 9]].mean()) for c in CI + ["delhi"]}).T
sea["winter_over_monsoon"] = sea.winter_DJF / sea.monsoon_JJAS; sea.round(2).to_csv(f"{O}/seasonality_summary.csv")
# synchrony: correlation of levels and of anomalies (after removing each city's own harmonic K=4 cycle)
t = np.arange(len(P), dtype=float); X = np.column_stack([np.ones(len(P)), T.fourier_terms(t, 4)])
A = pd.DataFrame({c: P[c].values - X @ np.linalg.lstsq(X, P[c].values, rcond=None)[0] for c in CI}, index=P.index)
cl, ca = P.corr(method="spearman"), A.corr(method="spearman"); cl.round(2).to_csv(f"{O}/corr_levels.csv"); ca.round(2).to_csv(f"{O}/corr_anomalies.csv")
# lead-lag: best lag of cross-correlation between anomaly series (|lag|<=5)
ll = []
for i, a in enumerate(CI):
    for b in CI[i + 1:]:
        cc = {k: np.corrcoef(A[a].values[max(0, -k):len(A) - max(0, k)], A[b].values[max(0, k):len(A) - max(0, -k)])[0, 1] for k in range(-5, 6)}
        k = max(cc, key=lambda z: abs(cc[z])); ll.append(dict(a=a, b=b, best_lag=k, corr_best=cc[k], corr_lag0=cc[0]))
pd.DataFrame(ll).round(3).to_csv(f"{O}/leadlag_anomalies.csv", index=False)
# year-by-year (complete calendar years) ranking
Y = pd.DataFrame({c: S[c].groupby(S[c].index.year).mean() for c in CI}).loc[2023:2025]; Y.round(2).to_csv(f"{O}/annual_means_complete_years.csv")
# variance decomposition: share of total variance explained by city, by month-of-year, interaction
long = P.stack().rename("v").reset_index(); long.columns = ["date", "city", "v"]; long["mo"] = long.date.dt.month
gm = long.v.mean(); sst = ((long.v - gm) ** 2).sum()
ss_city = long.groupby("city").v.transform("mean").sub(gm).pow(2).sum() / sst; ss_mo = long.groupby("mo").v.transform("mean").sub(gm).pow(2).sum() / sst
ss_cm = long.groupby(["city", "mo"]).v.transform("mean").sub(gm).pow(2).sum() / sst
json.dump(dict(kruskal=[kw.statistic, kw.pvalue], friedman=[fr.statistic, fr.pvalue], n_days=len(P), start=str(P.index[0].date()), end=str(P.index[-1].date()),
               share_city=ss_city, share_month=ss_mo, share_city_x_month=ss_cm - ss_city - ss_mo, share_residual=1 - ss_cm), open(f"{O}/tests.json", "w"), indent=1, default=float)
# figures
cols = dict(zip(CI + ["delhi"], ["#2b6cb0", "#2f855a", "#dd6b20", "#b7791f", "#6b46c1", "#c53030", "#111111"]))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c in CI + ["delhi"]: ax[0].plot(range(1, 13), mm[c], marker="o", ms=3, label=c.title() + (" (stations)" if c == "delhi" else ""), color=cols[c], ls="--" if c == "delhi" else "-")
ax[0].set_xticks(range(1, 13)); ax[0].set_xlabel("Month"); ax[0].set_ylabel("µg/m³"); ax[0].set_title("Mean PM2.5 by calendar month"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.25)
bp = ax[1].boxplot([P[c] for c in CI] + [dl], tick_labels=[c[:5].title() for c in CI] + ["Delhi*"], showfliers=False, patch_artist=True)
for p_, c in zip(bp["boxes"], CI + ["delhi"]): p_.set_facecolor(cols[c]); p_.set_alpha(.5)
for a in ax:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
ax[1].set_title("Distribution of daily PM2.5 (*Delhi: ground stations)"); ax[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(f"{O}/fig_city_profiles.png", dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for a, M, ti in ((ax[0], cl, "Spearman, daily levels"), (ax[1], ca, "Spearman, deseasonalised anomalies")):
    im = a.imshow(M.values, vmin=0, vmax=1, cmap="YlOrRd"); a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels([c[:5].title() for c in CI], rotation=45); a.set_yticklabels([c.title() for c in CI]); a.grid(False)
    for i in range(6):
        for j in range(6): a.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    a.set_title(ti, fontsize=10)
fig.tight_layout(); fig.savefig(f"{O}/fig_synchrony.png", dpi=150)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(Dsc.round(1).to_string()); print(kw, fr); print(pw.round(3).to_string()); print(sea.round(2)); print(ca.round(2)); print(pd.DataFrame(ll).round(2)); print(Y.round(1)); print(open(f"{O}/tests.json").read())
