"""How much has PM2.5 changed within the period covered by the data?
Six CAMS model cities (2022-08 .. 2026-02) and Delhi ground stations (2023-01 .. 2026-08).
Outputs: tables + one figure in <out>.
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, DEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
CITIES = ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata"]
rng = np.random.default_rng(7)


def nw_var_mean(e, L=14):
    e = np.asarray(e, float); e = e[~np.isnan(e)]; n = len(e); e = e - e.mean(); v = e @ e / n
    for j in range(1, L + 1):
        v += 2 * (1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n
    return v / n


def diff_test(a, b):
    """mean(b) - mean(a) with HAC standard error (windows are separate, so variances add)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    d = b.mean() - a.mean(); se = np.sqrt(nw_var_mean(a) + nw_var_mean(b))
    return d, se, d - 1.96 * se, d + 1.96 * se, 2 * (1 - stats.norm.cdf(abs(d / se)))


def mk(x):
    x = np.asarray(x); s = sum(np.sign(x[i + 1:] - x[i]).sum() for i in range(len(x) - 1)); n = len(x)
    z = (s - np.sign(s)) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18) if s else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def sen_boot(an, B=1500, blk=6):
    t = np.arange(len(an)) / 12; sl, ic, *_ = stats.theilslopes(an, t); fit = ic + sl * t; res = an - fit; n = len(an); out = []
    for _ in range(B):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n - blk + 1); idx += list(range(s, s + blk))
        out.append(stats.theilslopes(fit + res[idx[:n]], t)[0])
    return sl, *np.percentile(out, [2.5, 97.5])


def trend(series, start, end):
    m = series.resample("MS").mean(); m = m[(m.index >= start) & (m.index <= end)].dropna()
    an = m - m.groupby(m.index.month).transform("mean")
    sl, lo, hi = sen_boot(an.values)
    return sl, lo, hi, mk(an.values), len(an)


S = {c: pd.read_csv(f"{D}/{c}/air_quality_historical.csv", parse_dates=["date"]).set_index("date").pm2_5.dropna() for c in CITIES}
dl = pd.read_csv(f"{DEL}/delhi_air_quality_daily.csv", parse_dates=["date"]).set_index("date")

rows = []
def add(city, source, label, a, b, wa, wb):
    d, se, lo, hi, p = diff_test(a, b)
    rows.append(dict(city=city, source=source, comparison=label, window_A=wa, window_B=wb, mean_A=np.mean(a), mean_B=np.mean(b), change=d,
                     pct_change=d / np.mean(a) * 100, ci_low=lo, ci_high=hi, p=p, days_A=len(a), days_B=len(b)))

for c in CITIES:
    s = S[c]
    add(c, "CAMS model", "Full year 2023 -> 2025", s["2023"], s["2025"], "2023", "2025")
    add(c, "CAMS model", "Aug 4-Dec 31: 2022 -> 2025", s["2022-08-04":"2022-12-31"], s["2025-08-04":"2025-12-31"], "2022", "2025")
    add(c, "CAMS model", "Jan 1-Feb 18: 2023 -> 2026", s["2023-01-01":"2023-02-18"], s["2026-01-01":"2026-02-18"], "2023", "2026")
# Delhi: single fixed station (Dwarka) for 2023 -> 2025, six fixed stations for Jan-Feb 2024 -> 2026
dw = dl.pm2_5_dwarka.dropna(); core = dl.pm2_5_core6
add("delhi", "stations (Dwarka only)", "Full year 2023 -> 2025", dw["2023"], dw["2025"], "2023", "2025")
add("delhi", "stations (Dwarka only)", "Aug 4-Dec 31: 2023 -> 2025", dw["2023-08-04":"2023-12-31"], dw["2025-08-04":"2025-12-31"], "2023", "2025")
add("delhi", "stations (6 fixed)", "Full year 2024 -> 2025", core["2024"], core["2025"], "2024", "2025")
add("delhi", "stations (6 fixed)", "Jan 1-Feb 18: 2024 -> 2026", core["2024-01-01":"2024-02-18"], core["2026-01-01":"2026-02-18"], "2024", "2026")
R = pd.DataFrame(rows); R.round(3).to_csv(f"{OUT}/change_between_periods.csv", index=False)

T = []
for c in CITIES:
    sl, lo, hi, p, n = trend(S[c], "2022-09-01", "2026-01-01")
    T.append(dict(city=c, source="CAMS model", months=n, sen_per_year=sl, ci_low=lo, ci_high=hi, p_mannkendall=p, mean=S[c].mean()))
sl, lo, hi, p, n = trend(dw, "2023-01-01", "2025-12-01")
T.append(dict(city="delhi", source="stations (Dwarka only)", months=n, sen_per_year=sl, ci_low=lo, ci_high=hi, p_mannkendall=p, mean=dw.mean()))
sl, lo, hi, p, n = trend(core.dropna(), "2024-01-01", "2026-01-01")
T.append(dict(city="delhi", source="stations (6 fixed)", months=n, sen_per_year=sl, ci_low=lo, ci_high=hi, p_mannkendall=p, mean=core.mean()))
T = pd.DataFrame(T); T["pct_per_year"] = T.sen_per_year / T["mean"] * 100
T.round(3).to_csv(f"{OUT}/trend_tests_all_cities.csv", index=False)

# share of days above limits, complete years
E = []
for c in CITIES:
    for y in (2023, 2024, 2025):
        s = S[c][str(y)]; E.append(dict(city=c, year=y, mean=s.mean(), pct_days_above_15=(s > 15).mean() * 100, pct_days_above_60=(s > 60).mean() * 100))
for y in (2023, 2024, 2025):
    s = dw[str(y)]; E.append(dict(city="delhi (Dwarka stn)", year=y, mean=s.mean(), pct_days_above_15=(s > 15).mean() * 100, pct_days_above_60=(s > 60).mean() * 100))
E = pd.DataFrame(E); E.round(2).to_csv(f"{OUT}/annual_means_and_exceedance.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R[["city", "source", "comparison", "mean_A", "mean_B", "change", "pct_change", "ci_low", "ci_high", "p"]].round(2).to_string(index=False))
print(T.round(3).to_string(index=False))
print(E.pivot(index="city", columns="year", values="mean").round(1).to_string())

# figure: change 2023 -> 2025 (full year), with 95% interval
f = R[R.comparison.str.startswith("Full year")].copy(); f["lab"] = f.city + np.where(f.city == "delhi", " (" + f.source.str.extract(r"\((.*)\)")[0] + ")", "")
f = f.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for a, col, lab in ((ax[0], "change", "Change in mean PM2.5 (µg/m³)"), (ax[1], "pct_change", "Change (% of earlier year)")):
    for i, r in f.iterrows():
        lo, hi = (r.ci_low, r.ci_high) if col == "change" else (r.ci_low / r.mean_A * 100, r.ci_high / r.mean_A * 100)
        sig = r.p < 0.05
        a.errorbar(r[col], i, xerr=[[r[col] - lo], [hi - r[col]]], fmt="o", color="#c53030" if sig else "#2b6cb0", capsize=3)
    a.axvline(0, color="k", lw=.8); a.set_xlabel(lab); a.grid(alpha=.25)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_yticks(range(len(f))); ax[0].set_yticklabels(f.lab)
fig.suptitle("Change in annual mean PM2.5 between full years (95% HAC interval; red = significant at 5%)\nCAMS cities: 2023 vs 2025. Delhi: Dwarka station 2023 vs 2025; six-station network 2024 vs 2025", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_change_between_years.png", dpi=150, bbox_inches="tight")
