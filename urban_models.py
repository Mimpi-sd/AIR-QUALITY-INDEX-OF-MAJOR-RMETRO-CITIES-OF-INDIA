"""Objective 3: link air quality to urban characteristics. Run from the repository root."""
import itertools, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tslib as T
C = "data/cams"; OUT = "out/urban"
import os; [os.makedirs(f"{OUT}/{d}", exist_ok=True) for d in ("tables", "figures")]
CITIES = ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata"]
D = {c: pd.read_csv(f"{C}/{c}/air_quality_historical.csv", parse_dates=["date"]).set_index("date") for c in CITIES}
info = pd.concat([pd.read_csv(f"{C}/{c}/city_info.csv").assign(city=c) for c in CITIES]).set_index("city")
# supplementary, hand-coded (NOT from the dataset; general geography, approximate)
info["coastal"] = [0, 1, 0, 0, 1, 0]; info["elevation_m"] = [920, 6, 540, 53, 14, 9]
# city-level outcomes from the data (complete years 2023-2025)
rows = []
for c in CITIES:
    d = D[c]; s = d.pm2_5["2023":"2025"].dropna(); t = np.arange(len(s), dtype=float)
    X = np.column_stack([np.ones(len(s)), T.fourier_terms(t, 1)]); b = np.linalg.lstsq(X, s.values, rcond=None)[0]
    amp = np.hypot(b[1], b[2]); m = s.groupby(s.index.month).mean()
    rows.append(dict(city=c, mean_pm25=s.mean(), annual_amp=amp, winter_mean=s[s.index.month.isin([12, 1, 2])].mean(), monsoon_mean=s[s.index.month.isin([7, 8, 9])].mean(),
                     pct_days_above_60=(s > 60).mean() * 100, mean_no2=d.nitrogen_dioxide["2023":"2025"].mean(), mean_co=d.carbon_monoxide["2023":"2025"].mean(),
                     mean_dust=d.dust["2023":"2025"].mean(), mean_aod=d.aerosol_optical_depth["2023":"2025"].mean(), mean_o3=d.ozone["2023":"2025"].mean()))
O = pd.DataFrame(rows).set_index("city"); city = info[["population", "latitude", "longitude", "coastal", "elevation_m"]].join(O)
city["log_pop"] = np.log10(city.population); city.round(3).to_csv(f"{OUT}/tables/city_level_table.csv")

# ---- A. cross-city association, exact permutation test on Spearman rho (n=6 -> 720 permutations)
def perm_rho(x, y):
    r0 = stats.spearmanr(x, y)[0]; rs = np.array([stats.spearmanr(x, np.array(p))[0] for p in itertools.permutations(y)])
    return r0, np.mean(np.abs(rs) >= abs(r0) - 1e-12)
chars = ["population", "latitude", "longitude", "coastal", "elevation_m"]; outs = ["mean_pm25", "winter_mean", "annual_amp", "monsoon_mean"]
A = []
for ch in chars:
    for o in outs:
        r, p = perm_rho(city[ch].values, city[o].values); A.append(dict(characteristic=ch, outcome=o, spearman=r, exact_perm_p=p))
A = pd.DataFrame(A); A.round(3).to_csv(f"{OUT}/tables/crosscity_spearman.csv", index=False)

# leave-one-city-out CV for mean PM2.5: intercept-only vs one-predictor OLS vs ridge(2 predictors)
def loo(cols, o="mean_pm25", lam=0.0):
    err = []
    for i, c in enumerate(city.index):
        tr = city.drop(c); 
        if not cols: pred = tr[o].mean()
        else:
            mu, sd = tr[cols].mean(), tr[cols].std(); Z = ((tr[cols] - mu) / sd).values; z = ((city.loc[[c], cols] - mu) / sd).values
            Z1 = np.column_stack([np.ones(len(Z)), Z]); P = lam * np.eye(Z1.shape[1]); P[0, 0] = 0
            b = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ tr[o].values); pred = (np.column_stack([1, z]) @ b)[0]
        err.append(city.loc[c, o] - pred)
    return np.sqrt(np.mean(np.square(err))), np.array(err)
L = []
for nm, cols, lam in [("intercept only", [], 0), ("log population", ["log_pop"], 0), ("latitude", ["latitude"], 0), ("longitude", ["longitude"], 0), ("coastal", ["coastal"], 0),
                      ("log pop + latitude (ridge 1)", ["log_pop", "latitude"], 1), ("latitude + coastal (ridge 1)", ["latitude", "coastal"], 1)]:
    r, e = loo(cols, lam=lam); L.append(dict(model=nm, loo_rmse=r, **{f"err_{c}": v for c, v in zip(city.index, e)}))
L = pd.DataFrame(L); L.round(2).to_csv(f"{OUT}/tables/crosscity_loo_cv.csv", index=False)

# ---- B. daily panel: PM2.5 vs co-pollutants / dust / AOD with city fixed effects, HAC errors
P = pd.concat([D[c].assign(city=c) for c in CITIES]).reset_index()
P["doy_s"] = np.sin(2 * np.pi * P.date.dt.dayofyear / 365.25); P["doy_c"] = np.cos(2 * np.pi * P.date.dt.dayofyear / 365.25)
P = P.dropna(subset=["pm2_5", "nitrogen_dioxide", "carbon_monoxide", "dust", "aerosol_optical_depth", "ozone"])
P["co_mg"] = P.carbon_monoxide / 1000
preds = ["nitrogen_dioxide", "co_mg", "dust", "aerosol_optical_depth", "ozone"]
def ols_hac(df, cols, L=14, fe=True):
    X = df[cols].values.astype(float); y = df.pm2_5.values
    if fe:
        d = pd.get_dummies(df.city, drop_first=False).values.astype(float); X = np.column_stack([X, d])
    else: X = np.column_stack([X, np.ones(len(X))])
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; n = len(y); XtXi = np.linalg.pinv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for j in range(1, L + 1):
        w = 1 - j / (L + 1); G = (X[j:] * e[j:, None]).T @ (X[:-j] * e[:-j, None]); S += w * (G + G.T)
    V = XtXi @ S @ XtXi; se = np.sqrt(np.diag(V)); r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return b[:len(cols)], se[:len(cols)], r2
# standardise within city (z-scores) so coefficients are comparable
Z = P.copy()
for c in preds + ["pm2_5"]: Z[c] = Z.groupby("city")[c].transform(lambda v: (v - v.mean()) / v.std())
rows = []
for c in CITIES:
    zc = Z[Z.city == c]; b, se, r2 = ols_hac(zc, preds, fe=False)
    rows.append(dict(model=c, **{f"b_{p}": v for p, v in zip(preds, b)}, **{f"t_{p}": v / s for p, v, s in zip(preds, b, se)}, R2=r2, n=len(zc)))
b, se, r2 = ols_hac(Z, preds, fe=True); rows.append(dict(model="pooled (city fixed effects)", **{f"b_{p}": v for p, v in zip(preds, b)}, **{f"t_{p}": v / s for p, v, s in zip(preds, b, se)}, R2=r2, n=len(Z)))
B = pd.DataFrame(rows); B.round(3).to_csv(f"{OUT}/tables/panel_regression_standardised.csv", index=False)

# leave-one-city-out prediction of daily PM2.5 (raw units): pooled model without FE, can a model learned elsewhere predict a new city?
def lcv(cols):
    out = []
    for c in CITIES:
        tr, te = P[P.city != c], P[P.city == c]
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[k] for k in cols]); b = np.linalg.lstsq(Xtr, tr.pm2_5, rcond=None)[0]
        pr = np.column_stack([np.ones(len(te))] + [te[k] for k in cols]) @ b; e = te.pm2_5 - pr
        out.append(dict(held_out=c, features=" + ".join(cols), RMSE=np.sqrt(np.mean(e ** 2)), R2=1 - np.sum(e ** 2) / np.sum((te.pm2_5 - te.pm2_5.mean()) ** 2), bias=-e.mean(),
                        R2_vs_trainmean=1 - np.sum(e ** 2) / np.sum((te.pm2_5 - tr.pm2_5.mean()) ** 2)))
    return out
V = pd.DataFrame(lcv(["nitrogen_dioxide", "co_mg"]) + lcv(preds) + lcv(preds + ["doy_s", "doy_c"]))
V.round(3).to_csv(f"{OUT}/tables/panel_leave_one_city_out.csv", index=False)
# correlation PM2.5 vs predictors, by city (raw)
Corr = pd.DataFrame({c: P[P.city == c][preds + ["pm2_5"]].corr(method="spearman")["pm2_5"][preds] for c in CITIES}).T; Corr.round(2).to_csv(f"{OUT}/tables/spearman_pm25_vs_copollutants.csv")

# figures
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4))
for a, (ch, lab) in zip(ax, [("log_pop", "log10 population"), ("latitude", "Latitude (°N)"), ("elevation_m", "Elevation (m, approx.)")]):
    a.scatter(city[ch], city.mean_pm25, s=40, color="#2b6cb0")
    for c in city.index: a.annotate(c.title(), (city.loc[c, ch], city.loc[c, "mean_pm25"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    r, p = perm_rho(city[ch].values, city.mean_pm25.values); a.set_title(f"Spearman ρ={r:.2f}, exact p={p:.2f}", fontsize=9); a.set_xlabel(lab); a.grid(alpha=.25)
ax[0].set_ylabel("Mean PM2.5 2023–25 (µg/m³, CAMS model)"); fig.suptitle("Six cities: mean PM2.5 against urban characteristics (n = 6)", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig_crosscity_scatter.png", dpi=140)
fig, a = plt.subplots(figsize=(8, 3.8)); Bp = B.set_index("model")[[f"b_{p}" for p in preds]]; Bp.columns = ["NO2", "CO", "dust", "AOD", "ozone"]; Bp.plot.bar(ax=a, width=.8); a.axhline(0, color="k", lw=.7)
a.set_ylabel("standardised coefficient"); a.set_title("Daily PM2.5 on co-pollutants (z-scores within city; HAC errors)", fontsize=10); a.tick_params(axis="x", labelsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig_panel_coefficients.png", dpi=140)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(city.round(2).to_string()); print(A.round(3).to_string()); print(L.round(2).iloc[:, :2].to_string()); print(B.round(2).to_string()); print(V.round(2).to_string()); print(Corr.round(2))
