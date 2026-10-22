import json, numpy as np, pandas as pd, tslib
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CI = ["bengaluru", "chennai", "hyderabad", "ahmedabad", "mumbai", "kolkata", "delhi"]; O = "out/compare"
def pp_test(x, reps=4000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    Y, Xl = x[1:], np.column_stack([np.ones(len(x) - 1), x[:-1]]); b = np.linalg.lstsq(Xl, Y, rcond=None)[0]; u = Y - Xl @ b; n_ = len(Y)
    ssr = u @ u; s2 = ssr / (n_ - 2); se = np.sqrt(s2 * np.linalg.inv(Xl.T @ Xl)[1, 1]); tau = (b[1] - 1) / se
    l = int(np.floor(4 * (n_ / 100) ** (2 / 9))); g0 = ssr / n_; lam2 = g0
    for j in range(1, l + 1): lam2 += 2 * (1 - j / (l + 1)) * (u[j:] @ u[:-j]) / n_
    zt = np.sqrt(g0 / lam2) * tau - 0.5 * (lam2 - g0) / np.sqrt(lam2) * (n_ * se / np.sqrt(s2))
    dist = tslib._df_null_distribution(n_, "c", reps); return zt, (np.sum(dist <= zt) + 1) / (len(dist) + 1)
def pick(names, kind):
    for n in names:
        if kind == "naive" and n == "Naive": return n
        if kind == "snaive" and n == "Seasonal naive": return n
        if kind == "arima0" and n.startswith("ARIMA(") and "+const" in n: return n
        if kind == "arima1" and n.startswith("ARIMA(") and "+const" not in n: return n
        if kind == "hw" and n.startswith("Holt"): return n
        if kind == "sarima" and n.startswith("SARIMA"): return n
        if kind == "gbm" and n.startswith("Gradient"): return n
        if kind == "log" and "[log]" in n: return n
    return None
KINDS = [("Naive", "naive"), ("Seasonal naive", "snaive"), ("ARIMA d=0", "arima0"), ("ARIMA d=1", "arima1"), ("Holt-Winters", "hw"), ("SARIMA[365]", "sarima"), ("Harmonic+ARMA (log)", "log"), ("Gradient boosting", "gbm")]
S, ST, M, W, OS, FX, DG, FC, DMt = [], [], [], [], [], [], [], [], []
for c in CI:
    d = f"out/{c}"; R = json.load(open(f"{d}/results.json")); best = R["best_model"]
    ser = pd.read_csv(f"data/analysis/{c}/air_quality_historical.csv", parse_dates=["date"]).set_index("date").pm2_5.dropna(); n = len(ser); tr = R["data"]["train_n"]; ytr = ser.values[:tr]
    S.append(dict(city=c, n=n, start=R["data"]["start"], end=R["data"]["end"], train=tr, test=R["data"]["test_n"], mean=R["desc"]["mean"], sd=R["desc"]["std"], skew=R["desc"]["skew"], kurt=R["desc"]["kurt"],
                  seas_strength=R["decomp"]["fs_add"], trend_strength=R["decomp"]["ft_add"], lag1=R["acf"]["raw_lag1"], lag365=R["acf"]["raw_lag365"], top_period=R["periodogram_top"][0]["period_days"]))
    X = np.column_stack([np.ones(tr), tslib.fourier_terms(np.arange(tr, dtype=float), 4)]); ds = ytr - X @ np.linalg.lstsq(X, ytr, rcond=None)[0]
    for lab, x in (("level", ytr), ("deseasonalised", ds), ("first difference", np.diff(ytr))):
        a = tslib.adf_test(x, "c"); k = tslib.kpss_test(x, "c"); zt, pp = pp_test(x)
        ST.append(dict(city=c, series=lab, ADF_stat=a["stat"], ADF_p=a["pvalue"], PP_p=pp, KPSS_stat=k["stat"], KPSS_p=k["pvalue"], ADF_rej=a["reject_5pct"], KPSS_rej=k["reject_5pct"]))
    import re; mm_ = re.search(r"Harmonic\(K=(\d+)\) \+ ARMA\((\d+),(\d+)\)", best); K = [int(v) for v in mm_.groups()] if mm_ else [0, *map(int, re.search(r"ARIMA\((\d+),1,(\d+)\)", best).groups())]; dg = R["diag"]
    M.append(dict(city=c, model=best, log="[log]" in best, K=K[0], p=K[1], q=K[2], sigma=dg["sigma"], AIC=dg["aic"], BIC=dg["bic"], LB20_p=dg["LB20_p"], LB60_p=dg["LB60_p"], JB_p=dg["JB_p"], kurt=dg["kurt"], ARCH_p=dg["ARCH_p"],
                  resid_sd_ratio_winter_summer=np.mean([dg["resid_sd_by_month"][str(m)] for m in (11, 12, 1)]) / np.mean([dg["resid_sd_by_month"][str(m)] for m in (6, 7, 8)]), best_any=R["best_any"], pi=R["pi_type"], cov_const=R["pi_cov"]["constant_sd_wf"], cov_seas=R["pi_cov"]["seasonal_sd_wf"]))
    o1 = pd.read_csv(f"{d}/tables/oos_onestep_metrics.csv", index_col=0); fx = pd.read_csv(f"{d}/tables/oos_fixed_origin_metrics.csv", index_col=0); wf = pd.read_csv(f"{d}/tables/oos_walkforward_metrics.csv", index_col=0)
    ins = pd.read_csv(f"{d}/tables/insample_metrics.csv", index_col=0); dm = pd.read_csv(f"{d}/tables/diebold_mariano_vs_best.csv", index_col=0)
    names = list(o1.index); rows = [("Mean", "Mean")] + KINDS + [("Selected model", best)]
    for lab, key in rows:
        nm = key if key in names else pick(names, key)
        if nm is None: continue
        OS.append(dict(city=c, model=lab, spec=nm, RMSE_in=ins.loc[nm, "RMSE"], RMSE_1step=o1.loc[nm, "RMSE"], MASE_1step=o1.loc[nm, "MASE"], R2_1step=o1.loc[nm, "R2"], RMSE_fixed=fx.loc[nm, "RMSE"], R2_fixed=fx.loc[nm, "R2"],
                       WF_h1=wf.loc[nm, "RMSE h=1"], WF_h7=wf.loc[nm, "RMSE h=7"], WF_h30=wf.loc[nm, "RMSE h=30"], WF_avg=wf.loc[nm, "RMSE avg 1-30"], WF_MAE_avg=wf.loc[nm, "MAE avg 1-30"]))
    for lab, key in KINDS[:2] + [("Mean", "Mean")]:
        nm = key if key in dm.index else pick(list(dm.index), key)
        if nm: DMt.append(dict(city=c, vs=lab, DM1=dm.loc[nm, "DM 1-step"], p1=dm.loc[nm, "p 1-step"], DM7=dm.loc[nm, "DM h=7"], p7=dm.loc[nm, "p h=7"], DM30=dm.loc[nm, "DM h=30"], p30=dm.loc[nm, "p h=30"]))
    f = pd.read_csv(f"{d}/tables/final_forecast_90d.csv", parse_dates=["date"]); f["city"] = c; FC.append(f)
for nm, L in (("data_summary", S), ("stationarity_all", ST), ("selected_models", M), ("model_comparison", OS), ("dm_best_vs", DMt)): pd.DataFrame(L).round(4).to_csv(f"{O}/{nm}.csv", index=False)
F = pd.concat(FC); F[["lo80","lo95"]] = F[["lo80","lo95"]].clip(lower=0); F.round(2).to_csv(f"{O}/forecast_90d_all.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(S).round(2).to_string()); print(pd.DataFrame(ST).round(3).to_string()); print(pd.DataFrame(M).round(3).drop(columns=["model"]).to_string())
os_ = pd.DataFrame(OS); print(os_.pivot(index="city", columns="model", values="RMSE_1step").round(2).to_string()); print(os_.pivot(index="city", columns="model", values="WF_avg").round(2).to_string()); print(pd.DataFrame(DMt).round(3).to_string())
