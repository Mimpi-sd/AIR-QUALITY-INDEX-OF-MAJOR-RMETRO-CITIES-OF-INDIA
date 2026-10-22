"""
tslib.py - Time-series toolkit implemented from scratch with numpy / scipy / sklearn.

Contents
--------
* acf, pacf, ljung_box, arch_lm, jarque_bera
* adf_test  (augmented Dickey-Fuller, AIC lag selection, Monte-Carlo p-values)
* kpss_test
* classical_decompose, fourier_terms, periodogram
* Forecasting models with a common interface (fit / one_step / forecast):
    Mean, Naive, SeasonalNaive, HoltWintersAdd (ETS(A,N,A), m=365),
    RegArima (ARMA errors + optional Fourier regression / differencing / log),
    GBMLag (gradient boosting on lags + Fourier terms, recursive forecasts)
* metrics, dm_test, walk_forward
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize, signal

# ----------------------------------------------------------------------------
# Basic diagnostics
# ----------------------------------------------------------------------------

def acf(x, nlags):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    n = len(x)
    d = np.dot(x, x)
    return np.array([np.dot(x[:n - k], x[k:]) / d for k in range(nlags + 1)])


def pacf(x, nlags):
    """Partial autocorrelation via Durbin-Levinson."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = [1.0]
    phi[1, 1] = r[1]
    out.append(r[1])
    for k in range(2, nlags + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[k - 1:0:-1])
        den = 1 - np.dot(phi[k - 1, 1:k], r[1:k])
        phi[k, k] = num / den
        phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, k - 1:0:-1]
        out.append(phi[k, k])
    return np.array(out)


def ljung_box(x, lags, ddof=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    r = acf(x, lags)[1:]
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return Q, 1 - stats.chi2.cdf(Q, max(lags - ddof, 1))


def arch_lm(e, lags=5):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    e2 = e ** 2
    Y = e2[lags:]
    X = np.column_stack([np.ones(len(Y))] + [e2[lags - i:-i] for i in range(1, lags + 1)])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = Y - X @ b
    r2 = 1 - res @ res / ((Y - Y.mean()) @ (Y - Y.mean()))
    lm = len(Y) * r2
    return lm, 1 - stats.chi2.cdf(lm, lags)


def jarque_bera(e):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    jb, p = stats.jarque_bera(e)
    return jb, p, stats.skew(e), stats.kurtosis(e, fisher=False)


# ----------------------------------------------------------------------------
# Unit-root / stationarity tests
# ----------------------------------------------------------------------------

def _adf_design(y, k, start, regression):
    dy = np.diff(y)
    rows = np.arange(start, len(dy))
    cols = [y[rows]]
    for i in range(1, k + 1):
        cols.append(dy[rows - i])
    if regression in ("c", "ct"):
        cols.append(np.ones(len(rows)))
    if regression == "ct":
        cols.append(np.arange(1, len(rows) + 1, dtype=float))
    return np.column_stack(cols), dy[rows]


def _ols(X, Y):
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ b
    ssr = res @ res
    N, K = X.shape
    s2 = ssr / (N - K)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return b, np.sqrt(np.diag(cov)), ssr, N, K


_MC_CACHE = {}


def _df_null_distribution(n, regression, reps=4000, seed=123):
    """Monte-Carlo distribution of the Dickey-Fuller tau statistic under a random walk."""
    key = (n, regression, reps)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((reps, n))
    y = np.cumsum(eps, axis=1)
    ylag = y[:, :-1]
    dy = np.diff(y, axis=1)
    m = ylag.shape[1]
    if regression == "n":
        Z = None
    elif regression == "c":
        Z = np.ones((m, 1))
    else:
        Z = np.column_stack([np.ones(m), np.arange(1, m + 1)])
    if Z is not None:
        P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        ylag = ylag - ylag @ P
        dy = dy - dy @ P
        kz = Z.shape[1]
    else:
        kz = 0
    sxx = np.sum(ylag ** 2, axis=1)
    beta = np.sum(ylag * dy, axis=1) / sxx
    res = dy - beta[:, None] * ylag
    s2 = np.sum(res ** 2, axis=1) / (m - 1 - kz)
    tau = beta / np.sqrt(s2 / sxx)
    tau.sort()
    _MC_CACHE[key] = tau
    return tau


def adf_test(y, regression="c", maxlag=None, reps=4000):
    """Augmented Dickey-Fuller test (H0: unit root).
    Lag length chosen by AIC; p-value and critical values from a Monte-Carlo simulation
    of the Dickey-Fuller null distribution at the same sample size."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12 * (n / 100) ** 0.25))
    best = None
    for k in range(maxlag + 1):
        X, Y = _adf_design(y, k, maxlag, regression)
        b, se, ssr, N, K = _ols(X, Y)
        aic = N * np.log(ssr / N) + 2 * K
        if best is None or aic < best[0]:
            best = (aic, k)
    k = best[1]
    X, Y = _adf_design(y, k, k, regression)
    b, se, ssr, N, K = _ols(X, Y)
    tau = b[0] / se[0]
    dist = _df_null_distribution(N, regression, reps)
    p = (np.sum(dist <= tau) + 1) / (len(dist) + 1)
    crit = {"1%": np.quantile(dist, 0.01), "5%": np.quantile(dist, 0.05), "10%": np.quantile(dist, 0.10)}
    return {"stat": tau, "pvalue": p, "lags": k, "nobs": N, "crit": crit,
            "reject_5pct": bool(tau < crit["5%"]), "regression": regression}


def kpss_test(y, regression="c", nlags=None):
    """KPSS test (H0: stationary around a constant 'c' or a deterministic trend 'ct')."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = len(y)
    if regression == "c":
        e = y - y.mean()
        crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    else:
        X = np.column_stack([np.ones(n), np.arange(n)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        crit = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}
    if nlags is None:
        nlags = int(np.ceil(12 * (n / 100) ** 0.25))
    S = np.cumsum(e)
    s2 = e @ e / n
    for k in range(1, nlags + 1):
        w = 1 - k / (nlags + 1)
        s2 += 2 * w * (e[k:] @ e[:-k]) / n
    stat = (S @ S) / (n ** 2 * s2)
    pts = np.array([crit["10%"], crit["5%"], crit["2.5%"], crit["1%"]])
    ps = np.array([0.10, 0.05, 0.025, 0.01])
    p = float(np.interp(stat, pts, ps))  # clipped to [0.01, 0.10]
    return {"stat": stat, "pvalue": p, "lags": nlags, "crit": crit,
            "reject_5pct": bool(stat > crit["5%"]), "regression": regression}


# ----------------------------------------------------------------------------
# Seasonality helpers
# ----------------------------------------------------------------------------

def fourier_terms(t, K, period=365.25):
    t = np.asarray(t, float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols) if cols else np.empty((len(t), 0))


def _cal_index(idx):
    """Map dates to a 0..364 calendar position (29 Feb shares the slot of 28 Feb)."""
    doy = idx.dayofyear.values.astype(int)
    leap = idx.is_leap_year
    pos = doy - 1
    pos = np.where(leap & (doy > 59), pos - 1, pos)
    return np.clip(pos, 0, 364)


def classical_decompose(s, model="additive", period=365, smooth=15):
    """Classical decomposition with a centred moving-average trend and a smoothed
    calendar-day seasonal index (period 365)."""
    s = s.astype(float)
    trend = s.rolling(window=period, center=True).mean()
    detr = s - trend if model == "additive" else s / trend
    pos = _cal_index(s.index)
    raw = pd.Series(detr.values).groupby(pos).mean().reindex(range(365)).values
    ext = np.r_[raw[-smooth:], raw, raw[:smooth]]
    sm = pd.Series(ext).rolling(smooth, center=True, min_periods=1).mean().values[smooth:-smooth]
    sm = sm - sm.mean() if model == "additive" else sm / sm.mean()
    seas = pd.Series(sm[pos], index=s.index)
    rem = s - trend - seas if model == "additive" else s / (trend * seas)
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seas, "remainder": rem})


def strength(dec, model="additive"):
    d = dec.dropna()
    if model == "multiplicative":
        d = np.log(d)
        d["remainder"] = d["observed"] - d["trend"] - d["seasonal"]
    fs = max(0, 1 - d["remainder"].var() / (d["seasonal"] + d["remainder"]).var())
    ft = max(0, 1 - d["remainder"].var() / (d["trend"] + d["remainder"]).var())
    return fs, ft


def periodogram(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfftfreq(n, d=1.0)[1:]
    p = (np.abs(np.fft.rfft(x)) ** 2 / n)[1:]
    return f, p


# ----------------------------------------------------------------------------
# Metrics and tests on forecasts
# ----------------------------------------------------------------------------

def metrics(y, yhat, scale=None, y_ref=None):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat = y[m], yhat[m]
    e = y - yhat
    out = {"n": int(m.sum()),
           "MAE": np.mean(np.abs(e)),
           "RMSE": np.sqrt(np.mean(e ** 2)),
           "MAPE%": np.mean(np.abs(e) / np.abs(y)) * 100,
           "sMAPE%": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(yhat))) * 100,
           "R2": 1 - np.sum(e ** 2) / np.sum((y - (y_ref if y_ref is not None else y.mean())) ** 2)}
    if scale:
        out["MASE"] = out["MAE"] / scale
    return out


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test on squared-error loss with Harvey-Leybourne-Newbold correction.
    Negative statistic => model 1 has smaller loss than model 2."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    m = ~(np.isnan(e1) | np.isnan(e2))
    d = e1[m] ** 2 - e2[m] ** 2
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    g0 = dc @ dc / n
    var = g0
    for k in range(1, h):
        var += 2 * (dc[k:] @ dc[:-k]) / n
    var = max(var, 1e-12) / n
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), n - 1))
    return dm, p


# ----------------------------------------------------------------------------
# Forecasting models
# ----------------------------------------------------------------------------

def _pack(mean, se, zs=(1.2816, 1.96), log=False):
    out = {"mean": mean}
    for lvl, z in zip((80, 95), zs):
        lo, hi = mean - z * se, mean + z * se
        out[f"lo{lvl}"], out[f"hi{lvl}"] = lo, hi
    return out


class Mean:
    name = "Mean"

    def fit(self, y, t):
        self.mu, self.sd = np.mean(y), np.std(y, ddof=1)
        return self

    def one_step(self, y, t):
        return np.full(len(y), self.mu)

    def forecast(self, y, t, h):
        se = np.full(h, self.sd * np.sqrt(1 + 1 / len(y)))
        return _pack(np.full(h, self.mu), se)


class Naive:
    name = "Naive"

    def fit(self, y, t):
        self.sd = np.std(np.diff(y), ddof=1)
        return self

    def one_step(self, y, t):
        p = np.r_[np.nan, y[:-1]]
        return p

    def forecast(self, y, t, h):
        se = self.sd * np.sqrt(np.arange(1, h + 1))
        return _pack(np.full(h, y[-1]), se)


class SeasonalNaive:
    def __init__(self, m=365):
        self.m = m
        self.name = f"Seasonal naive (m={m})"

    def fit(self, y, t):
        d = y[self.m:] - y[:-self.m]
        self.sd = np.std(d, ddof=1)
        return self

    def one_step(self, y, t):
        p = np.full(len(y), np.nan)
        p[self.m:] = y[:-self.m]
        return p

    def forecast(self, y, t, h):
        n = len(y)
        yf = np.r_[y, np.zeros(h)]
        for k in range(h):
            yf[n + k] = yf[n + k - self.m]
        se = self.sd * np.sqrt(np.ceil(np.arange(1, h + 1) / self.m))
        return _pack(yf[n:], se)


class HoltWintersAdd:
    """ETS(A,N,A) with a very long season (m=365). Seasonal states are initialised from a
    Fourier regression on the training sample (a raw first-year profile is too noisy for m=365)."""

    def __init__(self, m=365, K_init=5):
        self.m = m
        self.K_init = K_init
        self.name = "Holt-Winters ETS(A,N,A), m=365"

    def _run(self, y, alpha, gamma, l0, s0, pos0):
        m = self.m
        s = s0.copy()
        l = l0
        n = len(y)
        pred = np.empty(n)
        for i in range(n):
            j = (pos0 + i) % m
            pred[i] = l + s[j]
            e = y[i] - pred[i]
            l += alpha * e
            s[j] += gamma * e
        return pred, l, s

    def fit(self, y, t):
        y = np.asarray(y, float)
        self.t0 = int(t[0])
        X = np.column_stack([np.ones(len(y)), fourier_terms(t, self.K_init)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        self.l0 = b[0]
        grid = np.arange(self.m)
        s0 = fourier_terms(grid, self.K_init, period=365.25) @ b[1:]
        # align: position j corresponds to calendar offset from t0
        pos_offsets = (grid + 0)  # position index == (t - t0) mod m ; Fourier evaluated at t0 + j
        s0 = fourier_terms(self.t0 + pos_offsets, self.K_init) @ b[1:]
        self.s0 = s0 - s0.mean()
        self.pos0 = 0

        def loss(p):
            a, g = 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1]))
            pred, _, _ = self._run(y, a, g, self.l0, self.s0, 0)
            return np.mean((y - pred) ** 2)

        best = None
        for start in ([-2.5, -4.0], [-1.0, -2.0], [-3.5, -6.0]):
            r = optimize.minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        self.alpha = 1 / (1 + np.exp(-best.x[0]))
        self.gamma = 1 / (1 + np.exp(-best.x[1]))
        pred, _, _ = self._run(y, self.alpha, self.gamma, self.l0, self.s0, 0)
        self.sigma2 = np.mean((y - pred) ** 2)
        self.t_train_start = self.t0
        return self

    def one_step(self, y, t):
        pred, _, _ = self._run(np.asarray(y, float), self.alpha, self.gamma, self.l0, self.s0, int(t[0]) - self.t0)
        return pred

    def forecast(self, y, t, h):
        y = np.asarray(y, float)
        pos0 = int(t[0]) - self.t0
        pred, l, s = self._run(y, self.alpha, self.gamma, self.l0, self.s0, pos0)
        n = len(y)
        mean = np.array([l + s[(pos0 + n + k) % self.m] for k in range(h)])
        c = np.array([self.alpha + (self.gamma if (j % self.m == 0) else 0) for j in range(1, h)])
        var = self.sigma2 * (1 + np.r_[0, np.cumsum(c ** 2)])
        return _pack(mean, np.sqrt(var))


class RegArima:
    """Regression with ARMA errors, or (S)ARIMA-type model with a single differencing lag.

    diff_lag=None : y_t = X_t b + u_t,  phi(B) u_t = theta(B) e_t   (X = const [+ trend] [+ K Fourier pairs])
    diff_lag=L    : (1-B^L) y_t = w_t,  phi(B) w_t = theta(B) e_t   (L=1 -> ARIMA(p,1,q); L=365 -> SARIMA(p,0,q)(0,1,0)_365)
    Estimated by conditional sum of squares (exact filtering with scipy.signal.lfilter)."""
    BURN = 5

    def __init__(self, p=0, q=0, diff_lag=None, K=None, trend=False, log=False):
        self.p, self.q, self.diff_lag, self.K, self.trend, self.log = p, q, diff_lag, K, trend, log
        parts = []
        if diff_lag == 365:
            parts.append(f"SARIMA({p},0,{q})(0,1,0)[365]")
        elif diff_lag == 1:
            parts.append(f"ARIMA({p},1,{q})")
        elif K:
            parts.append(f"Harmonic(K={K})+ARMA({p},{q})")
        else:
            parts.append(f"ARIMA({p},0,{q})+const")
        if trend:
            parts.append("+trend")
        if log:
            parts.append("[log]")
        self.name = " ".join(parts)

    # -- helpers
    def _z(self, y):
        y = np.asarray(y, float)
        return np.log(y) if self.log else y

    def _X(self, t):
        t = np.asarray(t, float)
        cols = [np.ones_like(t)]
        if self.trend:
            cols.append((t - self.t0) / 365.25)
        if self.K:
            cols.append(fourier_terms(t, self.K))
        return np.column_stack(cols)

    def _split(self, x):
        k = self.k
        return x[:k], x[k:k + self.p], x[k + self.p:k + self.p + self.q]

    @staticmethod
    def _valid(phi, th):
        ok = True
        if len(phi):
            ok &= np.all(np.abs(np.roots(np.r_[1, -phi])) < 0.999)
        if len(th):
            ok &= np.all(np.abs(np.roots(np.r_[1, th])) < 0.999)
        return ok

    def _series(self, z, t):
        """return (u, X) where u is the series that follows the ARMA process"""
        if self.diff_lag:
            L = self.diff_lag
            return z[L:] - z[:-L], None
        X = self._X(t)
        return z, X

    def _e(self, x, z, t):
        beta, phi, th = self._split(x)
        u, X = self._series(z, t)
        if X is not None:
            u = u - X @ beta
        return signal.lfilter(np.r_[1, -phi], np.r_[1, th], u)

    # -- API
    def fit(self, y, t):
        t = np.asarray(t)
        self.t0 = float(t[0])
        z = self._z(y)
        u, X = self._series(z, t)
        self.k = 0 if X is None else X.shape[1]
        beta0 = np.linalg.lstsq(X, z, rcond=None)[0] if X is not None else np.array([])
        u0 = u - X @ beta0 if X is not None else u
        if self.p:
            Xl = np.column_stack([u0[self.p - i - 1:len(u0) - i - 1] for i in range(self.p)])
            phi0 = np.linalg.lstsq(Xl, u0[self.p:], rcond=None)[0]
            while not self._valid(phi0, np.zeros(0)):
                phi0 = phi0 * 0.9
        else:
            phi0 = np.array([])
        x0 = np.r_[beta0, phi0, np.zeros(self.q)]
        burn = self.BURN

        def fun(x):
            _, phi, th = self._split(x)
            e = self._e(x, z, t)[burn:]
            if not self._valid(phi, th):
                return e * 1e3 + 1e3
            return e

        if len(x0) == self.k:  # regression only -> OLS
            xs = x0
            J = X
        else:
            r = optimize.least_squares(fun, x0, method="trf", x_scale="jac", max_nfev=200)
            xs, J = r.x, r.jac
        e = self._e(xs, z, t)
        ee = e[burn:]
        N = len(ee)
        self.sigma2 = float(ee @ ee / N)
        self.N = N
        npar = len(xs) + 1
        self.beta, self.phi, self.theta = self._split(xs)
        self.params = xs
        self.loglik = -0.5 * N * (np.log(2 * np.pi * self.sigma2) + 1)
        self.aic = -2 * self.loglik + 2 * npar
        self.bic = -2 * self.loglik + np.log(N) * npar
        try:
            cov = self.sigma2 * np.linalg.pinv(J[burn:].T @ J[burn:]) if J.shape[0] == len(e) else self.sigma2 * np.linalg.pinv(J.T @ J)
            self.se = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            self.se = np.full(len(xs), np.nan)
        self.resid = e
        # seasonal variance profile (used only for heteroscedasticity-aware prediction intervals)
        self._vg = None
        if not self.diff_lag:
            tt = t[burn:]
            Xv = np.column_stack([np.ones(len(tt)), fourier_terms(tt, 2)])
            self._vg = np.linalg.lstsq(Xv, ee ** 2, rcond=None)[0]
            self._vfloor = 0.15 * np.mean(ee ** 2)
        return self

    def _vscale(self, tf):
        Xv = np.column_stack([np.ones(len(tf)), fourier_terms(tf, 2)])
        return np.maximum(Xv @ self._vg, self._vfloor) / self.sigma2

    def param_table(self):
        names = []
        if self.k:
            names += ["const"] + (["trend/yr"] if self.trend else []) + \
                     [f"{f}{k}" for k in range(1, (self.K or 0) + 1) for f in ("sin", "cos")]
        names += [f"AR{i}" for i in range(1, self.p + 1)] + [f"MA{i}" for i in range(1, self.q + 1)]
        df = pd.DataFrame({"param": names, "coef": self.params, "se": self.se})
        df["z"] = df["coef"] / df["se"]
        return df

    def _tr_back(self, a):
        return np.exp(a) if self.log else a

    def one_step(self, y, t):
        z = self._z(y)
        t = np.asarray(t)
        x = self.params
        e = self._e(x, z, t)
        if self.diff_lag:
            zhat = np.full(len(z), np.nan)
            zhat[self.diff_lag:] = z[self.diff_lag:] - e
        else:
            zhat = z - e
        return self._tr_back(zhat)

    def std_resid(self):
        return self.resid[self.BURN:] / np.sqrt(self.sigma2)

    def _arma_fc(self, u, e, h):
        p, q = self.p, self.q
        n = len(u)
        uf = np.r_[u, np.zeros(h)]
        ef = np.r_[e, np.zeros(h)]
        for k in range(h):
            i = n + k
            v = 0.0
            for a in range(p):
                v += self.phi[a] * uf[i - 1 - a]
            for b in range(q):
                v += self.theta[b] * ef[i - 1 - b]
            uf[i] = v
        return uf[n:]

    def forecast(self, y, t, h):
        z = self._z(y)
        t = np.asarray(t)
        n = len(z)
        e = self._e(self.params, z, t)
        imp = np.zeros(h)
        imp[0] = 1
        A = np.r_[1, -self.phi]
        M = np.r_[1, self.theta]
        if self.diff_lag:
            L = self.diff_lag
            w = z[L:] - z[:-L]
            wf = self._arma_fc(w, e, h)
            zf = np.r_[z, np.zeros(h)]
            for k in range(h):
                zf[n + k] = zf[n + k - L] + wf[k]
            mean = zf[n:]
            dpoly = np.zeros(L + 1)
            dpoly[0], dpoly[L] = 1, -1
            psi = signal.lfilter(M, np.convolve(A, dpoly), imp)
        else:
            u = z - self._X(t) @ self.beta
            uf = self._arma_fc(u, e, h)
            tf = t[-1] + np.arange(1, h + 1)
            mean = uf + self._X(tf) @ self.beta
            psi = signal.lfilter(M, A, imp)
        se = np.sqrt(self.sigma2 * np.cumsum(psi ** 2))
        out = {"mean": self._tr_back(mean)}
        if self._vg is not None:
            se_h = se * np.sqrt(self._vscale(t[-1] + np.arange(1, h + 1)))
        else:
            se_h = se
        for lvl, zq in ((80, 1.2816), (95, 1.96)):
            out[f"lo{lvl}"] = self._tr_back(mean - zq * se)
            out[f"hi{lvl}"] = self._tr_back(mean + zq * se)
            out[f"lo{lvl}h"] = self._tr_back(mean - zq * se_h)
            out[f"hi{lvl}h"] = self._tr_back(mean + zq * se_h)
        return out


class GBMLag:
    """Gradient boosting on lagged values, rolling means and Fourier terms; recursive multi-step forecasts."""
    LAGS = (1, 2, 3, 7, 14, 28)

    def __init__(self, K=3, seed=0):
        self.K = K
        self.seed = seed
        self.name = "Gradient boosting (lags + Fourier)"

    def _row(self, z, i, t):
        f = [z[i - l] for l in self.LAGS]
        f += [z[i - 7:i].mean(), z[i - 30:i].mean()]
        f += list(fourier_terms([t], self.K)[0])
        return f

    def fit(self, y, t):
        from sklearn.ensemble import HistGradientBoostingRegressor
        z = np.asarray(y, float)
        rows = [self._row(z, i, t[i]) for i in range(30, len(z))]
        self.m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                               min_samples_leaf=20, random_state=self.seed)
        self.m.fit(np.array(rows), z[30:])
        pred = self.m.predict(np.array(rows))
        self.sd = np.std(z[30:] - pred, ddof=1)
        return self

    def one_step(self, y, t):
        z = np.asarray(y, float)
        out = np.full(len(z), np.nan)
        rows = [self._row(z, i, t[i]) for i in range(30, len(z))]
        out[30:] = self.m.predict(np.array(rows))
        return out

    def forecast(self, y, t, h):
        z = np.r_[np.asarray(y, float), np.zeros(h)]
        n = len(y)
        tf = t[-1] + np.arange(1, h + 1)
        for k in range(h):
            z[n + k] = self.m.predict(np.array([self._row(z, n + k, tf[k])]))[0]
        mean = z[n:]
        nan = np.full(h, np.nan)
        return {"mean": mean, "lo80": nan, "hi80": nan, "lo95": nan, "hi95": nan}


# ----------------------------------------------------------------------------
# Walk-forward (rolling-origin) evaluation
# ----------------------------------------------------------------------------

def walk_forward(factory, y, t, train_n, horizon=30, step=7, refit_every=8, verbose=False):
    """Expanding-window rolling-origin evaluation.
    Origins start at `train_n`, move `step` days at a time; the model is re-estimated every
    `refit_every` origins. Returns a long DataFrame (origin, h, actual, forecast, lo95, hi95)."""
    n = len(y)
    rows = []
    model = None
    for j, T in enumerate(range(train_n, n - 1, step)):
        if model is None or j % refit_every == 0:
            model = factory().fit(y[:T], t[:T])
        h = min(horizon, n - T)
        fc = model.forecast(y[:T], t[:T], h)
        for k in range(h):
            rows.append((T, k + 1, y[T + k], fc["mean"][k], fc["lo95"][k], fc["hi95"][k],
                         fc.get("lo95h", fc["lo95"])[k], fc.get("hi95h", fc["hi95"])[k]))
    return pd.DataFrame(rows, columns=["origin", "h", "actual", "fc", "lo95", "hi95", "lo95h", "hi95h"])
