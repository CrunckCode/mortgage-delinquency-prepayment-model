"""Credit-scorecard build: quantile pre-binning, monotone WoE merging, IV, Logit on WoE, points scaling.

Convention: WoE = ln(%good / %bad) (higher = safer), so the Logit of P(bad) has NEGATIVE WoE coefficients.
Score = offset + factor * ln(odds good:bad); doubling the good odds adds PDO points.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from .columns import FEATURES_PD, CATEGORICAL_FEATURES

# ===== CONFIG (user inputs) =====
PREBINS = 20
SMOOTH = 0.5
MAX_SIGN_DROPS = 10
EXCLUDED_NUMERIC = ()
# ===== END CONFIG =====


@dataclass
class BinSpec:
    variable: str
    edges: np.ndarray            # interior cut points; bin i = (edges[i-1], edges[i]]
    woe: np.ndarray              # one value per bin
    missing_woe: float
    iv: float
    n_bad: np.ndarray = field(default_factory=lambda: np.array([]))
    n_good: np.ndarray = field(default_factory=lambda: np.array([]))

    def bin_index(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.edges, x, side="left")
        return np.where(np.isnan(x), len(self.edges) + 1, idx)

    def transform(self, x) -> np.ndarray:
        return np.append(self.woe, self.missing_woe)[self.bin_index(x)]


def _woe_iv(bad, good, tot_bad, tot_good):
    db = (bad + SMOOTH) / (tot_bad + SMOOTH * len(bad))
    dg = (good + SMOOTH) / (tot_good + SMOOTH * len(bad))
    woe = np.log(dg / db)
    return woe, float(np.sum((dg - db) * woe))


def _merge(bad, good, edges, i):
    """Merge bin i with bin i+1."""
    bad = np.r_[bad[:i], bad[i] + bad[i + 1], bad[i + 2:]]
    good = np.r_[good[:i], good[i] + good[i + 1], good[i + 2:]]
    return bad, good, np.delete(edges, i)


def fit_bins(x, y, max_bins: int = 8, min_share: float = 0.05, monotone: bool = True) -> BinSpec:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    miss = np.isnan(x)
    xv, yv = x[~miss], y[~miss]
    cuts = np.unique(np.quantile(xv, np.linspace(0, 1, PREBINS + 1)[1:-1]))
    # a cut at the maximum would leave an empty top bin
    cuts = cuts[cuts < xv.max()]
    idx = np.searchsorted(cuts, xv, side="left")
    nb = len(cuts) + 1
    bad = np.bincount(idx, weights=yv, minlength=nb)
    good = np.bincount(idx, weights=1 - yv, minlength=nb)
    tb, tg = bad.sum(), good.sum()
    edges = cuts.copy()

    def rate(b, g):
        return (b + SMOOTH) / (b + g + 2 * SMOOTH)

    # 1) enforce minimum share by merging the thinnest bin into its closer-risk neighbour
    n_all = len(xv)
    while len(bad) > 1:
        share = (bad + good) / n_all
        j = int(np.argmin(share))
        if share[j] >= min_share:
            break
        r = rate(bad, good)
        if j == 0:
            k = 0
        elif j == len(bad) - 1:
            k = j - 1
        else:
            k = j - 1 if abs(r[j] - r[j - 1]) <= abs(r[j] - r[j + 1]) else j
        bad, good, edges = _merge(bad, good, edges, k)

    def best_pair(r):
        return int(np.argmin(np.abs(np.diff(r))))

    # 2) merge toward monotone bad rate (direction from the data trend)
    if monotone and len(bad) > 2:
        r = rate(bad, good)
        sign = 1.0 if np.corrcoef(np.arange(len(r)), r)[0, 1] >= 0 else -1.0
        while len(bad) > 2:
            r = rate(bad, good) * sign
            viol = np.flatnonzero(np.diff(r) < 0)
            if len(viol) == 0:
                break
            bad, good, edges = _merge(bad, good, edges, int(viol[0]))
    # 3) cap the bin count by merging the most similar neighbours
    while len(bad) > max_bins:
        bad, good, edges = _merge(bad, good, edges, best_pair(rate(bad, good)))

    woe, iv = _woe_iv(bad, good, tb, tg)
    miss_woe = 0.0
    if miss.any():
        mb, mg = y[miss].sum(), (1 - y[miss]).sum()
        db, dg = (mb + SMOOTH) / (tb + mb + SMOOTH), (mg + SMOOTH) / (tg + mg + SMOOTH)
        miss_woe = float(np.log(dg / db))
    return BinSpec(variable="", edges=np.asarray(edges), woe=woe, missing_woe=miss_woe, iv=iv, n_bad=bad, n_good=good)


def woe_transform(df, specs) -> pd.DataFrame:
    return pd.DataFrame({v: s.transform(df[v].to_numpy()) for v, s in specs.items()}, index=df.index)


@dataclass
class Scorecard:
    specs: dict
    variables: list
    coef: dict
    intercept: float
    factor: float
    offset: float
    iv: dict

    def margin(self, df) -> np.ndarray:
        w = woe_transform(df, {v: self.specs[v] for v in self.variables})
        z = np.full(len(df), self.intercept)
        for v in self.variables:
            z = z + self.coef[v] * w[v].to_numpy()
        return z

    def predict_pd(self, df) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.margin(df)))

    def points(self, df) -> np.ndarray:
        return self.offset - self.factor * self.margin(df)


def _numeric_candidates(df, features):
    return [f for f in features if f not in CATEGORICAL_FEATURES and f not in EXCLUDED_NUMERIC
            and pd.api.types.is_numeric_dtype(df[f])]


def fit_scorecard(df, y_col, features=None) -> Scorecard:
    features = list(features if features is not None else FEATURES_PD)
    y = df[y_col].to_numpy(dtype=np.float64)
    specs, ivs = {}, {}
    for f in _numeric_candidates(df, features):
        s = fit_bins(df[f].to_numpy(), y)
        s.variable = f
        specs[f], ivs[f] = s, s.iv
    keep = [f for f, v in ivs.items() if v >= config.SCORECARD_IV_MIN]
    W = woe_transform(df, {f: specs[f] for f in keep})
    for _ in range(MAX_SIGN_DROPS):
        fit = sm.Logit(y, sm.add_constant(W[keep], has_constant="add")).fit(disp=0, maxiter=100)
        bad = [f for f in keep if fit.params[f] >= 0]
        if not bad:
            break
        # drop the least informative wrong-signed variable and refit
        keep.remove(min(bad, key=lambda f: ivs[f]))
    factor = config.SCORECARD_PDO / np.log(2.0)
    offset = config.SCORECARD_BASE_SCORE - factor * np.log(config.SCORECARD_BASE_ODDS)
    return Scorecard(specs={f: specs[f] for f in keep}, variables=keep,
                     coef={f: float(fit.params[f]) for f in keep}, intercept=float(fit.params["const"]),
                     factor=float(factor), offset=float(offset), iv={f: ivs[f] for f in keep})


def export_scorecard_table(sc: Scorecard) -> pd.DataFrame:
    base = (sc.offset - sc.factor * sc.intercept) / len(sc.variables)
    rows = []
    for v in sc.variables:
        s, c = sc.specs[v], sc.coef[v]
        lo = np.r_[-np.inf, s.edges]
        hi = np.r_[s.edges, np.inf]
        for i in range(len(s.woe)):
            rows.append(dict(variable=v, bin_lo=lo[i], bin_hi=hi[i], woe=s.woe[i], coef=c,
                             points=base - sc.factor * c * s.woe[i]))
        rows.append(dict(variable=v, bin_lo=np.nan, bin_hi=np.nan, woe=s.missing_woe, coef=c,
                         points=base - sc.factor * c * s.missing_woe))
    return pd.DataFrame(rows, columns=["variable", "bin_lo", "bin_hi", "woe", "coef", "points"])
