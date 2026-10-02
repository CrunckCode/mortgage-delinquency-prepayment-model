"""Pure-numpy discrimination, calibration and stability metrics plus the benchmark and output writers."""
import json

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .columns import (LOAN_ID, PERIOD, VINTAGE, FICO_BAND, PURPOSE, OCCUPANCY, FEATURES_PD, CATEGORICAL_FEATURES,
                      TARGET_PD12, TARGET_PREPAY_1M, SNAPSHOT_DATE, CUR_UPB, TRUE_HP)

# ===== CONFIG (user inputs) =====
PSI_FLOOR = 1e-4
BASELINE_KEY = "logit"
SCORECARD_PD_COL, GBM_PD_COL, SCORECARD_POINTS_COL = "scorecard_pd", "gbm_pd", "scorecard_points"
GBM_REFERENCE_KEY = "lgbm"
SEGMENT_COLUMNS = (FICO_BAND, PURPOSE, OCCUPANCY)
SAMPLE_SEED = 7
OOT_AUC_BOOT_ROWS = 25_000  # row cap keeps the 500-draw bootstrap fast; CI is slightly conservative
CPR_EXPONENT = 12
# ===== END CONFIG =====


# ---------- core metrics ----------
def _arr(a):
    return np.asarray(a, dtype=np.float64)


def auc(y, p) -> float:
    y, p = _arr(y), _arr(p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gini(y, p) -> float:
    return 2.0 * auc(y, p) - 1.0


def ks(y, p) -> float:
    y, p = _arr(y), _arr(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    # evaluate CDFs only at the last index of each tied score
    last = np.r_[ps[1:] != ps[:-1], True]
    c1 = np.cumsum(ys) / n1
    c0 = np.cumsum(1 - ys) / n0
    return float(np.max(np.abs(c1[last] - c0[last])))


def brier(y, p) -> float:
    return float(np.mean((_arr(p) - _arr(y)) ** 2))


def _equal_count_groups(n_rows, n):
    return np.array_split(np.arange(n_rows), n)


def decile_table(y, p, n: int = 10) -> pd.DataFrame:
    y, p = _arr(y), _arr(p)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tot1, tot0 = ys.sum(), len(ys) - ys.sum()
    base = ys.mean() if len(ys) else np.nan
    rows, c1, c0 = [], 0.0, 0.0
    for d, idx in enumerate(_equal_count_groups(len(ys), n), start=1):
        e = ys[idx].sum()
        c1 += e
        c0 += len(idx) - e
        cum1 = c1 / tot1 if tot1 else np.nan
        cum0 = c0 / tot0 if tot0 else np.nan
        rows.append(dict(decile=d, n=len(idx), events=e, event_rate=e / len(idx), mean_score=ps[idx].mean(),
                         cum_events_share=cum1, lift=(e / len(idx)) / base if base else np.nan,
                         ks_at_decile=abs(cum1 - cum0)))
    return pd.DataFrame(rows)


def calibration_table(y, p, n: int = 10) -> pd.DataFrame:
    y, p = _arr(y), _arr(p)
    order = np.argsort(p, kind="mergesort")
    ys, ps = y[order], p[order]
    rows = []
    for b, idx in enumerate(_equal_count_groups(len(ys), n), start=1):
        rows.append(dict(bin=b, n=len(idx), mean_pred=ps[idx].mean(), obs_rate=ys[idx].mean(),
                         expected_events=ps[idx].sum(), events=ys[idx].sum()))
    return pd.DataFrame(rows)


def hosmer_lemeshow(y, p, n: int = 10):
    t = calibration_table(y, p, n)
    nn, mp, op = t["n"].to_numpy(), t["mean_pred"].to_numpy(), t["obs_rate"].to_numpy()
    den = nn * mp * (1 - mp)
    ok = den > 0
    stat = float(np.sum(nn[ok] * (op[ok] - mp[ok]) ** 2 / (mp[ok] * (1 - mp[ok]))))
    dof = n  # held-out data: g degrees of freedom, not g-2
    return stat, float(stats.chi2.sf(stat, dof))


# ---------- stability ----------
def _psi_from_shares(e, a):
    e, a = np.maximum(e, PSI_FLOOR), np.maximum(a, PSI_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_bins(expected, bins: int = 10) -> np.ndarray:
    """Interior cut points from the expected sample's quantiles."""
    x = _arr(expected)
    x = x[~np.isnan(x)]
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return cuts


def _shares(x, cuts):
    x = _arr(x)
    x = x[~np.isnan(x)]
    idx = np.searchsorted(cuts, x, side="left")
    return np.bincount(idx, minlength=len(cuts) + 1) / max(len(x), 1)


def psi(expected, actual, bins: int = 10) -> float:
    cuts = psi_bins(expected, bins)
    return _psi_from_shares(_shares(expected, cuts), _shares(actual, cuts))


def _psi_categorical(e, a):
    ce = pd.Series(e).astype(str).value_counts(normalize=True)
    ca = pd.Series(a).astype(str).value_counts(normalize=True)
    cats = ce.index.union(ca.index)
    return _psi_from_shares(ce.reindex(cats, fill_value=0).to_numpy(), ca.reindex(cats, fill_value=0).to_numpy())


def psi_band(v: float) -> str:
    lo, hi = config.PSI_BANDS
    return "stable" if v < lo else ("moderate" if v < hi else "significant")


def csi(train_df, oot_df, features) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in CATEGORICAL_FEATURES or not pd.api.types.is_numeric_dtype(train_df[f]):
            v = _psi_categorical(train_df[f], oot_df[f])
        else:
            v = psi(train_df[f], oot_df[f])
        rows.append(dict(feature=f, psi=v, band=psi_band(v)))
    return pd.DataFrame(rows)


# ---------- inference ----------
def _delong_components(y, p):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    allv = np.r_[pos, neg]
    r_all = stats.rankdata(allv)
    r_pos = stats.rankdata(pos)
    r_neg = stats.rankdata(neg)
    v10 = (r_all[:m] - r_pos) / n          # structural component per positive
    v01 = 1.0 - (r_all[m:] - r_neg) / m    # structural component per negative
    return float(v10.mean()), v10, v01


def delong_test(y, p1, p2):
    y, p1, p2 = _arr(y), _arr(p1), _arr(p2)
    a1, v10_1, v01_1 = _delong_components(y, p1)
    a2, v10_2, v01_2 = _delong_components(y, p2)
    m, n = len(v10_1), len(v01_1)
    s10 = np.cov(np.vstack([v10_1, v10_2])) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.vstack([v01_1, v01_2])) if n > 1 else np.zeros((2, 2))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = a1 - a2
    if var <= 0:
        return float(d), float("nan"), float("nan")
    z = d / np.sqrt(var)
    return float(d), float(z), float(2 * stats.norm.sf(abs(z)))


def bootstrap_ci(metric, y, p, groups=None, n: int = config.BOOTSTRAP_N, seed: int = config.SEED):
    y, p = _arr(y), _arr(p)
    rng = np.random.default_rng(seed)
    vals = []
    if groups is None:
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            vals.append(metric(y[i], p[i]))
    else:
        codes, uniq = pd.factorize(np.asarray(groups))
        order = np.argsort(codes, kind="mergesort")
        starts = np.r_[0, np.cumsum(np.bincount(codes))]
        sizes = np.diff(starts)
        for _ in range(n):
            g = rng.integers(0, len(uniq), len(uniq))
            sz = sizes[g]
            idx = np.repeat(starts[g] - np.r_[0, np.cumsum(sz)[:-1]], sz) + np.arange(sz.sum())
            i = order[idx]
            vals.append(metric(y[i], p[i]))
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def by_segment(df, seg_col, y_col, p_col) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(seg_col, observed=True):
        rows.append(dict(segment=seg, n=len(g), events=g[y_col].sum(), event_rate=g[y_col].mean(),
                         mean_pred=g[p_col].mean(), auc=auc(g[y_col], g[p_col])))
    return pd.DataFrame(rows)


# ---------- benchmark ----------
def _target_col(target):
    return TARGET_PD12 if target == "pd" else TARGET_PREPAY_1M


def _score(bundle, df):
    return bundle.predict_pd(df) if bundle.target == "pd" else bundle.predict_smm(df)


def _metrics_row(label, bundle, splits, ycol, base_scores, ci_rows):
    row = dict(model=label)
    oot_p = None
    for part in ("train", "calib", "oot"):
        d = splits[part]
        p = _score(bundle, d)
        y = d[ycol].to_numpy()
        row[f"{part}_auc"] = auc(y, p)
        row[f"{part}_gini"] = gini(y, p)
        row[f"{part}_ks"] = ks(y, p)
        row[f"{part}_brier"] = brier(y, p)
        if part == "oot":
            oot_p = p
    y = splits["oot"][ycol].to_numpy()
    if base_scores is None:
        row["delong_dauc_vs_logit"], row["delong_p_vs_logit"] = 0.0, np.nan
    else:
        dauc, _, pv = delong_test(y, oot_p, base_scores)
        row["delong_dauc_vs_logit"], row["delong_p_vs_logit"] = dauc, pv
    # cap rows for the bootstrap so large prepay panels stay fast
    n = len(y)
    sel = np.arange(n)
    if n > OOT_AUC_BOOT_ROWS:
        sel = np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, OOT_AUC_BOOT_ROWS, replace=False))
    g = splits["oot"][LOAN_ID].to_numpy()[sel]
    lo, hi = bootstrap_ci(auc, y[sel], oot_p[sel], groups=g, n=ci_rows)
    row["oot_auc_ci_lo"], row["oot_auc_ci_hi"] = lo, hi
    return row, oot_p


def benchmark_table(bundles, splits, target, n_boot: int = config.BOOTSTRAP_N) -> pd.DataFrame:
    ycol = _target_col(target)
    sel = {k: b for k, b in bundles.items() if b.target == target}
    base_key = next(k for k, b in sel.items() if b.kind == BASELINE_KEY)
    rows = []
    base_row, base_p = _metrics_row(base_key, sel[base_key], splits, ycol, None, n_boot)
    rows.append(base_row)
    for k, b in sel.items():
        if k == base_key:
            continue
        r, _ = _metrics_row(k, b, splits, ycol, base_p, n_boot)
        rows.append(r)
        for lab, wb in (getattr(b, "extras", None) or {}).items():
            if not hasattr(wb, "predict_pd"):
                continue
            r2, _ = _metrics_row(f"{k}_{lab}_sensitivity", wb, splits, ycol, base_p, n_boot)
            rows.append(r2)
    return pd.DataFrame(rows)


def prepay_cpr_comparison(bundle, oot_df) -> pd.DataFrame:
    d = pd.DataFrame({PERIOD: oot_df[PERIOD].to_numpy(), "act": oot_df[TARGET_PREPAY_1M].to_numpy(),
                      "prd": bundle.predict_smm(oot_df)})
    g = d.groupby(PERIOD).agg(act=("act", "mean"), prd=("prd", "mean")).reset_index()
    out = pd.DataFrame({PERIOD: g[PERIOD], "actual_cpr": 1 - (1 - g["act"]) ** CPR_EXPONENT,
                        "pred_cpr": 1 - (1 - g["prd"]) ** CPR_EXPONENT})
    out.attrs["rmse_cpr_pts"] = float(np.sqrt(np.mean((out["actual_cpr"] - out["pred_cpr"]) ** 2)) * 100)
    return out


# ---------- writer ----------
def _sample_oot(oot, n):
    if len(oot) <= n:
        return oot.copy()
    return oot.sample(n=n, random_state=SAMPLE_SEED).sort_index()


def evaluate_all(bundles, splits_pd, splits_pp, data_source=None) -> dict:
    out = config.OUT_DIR
    xdir = config.EXCEL_INPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    xdir.mkdir(parents=True, exist_ok=True)
    if data_source is None:
        data_source = "synthetic" if TRUE_HP in splits_pd["train"].columns else "freddie"

    bench_pd = benchmark_table(bundles, splits_pd, "pd")
    bench_pp = benchmark_table(bundles, splits_pp, "prepay")
    bench_pd.to_csv(out / "benchmark_pd.csv", index=False)
    bench_pp.to_csv(out / "benchmark_prepay.csv", index=False)

    oot = splits_pd["oot"]
    pd_models = {k: b for k, b in bundles.items() if b.target == "pd"}
    scores = {}
    for k, b in pd_models.items():
        p = b.predict_pd(oot)
        scores[k] = p
        decile_table(oot[TARGET_PD12], p).to_csv(out / f"deciles_pd_{k}.csv", index=False)
        calibration_table(oot[TARGET_PD12], p).to_csv(out / f"calibration_pd_{k}.csv", index=False)

    # score and feature stability
    ref_key = next(k for k, b in pd_models.items() if b.kind == GBM_REFERENCE_KEY)
    sc_key = next(k for k, b in pd_models.items() if b.kind == BASELINE_KEY)
    rows = []
    for k, b in pd_models.items():
        tr = b.predict_pd(splits_pd["train"])
        for part, d in (("calib", splits_pd["calib"]), ("oot", oot)):
            v = psi(tr, b.predict_pd(d))
            rows.append(dict(kind="score", name=f"{k}:train_vs_{part}", psi=v, band=psi_band(v)))
    c = csi(splits_pd["train"], oot, FEATURES_PD)
    c.insert(0, "kind", "csi")
    c = c.rename(columns={"feature": "name"})
    pd.concat([pd.DataFrame(rows), c], ignore_index=True).to_csv(out / "psi_csi.csv", index=False)

    # Excel PSI inputs from the train scorecard-PD distribution
    sc_train = pd_models[sc_key].predict_pd(splits_pd["train"])
    cuts = psi_bins(sc_train, 10)
    edges = np.r_[-np.inf, cuts, np.inf]
    pd.DataFrame(dict(bin=np.arange(1, len(edges)), lo=edges[:-1], hi=edges[1:],
                      train_share=_shares(sc_train, cuts))).to_csv(xdir / "psi_train_bins.csv", index=False)

    # vintage stability and segments on OOT
    tmp = oot[[VINTAGE, TARGET_PD12, *SEGMENT_COLUMNS]].copy()
    vrows, srows = [], []
    for k, p in scores.items():
        tmp[GBM_PD_COL] = p
        v = by_segment(tmp, VINTAGE, TARGET_PD12, GBM_PD_COL)
        v.insert(0, "model", k)
        vrows.append(v.rename(columns={"segment": "vintage"}))
        for sc in SEGMENT_COLUMNS:
            s = by_segment(tmp, sc, TARGET_PD12, GBM_PD_COL)
            s.insert(0, "segment_col", sc)
            s.insert(0, "model", k)
            srows.append(s)
    pd.concat(vrows, ignore_index=True).to_csv(out / "stability_by_vintage.csv", index=False)
    pd.concat(srows, ignore_index=True).to_csv(out / "segments.csv", index=False)

    # scorecard artefacts
    from .scorecard import export_scorecard_table
    sc_bundle = pd_models[sc_key]
    export_scorecard_table(sc_bundle.scorecard).to_csv(out / "scorecard_table.csv", index=False)
    pp_logit = next(b for b in bundles.values() if b.target == "prepay" and b.kind == BASELINE_KEY)
    pp_logit.coefs.to_csv(out / "prepay_logit_coefs.csv", index=False)
    cpr = prepay_cpr_comparison(next(b for b in bundles.values() if b.target == "prepay"
                                     and b.kind == GBM_REFERENCE_KEY), splits_pp["oot"])
    cpr.to_csv(out / "prepay_cpr_oot.csv", index=False)
    cpr_rmse = {}
    for k, b in bundles.items():
        if b.target == "prepay":
            cpr_rmse[k] = prepay_cpr_comparison(b, splits_pp["oot"]).attrs["rmse_cpr_pts"]

    # Excel loan sample
    samp = _sample_oot(oot, config.EXCEL_SAMPLE_LOANS)
    ls = samp[[LOAN_ID, SNAPSHOT_DATE, *FEATURES_PD, TARGET_PD12, CUR_UPB]].copy()
    ls[SCORECARD_PD_COL] = sc_bundle.predict_pd(samp)
    ls[SCORECARD_POINTS_COL] = sc_bundle.scorecard.points(samp)
    ls[GBM_PD_COL] = pd_models[ref_key].predict_pd(samp)
    ls.to_csv(xdir / "loan_sample.csv", index=False)

    meta = dict(data_source=data_source, pd_oot_auc={k: auc(oot[TARGET_PD12], p) for k, p in scores.items()},
                prepay_cpr_rmse_pts=cpr_rmse, n_pd_oot=int(len(oot)), n_prepay_oot=int(len(splits_pp["oot"])))
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return dict(benchmark_pd=bench_pd, benchmark_prepay=bench_pp, meta=meta)
