"""SHAP explanations for the GBM bundles, DGP alignment checks and reason codes.

SHAP explains the raw GBM margin (log-odds), not the calibrated probability, so base + sum(SHAP) equals
predict_margin and not predict_pd/predict_smm.
"""
import logging

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from . import charts, config
from .columns import (LOAN_ID, FICO, MTM_LTV, UNEMP, DTI, TIMES_30DPD_12M, INCENTIVE, BURNOUT,
                      DLQ_STATUS)

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
TARGET_PD, TARGET_PREPAY = "pd", "prepay"
PD_TOP_K = 5
PD_REQUIRED_TOP5 = (FICO, MTM_LTV, UNEMP)
PD_EITHER_TOP5 = (DTI, TIMES_30DPD_12M)
PREPAY_FIRST, PREPAY_TOP4 = INCENTIVE, BURNOUT
PREPAY_TOP_K = 4
# expected sign of Spearman(feature value, SHAP) implied by config.TRUE_DGP
EXPECTED_SIGN = {
    TARGET_PD: {FICO: -1, MTM_LTV: 1, UNEMP: 1, DTI: 1},
    TARGET_PREPAY: {INCENTIVE: 1, BURNOUT: -1, FICO: 1},
}
MIN_ABS_RHO = 0.05
ADDITIVITY_TOL = 1e-4
IMPORTANCE_COL, FEATURE_COL, RANK_COL, EXPECTED_COL, PASSED_COL = "mean_abs_shap", "feature", "rank", "expected", "passed"
SPEARMAN_COL = "spearman"
DEPENDENCE_TOP_K = 6
REASON_COUNT = 4
ONEHOT_SEP = "="
# ===== END CONFIG =====

_EXPLAINERS = {}


def prepare(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Original-feature view of X used for plots and sign checks (SHAP is aggregated back to these columns)."""
    return X[list(bundle.features)]


def _explainer(bundle):
    key = id(bundle.estimator)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(bundle.estimator)
    return _EXPLAINERS[key]


def shap_values(bundle, X: pd.DataFrame) -> shap.Explanation:
    """TreeExplainer on the raw margin; one-hot columns (XGBoost) are summed back to their source feature."""
    M = bundle.design(X)
    raw = _explainer(bundle)(M)
    vals, base = np.asarray(raw.values, dtype=np.float64), np.asarray(raw.base_values, dtype=np.float64)
    if vals.ndim == 3:                      # some versions return one slice per class
        vals, base = vals[:, :, -1], base[:, -1] if base.ndim == 2 else base
    if base.ndim == 0:
        base = np.full(vals.shape[0], float(base))
    src = pd.Series([str(c).split(ONEHOT_SEP)[0] for c in M.columns])
    agg = pd.DataFrame(vals).T.groupby(src.to_numpy(), sort=False).sum().T
    names = [f for f in bundle.features if f in agg.columns]
    return shap.Explanation(values=agg[names].to_numpy(), base_values=base, data=None, feature_names=names)


def additivity_gap(bundle, X: pd.DataFrame, expl: shap.Explanation | None = None) -> float:
    """Max |base + sum(SHAP) - margin| over rows."""
    expl = expl if expl is not None else shap_values(bundle, X)
    recon = expl.base_values + expl.values.sum(axis=1)
    return float(np.max(np.abs(recon - np.asarray(bundle.predict_margin(X), dtype=np.float64))))


def assert_additive(bundle, X, expl=None, tol=ADDITIVITY_TOL):
    gap = additivity_gap(bundle, X, expl)
    assert gap <= tol, f"SHAP additivity gap {gap:.2e} exceeds {tol}"
    return gap


def global_importance(expl: shap.Explanation, features=None) -> pd.DataFrame:
    names = list(features) if features is not None else list(expl.feature_names)
    imp = pd.DataFrame({FEATURE_COL: names, IMPORTANCE_COL: np.abs(expl.values).mean(axis=0)})
    return imp.sort_values(IMPORTANCE_COL, ascending=False).reset_index(drop=True)


def _rank_map(importance):
    ordered = importance.sort_values(IMPORTANCE_COL, ascending=False)[FEATURE_COL].tolist()
    return {f: i + 1 for i, f in enumerate(ordered)}


def sign_checks(expl, X: pd.DataFrame, target: str) -> pd.DataFrame:
    """Spearman(feature value, SHAP) against the sign implied by the true DGP."""
    rows = []
    for f, sign in EXPECTED_SIGN[target].items():
        j = list(expl.feature_names).index(f)
        v = pd.Series(X[f].to_numpy()).astype(float)
        rho = spearmanr(v, expl.values[:, j]).correlation
        rho = 0.0 if np.isnan(rho) else float(rho)
        rows.append({FEATURE_COL: f, EXPECTED_COL: f"sign {'+' if sign > 0 else '-'}", SPEARMAN_COL: rho,
                     PASSED_COL: bool(np.sign(rho) == sign and abs(rho) >= MIN_ABS_RHO)})
    return pd.DataFrame(rows)


def dgp_alignment(importance: pd.DataFrame, target: str, expl=None, X=None) -> pd.DataFrame:
    """Rank checks against the DGP; with expl and X also the Spearman sign checks. Returns feature, rank, expected, passed."""
    rk = _rank_map(importance)
    rows = []
    if target == TARGET_PD:
        for f in PD_REQUIRED_TOP5:
            rows.append((f, rk.get(f), f"top {PD_TOP_K}", rk.get(f, 99) <= PD_TOP_K))
        either = [rk.get(f, 99) for f in PD_EITHER_TOP5]
        best = PD_EITHER_TOP5[int(np.argmin(either))]
        rows.append(("|".join(PD_EITHER_TOP5), rk.get(best), f"either in top {PD_TOP_K}", min(either) <= PD_TOP_K))
    elif target == TARGET_PREPAY:
        rows.append((PREPAY_FIRST, rk.get(PREPAY_FIRST), "rank 1", rk.get(PREPAY_FIRST, 99) == 1))
        rows.append((PREPAY_TOP4, rk.get(PREPAY_TOP4), f"top {PREPAY_TOP_K}", rk.get(PREPAY_TOP4, 99) <= PREPAY_TOP_K))
    else:
        raise ValueError(f"unknown target {target}")
    out = pd.DataFrame(rows, columns=[FEATURE_COL, RANK_COL, EXPECTED_COL, PASSED_COL])
    if expl is not None and X is not None:
        sg = sign_checks(expl, X, target)
        sg[RANK_COL] = sg[FEATURE_COL].map(rk)
        out = pd.concat([out, sg[[FEATURE_COL, RANK_COL, EXPECTED_COL, PASSED_COL]]], ignore_index=True)
    return out


def dependence_plots(expl, X: pd.DataFrame, top_k: int = DEPENDENCE_TOP_K, out_dir=config.CHART_DIR,
                     prefix: str = "shap_dependence_pd") -> list:
    imp = global_importance(expl)
    paths = []
    for f in imp[FEATURE_COL].head(top_k):
        p = out_dir / f"{prefix}_{f}.png"
        charts.shap_dependence(expl, X, f, p)
        paths.append(p)
    return paths


def reason_codes(bundle, row_df: pd.DataFrame, k: int = REASON_COUNT) -> list:
    """Top-k risk-increasing contributions for the first row, as plain-language text."""
    expl = shap_values(bundle, row_df.iloc[[0]])
    contrib = pd.Series(expl.values[0], index=expl.feature_names).sort_values(ascending=False)
    out = []
    for f in contrib.index:
        text = config.REASON_TEXT.get(f, f"Elevated risk from {f}")
        if text not in out:
            out.append(text)
        if len(out) == k:
            break
    return out


def reason_code_table(bundle, df: pd.DataFrame, k: int = REASON_COUNT) -> pd.DataFrame:
    """Five illustrative loans: lowest, median, highest PD, one currently 30DPD, one high prepay incentive."""
    pdv = pd.Series(np.asarray(bundle.predict_pd(df), dtype=np.float64), index=df.index)
    order = pdv.sort_values()
    picks = {"lowest_pd": order.index[0], "median_pd": order.index[len(order) // 2], "highest_pd": order.index[-1]}
    dq = df.index[df[DLQ_STATUS].to_numpy() >= 1]
    if len(dq):
        picks["currently_30dpd"] = pdv.loc[dq].idxmax()
    picks["high_prepay_incentive"] = df[INCENTIVE].idxmax()
    rows = []
    for case, idx in picks.items():
        codes = reason_codes(bundle, df.loc[[idx]], k)
        row = {"case": case, LOAN_ID: df.at[idx, LOAN_ID], "pd": float(pdv.loc[idx])}
        row.update({f"reason_{i + 1}": c for i, c in enumerate(codes)})
        rows.append(row)
    return pd.DataFrame(rows)


def run_explain(bundle_pd, bundle_pp, pd_oot: pd.DataFrame, pp_oot: pd.DataFrame, out_dir=config.OUT_DIR,
                chart_dir=config.CHART_DIR, seed: int = config.SEED) -> dict:
    """Sample OOT rows, compute SHAP, write importance, alignment and reason-code CSVs, dependence plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    align = []
    for tag, bundle, df in ((TARGET_PD, bundle_pd, pd_oot), (TARGET_PREPAY, bundle_pp, pp_oot)):
        sample = df.sample(n=min(config.SHAP_SAMPLE, len(df)), random_state=seed).reset_index(drop=True)
        expl = shap_values(bundle, sample)
        gap = additivity_gap(bundle, sample, expl)
        imp = global_importance(expl)
        imp.to_csv(out_dir / f"shap_importance_{tag}.csv", index=False)
        al = dgp_alignment(imp, tag, expl, prepare(bundle, sample))
        align.append(al.assign(target=tag))
        prefix = "shap_dependence_pd" if tag == TARGET_PD else "shap_dependence_prepay"
        dependence_plots(expl, prepare(bundle, sample), DEPENDENCE_TOP_K if tag == TARGET_PD else 3, chart_dir, prefix)
        res[tag] = dict(expl=expl, X=prepare(bundle, sample), importance=imp, alignment=al, additivity_gap=gap)
    pd.concat(align, ignore_index=True).to_csv(out_dir / "dgp_alignment.csv", index=False)
    reason_code_table(bundle_pd, pd_oot.reset_index(drop=True)).to_csv(out_dir / "reason_codes.csv", index=False)
    return res
