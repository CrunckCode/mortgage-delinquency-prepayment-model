"""Macro-shock portfolio projection: model-based (PD + prepay bundles) and DGP-truth (synthetic only).

Delinquency status and TIMES_30DPD_12M are held at their snapshot values during the projection (simplification).
Survival is the UPB-weighted probability that a loan is still performing and unpaid, on the starting UPB.
"""
import logging

import numpy as np
import pandas as pd

from . import config, macro as macro_mod
from .columns import (LOAN_ID, PERIOD, AGE, FICO, ORIG_LTV, DTI, ORIG_UPB, ORIG_TERM, STATE, CUR_UPB, CUR_RATE,
                      DLQ_STATUS, IS_ACTIVE, PREPAY_FLAG, DEFAULT_FLAG, MKT_RATE, HPI, HPI_ORIG, UNEMP, MTM_LTV,
                      INCENTIVE, BURNOUT, SEASONING_RAMP, HPI_CHG_12M, UNEMP_CHG_12M, LOG_UPB, MONTH_OF_YEAR)
from .features import month_index, BURNOUT_THRESHOLD, SEASONING_MONTHS

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
ASOF_DATE = pd.Timestamp("2023-12-31")
SNAPSHOT_STATUSES = (0, 1)
HAZARD_CAP = 0.5            # monthly default hazard cap
SMM_CAP = 0.5
LAG_MONTHS = 12
CURVE_COLUMNS = ["month", "smm", "mdr", "cpr", "cdr", "survival_upb", "exp_defaults_upb", "exp_prepay_upb"]
SUMMARY_DEFAULT, SUMMARY_PREPAY = "default_12m_pct_upb", "prepay_12m_pct_upb"
SUMMARY_CPR, SUMMARY_CDR = "avg_cpr_12m", "avg_cdr_12m"
SCENARIO_COL = "scenario"
TRUTH_COLUMNS = ["scenario", "model_default_12m", "truth_default_12m", "default_rel_err",
                 "model_prepay_12m", "truth_prepay_12m", "prepay_rel_err"]
# ===== END CONFIG =====

STATUS_ABSORB_NOTE = "synthetic-only methodology check: truth is the simulation's own hazards"


def portfolio_asof(panel_feat: pd.DataFrame, asof: pd.Timestamp = ASOF_DATE) -> pd.DataFrame:
    """Loans active and still performing (status 0 or 1) at the as-of month end."""
    m = (panel_feat[PERIOD] == asof) & (panel_feat[IS_ACTIVE] == 1) & panel_feat[DLQ_STATUS].isin(SNAPSHOT_STATUSES)
    m &= (panel_feat[PREPAY_FLAG] == 0) & (panel_feat[DEFAULT_FLAG] == 0)
    return panel_feat[m].reset_index(drop=True)


def _extend_macro(macro: pd.DataFrame, asof, horizon: int) -> pd.DataFrame:
    end = asof + pd.offsets.MonthEnd(horizon)
    if macro[PERIOD].max() >= end:
        return macro
    log.warning("macro ends before projection horizon; extending with the synthetic macro path")
    return macro_mod.build_macro(macro[PERIOD].min(), end, states=macro[STATE].astype(str).unique())


class _Path:
    """Per-loan state arrays [horizon+1, n_loans] on month ends asof+j, under one shocked macro path."""

    def __init__(self, loans, macro, shock, asof, horizon):
        H, n = horizon, len(loans)
        shocked = macro_mod.apply_shock(_extend_macro(macro, asof, horizon), shock, asof)
        a0 = int(month_index([asof])[0])
        states = list(pd.unique(loans[STATE].astype(str)))
        sidx = pd.Categorical(loans[STATE].astype(str), categories=states).codes
        lo, hi = a0 - LAG_MONTHS, a0 + H
        cols = {}
        for c in (MKT_RATE, HPI, UNEMP):
            tbl = shocked.assign(_s=shocked[STATE].astype(str), _m=month_index(shocked[PERIOD]))
            tbl = tbl[tbl["_s"].isin(states) & (tbl["_m"] >= lo) & (tbl["_m"] <= hi)]
            piv = tbl.pivot(index="_s", columns="_m", values=c).reindex(states)
            if piv.isna().any().any() or piv.shape[1] != hi - lo + 1:
                raise ValueError("macro frame has gaps around the projection window")
            cols[c] = piv.to_numpy(np.float64)[sidx]                  # [n, H+13]
        off = LAG_MONTHS
        j = np.arange(H + 1)
        self.n, self.H = n, H
        self.mkt = cols[MKT_RATE][:, off + j].T
        self.hpi = cols[HPI][:, off + j].T
        self.unemp = cols[UNEMP][:, off + j].T
        self.hpi_chg = ((cols[HPI][:, off + j] / cols[HPI][:, off + j - LAG_MONTHS] - 1.0) * 100.0).T
        self.unemp_chg = (cols[UNEMP][:, off + j] - cols[UNEMP][:, off + j - LAG_MONTHS]).T

        rate = loans[CUR_RATE].to_numpy(np.float64)
        self.rate = rate
        self.inc = rate[None, :] - self.mkt
        age0 = loans[AGE].to_numpy(np.float64)
        self.age = age0[None, :] + j[:, None]
        r = rate / 1200.0
        n_rem = np.maximum(loans[ORIG_TERM].to_numpy(np.float64) - age0, 1.0)
        g = (1.0 + r) ** n_rem
        upb0 = loans[CUR_UPB].to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sched = np.where(r > 0, upb0 * (g - (1.0 + r) ** j[:, None]) / (g - 1.0),
                             upb0 * (1.0 - j[:, None] / n_rem))
        self.upb = np.maximum(sched, 0.0)
        burn0 = loans[BURNOUT].to_numpy(np.float64)
        add = np.vstack([np.zeros((1, n)), (self.inc[1:] > BURNOUT_THRESHOLD).astype(np.float64)])
        self.burn = burn0[None, :] + np.cumsum(add, axis=0)
        value0 = loans[ORIG_UPB].to_numpy(np.float64) / loans[ORIG_LTV].to_numpy(np.float64) * 100.0
        hpi_orig = loans[HPI_ORIG].to_numpy(np.float64)
        self.mtm = self.upb / (value0[None, :] * self.hpi / hpi_orig[None, :]) * 100.0
        self.month = np.array([(asof + pd.offsets.MonthEnd(k)).month for k in range(H + 1)])
        self.dates = [asof + pd.offsets.MonthEnd(k) for k in range(H + 1)]


def _monthly_hazard(pd12, cap=HAZARD_CAP):
    return np.minimum(1.0 - (1.0 - np.clip(pd12, 0.0, 1.0)) ** (1.0 / config.PD_HORIZON_MONTHS), cap)


def _curve_frame(rows, start_upb):
    out = pd.DataFrame(rows, columns=CURVE_COLUMNS)
    out["month"] = out["month"].astype(int)
    out.attrs["start_upb"] = float(start_upb)
    return out


def project_portfolio(bundle_pd, bundle_pp, loans_asof, macro, shock, horizon=config.PROJECTION_MONTHS,
                      asof=ASOF_DATE) -> pd.DataFrame:
    """Re-score each month on shocked macro; default hazard first, prepay on survivors in status C.

    Prepay is applied only to loans whose held status is current, because the prepay model is trained on
    current-status loan-months.
    """
    loans = loans_asof.reset_index(drop=True)
    path = _Path(loans, macro, shock, asof, horizon)
    X = loans.copy()
    upb0_total = float(loans[CUR_UPB].sum())
    w = np.ones(len(loans))
    current = (loans[DLQ_STATUS].to_numpy() == 0)
    rows = []
    for k in range(1, horizon + 1):
        j = k - 1
        X[AGE] = path.age[j].astype(X[AGE].dtype)
        X[CUR_UPB] = path.upb[j]
        X[LOG_UPB] = np.log(np.maximum(path.upb[j], 1.0))
        X[MKT_RATE], X[HPI], X[UNEMP] = path.mkt[j], path.hpi[j], path.unemp[j]
        X[MTM_LTV], X[INCENTIVE], X[BURNOUT] = path.mtm[j], path.inc[j], path.burn[j]
        X[SEASONING_RAMP] = np.minimum(path.age[j] / SEASONING_MONTHS, 1.0)
        X[HPI_CHG_12M], X[UNEMP_CHG_12M] = path.hpi_chg[j], path.unemp_chg[j]
        X[MONTH_OF_YEAR] = path.month[k]
        hd = _monthly_hazard(np.asarray(bundle_pd.predict_pd(X), dtype=np.float64))
        smm = np.minimum(np.asarray(bundle_pp.predict_smm(X), dtype=np.float64), SMM_CAP) * current
        exposure = w * path.upb[j]
        dflt = exposure * hd
        prep = (exposure - dflt) * smm
        e_tot = exposure.sum()
        surv = e_tot - dflt.sum()
        mdr = dflt.sum() / e_tot if e_tot > 0 else 0.0
        smm_w = prep.sum() / surv if surv > 0 else 0.0
        w = w * (1.0 - hd) * (1.0 - smm)
        rows.append((k, smm_w, mdr, 1.0 - (1.0 - smm_w) ** 12, 1.0 - (1.0 - mdr) ** 12,
                     float((w * loans[CUR_UPB].to_numpy()).sum() / upb0_total), dflt.sum(), prep.sum()))
    return _curve_frame(rows, upb0_total)


def dgp_hazards(age, upb, mtm, inc, burn, fico, dti, unemp, hpi_chg, month):
    """Faithful copy of the synthetic DGP monthly prepay (hp) and C-to-30 (hd) hazards."""
    P, T = config.TRUE_DGP["prepay"], config.TRUE_DGP["to30"]
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    f40, f50 = (fico - 740.0) / 40.0, (fico - 740.0) / 50.0
    refi = sig((inc - P["incentive_mid"]) / P["incentive_scale"]) * np.exp(-P["a_burn"] * burn) * (mtm < P["ltv_cap"])
    hp = sig(P["a0"] + P["a_ref"] * refi + P["a_seas"] * np.minimum(age / 30.0, 1.0) + P["a_fico"] * f40
             + P["a_size"] * np.log(upb / 200_000.0) + P["month_amp"] * np.cos(2 * np.pi * (month - 6) / 12.0))
    hump = np.exp(-(((age - 30.0) / 20.0) ** 2)) - 0.3
    hd = sig(T["b0"] + T["b_fico"] * f50 + T["b_ltv"] * np.maximum(mtm - 80.0, 0.0) / 10.0
             + T["b_dti"] * (dti - 36.0) / 10.0 + T["b_unemp"] * (unemp - 5.0)
             + T["b_hpi"] * np.minimum(hpi_chg, 0.0) / 10.0 + T["b_age"] * hump)
    return hp, hd


def roll_probs(mtm, fico):
    """Per-loan 30-to-60 and 60-to-90 roll probabilities (DGP), tilted by FICO and LTV."""
    R30, R60 = config.TRUE_DGP["roll30"], config.TRUE_DGP["roll60"]
    tilt = np.exp(R30["fico_k"] * (fico - 740.0) / 50.0 + R30["ltv_k"] * np.maximum(mtm - 80.0, 0.0) / 10.0)
    return np.clip(R30["base"] * tilt, 0.0, 0.9), np.clip(R60["base"] * tilt, 0.0, 0.9)


def project_truth(loans_asof, macro, shock, horizon=config.PROJECTION_MONTHS, asof=ASOF_DATE) -> pd.DataFrame:
    """Same projection with the DGP Markov chain (C, 30, 60 states) evaluated under the shocked macro."""
    loans = loans_asof.reset_index(drop=True)
    path = _Path(loans, macro, shock, asof, horizon)
    R30, R60 = config.TRUE_DGP["roll30"], config.TRUE_DGP["roll60"]
    fico, dti = loans[FICO].to_numpy(np.float64), loans[DTI].to_numpy(np.float64)
    upb0 = loans[CUR_UPB].to_numpy(np.float64)
    p0 = (loans[DLQ_STATUS].to_numpy() == 0).astype(np.float64)
    p1 = 1.0 - p0
    p2 = np.zeros(len(loans))
    rows = []
    for k in range(1, horizon + 1):
        hp, hd = dgp_hazards(path.age[k], path.upb[k], path.mtm[k], path.inc[k], path.burn[k], fico, dti,
                             path.unemp[k], path.hpi_chg[k], path.month[k])
        r60, r90 = roll_probs(path.mtm[k], fico)
        bal = path.upb[k - 1]
        e_tot = ((p0 + p1 + p2) * bal).sum()
        dflt_m, prep_m = p2 * r90, p0 * hp
        dflt, prep = (dflt_m * bal).sum(), (prep_m * bal).sum()
        n0 = p0 * (1.0 - hp - hd) + p1 * R30["cure"] + p2 * R60["cure"]
        n1 = p0 * hd + p1 * np.maximum(1.0 - r60 - R30["cure"], 0.0)
        n2 = p1 * r60 + p2 * np.maximum(1.0 - r90 - R60["cure"], 0.0)
        p0, p1, p2 = n0, n1, n2
        mdr = dflt / e_tot if e_tot > 0 else 0.0
        surv_after = e_tot - dflt
        smm = prep / surv_after if surv_after > 0 else 0.0
        rows.append((k, smm, mdr, 1.0 - (1.0 - smm) ** 12, 1.0 - (1.0 - mdr) ** 12,
                     float(((p0 + p1 + p2) * upb0).sum() / upb0.sum()), dflt, prep))
    return _curve_frame(rows, upb0.sum())


def _start_upb(curve):
    start = curve.attrs.get("start_upb")
    if start is None:  # month 1 exposure equals the starting UPB when survival is 1
        start = float(curve["exp_defaults_upb"].iloc[0] / curve["mdr"].iloc[0])
    return start


def summary_12m(curves, start_upb=None) -> pd.DataFrame:
    """curves: dict scenario -> curve frame (start UPB read from frame attrs), or a long frame with start_upb."""
    if not isinstance(curves, dict):
        curves = {s: g.reset_index(drop=True) for s, g in curves.groupby(SCENARIO_COL, sort=False)}
    rows = []
    for s, c in curves.items():
        g = c[c["month"] <= 12]
        start = start_upb if start_upb is not None else _start_upb(c)
        rows.append({SCENARIO_COL: s, SUMMARY_DEFAULT: g["exp_defaults_upb"].sum() / start * 100.0,
                     SUMMARY_PREPAY: g["exp_prepay_upb"].sum() / start * 100.0,
                     SUMMARY_CPR: g["cpr"].mean(), SUMMARY_CDR: g["cdr"].mean()})
    return pd.DataFrame(rows)


def run_scenarios(bundle_pd, bundle_pp, panel_feat, macro, horizon=config.PROJECTION_MONTHS, with_truth=True,
                  out_dir=config.OUT_DIR, asof=ASOF_DATE) -> dict:
    """Run all config.SHOCKS; write scenario_curves.csv, scenario_12m.csv and model_vs_truth.csv."""
    loans = portfolio_asof(panel_feat, asof)
    start_upb = float(loans[CUR_UPB].sum())
    model, truth = {}, {}
    for shock in config.SHOCKS:
        model[shock.name] = project_portfolio(bundle_pd, bundle_pp, loans, macro, shock, horizon, asof)
        if with_truth:
            truth[shock.name] = project_truth(loans, macro, shock, horizon, asof)
    s_model = summary_12m(model, start_upb)
    out_dir.mkdir(parents=True, exist_ok=True)
    long = pd.concat([c.assign(**{SCENARIO_COL: s}) for s, c in model.items()], ignore_index=True)
    long = long[[SCENARIO_COL, *CURVE_COLUMNS]]
    long.to_csv(out_dir / "scenario_curves.csv", index=False)
    s_model.to_csv(out_dir / "scenario_12m.csv", index=False)
    res = dict(curves=model, truth_curves=truth, summary=s_model, n_loans=len(loans), start_upb=start_upb)
    if with_truth:
        s_truth = summary_12m(truth, start_upb)
        cmp = pd.DataFrame({SCENARIO_COL: s_model[SCENARIO_COL],
                            "model_default_12m": s_model[SUMMARY_DEFAULT], "truth_default_12m": s_truth[SUMMARY_DEFAULT],
                            "model_prepay_12m": s_model[SUMMARY_PREPAY], "truth_prepay_12m": s_truth[SUMMARY_PREPAY]})
        cmp["default_rel_err"] = cmp["model_default_12m"] / cmp["truth_default_12m"] - 1.0
        cmp["prepay_rel_err"] = cmp["model_prepay_12m"] / cmp["truth_prepay_12m"] - 1.0
        cmp = cmp[TRUTH_COLUMNS]
        cmp.insert(len(cmp.columns), "note", STATUS_ABSORB_NOTE)
        cmp.to_csv(out_dir / "model_vs_truth.csv", index=False)
        res["truth_summary"], res["model_vs_truth"] = s_truth, cmp
    return res
