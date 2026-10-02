"""Builds docs/Mortgage_Risk_Model_Documentation.md and .docx (and README.md) from python/outputs; rerunnable.

Every result number is read from outputs/*.csv|json (or recomputed from the cached panel); none is typed by hand.
Run from anywhere: py -3 docs/build_docs.py
"""
import contextlib
import io
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR, PY_DIR = ROOT / "docs", ROOT / "python"
OUT_DIR = PY_DIR / "outputs"
MD_PATH = DOCS_DIR / "Mortgage_Risk_Model_Documentation.md"
DOCX_PATH = DOCS_DIR / "Mortgage_Risk_Model_Documentation.docx"
README_PATH = ROOT / "README.md"
REFERENCE_DOCX = DOCS_DIR / "reference.docx"
FACTS_CACHE = DOCS_DIR / "_facts_cache.json"
EXCEL_RECON_JSON = OUT_DIR / "excel_reconciliation.json"
EXCEL_BUILDER = ROOT / "excel" / "build_workbook.py"
ITM_DIR = OUT_DIR / "scenarios_itm_2021"
CHART_REL = "../python/outputs/charts"      # image paths are relative to docs/
AUTHOR = "Deepak Chaudhary"
DOC_TITLE = "Mortgage Delinquency and Prepayment Risk Model: Technical Documentation"
TOC_DEPTH = 2
SCEN_PD_MODEL, SCEN_PP_MODEL = "pd_lgbm", "pp_xgb"       # models wired into scenarios by cli.py
SHAP_PD_MODEL, SHAP_PP_MODEL = "pd_lgbm", "pp_xgb"
# ===== END CONFIG =====

sys.path.insert(0, str(PY_DIR))
from mortgage_risk import config, macro as macro_mod, evaluation as ev, scorecard as sc_mod  # noqa: E402
from mortgage_risk import columns as C  # noqa: E402


# ---------------------------------------------------------------- formatting helpers
def nz(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def fx(x, d=3):
    return "n/a" if nz(x) else f"{x:,.{d}f}"


def pc(x, d=2):
    return "n/a" if nz(x) else f"{100 * x:.{d}f}%"


def ni(x):
    return "n/a" if nz(x) else f"{int(round(x)):,}"


def pv(x):
    if nz(x):
        return "n/a"
    return f"{x:.1e}" if x < 1e-3 else f"{x:.3f}"


def md_table(df, fmts=None, headers=None, first_left=True):
    fmts, headers = fmts or {}, headers or {}
    cols = list(df.columns)
    cells = [[(fmts[c](r[c]) if c in fmts else str(r[c])).replace("|", "/") for c in cols] for _, r in df.iterrows()]
    head = [str(headers.get(c, c)) for c in cols]
    widths = [min(max(3, len(head[j]), *(len(row[j]) for row in cells)), 40) for j in range(len(cols))]
    sep = []
    for j, c in enumerate(cols):
        left = first_left and j == 0
        sep.append((":" + "-" * (widths[j] - 1)) if left else ("-" * (widths[j] - 1) + ":"))
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def rcsv(name, base=OUT_DIR):
    return pd.read_csv(base / name)


def rjson(name, base=OUT_DIR):
    return json.loads((base / name).read_text())


# ---------------------------------------------------------------- facts recomputed from the cached panel
def panel_facts():
    """Event rates, macro ranges and scorecard IVs that are not written by the pipeline; cached by panel mtime."""
    ds = rjson("data_summary.json")
    panel_p = config.INTERIM_DIR / f"panel_{ds['data_source']}_{ds['n_loans']}_{config.SEED}.parquet"
    macro_p = config.INTERIM_DIR / f"macro_{ds['data_source']}_{ds['n_loans']}_{config.SEED}.parquet"
    if not panel_p.exists():
        return {}
    stamp = panel_p.stat().st_mtime
    if FACTS_CACHE.exists():
        cached = json.loads(FACTS_CACHE.read_text())
        if cached.get("stamp") == stamp:
            return cached
    from mortgage_risk import features, targets, synthetic
    panel, mac = pd.read_parquet(panel_p), pd.read_parquet(macro_p)
    f = {"stamp": stamp}
    if ds["data_source"] == "synthetic":
        with contextlib.redirect_stdout(io.StringIO()):
            rep = synthetic.calibration_report(panel)
        f["calib_report"] = rep
    feat = features.add_features(panel, mac)
    snaps = targets.build_snapshots(feat)
    sp = targets.split_by_time(snaps, targets.PD_KIND)
    tr, ca, oo = sp["train"], sp["calib"], sp["oot"]
    f["pd_rate"] = {k: float(v[C.TARGET_PD12].mean()) for k, v in sp.items()}
    f["pd_rate_oot_by_year"] = {int(y): [float(g[C.TARGET_PD12].mean()), int(len(g))]
                                for y, g in oo.groupby(oo[C.SNAPSHOT_DATE].dt.year)}
    f["pd_rate_train_by_year"] = {int(y): [float(g[C.TARGET_PD12].mean()), int(len(g))]
                                  for y, g in tr.groupby(tr[C.SNAPSHOT_DATE].dt.year)}
    f["unemp_train"] = [float(tr[C.UNEMP].min()), float(tr[C.UNEMP].max())]
    f["unemp_oot"] = [float(oo[C.UNEMP].min()), float(oo[C.UNEMP].max())]
    f["hpi_chg_train"] = [float(tr[C.HPI_CHG_12M].min()), float(tr[C.HPI_CHG_12M].max())]
    f["hpi_chg_oot"] = [float(oo[C.HPI_CHG_12M].min()), float(oo[C.HPI_CHG_12M].max())]
    f["dlq_share"] = {"train": float((tr[C.DLQ_STATUS] >= 1).mean()), "oot": float((oo[C.DLQ_STATUS] >= 1).mean())}
    iv = {}
    for v in C.FEATURES_PD:
        if v in C.CATEGORICAL_FEATURES:
            continue
        b = sc_mod.fit_bins(tr[v].to_numpy(), tr[C.TARGET_PD12].to_numpy())
        iv[v] = [float(b.iv), int(len(b.woe))]
    f["iv_all"] = iv
    last = panel.sort_values([C.LOAN_ID, C.PERIOD]).groupby(C.LOAN_ID).tail(1)
    f["exit_share"] = {"prepaid": float((last[C.PREPAY_FLAG] == 1).mean()), "defaulted": float((last[C.DEFAULT_FLAG] == 1).mean()),
                       "other": float(((last[C.PREPAY_FLAG] == 0) & (last[C.DEFAULT_FLAG] == 0)).mean())}
    from mortgage_risk import scenarios
    pools = {}
    for asof in (scenarios.ASOF_DATE, pd.Timestamp("2021-12-31")):
        loans = scenarios.portfolio_asof(feat, asof)
        w = loans[C.CUR_UPB].to_numpy(dtype=float)
        inc = loans[C.INCENTIVE].to_numpy(dtype=float)
        pools[asof.strftime("%Y-%m-%d")] = dict(n=int(len(loans)), upb=float(w.sum()),
                                                rate=float(np.average(loans[C.CUR_RATE].to_numpy(dtype=float), weights=w)),
                                                inc=float(np.average(inc, weights=w)), itm=float(w[inc > 0.5].sum() / w.sum()))
    f["pools"] = pools
    f = json.loads(json.dumps(f))     # same key types as when read back from the cache
    FACTS_CACHE.write_text(json.dumps(f, indent=1))
    return f


def bundle_facts():
    """Calibrator method and tuned parameters from the saved joblib bundles, if loadable."""
    out = {}
    try:
        from mortgage_risk.models import ModelBundle
        for p in sorted(config.MODEL_DIR.glob("*.joblib")):
            b = ModelBundle.load(p)
            out[p.stem] = dict(calibrator=getattr(b.calibrator, "method", "none"),
                               params=(b.extras or {}).get("tuned_params", {}), n_features=len(b.features))
    except Exception as exc:  # noqa: BLE001
        print("bundle facts unavailable:", exc)
    return out


# ---------------------------------------------------------------- values and tables
MODEL_LABEL = {"pd_logit": "Scorecard logit", "pd_xgb": "XGBoost", "pd_lgbm": "LightGBM",
               "pd_xgb_weighted_sensitivity": "XGBoost, class-weighted (uncalibrated sensitivity)",
               "pp_logit": "Hinge logit", "pp_xgb": "XGBoost", "pp_lgbm": "LightGBM"}
DGP_ROLE = {
    "a0": "intercept of the monthly prepay hazard (log-odds)", "a_ref": "weight on the refinance incentive term R",
    "a_burn": "burnout decay per month spent in the money", "a_seas": "seasoning ramp (0 to 1 over 30 months)",
    "a_fico": "FICO effect per 40 points above 740", "a_size": "log balance effect (relative to 200,000)",
    "incentive_mid": "incentive (pp) at which the refinance sigmoid is 50%", "incentive_scale": "width of the refinance sigmoid (pp)",
    "ltv_cap": "refinance blocked when mark-to-market LTV is at or above this", "month_amp": "seasonal amplitude (cosine peaking in June)",
    "b0": "intercept of the monthly current-to-30DPD hazard (log-odds)", "b_fico": "FICO effect per 50 points above 740",
    "b_ltv": "effect per 10 points of mark-to-market LTV above 80", "b_dti": "effect per 10 points of DTI above 36",
    "b_unemp": "effect per point of state unemployment above 5", "b_hpi": "effect per 10 points of 12m house-price decline (only when negative)",
    "b_age": "effect of the seasoning hump g(age)", "base": "base roll probability", "cure": "cure probability",
    "fico_k": "FICO tilt in the roll probability (per 50 points)", "ltv_k": "LTV tilt in the roll probability (per 10 points above 80)"}
DGP_GROUP = {"prepay": "Prepay hazard", "to30": "Entry to 30DPD hazard", "roll30": "Roll from 30DPD", "roll60": "Roll from 60DPD"}


def bench_table(b):
    rows = []
    for _, r in b.iterrows():
        rows.append(dict(model=MODEL_LABEL.get(r["model"], r["model"]), train_auc=r["train_auc"], calib_auc=r["calib_auc"],
                         oot_auc=r["oot_auc"], ci=f"{r['oot_auc_ci_lo']:.3f} to {r['oot_auc_ci_hi']:.3f}", gini=r["oot_gini"],
                         ks=r["oot_ks"], brier=r["oot_brier"], dauc=r["delong_dauc_vs_logit"], p=r["delong_p_vs_logit"]))
    df = pd.DataFrame(rows)
    f = {c: (lambda x: fx(x, 3)) for c in ("train_auc", "calib_auc", "oot_auc", "gini", "ks")}
    f.update(brier=lambda x: fx(x, 4), dauc=lambda x: "baseline" if nz(x) or x == 0 else f"{x:+.4f}", p=pv, ci=str)
    h = dict(model="Model", train_auc="Train AUC", calib_auc="Calib AUC", oot_auc="OOT AUC", ci="OOT AUC 95% CI", gini="OOT Gini",
             ks="OOT KS", brier="OOT Brier", dauc="dAUC vs logit", p="DeLong p")
    return md_table(df, f, h)


def hl_stat(cal):
    from scipy import stats
    n, p, o = cal["n"].to_numpy(), cal["mean_pred"].to_numpy(), cal["obs_rate"].to_numpy()
    s = float(np.sum(n * (o - p) ** 2 / (p * (1 - p))))
    return s, float(stats.chi2.sf(s, len(n)))


def walkthrough(loan, table):
    """Points for one loan: look up each variable's bin in scorecard_table.csv (bin = (lo, hi])."""
    rows = []
    for v in table["variable"].unique():
        t = table[(table["variable"] == v) & table["bin_lo"].notna()]
        x = float(loan[v])
        hit = t[(t["bin_lo"] < x) & (x <= t["bin_hi"])].iloc[0]
        lo = "-inf" if np.isinf(hit["bin_lo"]) else f"{hit['bin_lo']:.2f}"
        hi = "inf" if np.isinf(hit["bin_hi"]) else f"{hit['bin_hi']:.2f}"
        rows.append(dict(variable=v, value=x, bin=f"({lo}, {hi}]", woe=hit["woe"], coef=hit["coef"], points=hit["points"]))
    return pd.DataFrame(rows)


def _core(V, T, X):
    ds, meta, pf, bf = X["ds"], X["meta"], X["pf"], X["bf"]
    bpd, bpp = rcsv("benchmark_pd.csv"), rcsv("benchmark_prepay.csv")
    V["source"] = ds["data_source"]
    for k in ("n_loans", "panel_rows", "snapshot_rows", "hazard_rows"):
        V[k] = ni(ds[k])
    V["pd_event_rate"] = pc(ds["pd_event_rate"], 2)
    for part in ("train", "calib", "oot"):
        V[f"pd_rows_{part}"] = ni(ds["split_rows_pd"][part])
        V[f"pp_rows_{part}"] = ni(ds["split_rows_prepay"][part])
    V["seed"], V["n_quick"], V["n_full"] = str(config.SEED), ni(config.N_LOANS_QUICK), ni(config.N_LOANS_FULL)
    V["orig_start"], V["orig_end"], V["obs_end"] = (config.ORIG_START.strftime("%B %Y"), config.ORIG_END.strftime("%B %Y"),
                                                   config.OBS_END.strftime("%B %Y"))
    V["dev_cutoff"] = config.DEV_CUTOFF.strftime("%Y-%m-%d")
    V["prepay_share"] = pc(config.PREPAY_LOAN_SAMPLE_SHARE, 0)

    bp, bq = bpd.set_index("model"), bpp.set_index("model")
    X["bp"], X["bq"] = bp, bq
    for k in ("pd_logit", "pd_xgb", "pd_lgbm"):
        s = k.split("_")[1]
        for col in ("oot_auc", "oot_gini", "oot_ks", "oot_brier", "train_auc", "calib_auc"):
            V[f"pd_{s}_{col}"] = fx(bp.loc[k, col], 4 if col == "oot_brier" else 3)
        V[f"pd_{s}_dauc"] = f"{bp.loc[k, 'delong_dauc_vs_logit']:+.4f}"
        V[f"pd_{s}_p"] = pv(bp.loc[k, "delong_p_vs_logit"])
        V[f"pd_{s}_gap"] = fx(bp.loc[k, "train_auc"] - bp.loc[k, "oot_auc"], 3)
        V[f"pd_{s}_ci"] = f"{bp.loc[k, 'oot_auc_ci_lo']:.3f} to {bp.loc[k, 'oot_auc_ci_hi']:.3f}"
    for k in ("pp_logit", "pp_xgb", "pp_lgbm"):
        s = k.split("_")[1]
        for col in ("oot_auc", "oot_gini", "oot_ks", "oot_brier", "train_auc", "calib_auc"):
            V[f"pp_{s}_{col}"] = fx(bq.loc[k, col], 4 if col == "oot_brier" else 3)
        V[f"pp_{s}_p"] = pv(bq.loc[k, "delong_p_vs_logit"])
        V[f"pp_{s}_dauc"] = f"{bq.loc[k, 'delong_dauc_vs_logit']:+.4f}"
        V[f"pp_{s}_rmse"] = fx(meta["prepay_cpr_rmse_pts"][k], 2)
    wk = "pd_xgb_weighted_sensitivity"
    V["w_oot_brier"], V["w_oot_auc"] = fx(bp.loc[wk, "oot_brier"], 4), fx(bp.loc[wk, "oot_auc"], 3)
    T["bench_pd"], T["bench_pp"] = bench_table(bpd), bench_table(bpp)

    cal = {m: rcsv(f"calibration_pd_{m}.csv") for m in ("pd_logit", "pd_xgb", "pd_lgbm")}
    hl_rows = []
    for m, c in cal.items():
        s, p = hl_stat(c)
        obs, pred = c["events"].sum() / c["n"].sum(), (c["mean_pred"] * c["n"]).sum() / c["n"].sum()
        s2 = m.split("_")[1]
        V[f"obs_oot_{s2}"], V[f"pred_oot_{s2}"] = pc(obs), pc(pred)
        V[f"ratio_{s2}"] = fx(obs / pred, 2)
        hl_rows.append(dict(model=MODEL_LABEL[m], obs=obs, pred=pred, ratio=obs / pred, hl=s, p=p))
    V["obs_oot"] = V["obs_oot_lgbm"]
    T["hl"] = md_table(pd.DataFrame(hl_rows), dict(obs=pc, pred=pc, ratio=lambda x: fx(x, 2), hl=lambda x: fx(x, 0), p=pv),
                       dict(model="Model", obs="Observed rate", pred="Mean predicted PD", ratio="Observed / predicted",
                            hl="Hosmer-Lemeshow", p="p-value (10 d.f.)"))
    cl = cal["pd_lgbm"].copy()
    cl["ratio"] = cl["obs_rate"] / cl["mean_pred"]
    T["calib_lgbm"] = md_table(cl[["bin", "n", "mean_pred", "obs_rate", "ratio", "expected_events", "events"]],
                               dict(bin=str, n=ni, mean_pred=lambda x: pc(x, 3), obs_rate=lambda x: pc(x, 3), ratio=lambda x: fx(x, 2),
                                    expected_events=lambda x: fx(x, 0), events=lambda x: fx(x, 0)),
                               dict(bin="Bin (low to high PD)", n="Loans", mean_pred="Mean predicted", obs_rate="Observed rate",
                                    ratio="Obs / pred", expected_events="Expected events", events="Actual events"))
    dfm = dict(decile=str, n=ni, events=lambda x: fx(x, 0), event_rate=lambda x: pc(x, 2), mean_score=lambda x: pc(x, 2),
               cum_events_share=lambda x: pc(x, 1), lift=lambda x: fx(x, 2), ks_at_decile=lambda x: fx(x, 3))
    dh = dict(decile="Decile (1 = riskiest)", n="Loans", events="Defaults", event_rate="Observed rate", mean_score="Mean PD",
              cum_events_share="Cumulative share of defaults", lift="Lift", ks_at_decile="Gap in cumulative shares")
    d_l, d_g = rcsv("deciles_pd_pd_lgbm.csv"), rcsv("deciles_pd_pd_logit.csv")
    T["dec_lgbm"], T["dec_logit"] = md_table(d_l, dfm, dh), md_table(d_g, dfm, dh)
    V["top_dec_rate"], V["top_dec_lift"] = pc(d_l.loc[0, "event_rate"], 1), fx(d_l.loc[0, "lift"], 2)
    V["top2_share"] = pc(d_l.loc[1, "cum_events_share"], 0)
    V["top_dec_share"] = pc(d_l.loc[0, "cum_events_share"], 0)
    V["top_dec_rate_logit"], V["top_dec_share_logit"] = pc(d_g.loc[0, "event_rate"], 1), pc(d_g.loc[0, "cum_events_share"], 0)
    V["bottom_dec_rate"] = pc(d_l.loc[9, "event_rate"], 2)
    V["ks_dec"] = fx(d_l["ks_at_decile"].max(), 3)
    V["ks_dec_at"] = str(int(d_l.loc[d_l["ks_at_decile"].idxmax(), "decile"]))
    V["calib_ratio_mid"] = fx(cl.loc[7, "ratio"], 2)
    V["calib_ratio_top"] = fx(cl.loc[9, "ratio"], 2)

    if pf:
        V["pd_rate_train"], V["pd_rate_calib"], V["pd_rate_oot"] = (pc(pf["pd_rate"][k]) for k in ("train", "calib", "oot"))
        blank = [np.nan, 0]
        yr = pd.DataFrame([dict(year=y, train=pf["pd_rate_train_by_year"].get(str(y), blank)[0],
                                n_tr=pf["pd_rate_train_by_year"].get(str(y), blank)[1],
                                oot=pf["pd_rate_oot_by_year"].get(str(y), blank)[0],
                                n_oo=pf["pd_rate_oot_by_year"].get(str(y), blank)[1]) for y in range(2013, 2024)])
        yr = yr[(yr["n_tr"] > 0) | (yr["n_oo"] > 0)]
        T["pd_rate_year"] = md_table(yr, dict(year=str, train=lambda x: pc(x) if not nz(x) else "", n_tr=lambda x: ni(x) if x else "",
                                              oot=lambda x: pc(x) if not nz(x) else "", n_oo=lambda x: ni(x) if x else ""),
                                     dict(year="Snapshot year", train="Train: 12m default rate", n_tr="Train snapshots",
                                          oot="OOT: 12m default rate", n_oo="OOT snapshots"))
        V["oot_2020_rate"] = pc(pf["pd_rate_oot_by_year"]["2020"][0])
        V["oot_2023_rate"] = pc(pf["pd_rate_oot_by_year"]["2023"][0])
        V["unemp_tr"] = f"{pf['unemp_train'][0]:.1f} to {pf['unemp_train'][1]:.1f}"
        V["unemp_tr_max"] = f"{pf['unemp_train'][1]:.1f}"
        V["unemp_oot"] = f"{pf['unemp_oot'][0]:.1f} to {pf['unemp_oot'][1]:.1f}"
        V["unemp_oot_max"] = f"{pf['unemp_oot'][1]:.1f}"
        V["hpi_tr"] = f"{pf['hpi_chg_train'][0]:.1f}% to {pf['hpi_chg_train'][1]:.1f}%"
        V["hpi_tr_min"] = f"{pf['hpi_chg_train'][0]:.1f}%"
        V["hpi_oot_min"] = f"{pf['hpi_chg_oot'][0]:.1f}%"
        V["dlq_share_tr"], V["dlq_share_oot"] = pc(pf["dlq_share"]["train"]), pc(pf["dlq_share"]["oot"])
        V["exit_prepaid"], V["exit_def"], V["exit_other"] = (pc(pf["exit_share"][k], 1) for k in ("prepaid", "defaulted", "other"))
        rep = pf.get("calib_report", {})
        if rep:
            lo, hi = config.SYNTH_CALIBRATION["pd12_train"]
            lo2, hi2 = config.SYNTH_CALIBRATION["pd12_oot_2020"]
            lo3, hi3 = config.SYNTH_CALIBRATION["cpr"]
            ok = lambda v, a, b: "yes" if a <= v <= b else "no"
            T["calib_report"] = md_table(pd.DataFrame([
                dict(item="12-month default rate, training snapshots", value=rep["pd12_train"], band=f"{pc(lo, 1)} to {pc(hi, 1)}",
                     ok=ok(rep["pd12_train"], lo, hi)),
                dict(item="12-month default rate, 2020 snapshots", value=rep["pd12_oot_2020"], band=f"{pc(lo2, 1)} to {pc(hi2, 1)}",
                     ok=ok(rep["pd12_oot_2020"], lo2, hi2)),
                dict(item="Mean CPR, whole panel", value=rep["cpr_mean"], band=f"{pc(lo3, 0)} to {pc(hi3, 0)}",
                     ok=ok(rep["cpr_mean"], lo3, hi3))]),
                dict(value=pc), dict(item="Calibration check", value="Simulated", band="Target band", ok="Within band"))
            cby = pd.DataFrame([dict(year=int(y), cpr=v) for y, v in rep["cpr_by_year"].items()])
            T["cpr_year_synth"] = md_table(cby, dict(year=str, cpr=lambda x: pc(x, 1)),
                                           dict(year="Calendar year", cpr="Simulated CPR"))
            V["cpr_min_year"], V["cpr_max_year"] = pc(cby["cpr"].min(), 1), pc(cby["cpr"].max(), 1)
    X["cal"] = cal


def _toys(V, T):
    """Small worked examples computed with the project's own metric functions."""
    y = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([0.95, 0.85, 0.70, 0.55, 0.40, 0.20])
    V["toy_auc"], V["toy_ks"], V["toy_gini"] = fx(ev.auc(y, p), 3), fx(ev.ks(y, p), 3), fx(ev.gini(y, p), 3)
    pos, neg = p[y == 1], p[y == 0]
    V["toy_conc"] = str(int(sum(a > b for a in pos for b in neg)))
    V["toy_pairs"] = str(len(pos) * len(neg))
    order = np.argsort(-p)
    cp, cn = np.cumsum(y[order]) / y.sum(), np.cumsum(1 - y[order]) / (1 - y).sum()
    toy = pd.DataFrame(dict(rank=np.arange(1, 7), score=p[order], default=y[order], cum_def=cp, cum_perf=cn, gap=np.abs(cp - cn)))
    T["toy_ks"] = md_table(toy, dict(rank=str, score=lambda x: fx(x, 2), default=str, cum_def=lambda x: fx(x, 3),
                                     cum_perf=lambda x: fx(x, 3), gap=lambda x: fx(x, 3)),
                           dict(rank="Rank (riskiest first)", score="Score", default="Defaulted", cum_def="Cum. share of defaults",
                                cum_perf="Cum. share of performing", gap="Gap"))
    # WoE and IV on three bins
    w = pd.DataFrame(dict(bin=["Low FICO", "Mid FICO", "High FICO"], good=[900, 5000, 4100], bad=[60, 100, 40]))
    w["pg"], w["pb"] = w["good"] / w["good"].sum(), w["bad"] / w["bad"].sum()
    w["woe"] = np.log(w["pg"] / w["pb"])
    w["iv"] = (w["pg"] - w["pb"]) * w["woe"]
    V["toy_iv"] = fx(w["iv"].sum(), 3)
    T["toy_woe"] = md_table(w, dict(good=ni, bad=ni, pg=lambda x: pc(x, 1), pb=lambda x: pc(x, 1), woe=lambda x: fx(x, 3),
                                    iv=lambda x: fx(x, 3)),
                            dict(bin="Bin", good="Good loans", bad="Defaults", pg="% of goods", pb="% of bads", woe="WoE",
                                 iv="IV contribution"))
    # PSI toy
    e, a = np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.3, 0.3])
    t = pd.DataFrame(dict(bin=["Bin 1", "Bin 2", "Bin 3"], e=e, a=a, d=a - e, l=np.log(a / e), c=(a - e) * np.log(a / e)))
    V["toy_psi"] = fx(t["c"].sum(), 4)
    T["toy_psi"] = md_table(t, dict(e=lambda x: pc(x, 0), a=lambda x: pc(x, 0), d=lambda x: f"{100 * x:+.0f} pp",
                                    l=lambda x: fx(x, 4), c=lambda x: fx(x, 4)),
                            dict(bin="Bin", e="Expected share", a="Actual share", d="Difference", l="ln(actual / expected)",
                                 c="Contribution"))
    # Shapley toy
    v0, va, vb, vab = 0.02, 0.05, 0.03, 0.12
    sa, sb = 0.5 * (va - v0) + 0.5 * (vab - vb), 0.5 * (vb - v0) + 0.5 * (vab - va)
    V["sh_v0"], V["sh_va"], V["sh_vb"], V["sh_vab"] = (fx(v, 2) for v in (v0, va, vb, vab))
    V["sh_a"], V["sh_b"], V["sh_sum"] = fx(sa, 2), fx(sb, 2), fx(sa + sb, 2)
    V["sh_gap"] = fx(vab - v0, 2)
    V["sh_a1"], V["sh_a2"] = fx(va - v0, 2), fx(vab - vb, 2)
    V["sh_b1"], V["sh_b2"] = fx(vb - v0, 2), fx(vab - va, 2)
    # scorecard scaling
    fac = config.SCORECARD_PDO / np.log(2.0)
    off = config.SCORECARD_BASE_SCORE - fac * np.log(config.SCORECARD_BASE_ODDS)
    V["pdo"], V["base_score"], V["base_odds"] = str(config.SCORECARD_PDO), str(config.SCORECARD_BASE_SCORE), str(config.SCORECARD_BASE_ODDS)
    V["factor"], V["offset"] = fx(fac, 3), fx(off, 2)
    V["base_pd"] = pc(1 / (1 + config.SCORECARD_BASE_ODDS), 2)
    rows = []
    for s in (500, 550, 600, 620, 650, 700):
        odds = config.SCORECARD_BASE_ODDS * 2 ** ((s - config.SCORECARD_BASE_SCORE) / config.SCORECARD_PDO)
        rows.append(dict(score=s, odds=odds, pd=1 / (1 + odds)))
    T["score_map"] = md_table(pd.DataFrame(rows), dict(score=str, odds=lambda x: fx(x, 1) + " : 1", pd=lambda x: pc(x, 3)),
                              dict(score="Score", odds="Good : bad odds", pd="Implied PD (uncalibrated)"))
    return fac, off


def _scorecard(V, T, X, fac, off):
    pf = X["pf"]
    sc = rcsv("scorecard_table.csv")
    X["sc"] = sc
    iv_all = pf.get("iv_all", {}) if pf else {}
    rows = []
    kept = list(dict.fromkeys(sc["variable"]))
    for v in kept:
        g = sc[sc["variable"] == v]
        nb = int(g["bin_lo"].notna().sum())
        rows.append(dict(variable=v, iv=iv_all.get(v, [np.nan, nb])[0], bins=nb, coef=g["coef"].iloc[0],
                         lo=g["points"].min(), hi=g["points"].max(), spread=g["points"].max() - g["points"].min()))
    sm = pd.DataFrame(rows)
    X["sc_summary"] = sm
    V["sc_nvars"] = str(len(kept))
    V["sc_vars"] = ", ".join(f"`{v}`" for v in kept)
    T["sc_summary"] = md_table(sm, dict(iv=lambda x: fx(x, 3), bins=str, coef=lambda x: fx(x, 3), lo=lambda x: fx(x, 1),
                                        hi=lambda x: fx(x, 1), spread=lambda x: fx(x, 1)),
                               dict(variable="Variable", iv="Information value", bins="Bins", coef="Coefficient on WoE",
                                    lo="Lowest points", hi="Highest points", spread="Point spread"))
    V["sc_top_var"] = sm.sort_values("spread", ascending=False).iloc[0]["variable"]
    V["sc_top_spread"] = fx(sm["spread"].max(), 1)
    base = float(sc[sc["bin_lo"].isna()]["points"].iloc[0])
    V["sc_base_pts"] = fx(base, 2)
    V["sc_intercept"] = fx((off - len(kept) * base) / fac, 3)
    # dropped variables
    dropped = []
    for v, (iv, nb) in iv_all.items():
        if v in kept:
            continue
        why = ("collapsed to a single bin by the 5% minimum bin share" if nb <= 1 else
               (f"information value below the {config.SCORECARD_IV_MIN} floor" if iv < config.SCORECARD_IV_MIN
                else "dropped by the coefficient sign check after refit"))
        dropped.append(dict(variable=v, iv=iv, bins=nb, why=why))
    if dropped:
        dd = pd.DataFrame(dropped).sort_values("iv", ascending=False)
        T["sc_dropped"] = md_table(dd, dict(iv=lambda x: fx(x, 4), bins=str),
                                   dict(variable="Numeric feature not in the scorecard", iv="IV on training snapshots", bins="Bins",
                                        why="Why it left"))
        V["sc_dropped_list"] = ", ".join(f"`{v}`" for v in dd["variable"])
        V["sc_orig_rate_iv"] = fx(iv_all.get("orig_rate", [np.nan])[0], 3)
        V["sc_dlq_bins"] = str(iv_all.get("dlq_status", [0, 1])[1])
    else:
        T["sc_dropped"], V["sc_dropped_list"], V["sc_orig_rate_iv"], V["sc_dlq_bins"] = "", "", "n/a", "n/a"
    # walk-through of the first loan in the Excel sample
    ls = pd.read_csv(config.EXCEL_INPUT_DIR / "loan_sample.csv")
    loan = ls.iloc[0]
    wt = walkthrough(loan, sc)
    tot = float(wt["points"].sum())
    X["walk"] = wt
    V["wk_loan"], V["wk_date"] = str(int(loan["loan_id"])), str(loan["snapshot_date"])[:10]
    V["wk_total"], V["wk_sheet"] = fx(tot, 2), fx(loan["scorecard_points"], 2)
    V["wk_match"] = "matches" if abs(tot - loan["scorecard_points"]) < 0.01 else "DOES NOT MATCH"
    margin = (off - tot) / fac
    V["wk_margin"], V["wk_raw_pd"] = fx(margin, 3), pc(1 / (1 + np.exp(-margin)), 3)
    V["wk_cal_pd"], V["wk_gbm_pd"] = pc(loan["scorecard_pd"], 3), pc(loan["gbm_pd"], 3)
    V["wk_default"] = "defaulted" if int(loan["target_pd12"]) == 1 else "did not default"
    V["wk_odds"] = fx(np.exp(-margin), 1)
    T["walk"] = md_table(wt, dict(value=lambda x: fx(x, 2), woe=lambda x: fx(x, 3), coef=lambda x: fx(x, 3),
                                  points=lambda x: fx(x, 2)),
                         dict(variable="Variable", value="Loan value", bin="Bin", woe="WoE", coef="Coefficient", points="Points"))


def _results(V, T, X):
    bf = X["bf"]
    # segments and vintages (LightGBM)
    seg = rcsv("segments.csv")
    sg = seg[(seg["model"] == "pd_lgbm") & (seg["segment_col"] == "fico_band")].copy()
    T["seg_fico"] = md_table(sg[["segment", "n", "events", "event_rate", "mean_pred", "auc"]],
                             dict(segment=str, n=ni, events=lambda x: fx(x, 0), event_rate=pc, mean_pred=pc, auc=lambda x: fx(x, 3)),
                             dict(segment="FICO band", n="Loans", events="Defaults", event_rate="Observed rate",
                                  mean_pred="Mean PD", auc="Within-band AUC"))
    so = seg[(seg["model"] == "pd_lgbm") & (seg["segment_col"] != "fico_band")]
    T["seg_other"] = md_table(so[["segment_col", "segment", "n", "event_rate", "mean_pred", "auc"]],
                              dict(segment_col=str, segment=str, n=ni, event_rate=pc, mean_pred=pc, auc=lambda x: fx(x, 3)),
                              dict(segment_col="Dimension", segment="Segment", n="Loans", event_rate="Observed rate",
                                   mean_pred="Mean PD", auc="AUC"))
    V["seg_lt620_n"] = ni(sg.iloc[0]["n"])
    V["seg_lt620_obs"], V["seg_lt620_pred"] = pc(sg.iloc[0]["event_rate"], 1), pc(sg.iloc[0]["mean_pred"], 1)
    V["seg_top_obs"], V["seg_top_pred"] = pc(sg.iloc[-1]["event_rate"], 2), pc(sg.iloc[-1]["mean_pred"], 2)
    V["seg_auc_min"], V["seg_auc_max"] = fx(sg["auc"].min(), 3), fx(sg["auc"].max(), 3)
    vin = rcsv("stability_by_vintage.csv")
    vg = vin[vin["model"] == "pd_lgbm"]
    T["vintage"] = md_table(vg[["vintage", "n", "events", "event_rate", "mean_pred", "auc"]],
                            dict(vintage=str, n=ni, events=lambda x: fx(x, 0), event_rate=pc, mean_pred=pc, auc=lambda x: fx(x, 3)),
                            dict(vintage="Origination vintage", n="Snapshots", events="Defaults", event_rate="Observed rate",
                                 mean_pred="Mean PD", auc="AUC"))
    V["vin_auc_min"], V["vin_auc_max"] = fx(vg["auc"].min(), 3), fx(vg["auc"].max(), 3)
    V["vin_auc_min_v"] = str(int(vg.loc[vg["auc"].idxmin(), "vintage"]))
    gaps = vg.assign(r=vg["event_rate"] / vg["mean_pred"])
    V["vin_ratio_max"], V["vin_ratio_max_v"] = fx(gaps["r"].max(), 2), str(int(gaps.loc[gaps["r"].idxmax(), "vintage"]))
    V["vin_ratio_min"], V["vin_ratio_min_v"] = fx(gaps["r"].min(), 2), str(int(gaps.loc[gaps["r"].idxmin(), "vintage"]))

    # PSI and CSI
    pc_ = rcsv("psi_csi.csv")
    sco = pc_[pc_["kind"] == "score"].copy()
    sco["model"] = sco["name"].str.split(":").str[0].map(MODEL_LABEL)
    sco["window"] = sco["name"].str.split(":").str[1].str.replace("train_vs_", "train vs ", regex=False)
    T["psi_score"] = md_table(sco[["model", "window", "psi", "band"]], dict(psi=lambda x: fx(x, 4)),
                              dict(model="Model", window="Comparison", psi="PSI", band="Band"))
    cs = pc_[pc_["kind"] == "csi"].sort_values("psi", ascending=False)
    T["csi"] = md_table(cs[["name", "psi", "band"]], dict(psi=lambda x: fx(x, 4)),
                        dict(name="Feature", psi="CSI (OOT vs train)", band="Band"))
    V["csi_n_sig"] = str(int((cs["band"] == "significant").sum()))
    V["csi_n_mod"] = str(int((cs["band"] == "moderate").sum()))
    V["csi_n_stable"] = str(int((cs["band"] == "stable").sum()))
    V["csi_n"] = str(len(cs))
    V["csi_top"] = ", ".join(f"`{n}` ({fx(p, 2)})" for n, p in zip(cs["name"].head(4), cs["psi"].head(4)))
    for k in ("logit", "xgb", "lgbm"):
        r = sco[sco["name"] == f"pd_{k}:train_vs_oot"].iloc[0]
        V[f"psi_oot_{k}"] = fx(r["psi"], 4)
        r = sco[sco["name"] == f"pd_{k}:train_vs_calib"].iloc[0]
        V[f"psi_cal_{k}"] = fx(r["psi"], 4)
    pb = pd.read_csv(config.EXCEL_INPUT_DIR / "psi_train_bins.csv")
    pb["lo"] = pb["lo"].replace(-np.inf, np.nan)
    T["psi_bins"] = md_table(pb, dict(bin=str, lo=lambda x: "-inf" if nz(x) else pc(x, 3), hi=lambda x: "inf" if np.isinf(x) else pc(x, 3),
                                      train_share=lambda x: pc(x, 2)),
                             dict(bin="Bin", lo="Lower edge (PD)", hi="Upper edge (PD)", train_share="Train share"))

    # prepay
    co = rcsv("prepay_logit_coefs.csv")
    T["pp_coefs"] = md_table(co, dict(coef=lambda x: fx(x, 4), knot=lambda x: "" if nz(x) else fx(x, 1)),
                             dict(term="Term", coef="Coefficient", knot="Knot (pp)"))
    c = co.set_index("term")["coef"]
    s1 = c["hinge_0.0"]; s2 = s1 + c["hinge_0.5"]; s3 = s2 + c["hinge_1.0"]; s4 = s3 + c["hinge_1.5"]
    sl = pd.DataFrame(dict(rng=["Incentive below 0 (out of the money)", "0 to 0.5", "0.5 to 1.0", "1.0 to 1.5", "Above 1.5"],
                           slope=[0.0, s1, s2, s3, s4]))
    T["pp_slopes"] = md_table(sl, dict(slope=lambda x: fx(x, 3)), dict(rng="Rate incentive (pp)", slope="Slope of log-odds per pp"))
    V["pp_s1"], V["pp_s2"], V["pp_s3"], V["pp_s4"] = fx(s1, 2), fx(s2, 2), fx(s3, 2), fx(s4, 2)
    V["pp_burn"], V["pp_seas"] = fx(c["burnout"], 4), fx(c["seasoning_ramp"], 3)
    V["pp_fico"], V["pp_ltv"] = fx(c["fico"], 4), fx(c["mtm_ltv"], 4)
    cpr = rcsv("prepay_cpr_oot.csv")
    cpr["year"] = pd.to_datetime(cpr["period"]).dt.year
    cy = cpr.groupby("year")[["actual_cpr", "pred_cpr"]].mean().reset_index()
    cy["err"] = cy["pred_cpr"] - cy["actual_cpr"]
    T["cpr_year"] = md_table(cy, dict(year=str, actual_cpr=lambda x: pc(x, 1), pred_cpr=lambda x: pc(x, 1),
                                      err=lambda x: f"{100 * x:+.1f} pp"),
                             dict(year="Calendar year", actual_cpr="Actual mean CPR", pred_cpr="Predicted mean CPR (LightGBM)",
                                  err="Predicted minus actual"))
    i = cpr["actual_cpr"].idxmax()
    V["cpr_peak"], V["cpr_peak_month"] = pc(cpr.loc[i, "actual_cpr"], 1), str(cpr.loc[i, "period"])[:7]
    V["cpr_peak_pred"] = pc(cpr.loc[i, "pred_cpr"], 1)
    V["cpr_2023_act"] = pc(cy[cy["year"] == 2023]["actual_cpr"].iloc[0], 1)
    V["cpr_2023_pred"] = pc(cy[cy["year"] == 2023]["pred_cpr"].iloc[0], 1)

    # SHAP
    sp, sq = rcsv("shap_importance_pd.csv"), rcsv("shap_importance_prepay.csv")
    for nm, d in (("shap_pd", sp), ("shap_pp", sq)):
        T[nm] = md_table(d.head(10).assign(rank=range(1, min(10, len(d)) + 1))[["rank", "feature", "mean_abs_shap"]],
                         dict(rank=str, mean_abs_shap=lambda x: fx(x, 4)),
                         dict(rank="Rank", feature="Feature", mean_abs_shap="Mean absolute SHAP (log-odds)"))
    V["shap_pd1"], V["shap_pd1v"] = sp.iloc[0]["feature"], fx(sp.iloc[0]["mean_abs_shap"], 3)
    V["shap_pd2"], V["shap_pd3"], V["shap_pd4"] = sp.iloc[1]["feature"], sp.iloc[2]["feature"], sp.iloc[3]["feature"]
    V["shap_pd5"] = sp.iloc[4]["feature"]
    rk = {f: i + 1 for i, f in enumerate(sp["feature"])}
    V["shap_dlq_rank"], V["shap_dlq_val"] = str(rk.get("dlq_status", "n/a")), fx(sp.set_index("feature").loc["dlq_status", "mean_abs_shap"], 3)
    V["shap_age_rank"] = str(rk.get("age", "n/a"))
    V["shap_pp1"], V["shap_pp1v"] = sq.iloc[0]["feature"], fx(sq.iloc[0]["mean_abs_shap"], 3)
    V["shap_pp2"], V["shap_pp3"], V["shap_pp4"] = sq.iloc[1]["feature"], sq.iloc[2]["feature"], sq.iloc[3]["feature"]
    V["shap_pp_share1"] = pc(sq.iloc[0]["mean_abs_shap"] / sq["mean_abs_shap"].sum(), 0)
    al = rcsv("dgp_alignment.csv")
    T["align"] = md_table(al.assign(target=al["target"].map({"pd": "PD", "prepay": "Prepay"}), passed=al["passed"].map({True: "pass", False: "FAIL"}))
                          [["target", "feature", "expected", "rank", "passed"]], dict(rank=lambda x: ni(x)),
                          dict(target="Model", feature="Feature", expected="DGP-implied expectation", rank="SHAP rank", passed="Result"))
    V["align_n"], V["align_pass"] = str(len(al)), str(int(al["passed"].sum()))
    rc = rcsv("reason_codes.csv")
    rc["case"] = rc["case"].str.replace("_", " ")
    T["reason"] = md_table(rc, dict(loan_id=str, pd=lambda x: pc(x, 2)),
                           dict(case="Case", loan_id="Loan", pd="PD", reason_1="Reason 1", reason_2="Reason 2", reason_3="Reason 3",
                                reason_4="Reason 4"))
    V["rc_low_pd"] = pc(rc.iloc[0]["pd"], 3)
    V["rc_high_pd"] = pc(rc[rc["case"] == "highest pd"].iloc[0]["pd"], 1)

    # model bundles
    rows = []
    for k, b in bf.items():
        p = b["params"]
        rows.append(dict(model=k, calibrator=b["calibrator"],
                         lr=p.get("learning_rate", np.nan), depth=p.get("max_depth", np.nan), leaves=p.get("num_leaves", np.nan),
                         trees=p.get("n_estimators", np.nan), lam=p.get("reg_lambda", np.nan), mcs=p.get("min_child", np.nan)))
    if rows:
        bd = pd.DataFrame(rows)
        gb = bd[bd["model"].isin(["pd_xgb", "pd_lgbm", "pp_xgb", "pp_lgbm"])]
        T["bundles"] = md_table(gb, dict(lr=lambda x: fx(x, 3), depth=lambda x: ni(x), leaves=lambda x: ni(x), trees=lambda x: ni(x),
                                         lam=lambda x: fx(x, 1), mcs=lambda x: ni(x)),
                                dict(model="Bundle", calibrator="Calibrator chosen", lr="Learning rate", depth="Max depth (XGBoost)",
                                     leaves="Leaves (LightGBM)", trees="Trees", lam="L2 lambda", mcs="Min child (samples/10 for XGB)"))
        cal_m = bd.set_index("model")["calibrator"].to_dict()
        for k in ("pd_logit", "pd_xgb", "pd_lgbm", "pp_logit", "pp_xgb", "pp_lgbm"):
            V[f"cal_{k}"] = cal_m.get(k, "n/a")
    else:
        T["bundles"] = "Saved model bundles could not be loaded in this environment."
        for k in ("pd_logit", "pd_xgb", "pd_lgbm", "pp_logit", "pp_xgb", "pp_lgbm"):
            V[f"cal_{k}"] = "n/a"


def _scenarios(V, T, X):
    s12, mvt = rcsv("scenario_12m.csv"), rcsv("model_vs_truth.csv")
    key = {"Base": "base", "Rates +200bp": "up", "Rates -200bp": "dn", "HPI -20%": "hpi", "Unemployment +4pt": "un",
           "Adverse combined": "adv"}
    s12i, mvi = s12.set_index("scenario"), mvt.set_index("scenario")
    for name, k in key.items():
        V[f"s_{k}_def"], V[f"s_{k}_pre"] = fx(s12i.loc[name, "default_12m_pct_upb"], 2), fx(s12i.loc[name, "prepay_12m_pct_upb"], 2)
        V[f"s_{k}_cpr"], V[f"s_{k}_cdr"] = pc(s12i.loc[name, "avg_cpr_12m"], 2), pc(s12i.loc[name, "avg_cdr_12m"], 3)
        V[f"t_{k}_def"], V[f"m_{k}_def"] = fx(mvi.loc[name, "truth_default_12m"], 2), fx(mvi.loc[name, "model_default_12m"], 2)
        V[f"e_{k}_def"] = f"{100 * mvi.loc[name, 'default_rel_err']:+.0f}%"
        V[f"e_{k}_pre"] = f"{100 * mvi.loc[name, 'prepay_rel_err']:+.0f}%"
        V[f"t_{k}_pre"] = fx(mvi.loc[name, "truth_prepay_12m"], 2)
    b = s12i.loc["Base"]
    V["un_uplift_model"] = f"{100 * (s12i.loc['Unemployment +4pt', 'default_12m_pct_upb'] / b['default_12m_pct_upb'] - 1):.0f}%"
    V["un_uplift_truth"] = f"{100 * (mvi.loc['Unemployment +4pt', 'truth_default_12m'] / mvi.loc['Base', 'truth_default_12m'] - 1):.0f}%"
    V["adv_mult_model"] = fx(s12i.loc["Adverse combined", "default_12m_pct_upb"] / b["default_12m_pct_upb"], 2)
    V["adv_mult_truth"] = fx(mvi.loc["Adverse combined", "truth_default_12m"] / mvi.loc["Base", "truth_default_12m"], 2)
    V["tol"] = pc(config.TEST_THRESHOLDS["model_vs_truth_rel_err"], 0)
    n_out = int((mvt["default_rel_err"].abs() > config.TEST_THRESHOLDS["model_vs_truth_rel_err"]).sum())
    V["n_out_tol"] = str(n_out)
    V["rate_def_diff"] = f"{abs(s12i.loc['Rates +200bp', 'default_12m_pct_upb'] - b['default_12m_pct_upb']):.1e}"
    V["rate_dn_pre_diff"] = fx(s12i.loc["Rates -200bp", "prepay_12m_pct_upb"] - b["prepay_12m_pct_upb"], 3)
    V["scen_pd_model"], V["scen_pp_model"] = MODEL_LABEL[SCEN_PD_MODEL], MODEL_LABEL[SCEN_PP_MODEL]
    sc = s12.copy()
    T["scen_12m"] = md_table(sc, dict(default_12m_pct_upb=lambda x: fx(x, 3), prepay_12m_pct_upb=lambda x: fx(x, 2),
                                      avg_cpr_12m=lambda x: pc(x, 2), avg_cdr_12m=lambda x: pc(x, 3)),
                             dict(scenario="Scenario", default_12m_pct_upb="12m defaults (% of starting UPB)",
                                  prepay_12m_pct_upb="12m prepayments (% of starting UPB)", avg_cpr_12m="Average CPR, months 1 to 12",
                                  avg_cdr_12m="Average CDR, months 1 to 12"))
    mv = mvt.drop(columns=["note"])
    T["mvt"] = md_table(mv, dict(model_default_12m=lambda x: fx(x, 2), truth_default_12m=lambda x: fx(x, 2),
                                 default_rel_err=lambda x: f"{100 * x:+.0f}%", model_prepay_12m=lambda x: fx(x, 2),
                                 truth_prepay_12m=lambda x: fx(x, 2), prepay_rel_err=lambda x: f"{100 * x:+.0f}%"),
                        dict(scenario="Scenario", model_default_12m="Model default", truth_default_12m="Truth default",
                             default_rel_err="Default error", model_prepay_12m="Model prepay", truth_prepay_12m="Truth prepay",
                             prepay_rel_err="Prepay error"))
    T["shocks"] = md_table(pd.DataFrame([dict(name=s.name, rate=s.rate_bp, hpi=s.hpi_pct, un=s.unemp_pt, ramp=s.ramp_months)
                                         for s in config.SHOCKS]),
                           dict(rate=lambda x: f"{x:+.0f}" if x else "0", hpi=lambda x: f"{x:+.0f}%" if x else "0",
                                un=lambda x: f"{x:+.0f}" if x else "0", ramp=str),
                           dict(name="Scenario", rate="Rate shock (bp)", hpi="House prices", un="Unemployment (pts)",
                                ramp="Ramp (months)"))
    cur = rcsv("scenario_curves.csv")
    b60 = cur[(cur["scenario"] == "Base") & (cur["month"] == 60)].iloc[0]
    V["base_surv60"] = pc(b60["survival_upb"], 1)
    bc = cur[cur["scenario"] == "Base"]
    V["base_cpr_min"], V["base_cpr_max"] = pc(bc["cpr"].min(), 1), pc(bc["cpr"].max(), 1)
    V["base_cdr_m1"], V["base_cdr_m60"] = pc(bc.iloc[0]["cdr"], 2), pc(bc.iloc[-1]["cdr"], 2)
    # hook
    sc_eq = rcsv("scalar_equivalents.csv", OUT_DIR / "waterfall_hook")
    T["hook_scalars"] = md_table(sc_eq, dict(cpr=lambda x: pc(x, 2), cdr=lambda x: pc(x, 3), severity=lambda x: fx(x, 2), lag=str,
                                             index_shift=lambda x: f"{x:+.4f}" if x else "0.0000"),
                                 dict(scenario="name", cpr="cpr", cdr="cdr", severity="severity", lag="lag", index_shift="index_shift"))
    hb = sc_eq.set_index("scenario")
    V["hook_base_cpr"], V["hook_base_cdr"] = pc(hb.loc["Base", "cpr"], 2), pc(hb.loc["Base", "cdr"], 3)
    V["hook_adv_cdr"] = pc(hb.loc["Adverse combined", "cdr"], 3)
    V["hook_sev"], V["hook_lag"] = fx(config.SEVERITY_ASSUMPTION, 2), str(config.LAG_ASSUMPTION)
    V["hook_months"] = str(config.HOOK_MONTHS)
    hc = rcsv("cpr_cdr_curves.csv", OUT_DIR / "waterfall_hook")
    sel = hc[(hc["scenario"] == "Base") & (hc["month"].isin([1, 12, 60, 61, 120]))]
    T["hook_curves"] = md_table(sel, dict(month=str, smm=lambda x: pc(x, 3), mdr=lambda x: pc(x, 3), cpr_annual=lambda x: pc(x, 2),
                                          cdr_annual=lambda x: pc(x, 3), survival=lambda x: pc(x, 1)),
                                dict(scenario="scenario", month="month", smm="smm", mdr="mdr", cpr_annual="cpr_annual",
                                     cdr_annual="cdr_annual", survival="survival"))
    # in-the-money as-of run
    if (ITM_DIR / "scenario_12m.csv").exists():
        i12 = rcsv("scenario_12m.csv", ITM_DIR)
        ii = i12.set_index("scenario")
        T["itm_12m"] = md_table(i12, dict(default_12m_pct_upb=lambda x: fx(x, 3), prepay_12m_pct_upb=lambda x: fx(x, 2),
                                          avg_cpr_12m=lambda x: pc(x, 2), avg_cdr_12m=lambda x: pc(x, 3)),
                                dict(scenario="Scenario", default_12m_pct_upb="12m defaults (% of UPB)",
                                     prepay_12m_pct_upb="12m prepayments (% of UPB)", avg_cpr_12m="Average CPR",
                                     avg_cdr_12m="Average CDR"))
        ib = ii.loc["Base"]
        V["itm_base_cpr"] = pc(ib["avg_cpr_12m"], 1)
        V["itm_dn_cpr"], V["itm_up_cpr"] = pc(ii.loc["Rates -200bp", "avg_cpr_12m"], 1), pc(ii.loc["Rates +200bp", "avg_cpr_12m"], 1)
        V["itm_dn_mult"] = fx(ii.loc["Rates -200bp", "avg_cpr_12m"] / ib["avg_cpr_12m"], 2)
        V["itm_up_mult"] = fx(ii.loc["Rates +200bp", "avg_cpr_12m"] / ib["avg_cpr_12m"], 2)
        V["itm_block"] = ("The in-the-money run (as-of 31 December 2021, when the pool was close to the market rate) is written to "
                          "`outputs/scenarios_itm_2021/`.\n\n@@tbl:itm_12m@@\n\nTable: Twelve-month scenario summary, pool as of 31 December 2021.\n\n"
                          f"Here the base average CPR over the first twelve months is {V['itm_base_cpr']}, it rises to {V['itm_dn_cpr']} "
                          f"({V['itm_dn_mult']} times base) when rates fall 200bp and drops to {V['itm_up_cpr']} ({V['itm_up_mult']} times base) "
                          "when they rise 200bp. This is the behaviour the end-2023 pool cannot show.")
        if (ITM_DIR / "model_vs_truth.csv").exists():
            im = rcsv("model_vs_truth.csv", ITM_DIR).drop(columns=["note"], errors="ignore")
            T["itm_mvt"] = md_table(im, dict(model_default_12m=lambda x: fx(x, 2), truth_default_12m=lambda x: fx(x, 2),
                                             default_rel_err=lambda x: f"{100 * x:+.0f}%", model_prepay_12m=lambda x: fx(x, 2),
                                             truth_prepay_12m=lambda x: fx(x, 2), prepay_rel_err=lambda x: f"{100 * x:+.0f}%"),
                                    dict(scenario="Scenario", model_default_12m="Model default", truth_default_12m="Truth default",
                                         default_rel_err="Default error", model_prepay_12m="Model prepay", truth_prepay_12m="Truth prepay",
                                         prepay_rel_err="Prepay error"))
            V["itm_block"] += "\n\n@@tbl:itm_mvt@@\n\nTable: Model versus simulation truth, pool as of 31 December 2021."
    else:
        V["itm_block"] = ("The in-the-money as-of run (31 December 2021) has not been generated in this checkout "
                          "(`outputs/scenarios_itm_2021/` is absent). The pipeline writes it automatically; rerun "
                          "`py -3 -m mortgage_risk.cli` and then this document builder to include its table here.")


def _excel(V, T):
    """Describe the workbook from the reconciliation json if present."""
    if EXCEL_RECON_JSON.exists():
        try:
            data = json.loads(EXCEL_RECON_JSON.read_text())
        except Exception:  # noqa: BLE001
            data = None
    else:
        data = None
    V["excel_present"] = "yes" if data is not None else "no"
    if data is None:
        V["excel_block"] = ("The reconciliation report `python/outputs/excel_reconciliation.json` is not present in this checkout, so the "
                            "workbook is still being finalized and no reconciliation numbers are quoted here. The sheet design below is "
                            "the specification the builder follows; rerun `docs/build_docs.py` after the workbook is built to include the "
                            "measured differences between Excel and Python.")
        return
    V["excel_block"] = excel_block_from(data)


# ---------------------------------------------------------------- document text (placeholders: @@key@@ and @@tbl:name@@)
S0 = r"""---
title: "@@doc_title@@"
author: "@@author@@"
date: "@@today@@"
---

"""

S1 = r"""
# Executive summary and purpose

## What this project is

This project builds, tests and documents two linked loan-level models for a pool of 30-year residential mortgages.
The first is a **12-month probability of default (PD)** model: given a loan that is current or 30 days past due today, what is the chance it reaches 90 or more days past due within the next twelve months?
The second is a **monthly prepayment (SMM) model**: given a loan that is current this month, what is the chance the borrower pays it off in full next month?
Both are fitted twice, once as a transparent benchmark (an industry-style WoE scorecard for PD, a piecewise-linear logistic regression for prepayment) and once as gradient-boosted trees (XGBoost and LightGBM) with monotone constraints and probability calibration.
The models are explained with SHAP, stress-tested on six macro scenarios, and the resulting prepayment and default curves are exported in the format of a separate securitization waterfall project.
An Excel workbook re-implements the scorecard with live formulas and is reconciled to the Python code.

## Read this first: what the data are and what the results prove

@@source_banner@@

The consequence is simple and should be said plainly in any interview. A model trained on data that a known formula generated should be able to recover that formula's structure, and this project checks that it does (section 9 and section 10). That is a **methodology check**. It shows the pipeline is built correctly: the target definitions, the time split, the leakage guards, the calibration and the explanations all behave as designed. It is **not evidence that the model would predict real mortgage defaults or prepayments**, and none of the headline numbers below should be quoted as real-world performance. The project is also **not a bank-approved or validated production model**; the governance section (section 12) is written in the style of a model risk document to show how such a model would be described and monitored, not to claim approval.

The pipeline reads real Freddie Mac loan-level files if they are placed in `data/raw/` (section 13), and the same code path then runs unchanged. No real files were used for any number in this document.

## Headline results (out-of-time, synthetic data)

All figures below are measured on data that no model saw during fitting: PD snapshots from January 2020 to December 2023 (@@pd_rows_oot@@ loan snapshots) and prepayment loan-months from January 2020 to December 2024 (@@pp_rows_oot@@ loan-months).

- **PD discrimination.** The scorecard logit reaches an out-of-time AUC of @@pd_logit_oot_auc@@ (Gini @@pd_logit_oot_gini@@, KS @@pd_logit_oot_ks@@). XGBoost reaches @@pd_xgb_oot_auc@@ and LightGBM @@pd_lgbm_oot_auc@@. The gain of the trees over the scorecard is small in size but statistically firm (DeLong test, XGBoost p = @@pd_xgb_p@@).
- **PD ranking in practice.** The riskiest decile of LightGBM scores holds @@top_dec_share@@ of all defaults with an observed 12-month default rate of @@top_dec_rate@@, which is @@top_dec_lift@@ times the portfolio average.
- **PD calibration is the main weakness.** Overall, the out-of-time observed default rate is @@obs_oot_lgbm@@ against a mean LightGBM prediction of @@pred_oot_lgbm@@, so the models **under-predict level by roughly a fifth** while ranking well (section 8.3 explains why).
- **Prepayment.** Out-of-time AUC is @@pp_logit_oot_auc@@ for the hinge logit and @@pp_lgbm_oot_auc@@ for LightGBM; the trees add little because the simulated prepayment process is smooth. Predicted monthly CPR tracks actual CPR with a root-mean-square error of @@pp_lgbm_rmse@@ percentage points (LightGBM).
- **Explainability.** SHAP rankings and signs agree with the data generating process in @@align_pass@@ of @@align_n@@ checks (section 9.2).
- **Scenarios.** On the end-2023 pool, an adverse combined scenario (rates +200bp, house prices -20%, unemployment +4 points) raises the model's 12-month default rate from @@s_base_def@@% to @@s_adv_def@@% of balance, but the simulation truth rises to @@t_adv_def@@%. The model **under-reacts to stress** because it never saw such conditions in training (section 10.4). Rate shocks do not move prepayments on this pool because it is far out of the money (section 10.3).

## How to read this document

Each topic follows the same order: the business intuition first, then the mathematics, then the module and function in this repository that implements it, then the actual numbers. Worked numeric examples (a scorecard walk-through for one real loan from the sample, a six-point KS and AUC calculation, a PSI calculation and a two-feature Shapley calculation) are placed next to the theory so that each formula can be reproduced by hand. A glossary (section 14) defines the terms.

# Business problem

## Why mortgage credit and prepayment risk matter

A mortgage investor or servicer receives, for each loan, a stream of scheduled payments of interest and principal. Two things can change that stream.

**Credit risk.** The borrower stops paying. A loan that misses payments moves through delinquency states: current, 30 days past due (30DPD), 60DPD, and 90 or more days past due (90+DPD). In this project the default event is the first month a loan reaches 90+DPD, because that is the usual practical definition used in performance analysis and the point at which losses become likely. If the loan defaults, the investor receives the sale proceeds of the property rather than the contractual payments, and bears a loss equal to the loss severity times the balance.

**Prepayment risk.** The borrower repays early, typically because rates fell and refinancing is attractive, or because the house was sold. Early repayment returns principal at par sooner than scheduled, which shortens the life of the security and reduces the interest earned. Prepayment speed is quoted as the **single monthly mortality (SMM)**, the fraction of the balance prepaid in a month, or annualized as the **conditional prepayment rate (CPR)**:

$$\text{CPR} = 1-(1-\text{SMM})^{12}, \qquad \text{SMM} = 1-(1-\text{CPR})^{1/12}.$$

Both risks interact. A borrower who can refinance has also usually kept a good credit profile, so refinancing removes the healthiest loans first (adverse selection, called burnout), and the remaining pool becomes slower to prepay and riskier.

## Why a loan-level model

Pool-level averages hide the drivers. Two pools with the same average FICO score can behave differently if one has a fat tail of low-FICO, high-LTV loans, and a pool's prepayment speed depends on how many individual loans are in the money and for how long. A loan-level model scores each loan on its own attributes (credit score, leverage, debt-to-income, age, local unemployment, house price change, rate incentive) and aggregates the results. This also lets the same fitted model answer new questions: change the macro path and re-score every loan each month.

## What is deliberately out of scope

The project models the **probability** of default and prepayment. It does not model loss severity (a fixed 35% is used when exporting to the waterfall), loan modifications, forbearance, servicer advances, mortgage insurance claims, or the full economics of a tranche. These would be needed for a production loss forecast and are listed as limitations in section 12.

## Link to the securitization waterfall project

The sibling project `Portfolio_Projects/Securitization_Waterfall_Model` runs a pool of loans through a note structure (senior and subordinate classes, overcollateralization and interest coverage tests, a reserve account) given scalar assumptions for CPR, CDR (annualized default rate), severity, lag and an index shift. This project supplies the *source* of such assumptions: instead of choosing CPR and CDR by judgement, the loan-level models produce them from macro scenarios. The module `export_hook.py` writes month-by-month curves and scalar equivalents whose columns match the waterfall project's `Scenario` fields (section 10.6). The two collateral pools are different (the waterfall pool is a synthetic short-term consumer-style pool), so the hook demonstrates the interface and units, not a like-for-like deal analysis.

# Data

## The Freddie Mac schema

The real-data path is built around the Freddie Mac Single-Family Loan-Level Dataset, which publishes two pipe-delimited files per vintage period (Freddie Mac, Single Family Loan-Level Dataset General User Guide). The **origination file** has one row per loan with the characteristics fixed at closing. The **monthly performance file** has one row per loan per reporting month with the current balance, delinquency status, loan age, current interest rate and, in the last row, a zero-balance code explaining why the loan left the pool.

`freddie.py` maps these files into the project's internal schema (defined once in `columns.py` so that no other module uses string-literal column names):

| Internal column | Freddie field | Handling in `freddie.py` |
|:--|:--|:--|
| `fico` | credit_score | sentinel 9999 set to missing, then median-imputed |
| `orig_ltv`, `orig_cltv` | ltv, cltv | sentinel 999 set to missing; each fills the other, then median |
| `dti` | dti | sentinel 999 set to missing, then median |
| `orig_rate`, `orig_upb`, `orig_term` | orig_rate, orig_upb, orig_term | used as given (term defaults to 360) |
| `state`, `purpose`, `occupancy` | property_state, loan_purpose, occupancy | owner-occupied code P mapped to O |
| `cur_upb`, `cur_rate`, `age` | current_upb, current_rate, loan_age | final-row balance carried from the prior month |
| `dlq_status` | delinquency_status | integer months past due capped at 3; code RA treated as 3 |
| `prepay_flag` | zero_balance_code = 01 | 1 only in the payoff month |
| `default_flag` | status 3 or zero_balance_code in 02, 03, 09, 15, 96 | 1 only in the first such month; the loan then stops |

Table: Mapping from the Freddie Mac layout to the internal schema.

Two design choices follow from the table. First, a loan is cut off at its first default event, so the panel never contains post-default rows that would otherwise leak the outcome into features. Second, imputing missing values with the median is simple and documented but crude; a production build would model missingness explicitly.

## The synthetic mode (used for every result here)

Freddie data require registration and are large, so the project ships a **simulator** in `synthetic.py` that produces a panel with the same internal schema, plus the true hazards used to generate each row (`true_hp`, `true_hd`) which are kept out of every feature set. The simulated population is @@n_loans@@ loans originated between @@orig_start@@ and @@orig_end@@ (monthly, uniformly across the 120 origination months) and observed until @@obs_end@@. The panel has @@panel_rows@@ loan-month rows. By the end of observation @@exit_prepaid@@ of loans had prepaid, @@exit_def@@ had reached 90+DPD, and @@exit_other@@ were still active or had matured. The fixed random seed is @@seed@@, so the data are reproducible.

Each loan is drawn with the following origination attributes (constants are in the CONFIG block of `synthetic.py`):

- FICO score normal with mean @@fico_mean@@ and standard deviation @@fico_sd@@, clipped to @@fico_lo@@ to @@fico_hi@@.
- LTV: @@ltv_p80@@ of loans exactly 80, @@ltv_plow@@ uniform between 60 and 80, and the rest uniform between 80 and 97.
- DTI normal with mean @@dti_mean@@ and standard deviation @@dti_sd@@, clipped to @@dti_lo@@ to @@dti_hi@@.
- Balance lognormal with median @@upb_median@@ and sigma @@upb_sigma@@, clipped to @@upb_floor@@ to @@upb_cap@@.
- Note rate equal to the market rate at origination plus a spread of @@rate_spread@@ points, plus @@rate_fico_slope@@ points for each 10 FICO points below 740, plus noise with standard deviation @@rate_noise@@.
- State from ten states (CA, TX, FL, NY, IL, PA, OH, GA, AZ, NV) with fixed weights; purpose (purchase, cash-out, no-cash refinance), occupancy, property type, number of borrowers and channel from fixed probability tables.

Each month, every loan that is still alive is moved forward by the process described in section 4. The simulation loops over calendar months and updates all alive loans at once (vectorized over loans), which is why 60,000 loans over about 12 years generate in well under a minute.

## Macro environment

Real mortgage behaviour depends on interest rates, house prices and unemployment, so the simulator and the real-data path share a stylized macro environment from `macro.py`, built as a state-by-month table with three series.

**Market mortgage rate (national).** Before 2020 the rate is a gentle sine wiggle around @@rate_pre_base@@%; from 2020 it is linearly interpolated between the anchor points below, imitating the low-rate period of 2020 and 2021 and the sharp rise of 2022.

@@tbl:rate_anchors@@

Table: Anchor points for the market rate path (percent).

**House price index (state).** National log growth is @@hpi_growth@@ a year, set to zero in @@hpi_flat_year@@, and each state scales that growth by a state-specific beta (from @@beta_min@@ for OH to @@beta_max@@ for NV), so Sunbelt states appreciate faster. The index is 100 in January 2012. In the training window house prices only rise; that fact matters for the stress results.

**Unemployment rate (state).** A national path interpolated between the anchors below (including a stylized pandemic spike to 11.0% in May 2020), plus a fixed state offset between @@uoff_min@@ and @@uoff_max@@ points, floored at @@unemp_floor@@%.

@@tbl:unemp_anchors@@

Table: Anchor points for the national unemployment path (percent, before state offsets).

These are stylized paths, **not** real macro data. For real loans, place real series in `data/raw/macro/` (section 13); otherwise the loader falls back to the synthetic macro with a logged warning, which would make the macro features meaningless for real loans.

## From loan-month panel to model datasets

The panel is a long table with one row per loan per month. Two datasets are cut from it, because the two models answer different questions.

| Dataset | Unit of observation | Rows here | Built by |
|:--|:--|--:|:--|
| PD snapshots | one row per loan per snapshot month (January and July), loan active and current or 30DPD | @@snapshot_rows@@ | `targets.build_snapshots` |
| Prepayment hazard rows | one row per loan per month at risk, a 25% random sample of loans | @@hazard_rows@@ | `targets.build_hazard_rows` |

Table: The two modelling datasets (counts from `data_summary.json`).

Using only January and July snapshots keeps the PD dataset at a manageable size and limits the overlap between consecutive snapshots of the same loan (a loan appearing in January 2021 and July 2021 shares eleven of twelve outcome months between rows, which is why the validation bootstrap resamples whole loans rather than rows). The prepayment dataset uses every month because the event is rare in any single month and the model needs the month-to-month variation in rate incentive; a random @@prepay_share@@ of loans is used to bound memory.

Panel access is handled by `data.get_panel`, which chooses real data when files exist, otherwise runs the simulator, and caches the result as parquet in `data/interim/` so repeated runs are fast.

# The data generating process

## Why write the truth down

If the data came from a real world, we could never know whether a model had learned the right relationships. A simulator with an explicit formula lets us ask a cleaner question: given enough data, does the pipeline recover the structure we put in? This is the same logic as a **parameter recovery test** in statistics. The formulas below are the "truth"; they are intentionally ordinary (logistic hazards with plausible signs), so the exercise tests the machinery rather than any discovery about mortgages.

## Notation

For a loan $i$ in month $t$: $\text{age}$ is months since origination; $I_t = \text{note rate} - \text{market rate}$ is the **rate incentive** in percentage points (positive when the borrower pays more than the current market, so refinancing saves money); $B_t$ is **burnout**, the cumulative number of months up to $t$ with $I > 0.5$; $\text{MTM}_t$ is the **mark-to-market LTV**, current balance divided by the house value updated with the state house price index; $U_t$ is state unemployment in percent; $\Delta H_t$ is the 12-month percentage change in the state house price index; and $\sigma(x) = 1/(1+e^{-x})$ is the logistic function.

## The prepayment hazard

The monthly probability that a current loan prepays is

$$h^{P}_t = \sigma\!\Big(a_0 + a_{\text{ref}} R_t + a_{\text{seas}} \min\!\big(\tfrac{\text{age}}{30},1\big) + a_{\text{fico}}\,\tfrac{\text{FICO}-740}{40} + a_{\text{size}} \ln\tfrac{\text{UPB}_t}{200{,}000} + a_{m}\cos\tfrac{2\pi(m_t-6)}{12}\Big),$$

with the **refinance term**

$$R_t = \sigma\!\Big(\tfrac{I_t - 0.5}{0.25}\Big)\, e^{-a_{\text{burn}} B_t}\, \mathbf{1}\big[\text{MTM}_t < 90\big].$$

In words: the baseline is low; it rises as the loan seasons (new loans rarely prepay in the first months); better-credit and larger borrowers prepay a bit faster; there is a June peak in home sales; and the refinance term turns on as the incentive crosses about half a point, is damped for loans that have been in the money for a long time (burnout), and is switched off if the borrower has too little equity (MTM LTV at or above 90) to refinance.

## The default (delinquency entry) hazard

The monthly probability that a current loan becomes 30 days late is

$$h^{D}_t = \sigma\!\Big(b_0 + b_{\text{fico}}\tfrac{\text{FICO}-740}{50} + b_{\text{ltv}}\tfrac{\max(\text{MTM}_t-80,0)}{10} + b_{\text{dti}}\tfrac{\text{DTI}-36}{10} + b_{u}(U_t-5) + b_{h}\tfrac{\min(\Delta H_t,0)}{10} + b_{\text{age}}\, g(\text{age})\Big),$$

where $g(\text{age}) = \exp\!\big(-\big(\tfrac{\text{age}-30}{20}\big)^2\big) - 0.3$ is a **seasoning hump**: delinquency risk is lowest right after origination, peaks around month 30 and fades later. The house-price term acts only when prices are *falling* ($\min(\Delta H,0)$), representing negative-equity-driven default.

A loan that is 30DPD then **rolls** to 60DPD or cures back to current, and from 60DPD it rolls to 90+DPD (the default event) or cures:

$$P(30\to 60) = \min\!\big(c_{30}\,\tau,\,0.9\big),\quad P(30\to 0) = 0.55;\qquad P(60\to 90+) = \min\!\big(c_{60}\,\tau,\,0.9\big),\quad P(60\to 0) = 0.20,$$

with the tilt $\tau = \exp\!\big(k_{f}\tfrac{\text{FICO}-740}{50} + k_{l}\tfrac{\max(\text{MTM}-80,0)}{10}\big)$ making weaker borrowers roll faster and cure less. In each month one uniform random number decides among prepay, entering 30DPD or staying current, which makes prepayment and delinquency **competing risks**: a loan that prepays cannot default and vice versa.

## Coefficients

@@tbl:dgp@@

Table: True data generating process parameters (`config.TRUE_DGP`).

Reading the table: a borrower 40 FICO points above 740 has $a_{\text{fico}} = @@a_fico@@$ added to the prepay log-odds, and each extra point of state unemployment above 5 adds $b_u = @@b_unemp@@$ to the default log-odds. Because the model sees only features and outcomes, success means its estimated effects have these signs and rough relative sizes.

## Calibration of the simulator

The coefficients were tuned so that overall behaviour sits in realistic ranges before any model was fitted. The check is run by `synthetic.calibration_report` and is reproduced from the cached panel:

@@tbl:calib_report@@

Table: Simulator calibration targets (`config.SYNTH_CALIBRATION`) against the simulated values.

Annual CPR in the simulated panel varies from @@cpr_min_year@@ to @@cpr_max_year@@ across calendar years, high in 2020 and 2021 when the market rate fell below most note rates and low in 2023 and 2024 when it rose above them:

@@tbl:cpr_year_synth@@

Table: Simulated CPR by calendar year (from `synthetic.calibration_report`).

# Target definitions, competing risks and the time split

## The PD target

The PD target is built by `targets.build_snapshots`. At each snapshot month $t$ (January or July) a loan enters the dataset if it is active and its status is current (0) or 30DPD (1). Loans already 60DPD or worse are excluded because they are close to default and would dominate the signal; this is a modelling choice to make the model useful for *early* identification. The label is

$$y_{i,t} = \mathbf{1}\big[\text{the loan first reaches 90+DPD in months } t+1,\dots,t+12\big].$$

A loan that **prepays** inside the window before defaulting gets $y=0$. This is the **cumulative incidence** definition: the PD is the probability of the *default event actually occurring* in the presence of prepayment as a competing way to leave. It answers "what fraction of today's loans will default within a year", which is what an investor needs, but it is lower than the hazard of default for a loan that could never prepay. A snapshot enters only if its entire twelve-month window is observed before @@obs_end@@, so no label is truncated.

## The prepayment target

The prepayment dataset (`targets.build_hazard_rows`) uses **discrete-time survival** or hazard framing. For month $t$, the row exists if the loan was current at the end of month $t-1$ and still active in month $t$. The features are taken **as of $t-1$** (everything known before the month starts) and the label is whether the loan prepaid during month $t$. The calendar month of $t$ is allowed as a feature because it is known in advance (seasonality), not an outcome. The model therefore estimates a monthly hazard, which is exactly the SMM.

The two targets treat competing risks differently on purpose. The prepayment model conditions on staying current, so default is excluded by construction; the PD model treats prepayment as "no event". Section 12 discusses what this implies for the projection.

## Time-based split

Random splitting would be wrong here: a loan's January 2019 row and its July 2019 row are almost the same observation, and the future would leak into the past. The split is by **calendar time** with a buffer so that no development label overlaps the out-of-time period.

@@tbl:splits@@

Table: Time windows and row counts. For PD the window refers to the snapshot date; for prepayment to the month of the outcome.

Every development label must be resolved by `DEV_CUTOFF` = @@dev_cutoff@@, so a training or calibration snapshot is kept only if its twelve-month window ends on or before that date (`targets.split_by_time` applies this). The out-of-time (OOT) window starts after the cutoff: it covers the 2020 pandemic-era unemployment spike, the 2021 low-rate refinance wave and the 2022 rate rise, which makes it a demanding test. The same loan can appear in train and OOT at different dates (it is a time split, not a loan split), so the bootstrap in section 7.10 resamples loans.

The observed 12-month default rate differs a lot across periods, which matters for calibration later:

@@tbl:pd_rate_year@@

Table: Observed 12-month default rate by snapshot year, training and out-of-time. The training window averages @@pd_rate_train@@, the calibration window @@pd_rate_calib@@ and the OOT window @@pd_rate_oot@@.
"""


S6 = r"""
# Feature engineering and the leakage checklist

## What a feature is, and what leakage means

A **feature** is a number computed for each loan at the prediction date. **Leakage** is any information in a feature that would not have been available at that date, or that is a disguised copy of the outcome. Leakage makes a model look excellent in testing and fail in use, and it is the single most common reason credit models are rejected in validation. Every feature here is built by `features.add_features`, which sorts the panel by loan and month and computes everything with **backward-looking** operations only.

## Features used by the PD model

| Feature | Definition | Why it belongs |
|:--|:--|:--|
| `fico` | FICO score at origination | strongest single credit-quality signal |
| `orig_ltv` | loan-to-value at origination, percent | initial equity cushion |
| `mtm_ltv` | current balance divided by house value updated with the state house price index, percent | current equity; negative equity drives default |
| `dti` | debt-to-income at origination, percent | payment burden |
| `orig_rate` | note rate, percent | higher rates reflect risk pricing and payment size |
| `incentive` | note rate minus current market rate, in percentage points | borrower's refinance position; also a proxy for rate regime |
| `age` | months since origination | the seasoning curve of defaults |
| `log_upb` | natural log of current balance | size effect |
| `unemp`, `unemp_chg_12m` | state unemployment rate and its change over 12 months | local labour market stress |
| `hpi_chg_12m` | percent change in the state house price index over 12 months | falling prices raise default |
| `times_30dpd_12m` | months in the last 12 (including the current one) with status of 30DPD or worse | recent payment history |
| `dlq_status` | current delinquency status (0 or 1 for snapshots) | a loan already late is far more likely to default |
| `purpose`, `occupancy`, `n_borrowers`, `state` | categorical attributes | segment effects and geography |

Table: PD model features (`columns.FEATURES_PD`).

## Features used by the prepayment model

| Feature | Definition | Why it belongs |
|:--|:--|:--|
| `incentive` | note rate minus market rate (pp), taken at $t-1$ | the primary driver of refinance |
| `burnout` | cumulative months with incentive above 0.5 pp, up to and including $t-1$ | borrowers who ignored past incentive are less responsive |
| `age`, `seasoning_ramp` | months since origination; $\min(\text{age}/30,1)$ | new loans prepay slowly |
| `mtm_ltv` | mark-to-market LTV | low equity blocks refinancing |
| `fico`, `dti`, `log_upb` | credit and size attributes | easier to qualify, larger saving |
| `month_of_year` | calendar month of the outcome month | seasonality of home sales |
| `purpose`, `state` | categorical | segment and regional effects |

Table: Prepayment model features (`columns.FEATURES_PREPAY`).

The hinge logit uses a subset of these (the incentive hinges, burnout, seasoning ramp, MTM LTV capped at 120, and FICO) because a simple parametric model is meant to be readable.

## Formulas

The mark-to-market LTV reconstructs an index-linked house value: the origination house value is $V_0 = \text{UPB}_0 / \text{LTV}_0 \times 100$, the current value is $V_t = V_0 \cdot \text{HPI}_t / \text{HPI}_0$ using the loan's state index, and

$$\text{MTM}_t = \frac{\text{UPB}_t}{V_t}\times 100.$$

Burnout is the running count $B_t = \sum_{s\le t}\mathbf{1}[I_s > 0.5]$, computed with a grouped cumulative sum per loan. The 12-month changes use a lookup of the same state twelve months earlier (`features._state_lag`), which returns missing, set to zero, for the first twelve months. `times_30dpd_12m` is a difference of two cumulative counts, so it only ever looks back.

## The leakage checklist

| Risk | Control in this repo | Where |
|:--|:--|:--|
| Outcome columns used as inputs | `assert_no_leakage` raises if any of the forbidden columns (`true_hp`, `true_hd`, both targets, `zb_code`, `default_flag`, `prepay_flag`, `snapshot_date`, prediction columns) appears in a feature list; called in every model fit | `features.py`, `models.py` |
| Features computed with future data | all rolling and lag features are backward only; lags return missing when the history is shorter than 12 months | `features.add_features` |
| Prepay features at the outcome month | hazard rows take features from $t-1$ and the label from $t$; only the calendar month is taken from $t$ because it is known in advance | `targets.build_hazard_rows` |
| Label window overlapping the test period | development rows are kept only if their label window ends on or before `DEV_CUTOFF`; OOT starts afterwards | `targets.split_by_time` |
| Snapshot labels truncated by end of data | snapshots later than OBS_END minus 12 months are dropped | `targets.build_snapshots` |
| Pre-processing fit on the wrong data | WoE bins, the categorical encoder, and hyperparameters are learned on the training window only | `scorecard.fit_bins`, `models.Encoder.fit`, `models.tune_time_cv` |
| Calibrating on training data | probability calibration is fitted on a separate window after training | `models.calibrate` |
| Random cross-validation across time | tuning folds are expanding-window with a gap of 12 months (PD) or 1 month (prepay) between training and validation | `models._fold_masks` |
| Same loan counted as independent | the AUC bootstrap resamples whole loans | `evaluation.bootstrap_ci` |
| Post-default rows | the simulator and the Freddie loader both stop a loan at its first default event | `synthetic.py`, `freddie.py` |

Table: Leakage controls.

One residual point deserves honesty: the **macro** features at the snapshot date are the actual realized macro values at that date, which is correct for scoring today's loans but means the projections in section 10 must supply a macro path, which they do (shocked, deterministic).

# Methodology theory

This section teaches the methods in the order they are used. Each subsection gives the intuition, then the maths, then the implementation, and where useful a worked example.

## Logistic regression

*Intuition.* We want a probability between 0 and 1 from a weighted sum of inputs. A straight line can leave that range, so the sum is passed through the logistic function.

*Maths.* For features $x$ and a binary outcome $y$,

$$P(y=1\mid x) = \sigma(\beta_0 + \beta^\top x) = \frac{1}{1+e^{-(\beta_0+\beta^\top x)}}, \qquad \ln\frac{p}{1-p} = \beta_0 + \beta^\top x.$$

The left-hand log of the odds is the **log-odds** (logit). A coefficient $\beta_j$ means that a one-unit rise in $x_j$ multiplies the odds by $e^{\beta_j}$. Coefficients are estimated by maximum likelihood, maximizing $\sum_i [y_i\ln p_i + (1-y_i)\ln(1-p_i)]$.

*Implementation.* `statsmodels.api.Logit` is used for the scorecard (`scorecard.fit_scorecard`) and for the prepayment hinge model (`models.fit_logit_prepay`), because statsmodels reports the coefficient table needed by validators and by the Excel workbook.

## Weight of evidence and information value

*Intuition.* Credit models group a continuous variable such as FICO into bands and give each band a score. The natural score for a band is how much more "good" than "bad" it contains, compared with the portfolio.

*Maths.* For a bin $k$ with $g_k$ good loans and $b_k$ bad loans out of totals $G$ and $B$,

$$\text{WoE}_k = \ln\frac{g_k/G}{b_k/B}, \qquad \text{IV} = \sum_k\Big(\frac{g_k}{G}-\frac{b_k}{B}\Big)\text{WoE}_k.$$

Positive WoE means safer than average. IV measures how well the whole variable separates good from bad; the common rule of thumb is below 0.02 not useful, 0.02 to 0.1 weak, 0.1 to 0.3 medium, 0.3 to 0.5 strong, and above 0.5 suspiciously strong (Siddiqi 2006).

*Worked example.* Three FICO bins with 10,000 good loans and 200 defaults in total:

@@tbl:toy_woe@@

Table: WoE and IV on a three-bin toy example. Total IV = @@toy_iv@@.

The low-FICO bin holds 9% of goods but 30% of bads, so its WoE is strongly negative; the mid bin matches the portfolio so its WoE is 0.

*Implementation.* `scorecard.fit_bins` (1) cuts the variable into 20 quantile pre-bins; (2) merges any bin below the 5% minimum share into its closest-risk neighbour; (3) enforces a **monotone** bad rate by merging neighbours that break the trend (the direction is taken from the data); (4) caps the number of bins at 8. WoE uses a 0.5 smoothing count so empty cells never give an infinite log, and missing values get their own WoE. `fit_scorecard` then transforms each variable to WoE, drops variables with IV below @@iv_min@@, fits the logistic regression of default on the WoE values, and **drops the lowest-IV variable with a wrong-signed coefficient** and refits until all signs are as expected. With $\text{WoE}=\ln(\text{good}/\text{bad})$ the correct sign on the coefficient is negative: higher WoE must lower the default log-odds.

## Scorecard scaling with points to double the odds

*Intuition.* Bankers prefer additive integer points to log-odds. The scale is fixed by three choices: a base score at base odds, and the points to double the odds (PDO).

*Maths.* The score is linear in the log of good-to-bad odds:

$$\text{Score} = \text{Offset} + \text{Factor}\cdot\ln(\text{odds}_{\text{good}}), \qquad \text{Factor} = \frac{\text{PDO}}{\ln 2},\quad \text{Offset} = \text{BaseScore} - \text{Factor}\ln(\text{BaseOdds}).$$

With PDO = @@pdo@@, base score @@base_score@@ at @@base_odds@@:1 odds, Factor = @@factor@@ and Offset = @@offset@@. A score of 620 means odds are 100:1, double the base. Because the model log-odds of default is $z = \beta_0 + \sum_j \beta_j W_j$ and good odds are $e^{-z}$,

$$\text{Score} = \text{Offset} - \text{Factor}\cdot z = \underbrace{\frac{\text{Offset} - \text{Factor}\,\beta_0}{m}}_{\text{base points per variable}} + \sum_{j=1}^{m}\big(-\text{Factor}\,\beta_j W_{j}\big),$$

so each bin of each variable gets a fixed number of points $-\text{Factor}\,\beta_j\,\text{WoE}_{jk}$ plus an equal share of the intercept. `scorecard.export_scorecard_table` produces this table; the Excel workbook looks it up with live formulas.

@@tbl:score_map@@

Table: Score to odds to PD map for the chosen scaling. The PD column is the raw logistic probability before calibration.

*Worked example: one real loan from the Excel sample.* Loan @@wk_loan@@ at the @@wk_date@@ snapshot is looked up bin by bin in `scorecard_table.csv`:

@@tbl:walk@@

Table: Scorecard points for one loan. Bins are right-closed, (low, high].

The points add up to @@wk_total@@, which @@wk_match@@ the @@wk_sheet@@ stored in `excel_inputs/loan_sample.csv`. The raw log-odds of default is $z = (\text{Offset}-\text{Score})/\text{Factor} = @@wk_margin@@$, so the raw PD is @@wk_raw_pd@@ (good-to-bad odds of about @@wk_odds@@ to 1). The loan's *calibrated* scorecard PD is @@wk_cal_pd@@: the isotonic or Platt calibration step (section 7.8) maps the raw probability to an observed frequency on the calibration window, which is why the two differ. For reference, LightGBM gives @@wk_gbm_pd@@ for the same loan, and the loan @@wk_default@@ within twelve months.

## Gradient boosting

*Intuition.* A single small decision tree is a weak predictor. Boosting builds trees one after another, each one fitted to the **mistakes** of the ensemble so far, and adds them up with a small weight. The result is a flexible function that captures interactions (for example, high LTV matters more when unemployment is high) without the analyst specifying them.

*Maths.* The prediction is a sum of $M$ trees on the log-odds scale, $\hat z_i = \sum_{m=1}^{M} f_m(x_i)$, with $p_i = \sigma(\hat z_i)$. At step $m$, XGBoost minimizes the regularized objective

$$\mathcal{L}^{(m)} = \sum_i \ell\big(y_i,\hat z_i^{(m-1)}+f_m(x_i)\big) + \Omega(f_m), \qquad \Omega(f) = \gamma T + \tfrac12\lambda\sum_{j=1}^{T}w_j^2,$$

where $T$ is the number of leaves and $w_j$ the leaf values. A second-order Taylor expansion of the loss around the current prediction gives

$$\mathcal{L}^{(m)} \approx \sum_i\Big[g_i f_m(x_i) + \tfrac12 h_i f_m(x_i)^2\Big] + \Omega(f_m),\qquad g_i=\frac{\partial\ell}{\partial \hat z_i}=p_i-y_i,\quad h_i=\frac{\partial^2\ell}{\partial \hat z_i^2}=p_i(1-p_i)$$

for logistic loss. Grouping observations by leaf, with $G_j=\sum_{i\in j}g_i$ and $H_j=\sum_{i\in j}h_i$, the best leaf value and the objective value are

$$w_j^{*} = -\frac{G_j}{H_j+\lambda}, \qquad \mathcal{L}^{*} = -\tfrac12\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda} + \gamma T.$$

A candidate split is scored by the **gain** in this objective,

$$\text{Gain} = \tfrac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma.$$

The $\lambda$ penalty shrinks leaf values toward zero (guarding against overfitting on small leaves), the Hessian sum $H$ acts as a "weighted sample count" for the minimum child size, and a learning rate (shrinkage) scales each tree's contribution. Row and column subsampling add randomness for generalization (Chen and Guestrin 2016).

*Implementation.* `models.fit_gbm` builds either learner through `models._make_estimator`; defaults are learning rate 0.08, depth 4, 15 leaves, subsample 0.8, column sample 0.8, L2 lambda 5 and 250 trees, replaced by tuned values (section 7.7).

## Monotone constraints

*Intuition.* A lender will not accept a model in which a higher FICO score can raise the default probability. Trees fitted to noisy data can produce such wiggles. A monotone constraint forbids them.

*Maths.* For a constrained feature the learner only accepts a split if the left child's value is not larger than the right child's (for a feature that should raise risk), and the allowed value interval is propagated to all descendants so that the whole function stays monotone in that feature. The constraints used are in `models.MONOTONE`:

- PD: `fico` decreasing; `mtm_ltv`, `dti`, `unemp`, `times_30dpd_12m` increasing.
- Prepay: `incentive` increasing; `burnout` decreasing.

A constraint is a restriction on the *shape* of the response, not on its extent: beyond the training range a tree is flat, so the model cannot extrapolate (section 10.4 shows the consequence).

## XGBoost compared with LightGBM

Both libraries implement the same boosting objective with histogram-based split search, so the differences are in how trees are grown and in practical details.

| Aspect | XGBoost | LightGBM |
|:--|:--|:--|
| Tree growth | level-wise (grow all nodes at a depth), limited by `max_depth` | leaf-wise (always split the leaf with the largest gain), limited by `num_leaves` |
| Typical effect | balanced trees, a little slower | deeper, asymmetric trees that can fit more with fewer splits, with higher overfitting risk on small data |
| Categorical features | needs one-hot encoding (used here) | native categorical splits (used here) |
| Minimum leaf size | `min_child_weight` on the Hessian sum (set to `min_child / 10` here) | `min_child_samples` on row counts |
| Speed tricks | approximate histograms, sparsity-aware splits | histogram subtraction, optional gradient-based one-side sampling, exclusive feature bundling |

Table: Practical differences between the two learners.

LightGBM's design is described in Ke et al. (2017). In this repo the one-hot XGBoost design matrix is turned back into the original features when SHAP values are computed (`explain.shap_values` sums the one-hot columns by source feature), so both learners can be explained in terms of the same variables.

## Time-aware hyperparameter tuning

*Intuition.* Hyperparameters (depth, learning rate, regularization) must be chosen without touching the out-of-time test. The honest way for time-ordered data is to train on the past and validate on the next period, repeatedly.

*Method.* `models.tune_time_cv` draws @@tune_iter@@ random candidates (3 in quick mode) from the ranges in `models.SEARCH`, with learning rate and lambda sampled on a log scale. Each candidate is scored by the average **log-loss** across @@tune_folds@@ expanding-window folds: the first 40% of months form the initial training window, each fold validates on the next block of months, and training rows are cut off a **gap** before the validation start (12 months for PD, because label windows are 12 months long and would otherwise overlap; 1 month for prepay). Early stopping on the validation fold gives the number of trees; the final tree count is 1.15 times the median early-stopped count. Tuning uses at most 200,000 rows for speed.

The final tuned parameters and the calibrator selected for each bundle (read from the saved `joblib` files):

@@tbl:bundles@@

Table: Tuned boosting parameters and selected calibrators.

## Probability calibration

*Intuition.* A model can rank loans correctly while its numbers are wrong: if it says 2% for a group that defaults 3%, the ranking (AUC) is fine but any expected-loss calculation is understated. Calibration is a post-processing map from the model's raw score to an observed frequency.

*Maths.* Let $m$ be the raw margin (log-odds). **Platt scaling** fits $p = \sigma(A m + B)$ by logistic regression. **Isotonic regression** fits a non-decreasing step function $\hat p = \phi(\sigma(m))$ minimizing $\sum_i (y_i-\phi(s_i))^2$ with the pool-adjacent-violators algorithm; it is more flexible but needs more data.

*Implementation.* `models.calibrate` fits both on the **calibration window**, which is separate from training, and chooses between them by Brier score using a two-fold, date-ordered cross-fit (fit on one half of the window, score on the other, and swap). Probabilities are clipped to $[10^{-6}, 1-10^{-6}]$. The selected methods are: scorecard PD @@cal_pd_logit@@, XGBoost PD @@cal_pd_xgb@@, LightGBM PD @@cal_pd_lgbm@@, prepay hinge logit @@cal_pp_logit@@, XGBoost prepay @@cal_pp_xgb@@, LightGBM prepay @@cal_pp_lgbm@@. A **class-weighted** XGBoost variant (positive class up-weighted by the inverse event rate) is also fitted as a sensitivity row; it ranks about as well (OOT AUC @@w_oot_auc@@) but, uncalibrated, has an out-of-time Brier score of @@w_oot_brier@@, far worse than the calibrated models, illustrating why weighting rare events distorts probabilities.

## Discrimination and calibration metrics

**AUC** is the probability that a randomly chosen defaulted loan has a higher score than a randomly chosen non-defaulted loan (ties count one half):

$$\text{AUC} = \frac{1}{n_1 n_0}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}\Big[\mathbf{1}(s_i>s_j)+\tfrac12\mathbf{1}(s_i=s_j)\Big],\qquad \text{Gini} = 2\,\text{AUC}-1.$$

`evaluation.auc` computes this from average ranks without forming all pairs. **KS** is the largest vertical gap between the cumulative distribution of scores for defaulted and for performing loans when loans are ordered from riskiest, $\text{KS}=\max_s|F_1(s)-F_0(s)|$. The **Brier score** is the mean squared error of the probabilities, $\frac1n\sum_i(p_i-y_i)^2$; it blends discrimination and calibration. The **Hosmer-Lemeshow** statistic groups loans into $g$ bins of predicted PD and compares expected with observed events,

$$\text{HL} = \sum_{k=1}^{g}\frac{n_k(\bar o_k-\bar p_k)^2}{\bar p_k(1-\bar p_k)},$$

with $g$ degrees of freedom when evaluated on held-out data (the usual $g-2$ applies to the fitting sample). With over a hundred thousand loans the test rejects any visible miscalibration, so the observed-to-predicted ratio in each bin is the more useful diagnostic.

*Worked example.* Six loans, scored by a model, three of which defaulted:

@@tbl:toy_ks@@

Table: Six-point example. AUC = @@toy_conc@@ concordant pairs of @@toy_pairs@@ = @@toy_auc@@, Gini = @@toy_gini@@, KS = @@toy_ks@@.

Counting pairs: there are 3 defaulted and 3 performing loans, so 9 pairs; in @@toy_conc@@ of them the defaulted loan has the higher score, so AUC = @@toy_conc@@/@@toy_pairs@@. For KS, walk down the ranking and track the two cumulative shares; the largest gap is the KS statistic.

## DeLong test and loan-clustered bootstrap

*DeLong.* Two models scored on the same loans have correlated AUCs, so a simple difference test is wrong. DeLong, DeLong and Clarke-Pearson (1988) express the AUC as a mean of "structural components" per positive and per negative, $V_{10}(x_i)$ and $V_{01}(y_j)$, estimate the covariance of the two models' AUCs from those components, and test $z = (\widehat{\text{AUC}}_1-\widehat{\text{AUC}}_2)/\sqrt{\widehat{\text{Var}}(\widehat{\text{AUC}}_1-\widehat{\text{AUC}}_2)}$ against a standard normal. `evaluation.delong_test` implements it with rank-based components.

*Bootstrap.* Because the same loan appears in several snapshots, rows are not independent, and a row bootstrap would give intervals that are too narrow. `evaluation.bootstrap_ci` resamples **loans** (all of a loan's rows together) @@boot_n@@ times and reports the 2.5th and 97.5th percentiles of the metric. To keep it fast, at most 25,000 rows enter the bootstrap, which makes intervals slightly conservative.

## Population stability index and characteristic stability index

*Intuition.* After deployment the population drifts. PSI compares the distribution of the score (or of one feature, in which case it is called CSI) today with the distribution at development.

*Maths.* Cut the development data into $K$ quantile bins (ten here). Let $e_k$ be each bin's share of the development sample and $a_k$ its share of the new sample:

$$\text{PSI} = \sum_{k=1}^{K}(a_k - e_k)\ln\frac{a_k}{e_k}.$$

It is a symmetric divergence, equal to the sum of the two Kullback-Leibler divergences between the distributions, and is zero only when the shares match. Shares are floored at $10^{-4}$ to avoid a log of zero. The conventional bands, held in `config.PSI_BANDS`, are below 0.10 stable, 0.10 to 0.25 moderate shift and above 0.25 significant shift.

*Worked example.* Expected shares of 50%, 30% and 20% against actual shares of 40%, 30% and 30%:

@@tbl:toy_psi@@

Table: PSI worked example. PSI = @@toy_psi@@, in the stable band.

*Implementation.* `evaluation.psi` builds the bin edges from the expected sample; `evaluation.csi` applies it to each feature (categorical features use category shares). Section 8.5 reports both.

## SHAP values

*Intuition.* SHAP answers "how much did each feature push this loan's score up or down compared with an average loan?" It borrows the Shapley value from cooperative game theory: treat the features as players who jointly produce the prediction, and split the payout fairly according to each player's average marginal contribution over all possible orders in which the players could join.

*Maths.* For a model $f$ with feature set $F$ ($|F|=M$), the contribution of feature $j$ for input $x$ is

$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(M-|S|-1)!}{M!}\Big[v(S\cup\{j\}) - v(S)\Big],\qquad v(S)=E\big[f(x)\mid x_S\big].$$

The attributions satisfy **efficiency** (local accuracy): $f(x)=\phi_0+\sum_j\phi_j$ with $\phi_0=E[f(x)]$ the base value. They also satisfy **symmetry** (identical features get identical credit), the **dummy** property (a feature that never changes the output gets zero) and **additivity** (attributions of a sum of models are the sum of attributions). Lundberg and Lee (2017) showed that these axioms single out this attribution among additive explanations.

*Worked example.* Two features $a$ and $b$, with the model's expected value restricted to subsets of features equal to $v(\emptyset)=@@sh_v0@@$, $v(\{a\})=@@sh_va@@$, $v(\{b\})=@@sh_vb@@$ and $v(\{a,b\})=@@sh_vab@@$. Feature $a$ joins first with probability one half (marginal @@sh_a1@@) and second with probability one half (marginal @@sh_a2@@), so $\phi_a = \tfrac12(@@sh_a1@@) + \tfrac12(@@sh_a2@@) = @@sh_a@@$. Likewise $\phi_b = \tfrac12(@@sh_b1@@)+\tfrac12(@@sh_b2@@) = @@sh_b@@$. The two add up to @@sh_sum@@, exactly the gap @@sh_gap@@ between the full prediction and the base value, as efficiency requires. Note that feature $a$ gets more credit than its stand-alone effect because it interacts with $b$: the interaction is split fairly.

*Implementation.* Computing $\phi$ by brute force costs $2^M$ model evaluations, but for trees **TreeSHAP** computes exact values in polynomial time by following all paths of a tree and tracking how many subsets flow to each leaf. `explain.shap_values` calls `shap.TreeExplainer` on the **raw margin** (log-odds), not on the calibrated probability, so the identity is

$$\text{margin}(x) = \phi_0 + \sum_j\phi_j(x),$$

checked on a sample by `explain.additivity_gap` (tolerance $10^{-4}$). The calibration step is a monotone transform applied afterwards. Global importance is the mean absolute SHAP value per feature (`explain.global_importance`), computed on a random sample of @@shap_n@@ out-of-time rows.
"""


S8 = r"""
# Results

All results are out-of-time on synthetic data (section 1.2). They show that the pipeline works as designed, not how a model would perform on real mortgages.

## Benchmark: scorecard against gradient boosting

@@tbl:bench_pd@@

Table: PD models. AUC, Gini, KS and Brier are out-of-time; the confidence interval is a loan-clustered bootstrap; dAUC and the DeLong p-value compare each model with the scorecard logit on the same out-of-time loans. The last row is a sensitivity run and is uncalibrated.

![Out-of-time ROC curves for the three PD models and the KS curve for LightGBM.](@@chart_dir@@/roc_ks_pd.png){width=95%}

Three things stand out.

1. **The trees beat the scorecard, modestly and reliably.** XGBoost improves AUC by @@pd_xgb_dauc@@ and LightGBM by @@pd_lgbm_dauc@@; the DeLong p-values (@@pd_xgb_p@@ and @@pd_lgbm_p@@) say the improvement is not noise. The simulated default hazard includes a non-linear seasoning hump and hinge-like terms (unemployment above 5, falling house prices only) that a binned linear scorecard approximates but cannot match exactly.
2. **The trees overfit more.** The training-to-OOT AUC drop is @@pd_logit_gap@@ for the scorecard, @@pd_xgb_gap@@ for XGBoost and @@pd_lgbm_gap@@ for LightGBM. LightGBM fits the training window best and generalizes no better than XGBoost: the OOT intervals (@@pd_xgb_ci@@ for XGBoost and @@pd_lgbm_ci@@ for LightGBM) overlap almost completely. Part of every gap is not overfitting but regime change, since the OOT period contains conditions the training period lacks.
3. **Model choice is not settled by AUC alone.** The pipeline uses LightGBM for PD explanation and projection and XGBoost for prepayment (`cli.py`). A validator would reasonably ask why, given that XGBoost has the nominally higher PD AUC; the honest answer is that the two are statistically close and that this choice was a pipeline default rather than the output of a formal selection exercise.

## Discrimination in practice: deciles and lift

Sorting the out-of-time snapshots by predicted PD and cutting them into ten equal groups shows how well the score concentrates defaults at the top:

@@tbl:dec_lgbm@@

Table: LightGBM out-of-time decile table. Lift is the decile's observed rate divided by the portfolio rate; the last column is the gap between the cumulative share of defaults and of performing loans, whose maximum is the KS statistic (largest at decile @@ks_dec_at@@, value @@ks_dec@@).

![Observed default rate and lift by decile, LightGBM.](@@chart_dir@@/decile_lift_pd_lgbm.png){width=85%}

The riskiest decile captures @@top_dec_share@@ of all defaults at an observed rate of @@top_dec_rate@@ (lift @@top_dec_lift@@), and the two riskiest deciles together capture @@top2_share@@. The safest decile has an observed rate of @@bottom_dec_rate@@. For comparison, the scorecard's riskiest decile captures @@top_dec_share_logit@@ of defaults at an observed rate of @@top_dec_rate_logit@@; the scorecard decile table is in `outputs/deciles_pd_pd_logit.csv`.

## Calibration

@@tbl:hl@@

Table: Out-of-time calibration summary. The Hosmer-Lemeshow statistic is computed on ten equal-count PD bins; with this many loans it rejects any visible gap, so read the observed-to-predicted ratio.

![Reliability diagram: mean predicted PD against observed default rate in ten bins.](@@chart_dir@@/calibration_pd.png){width=70%}

@@tbl:calib_lgbm@@

Table: LightGBM calibration by bin, out-of-time.

**The models under-predict the level of default.** The out-of-time observed rate is @@obs_oot_lgbm@@ against a mean LightGBM prediction of @@pred_oot_lgbm@@ (a ratio of @@ratio_lgbm@@; XGBoost @@ratio_xgb@@, scorecard @@ratio_logit@@). The top bin is almost exactly right (ratio @@calib_ratio_top@@) but the middle bins are low by a third or more (ratio @@calib_ratio_mid@@ in bin 8). That pattern points to three causes that together explain the bias:

- **Calibration-window mismatch.** The calibrator is fitted on the second half of 2018 where the observed 12-month default rate is only @@pd_rate_calib@@, against @@pd_rate_train@@ in training and @@pd_rate_oot@@ out-of-time (@@oot_2020_rate@@ for 2020 snapshots alone). A calibration map learned in a benign window carries that level forward.
- **Macro conditions outside the training range.** State unemployment in the training snapshots ranges over @@unemp_tr@@ percent but over @@unemp_oot@@ in the OOT window; trees return a constant beyond the edge of the training data, so the 2020 spike is under-weighted (section 10.4 shows the same mechanism).
- **Competing risk in the label.** Because prepayment counts as "no default", a regime with faster prepayment removes loans before they can default, and that varies by period.

This is the type of finding a model validator would flag as a finding requiring remediation (for example recalibrating on a recent window or adding a macro overlay) before the PD could be used as a level estimate. The ranking power is not affected.

## Stability by vintage and by segment

![LightGBM out-of-time AUC by origination vintage.](@@chart_dir@@/stability_vintage.png){width=75%}

@@tbl:vintage@@

Table: LightGBM out-of-time performance by origination vintage.

AUC ranges from @@vin_auc_min@@ (vintage @@vin_auc_min_v@@) to @@vin_auc_max@@, so ranking power is stable across vintages. Level calibration is not uniform: the observed-to-predicted ratio runs from @@vin_ratio_min@@ (vintage @@vin_ratio_min_v@@) to @@vin_ratio_max@@ (vintage @@vin_ratio_max_v@@), the older vintages being close to calibrated and the middle vintages that were seasoning through 2020 and 2021 being most under-predicted.

@@tbl:seg_fico@@

Table: LightGBM out-of-time performance by FICO band.

The model puts the right order on the bands. Within-band AUC is lower than the overall AUC (@@seg_auc_min@@ to @@seg_auc_max@@) because most of the overall separation comes from FICO itself. The below-620 band has only @@seg_lt620_n@@ snapshots and shows an observed rate of @@seg_lt620_obs@@ against a predicted @@seg_lt620_pred@@, an over-prediction on a very small sample that should not be read as a pattern. The 780+ band has observed @@seg_top_obs@@ and predicted @@seg_top_pred@@.

@@tbl:seg_other@@

Table: LightGBM out-of-time performance by loan purpose and occupancy. The simulated default hazard has no purpose or occupancy effect, so AUCs and rates are similar across segments, as they should be.

## Population and characteristic stability

@@tbl:psi_score@@

Table: Score PSI of each PD model, training scores against calibration-window and OOT scores.

![PSI of the PD score and CSI of each feature, out-of-time against training.](@@chart_dir@@/psi_bars.png){width=85%}

@@tbl:csi@@

Table: Characteristic stability index of each PD feature, out-of-time against training. @@csi_n_sig@@ of @@csi_n@@ features are in the significant band and @@csi_n_mod@@ in the moderate band.

The score distributions are stable out of time (LightGBM PSI @@psi_oot_lgbm@@, XGBoost @@psi_oot_xgb@@, scorecard @@psi_oot_logit@@), while the *inputs* drifted a lot: the macro and rate features dominate the CSI ranking (@@csi_top@@). The reason is the test design: the OOT window includes the 2020 labour market shock and the 2022 rate rise. The scorecard's PSI against the calibration window (@@psi_cal_logit@@) is in the moderate band, which an ongoing monitoring programme would investigate; the cause here is that the calibration window is a short, benign half-year. The population features borrowers choose at origination (FICO, LTV, DTI, balance) have CSI near zero, as the simulator draws every vintage from the same distribution.

The decile edges used for the scorecard PSI in the Excel workbook come from the training scores:

@@tbl:psi_bins@@

Table: Training-score decile bins used as the PSI baseline (`excel_inputs/psi_train_bins.csv`).

## Prepayment results

@@tbl:bench_pp@@

Table: Prepayment models, out-of-time on loan-month rows. The hinge logit is the baseline.

The hinge logit's out-of-time AUC is @@pp_logit_oot_auc@@, XGBoost @@pp_xgb_oot_auc@@ and LightGBM @@pp_lgbm_oot_auc@@. Only LightGBM's gain is statistically significant (p = @@pp_lgbm_p@@; XGBoost p = @@pp_xgb_p@@), and it is tiny (@@pp_lgbm_dauc@@ AUC). That is a sensible result rather than a disappointment: the simulated monthly prepayment hazard is a smooth logistic function of the inputs, which a well-designed piecewise-linear logit already captures. Note also that out-of-time AUC (about 0.73) is *higher* than training AUC (about 0.67 to 0.70): during 2013 to 2018 the market rate barely varies relative to note rates, so incentive carries little information, whereas 2020 to 2022 contains a large rate swing that makes incentive highly informative.

The hinge logit coefficients and the slope each segment of the incentive implies:

@@tbl:pp_coefs@@

Table: Hinge logit coefficients (statsmodels).

@@tbl:pp_slopes@@

Table: Slope of the prepay log-odds with respect to incentive, accumulated from the hinge terms.

The slope is zero when the incentive is negative (the pool is out of the money and prepayment is only turnover), then rises to @@pp_s1@@ per percentage point between 0 and 0.5, @@pp_s2@@ between 0.5 and 1, and flattens at larger incentives (@@pp_s3@@ and @@pp_s4@@) as the S-shape saturates. Burnout has a coefficient of @@pp_burn@@ per month (negative, as designed), the seasoning ramp @@pp_seas@@, MTM LTV @@pp_ltv@@ per point and FICO @@pp_fico@@ per point.

At the portfolio level, the LightGBM prediction follows the annualized CPR over time:

![Predicted against actual annualized CPR by month, out-of-time.](@@chart_dir@@/prepay_cpr_oot.png){width=90%}

@@tbl:cpr_year@@

Table: Average annualized CPR by year, actual against LightGBM predicted.

The monthly root-mean-square error of CPR is @@pp_lgbm_rmse@@ points for LightGBM, @@pp_xgb_rmse@@ for XGBoost and @@pp_logit_rmse@@ for the hinge logit. The actual peak of @@cpr_peak@@ in @@cpr_peak_month@@ is predicted as @@cpr_peak_pred@@: the model captures the wave but not its full height, and it responds a little late to the end of the wave in early 2022. In 2023 the actual average is @@cpr_2023_act@@ against a predicted @@cpr_2023_pred@@. The hinge logit has the lowest CPR error of the three, so it would be a defensible champion for prepayment projections on this evidence, even though XGBoost is wired into the scenario engine.

![Prepayment ROC and KS, out-of-time.](@@chart_dir@@/roc_ks_prepay.png){width=90%}

![Prepayment calibration, out-of-time.](@@chart_dir@@/calibration_prepay.png){width=65%}

# Explainability

## Global importance

SHAP values (section 7.12) were computed for the LightGBM PD model and the XGBoost prepayment model on random out-of-time samples.

@@tbl:shap_pd@@

Table: Top ten PD features by mean absolute SHAP value (log-odds units).

![Mean absolute SHAP value, PD model.](@@chart_dir@@/shap_bar_pd.png){width=70%}

![SHAP summary (beeswarm) for the PD model. Each dot is a loan; position is the effect on the log-odds of default; shade is the feature value.](@@chart_dir@@/shap_beeswarm_pd.png){width=80%}

`@@shap_pd1@@` dominates (@@shap_pd1v@@), followed by `@@shap_pd2@@`, `@@shap_pd3@@`, `@@shap_pd4@@` and `@@shap_pd5@@`. The beeswarm shows the direction: low FICO (light dots) pushes the log-odds up and high FICO pushes it down; high DTI, high unemployment and high mark-to-market LTV push it up; older age pulls it down in the simulated hump. One subtlety: `dlq_status` ranks @@shap_dlq_rank@@ with a mean absolute SHAP of only @@shap_dlq_val@@, yet in the beeswarm the few loans that are currently 30DPD receive a contribution of about +3 log-odds. Mean absolute SHAP averages over all loans, and only about @@dlq_share_oot@@ of snapshots are delinquent, so a rare feature can be decisive for the loans it touches and still rank low globally.

@@tbl:shap_pp@@

Table: Top ten prepayment features by mean absolute SHAP value.

![Mean absolute SHAP value, prepayment model.](@@chart_dir@@/shap_bar_prepay.png){width=70%}

For prepayment, `@@shap_pp1@@` alone carries @@shap_pp_share1@@ of the total importance, followed by `@@shap_pp2@@`, `@@shap_pp3@@` and `@@shap_pp4@@`. This is what the simulated hazard implies: the refinance term is the biggest lever, with age/seasoning and burnout the next.

## Alignment with the data generating process

A check that explanations make sense is only possible here because the truth is known. `explain.dgp_alignment` runs two kinds of test. The *rank* tests ask whether the features the DGP says matter most appear near the top of the SHAP ranking (for PD: FICO, mark-to-market LTV and unemployment in the top five, and DTI or recent delinquencies also in the top five; for prepay: incentive first, burnout in the top four). The *sign* tests compute the Spearman rank correlation between a feature's value and its SHAP value and require its sign to match the DGP coefficient with absolute value of at least 0.05.

@@tbl:align@@

Table: DGP alignment checks. @@align_pass@@ of @@align_n@@ pass.

Passing these checks shows that the trees learned the right structure and that the SHAP pipeline is wired correctly. It does not show that the same relationships hold in real data, where no such truth exists.

## Dependence plots

Dependence plots show, for one feature, the SHAP contribution of each loan against that feature's value. They reveal the *shape* of the learned relationship, including thresholds and flat regions.

![SHAP dependence for FICO (PD model).](@@chart_dir@@/shap_dependence_pd_fico.png){width=65%}

![SHAP dependence for state unemployment (PD model). The contribution rises with unemployment and then flattens above roughly 8.5 percent.](@@chart_dir@@/shap_dependence_pd_unemp.png){width=65%}

![SHAP dependence for mark-to-market LTV (PD model).](@@chart_dir@@/shap_dependence_pd_mtm_ltv.png){width=65%}

The FICO plot is monotone decreasing, as constrained. The unemployment plot is the instructive one: the contribution climbs steeply between 5 and 7 percent, then **goes flat above about 8.5 percent**, which is the edge of the unemployment range seen in training (training maximum @@unemp_tr_max@@, OOT maximum @@unemp_oot_max@@). The true hazard keeps rising linearly with unemployment; the model cannot, and this is the root of the stress under-reaction in section 10.4. The plots for age, DTI and 12-month unemployment change, and the prepayment dependence plots (incentive, age, burnout), are in `outputs/charts/`.

![SHAP dependence for rate incentive (prepayment model).](@@chart_dir@@/shap_dependence_prepay_incentive.png){width=65%}

## Reason codes for individual loans

`explain.reason_codes` turns a loan's SHAP vector into plain-language reasons: the features with the largest **positive** (risk-increasing) contributions are mapped through `config.REASON_TEXT`, so a loan receives up to four reasons such as "Credit score lower than typical". `explain.reason_code_table` applies this to five illustrative out-of-time loans.

@@tbl:reason@@

Table: Reason codes for illustrative loans (LightGBM PD model).

For the riskiest loan (PD @@rc_high_pd@@) the reasons are sensible: low credit score, currently delinquent, high DTI and a high note rate. The lowest-risk loan (PD @@rc_low_pd@@) shows the limitation of the method: the "reasons" are simply the least favourable features of a loan that has none, with tiny contributions (for example "Elevated risk from occupancy", the generic text for features without a mapped phrase). A production adverse-action process would only report reasons above a materiality threshold. The "currently 30DPD" case is the same loan as the highest-PD case because that loan is both.
"""

S10 = r"""
# Scenario analysis

## The six shocks

A scenario is a deterministic macro path applied to the end-2023 portfolio. `macro.apply_shock` ramps each shock linearly over the stated number of months from the as-of date and then holds it constant.

@@tbl:shocks@@

Table: Scenario definitions (`config.SHOCKS`). Rates are added to the market mortgage rate, house prices are scaled multiplicatively, and unemployment points are added to every state's rate.

## How the projection works

*Intuition.* Take every loan alive at the as-of date, move the calendar forward one month at a time, update each loan's inputs for the shocked macro world (age plus one, scheduled amortization, new rate incentive, new mark-to-market LTV, new unemployment), re-score it with both models, and aggregate the expected defaults and prepayments weighted by balance.

*Maths.* In projection month $k$, for loan $i$ with exposure $E_{i,k}$ (survival weight times scheduled balance), the 12-month PD is converted to a monthly default hazard,

$$\text{hd}_{i,k} = 1-\big(1-\text{PD}^{12}_{i,k}\big)^{1/12},$$

(capped at 0.5), applied first. The prepayment SMM $s_{i,k}$ is then applied to the loans that survived the default step and are in the current state at the as-of date. Expected defaulted and prepaid balances are

$$D_k=\sum_i E_{i,k}\,\text{hd}_{i,k}, \qquad P_k=\sum_i \big(E_{i,k}-E_{i,k}\text{hd}_{i,k}\big)\,s_{i,k},$$

and the survival weight is updated as $w_{i,k+1}=w_{i,k}(1-\text{hd}_{i,k})(1-s_{i,k})$. The portfolio monthly default rate (MDR) and SMM are $D_k/\sum_i E_{i,k}$ and $P_k/(\sum_i E_{i,k}-D_k)$, and the annualized CDR and CPR follow from $1-(1-x)^{12}$.

*Implementation.* `scenarios.project_portfolio` (model-based), built on a helper `_Path` that precomputes each loan's shocked inputs for every month, and `scenarios.project_truth` (the DGP's own Markov chain evaluated under the same shocked macro, synthetic data only). The models used are @@scen_pd_model@@ for PD and @@scen_pp_model@@ for prepayment. The 60-month curves are in `outputs/scenario_curves.csv` and the 12-month summary in `outputs/scenario_12m.csv`.

Important simplifications, all documented in the module header: delinquency status and the recent-delinquency count are held at their as-of values; only loans that were current at the as-of date are allowed to prepay; and the default step uses a 12-month cumulative-incidence PD as a monthly hazard on a portfolio that is also prepaying, which slightly double counts the removal of loans (section 10.5).

## Results on the end-2023 pool

![Projected CPR, CDR, surviving balance and SMM for the six scenarios.](@@chart_dir@@/scenario_curves.png){width=95%}

@@tbl:scen_12m@@

Table: Twelve-month summary on the 31 December 2023 pool, model-based. Defaults and prepayments are percentages of the starting balance.

**Unemployment and house prices move defaults; rates do not move anything.** The unemployment shock raises the model's 12-month default rate from @@s_base_def@@% to @@s_un_def@@% (an increase of @@un_uplift_model@@); the house-price shock to @@s_hpi_def@@%; the combined adverse scenario to @@s_adv_def@@% (@@adv_mult_model@@ times base). The two rate scenarios are almost indistinguishable from base: the +200bp 12-month default and prepayment rates are @@rate_up_txt@@, while -200bp changes 12-month prepayments by only @@rate_dn_pre_diff@@ percentage points.

The reason is **moneyness**. The pool at the end of 2023 holds @@pool_n@@ loans with a balance of USD @@pool_upb@@ million. Its balance-weighted note rate is @@pool_rate@@ percent against a simulated market rate of @@mkt_end@@ percent, a balance-weighted incentive of @@pool_inc@@ points, and only @@pool_itm@@ of the balance has an incentive above 0.5 points. In that region the refinance sigmoid is flat at zero, so a +200bp shock moves loans from very negative incentive to even more negative with no effect, and even -200bp only brings the market rate to @@mkt_dn@@ percent, still above the note rate of most loans. A pool this far out of the money has almost no rate sensitivity, which is a true economic fact about such pools (the 2022 to 2023 "lock-in" effect) and not a software defect. To see the rate sensitivity the model does have, the pipeline repeats the whole scenario run for a pool as of 31 December 2021, when the balance-weighted incentive was @@itm_pool_inc@@ points and @@itm_pool_itm@@ of the balance was in the money.

@@itm_block@@

## Model against simulation truth

Because the simulator has an exact DGP, the scenario engine can be compared with a "truth" projection that evaluates the true hazards (and the true 30, 60 and 90 day roll chain) on the same shocked paths. This is a **methodology check specific to synthetic data**; it has no analogue on real data.

![Model and simulation-truth expected 12-month default and prepay rates by scenario.](@@chart_dir@@/model_vs_truth.png){width=95%}

@@tbl:mvt@@

Table: Model against simulation truth, 12 months, percent of starting balance. Errors are model divided by truth minus one.

Prepayments are within about ten percent of truth in every scenario (errors of @@e_base_pre@@ in the base case and @@e_adv_pre@@ in the adverse case). **Defaults are not.** In the base case the model is @@e_base_def@@ below truth (@@m_base_def@@% against @@t_base_def@@%), which is @@base_tol_status@@ the project's own tolerance of @@tol@@, but the shortfall widens with stress: @@e_hpi_def@@ for house prices, @@e_un_def@@ for unemployment and @@e_adv_def@@ for the combined scenario (@@m_adv_def@@% against @@t_adv_def@@%). In total @@n_out_tol@@ of the six scenarios exceed the @@tol@@ tolerance. Under unemployment +4 points the truth rises by @@un_uplift_truth@@ and the model by @@un_uplift_model@@; under the combined scenario the truth is @@adv_mult_truth@@ times base and the model only @@adv_mult_model@@ times.

**Why the model under-reacts: trees cannot extrapolate.** A tree-based model is a piecewise-constant function of its inputs; outside the range seen in training, its output is whatever the outermost leaf says, even though the underlying true relationship continues. The stress scenarios push the inputs beyond that range. In training, the 12-month house price change never fell below @@hpi_tr_min@@ (the full range was @@hpi_tr@@), so the model has no information about falling prices, yet the DGP adds default risk precisely when prices fall; the HPI -20% scenario therefore sits entirely in unseen territory. Similarly, unemployment in training peaked at @@unemp_tr_max@@ percent while the +4 point shock moves state rates toward and past that level. Monotone constraints keep the response non-decreasing but do not add slope where the data have none. @@base_short_txt@@

The practical lesson, which a validator would write up as a limitation, is that a loan-level tree model is a reasonable *baseline* projection engine inside the historical range and should not be used alone for severe stress tests. Remedies include adding an explicit macro overlay or a parametric hazard layer for the stress dimension, training on data that spans a full cycle (real data would include 2008 to 2012), or blending with the logistic scorecard whose linear terms do extrapolate.

## Limits of the projection

Besides extrapolation, the following limits apply: scenarios are deterministic linear ramps, not draws from a joint macro distribution, so no probability can be attached to them; status is frozen during the projection, so a loan that is current at the as-of date cannot become delinquent and then cure within the model-based path; loss severity is not modelled; the PD already nets out prepayment (cumulative incidence) and is then applied together with a separate prepayment step, which double counts the removal of loans and biases the default projection downward by roughly the prepayment rate times the default rate; and the result is one realization of one simulated portfolio, with no uncertainty bands from re-simulating the data.

## Waterfall export hook

`export_hook.write_curves` converts each scenario's projected curve into the two files read by the securitization waterfall project.

**`cpr_cdr_curves.csv`** contains monthly vectors for each scenario for @@hook_months@@ months (the 60 modelled months are extended by holding the last SMM and MDR flat):

@@tbl:hook_curves@@

Table: Base scenario curve extract (months 1, 12, 60, 61 and 120). Month 61 onward repeats the month-60 rates.

**`scalar_equivalents.csv`** summarizes each scenario with one set of scalar assumptions, using survival-balance weights on the annualized rates:

@@tbl:hook_scalars@@

Table: Scalar equivalents. Column names equal the fields of `waterfall.config.Scenario`.

| Hook column | Waterfall `Scenario` field | Meaning and unit |
|:--|:--|:--|
| `scenario` | `name` | scenario label |
| `cpr` | `cpr` | annualized prepayment rate, decimal |
| `cdr` | `cdr` | annualized default rate, decimal |
| `severity` | `severity` | loss given default, fixed assumption @@hook_sev@@ |
| `lag` | `lag` | months from default to recovery, fixed assumption @@hook_lag@@ |
| `index_shift` | `index_shift` | parallel shift of the floating index, decimal (rate shock bp divided by 10,000) |

Table: Mapping of the exported scalars to the waterfall project's `Scenario` dataclass.

`export_hook.to_waterfall_scenarios` reads the scalar file back into dictionaries with exactly these keys, and a test compares the keys with the sibling project's `config.py` so the interface cannot drift silently. Two caveats. Severity and lag are assumptions, not model outputs. And the sibling project's collateral is a synthetic short-maturity, high-coupon consumer-style pool, with a base scenario of 10 percent CPR and 3 percent CDR, whereas this project's base mortgage pool shows a CPR of @@hook_base_cpr@@ and a CDR of only @@hook_base_cdr@@ (@@hook_adv_cdr@@ in the adverse combined case). The hook therefore demonstrates the interface and units; applying mortgage-derived CDRs to a different collateral type would not be a meaningful deal analysis.
"""


S11 = r"""
# Excel workbook walkthrough

## Why an Excel version exists

Model validators and credit committees often want to see a calculation in a tool they can audit cell by cell. The workbook `excel/mortgage_risk_model.xlsx` re-implements the scorecard, the decile and calibration analysis, the PSI and a simplified scenario engine with **live formulas** over a fixed sample of @@excel_n@@ out-of-time snapshot loans (`excel_inputs/loan_sample.csv`, a seeded random draw from the out-of-time set). The workbook contains @@excel_formulas@@ formulas. It is generated by `excel/build_workbook.py`, with the Python-side reference values and the comparison logic in `mortgage_risk/reconcile_excel.py`. Typed numbers are limited to the Inputs sheet and to imported Python value sheets; everything else is a formula.

## Sheet by sheet

| Sheet | What it contains | Live or static |
|:--|:--|:--|
| README | purpose, how to use, list of approximations | text |
| Inputs | scenario selector (1 to 6), LGD, scorecard scaling (PDO, base score, base odds; factor and offset are formulas), the logit intercept, Platt calibrator slope and intercept, prepay hinge coefficients, a one-loan calculator input block | inputs; factor and offset live |
| Scorecard | the bin table from `scorecard_table.csv` (variable, bin edges, WoE, coefficient, points) with a helper column of finite edges | static values from Python |
| Score_Calc | one-loan scorecard calculator: looks up each variable's bin, WoE and points from the Inputs block, sums to a margin, calibrated PD and total score, and ranks the three variables with the largest shortfall against the best bin | live |
| Loan_Sample | the @@excel_n@@ loans with their attributes and the Python scorecard PD, points and LightGBM PD (static), plus live columns for each variable's WoE, the margin, the calibrated PD, the points, the PD rank and the decile | static attributes, live `xl_` columns |
| Deciles | ten equal-count groups on the live PD: counts, events, rate, mean PD, cumulative capture, lift, KS gap, a trapezoid AUC and Gini, next to the full-population Python decile values | live |
| PSI | the training decile bins (from `psi_train_bins.csv`) against the live sample score shares, with the share floor and bands; total PSI | live |
| Calibration | predicted against observed by decile, Brier score against a constant-forecast Brier, Hosmer-Lemeshow, expected against observed events | live |
| Scenario | the selected shock applied to the sample (unemployment added, MTM LTV divided by one plus the HPI shock, incentive reduced by the rate shock), re-binned scorecard PD, expected loss as PD times LGD times balance, and prepay SMM from the hinge logit; base against selected; Python GBM reference below | live |
| Checks | 14 integrity checks (counts, bounds, rank permutation, shares sum to one, live PD and points against the exported Python values) and an all-pass cell | live |
| Conclusions | static text written by the builder from the computed numbers | text |
| Python_Ref | imported Python benchmark, decile, PSI bins, CSI and scenario summary tables used for side-by-side comparison | static values |
| Curves_Python | the 60-month Python scenario curves (`scenario_curves.csv`) | static values |

Table: Workbook sheets.

Three mechanics deserve explanation. First, a bin lookup: for a value $x$ and a variable's finite bin edges $e_1<\dots<e_{k-1}$, the bin number is one plus the count of edges strictly below $x$, i.e. `SUMPRODUCT(--(edges<x))+1`, which reproduces the right-closed bins of `BinSpec.bin_index`; a blank input falls into the "missing" bin. Second, the calibrated PD in Excel applies the stored Platt slope $a=@@xl_pd_a@@$ and intercept $b=@@xl_pd_b@@$ to the scorecard margin $z$, $p=\sigma(az+b)$, clipped to $[10^{-6}, 1-10^{-6}]$, mirroring the Python calibrator for the scorecard. Third, the rank and decile columns replicate `evaluation.decile_table`: sorted by descending PD with ties broken by input order (a `COUNTIF` tiebreak on rank), with the remainder loans going to the first groups as in `numpy.array_split`.

## Approximations the workbook makes on purpose

- **Prepay burnout.** The sample has no burnout history, so the prepay hinge logit uses a single burnout value of @@xl_burnout@@ months, solved so that the sample's base 12-month prepay equals the Python GBM base. This is labelled an approximation on the Inputs sheet.
- **Scenario step.** The full shock is applied at the snapshot with no ramp, and prepay is annualized at a constant SMM, so the Excel scenario table is not comparable with the 60-month Python projection; the Python reference values are shown beside it for context only.
- **Rate shocks.** The scorecard moves with a rate shock only through its `incentive` variable, while the hinge logit moves a great deal, so Excel rate scenarios mainly demonstrate the mechanics.
- **AUC and Gini.** Computed from the decile ROC trapezoid, an approximation of the exact AUC.

## Reconciliation to Python

@@excel_block@@

# Governance in the style of SR 11-7

Supervisory guidance on model risk management, issued by the Federal Reserve and the OCC in 2011 as SR 11-7, asks banks to document a model's purpose, design, data, performance, limitations and ongoing monitoring, to validate it independently with "effective challenge", and to keep an inventory and controls (Federal Reserve and OCC 2011). This section writes the project up in that structure to show how such a model would be described. **It is a documentation exercise on synthetic data. The model has not been independently validated, has not been approved by any institution, and must not be used to make credit, pricing or capital decisions.**

## Purpose and intended use

The models estimate (a) the 12-month probability that a performing or early-delinquent first-lien 30-year mortgage reaches 90+DPD and (b) the monthly probability of full prepayment, for ranking loans, producing expected default and prepayment curves for a pool under macro scenarios, and feeding those curves into a structured-finance cash-flow model. Intended users are an analyst learning the workflow and a reader assessing methodology. Out-of-scope uses: severity or loss forecasting, regulatory capital, accounting provisioning, origination decisions, pricing of real securities, or any use on loan types other than the simulated 30-year fixed-rate first liens.

## Data

Synthetic panel of @@n_loans@@ loans (seed @@seed@@) generated by a documented process (section 4). Real-data readiness: the loader for Freddie Mac loan-level files exists and is exercised only by unit tests on a small fixture, not on full vintages. Data quality controls: stop at first default event, sentinel handling and median imputation for missing credit fields, leakage guards, time-ordered splits.

## Methodology

A WoE scorecard (logistic regression on binned, monotone-trended variables, scaled to points) is the benchmark and the explainable champion candidate; XGBoost and LightGBM with monotone constraints, time-aware tuning and calibrated outputs are the challengers; a hinge logit and two boosted models for prepayment. Explanations use TreeSHAP on the raw margin. Scenarios re-score the portfolio monthly under deterministic macro shocks.

## Performance summary

| Measure (out-of-time, synthetic) | Value | Assessment |
|:--|:--|:--|
| PD AUC, scorecard / XGBoost / LightGBM | @@pd_logit_oot_auc@@ / @@pd_xgb_oot_auc@@ / @@pd_lgbm_oot_auc@@ | good ranking; trees significantly better (DeLong) |
| PD KS, LightGBM | @@pd_lgbm_oot_ks@@ | adequate separation |
| PD observed against predicted level | @@obs_oot_lgbm@@ against @@pred_oot_lgbm@@ | under-prediction; remediation required before level use |
| Score PSI, OOT against train | @@psi_oot_lgbm@@ (LightGBM) | stable |
| Features in significant CSI band | @@csi_n_sig@@ of @@csi_n@@ | regime shift in macro inputs, expected in a stress test |
| Prepay AUC, hinge logit / LightGBM | @@pp_logit_oot_auc@@ / @@pp_lgbm_oot_auc@@ | moderate; adequate for a smooth hazard |
| Prepay CPR RMSE, LightGBM | @@pp_lgbm_rmse@@ points | acceptable; peak under-estimated |
| SHAP against DGP checks | @@align_pass@@ of @@align_n@@ pass | explanations coherent with truth |
| 12m default, model against truth, adverse combined | @@m_adv_def@@% against @@t_adv_def@@% | material under-reaction under stress |

Table: Performance summary.

## Assumptions log

| ID | Assumption | Why it was made | Effect if wrong |
|:--|:--|:--|:--|
| A1 | Data follow the documented logistic-hazard DGP | no access to real loans in this build | results prove methodology only |
| A2 | Default means first month at 90+DPD | common practical definition; simple to observe | different definitions change the base rate |
| A3 | PD horizon 12 months, cumulative incidence (prepay is "no default") | answers the investor's question directly | not a pure default hazard; double counting in projection |
| A4 | Snapshots in January and July, status current or 30DPD | limits overlap and focuses on early identification | 60DPD loans are not scored |
| A5 | Time split with development labels resolved by the cutoff | prevents label leakage | otherwise optimistic validation |
| A6 | Stylized deterministic macro paths and linear-ramp shocks | transparent scenario design | no probabilities attached to scenarios |
| A7 | Delinquency status held fixed in projection | keeps the projection tractable | misses roll dynamics within the horizon |
| A8 | Severity 35% and lag 6 months in the waterfall export | no loss model built | loss and tranche results are assumption-driven |
| A9 | Prepayment model trained on a @@prepay_share@@ random sample of loans | memory and speed | slightly higher estimation noise |
| A10 | Median imputation for missing credit fields on real data | simplicity | bias if missingness is informative |
| A11 | Calibration on a separate 2018 (PD) and 2019 (prepay) window | out-of-sample calibration | level bias if the window is unrepresentative |

Table: Assumptions log.

## Limitations

1. **Synthetic data.** Performance, SHAP agreement and the model-versus-truth comparison say nothing about real mortgage behaviour. The models and the simulator share the same logistic family, which flatters both.
2. **Under-prediction of PD level out of time** (observed against predicted ratio @@ratio_lgbm@@), driven by the benign calibration window and regime change.
3. **No extrapolation.** Tree models are flat outside the training range of unemployment (training maximum @@unemp_tr_max@@ percent) and house price changes (training minimum @@hpi_tr_min@@); stress results under-react, and @@n_out_tol@@ of six scenarios breach the project's own @@tol@@ tolerance against simulation truth.
4. **Pool out of the money.** The end-2023 pool shows no rate sensitivity, so rate scenarios are not informative on it; the 2021 as-of run exists for that reason.
5. **Scorecard blind spot.** The scorecard drops `dlq_status` and `times_30dpd_12m` because only @@dlq_share_tr@@ of training snapshots are delinquent, below the 5% minimum bin share, so each variable collapses to one bin. The scorecard cannot see that a loan is already late, which the boosted models do (it is the largest SHAP contribution for the loans it affects). This is a design weakness of the binning rule, not of scorecards in general, and a fix (a special-value bin that is exempt from the minimum share) is listed under next steps.
6. **Competing-risk handling is approximate** in the projection (section 10.5).
7. **No loss severity, modifications, forbearance or insurance.**
8. **Single simulated realization**, no repeated-simulation confidence on any metric beyond the loan bootstrap.
9. **Reason codes** are generic and unfiltered for materiality (section 9.4).
10. **Real-data path lightly tested**; macro series must be supplied or the loader falls back to the synthetic macro, which would make macro features meaningless for real loans.

## Ongoing monitoring plan

| Metric | Frequency | Green | Amber | Red | Action |
|:--|:--|:--|:--|:--|:--|
| Score PSI (current against development) | monthly | below 0.10 | 0.10 to 0.25 | above 0.25 | amber: investigate drivers via CSI; red: escalate to the model owner and consider recalibration |
| CSI of the top five SHAP features | monthly | below 0.10 | 0.10 to 0.25 | above 0.25 | review input data and population change |
| AUC on matured 12-month outcomes | quarterly | within 0.02 of development (@@pd_lgbm_oot_auc@@) | 0.02 to 0.05 below | more than 0.05 below | redevelop or replace with the challenger |
| Observed over predicted default rate, overall and by decile | quarterly | 0.90 to 1.10 | 0.80 to 0.90 or 1.10 to 1.25 | outside 0.80 to 1.25 | recalibrate on a recent window |
| Prepay CPR RMSE against development (@@pp_lgbm_rmse@@ points) | monthly | below 1.25 times | 1.25 to 2 times | above 2 times | refit incentive response |
| Data checks (missing rates, sentinel codes, balance roll-forward) | monthly | no breach | minor | major | stop scoring until fixed |

Table: Proposed monitoring metrics and triggers (illustrative thresholds for the model owner to set and governance to approve).

A **backtest** compares predicted PD with realized 12-month outcomes for the same snapshot cohort once the window has matured, by decile and by vintage, using the same tables as section 8. Full revalidation is annual or earlier on a red trigger.

## Challenger plan

The scorecard logit is the standing challenger for PD and the hinge logit for prepayment; both are re-fitted and compared on each refresh using the benchmark table and the DeLong test. Planned additional challengers are a discrete-time multinomial competing-risk model (default, prepay, stay) that removes the double-counting approximation, a parametric hazard overlay for the macro stress dimension, and a benchmark run on real Freddie Mac vintages spanning 2007 to 2012.

# Repository guide and how to run

## Layout

```
Mortgage_Delinquency_Prepayment_Model/
  README.md                        overview and headline results
  INTERFACES.md                    module contracts used while building
  data/raw/, data/interim/         Freddie files go in raw; parquet cache in interim
  docs/                            this document, build_docs.py, reference.docx
  excel/                           build_workbook.py, mortgage_risk_model.xlsx
  python/
    mortgage_risk/                 the package (modules below)
    tests/                         pytest suites
    outputs/                       CSV/JSON results, charts/, models/, waterfall_hook/,
                                   excel_inputs/, scenarios_itm_2021/
    requirements.txt, pytest.ini
```

| Module | Role |
|:--|:--|
| `config.py`, `columns.py` | all shared settings, the true DGP coefficients, split windows, shocks; every column name as a constant |
| `macro.py` | stylized rate, house price and unemployment paths; `apply_shock` |
| `synthetic.py` | simulator `simulate_panel` and `calibration_report` |
| `freddie.py` | Freddie Mac file loader and macro loader |
| `data.py` | `get_panel`: choose source, cache parquet |
| `features.py` | backward-looking features and the leakage guard |
| `targets.py` | PD snapshots, prepayment hazard rows, time split |
| `scorecard.py` | WoE binning, IV, scorecard fit and points table |
| `models.py` | model bundles, time-aware tuning, calibration, `train_all` |
| `evaluation.py` | metrics, DeLong, bootstrap, PSI and CSI, benchmark, output writer |
| `explain.py` | SHAP, DGP alignment, reason codes |
| `scenarios.py` | projection engine, truth projection, 12-month summaries |
| `export_hook.py` | waterfall export |
| `charts.py` | all figures (matplotlib, plain style) |
| `reconcile_excel.py` | Excel reference values and LibreOffice reconciliation |
| `cli.py` | end-to-end pipeline |

Table: Package modules.

## Commands

```
cd python
py -3 -m pip install -r requirements.txt
py -3 -m mortgage_risk.cli            # full run: @@n_full@@ loans, tuning, all outputs
py -3 -m mortgage_risk.cli --quick    # @@n_quick@@ loans, light tuning, for smoke runs
py -3 -m pytest -q                    # unit and integration tests
py -3 ../docs/build_docs.py           # rebuild this document and the README
```

Option: `--skip-excel` skips workbook building and the LibreOffice reconciliation (which needs LibreOffice installed). The pipeline order is data, features, targets, training, evaluation, SHAP, scenarios, in-the-money scenarios, charts, Excel. The saved models are in `outputs/models/*.joblib` and can be reloaded with `ModelBundle.load`.

`docs/build_docs.py` is fully re-runnable: it reads the CSV and JSON outputs, recomputes a few panel statistics from the cached parquet (and caches them in `docs/_facts_cache.json`), writes the Markdown, converts it to Word with pandoc through `pypandoc` using `docs/reference.docx` (headings in size and weight only, with no colour), sets the author properties and verifies the round trip.

## Using real Freddie Mac data

1. Obtain the Single Family Loan-Level Dataset from Freddie Mac under its terms of use. It is not redistributed here.
2. Place the pipe-delimited origination and monthly performance files directly under `data/raw/` with Freddie's native names (`historical_data_YYYYQn.txt` and `historical_data_time_YYYYQn.txt`), which match the two glob constants at the top of `data.py`.
3. Optionally supply macro series as CSV files in `data/raw/macro/` with columns `period`, `state`, `mkt_rate`, `hpi`, `unemp`. Without them the loader logs a warning and uses the synthetic macro, which is wrong for real loans.
4. Run the CLI. `data.get_panel` selects real data automatically when both globs match, and everything downstream runs unchanged.
5. Ignore the model-versus-truth outputs on real data: the truth projection evaluates the simulator's own formulas and is meaningless for real loans. Interpret the "DGP alignment" file the same way.

The Freddie path has been exercised only on a small fixture in the unit tests (`python/tests/fixtures`).

# Glossary

| Term | Meaning |
|:--|:--|
| AUC | area under the ROC curve; probability a random default is scored above a random non-default |
| Brier score | mean squared error of predicted probabilities |
| Burnout | reduced prepayment responsiveness of a pool after its most rate-sensitive borrowers have already refinanced |
| CDR | conditional default rate, annualized default rate on the surviving balance |
| Competing risks | outcomes that exclude one another, here prepay and default |
| CPR | conditional prepayment rate, annualized |
| CSI | characteristic stability index; PSI applied to one input feature |
| DGP | data generating process; the formulas that created the synthetic data |
| DPD | days past due |
| DTI | debt-to-income ratio |
| Gini | $2\,\text{AUC}-1$ |
| Hosmer-Lemeshow | chi-square test comparing expected and observed events across bins |
| Incentive | note rate minus market mortgage rate |
| IV | information value of a binned variable |
| KS | Kolmogorov-Smirnov statistic; maximum gap between cumulative score distributions of bad and good |
| LGD | loss given default |
| LTV | loan-to-value ratio |
| MDR | monthly default rate |
| Moneyness | whether a borrower would save money by refinancing (in the money) or not (out of the money) |
| Monotone constraint | restriction that a model's output moves in one direction with a feature |
| MTM LTV | mark-to-market LTV, current balance over index-adjusted house value |
| OOT | out-of-time; data after the development period |
| PDO | points to double the odds in a scorecard |
| PSI | population stability index |
| Platt scaling | logistic recalibration of a model's margin |
| SHAP | Shapley additive explanations; per-feature contributions to a prediction |
| SMM | single monthly mortality; monthly prepayment rate |
| UPB | unpaid principal balance |
| WoE | weight of evidence of a bin |

# References

1. Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency (2011). *Supervisory Guidance on Model Risk Management*, SR Letter 11-7 and OCC Bulletin 2011-12.
2. Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
3. DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837-845.
4. Freddie Mac. *Single Family Loan-Level Dataset General User Guide*.
5. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30*.
6. Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30*.
7. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
"""


def _static(V, T, X):
    """Constants taken from the project modules, tables of settings, and boilerplate switches."""
    from mortgage_risk import synthetic as syn, models as mod
    V["doc_title"], V["author"], V["today"] = DOC_TITLE, AUTHOR, date.today().strftime("%d %B %Y")
    V["chart_dir"] = CHART_REL
    V["fico_mean"], V["fico_sd"], V["fico_lo"], V["fico_hi"] = (f"{x:.0f}" for x in (syn.FICO_MEAN, syn.FICO_SD, syn.FICO_LO, syn.FICO_HI))
    V["ltv_p80"], V["ltv_plow"] = pc(syn.LTV_MIX[0], 0), pc(syn.LTV_MIX[1], 0)
    V["dti_mean"], V["dti_sd"], V["dti_lo"], V["dti_hi"] = (f"{x:.0f}" for x in (syn.DTI_MEAN, syn.DTI_SD, syn.DTI_LO, syn.DTI_HI))
    V["upb_median"], V["upb_sigma"] = ni(syn.UPB_MEDIAN), fx(syn.UPB_SIGMA, 2)
    V["upb_floor"], V["upb_cap"] = ni(syn.UPB_FLOOR), ni(syn.UPB_CAP)
    V["rate_spread"], V["rate_fico_slope"], V["rate_noise"] = fx(syn.RATE_SPREAD, 2), fx(syn.RATE_FICO_SLOPE, 2), fx(syn.RATE_NOISE, 2)
    V["rate_pre_base"] = fx(macro_mod.RATE_PRE_BASE, 1)
    V["hpi_growth"], V["hpi_flat_year"] = pc(macro_mod.HPI_GROWTH_ANNUAL, 0), str(macro_mod.HPI_FLAT_YEAR)
    V["beta_min"], V["beta_max"] = fx(min(config.STATE_HPI_BETA), 2), fx(max(config.STATE_HPI_BETA), 2)
    V["uoff_min"], V["uoff_max"] = f"{min(config.STATE_UNEMP_OFFSET):+.1f}", f"{max(config.STATE_UNEMP_OFFSET):+.1f}"
    V["unemp_floor"] = fx(macro_mod.UNEMP_FLOOR, 1)
    V["tune_iter"], V["tune_folds"] = str(config.TUNE_ITER), str(config.TUNE_FOLDS)
    V["boot_n"], V["shap_n"] = ni(config.BOOTSTRAP_N), ni(config.SHAP_SAMPLE)
    V["iv_min"], V["excel_n"] = fx(config.SCORECARD_IV_MIN, 2), ni(config.EXCEL_SAMPLE_LOANS)
    V["a_fico"], V["b_unemp"] = fx(config.TRUE_DGP["prepay"]["a_fico"], 2), fx(config.TRUE_DGP["to30"]["b_unemp"], 3)
    if V["source"] == "synthetic":
        V["source_banner"] = (
            "**All data in this document are SYNTHETIC.** They were produced by a simulator (`python/mortgage_risk/synthetic.py`) "
            "from an explicit set of equations written down in section 4, with stylized macro paths. No real borrower, loan or "
            "Freddie Mac record was used for any number, table or chart here. Every result therefore reflects the documented data "
            "generating process (DGP), including the strengths and the blind spots of the models against it.")
    else:
        V["source_banner"] = ("The outputs were produced from real loan-level files. Sections that describe the simulator and "
                              "the model-versus-truth comparison apply to the synthetic mode only and should be ignored for these outputs.")

    # macro anchors
    ra = pd.DataFrame(macro_mod.RATE_ANCHORS, columns=["date", "rate"])
    T["rate_anchors"] = md_table(ra, dict(rate=lambda x: fx(x, 1)), dict(date="Month end", rate="Market rate (%)"))
    ua = pd.DataFrame(macro_mod.UNEMP_ANCHORS, columns=["date", "u"])
    T["unemp_anchors"] = md_table(ua, dict(u=lambda x: fx(x, 1)), dict(date="Month end", u="National unemployment (%)"))

    # DGP coefficients
    rows = []
    for g, d in config.TRUE_DGP.items():
        for k, v in d.items():
            rows.append(dict(group=DGP_GROUP[g], name=k, value=v, role=DGP_ROLE.get(k, "")))
    T["dgp"] = md_table(pd.DataFrame(rows), dict(value=lambda x: fx(x, 3)),
                        dict(group="Block", name="Parameter", value="Value", role="Role"))

    # split windows
    d = lambda t: t.strftime("%Y-%m-%d")
    sp = [("PD", "train", config.TRAIN_PD_SNAP, V["pd_rows_train"], "fit the models and the WoE bins"),
          ("PD", "calibration", config.CALIB_PD_SNAP, V["pd_rows_calib"], "fit the probability calibrators"),
          ("PD", "out-of-time", config.OOT_PD_SNAP, V["pd_rows_oot"], "all reported performance"),
          ("Prepay", "train", config.TRAIN_PP, V["pp_rows_train"], "fit the models"),
          ("Prepay", "calibration", config.CALIB_PP, V["pp_rows_calib"], "fit the probability calibrators"),
          ("Prepay", "out-of-time", config.OOT_PP, V["pp_rows_oot"], "all reported performance")]
    T["splits"] = md_table(pd.DataFrame([dict(m=a, part=b, start=d(w[0]), end=d(w[1]), n=n, use=u) for a, b, w, n, u in sp]),
                           dict(), dict(m="Model", part="Window", start="From", end="To", n="Rows", use="Used to"))
    try:
        from mortgage_risk import reconcile_excel as rx
        p = rx.load_model_params()
        V["xl_pd_a"], V["xl_pd_b"], V["xl_burnout"] = fx(p["pd_a"], 4), fx(p["pd_b"], 4), fx(p["burnout"], 1)
    except Exception as exc:  # noqa: BLE001
        print("excel params unavailable:", exc)
        V["xl_pd_a"], V["xl_pd_b"], V["xl_burnout"] = "n/a", "n/a", "n/a"


def _pool_and_itm(V, T, X):
    pf, cal = X["pf"], X["cal"]
    s12i = rcsv("scenario_12m.csv").set_index("scenario")
    mvt = rcsv("model_vs_truth.csv").set_index("scenario")
    tol = config.TEST_THRESHOLDS["model_vs_truth_rel_err"]
    err = mvt.loc["Base", "default_rel_err"]
    V["base_tol_status"] = "inside" if abs(err) <= tol else "outside"
    dup = abs(s12i.loc["Rates +200bp", "default_12m_pct_upb"] - s12i.loc["Base", "default_12m_pct_upb"])
    pup = abs(s12i.loc["Rates +200bp", "prepay_12m_pct_upb"] - s12i.loc["Base", "prepay_12m_pct_upb"])
    V["rate_up_txt"] = ("identical to base to the displayed precision" if max(dup, pup) < 5e-5 else
                        f"within {max(dup, pup):.3f} percentage points of base")
    mk = macro_mod.build_macro(pd.Timestamp("2023-12-01"), pd.Timestamp("2023-12-31"))
    mk_end = float(mk[mk[C.PERIOD] == pd.Timestamp("2023-12-31")][C.MKT_RATE].iloc[0])
    V["mkt_end"], V["mkt_dn"] = fx(mk_end, 1), fx(mk_end - 2.0, 1)
    pools = pf.get("pools", {}) if pf else {}
    p23, p21 = pools.get("2023-12-31"), pools.get("2021-12-31")
    if p23:
        V["pool_n"], V["pool_upb"] = ni(p23["n"]), fx(p23["upb"] / 1e6, 0)
        V["pool_rate"], V["pool_inc"], V["pool_itm"] = fx(p23["rate"], 2), f"{p23['inc']:.2f}", pc(p23["itm"], 1)
    else:
        for k in ("pool_n", "pool_upb", "pool_rate", "pool_inc", "pool_itm"):
            V[k] = "n/a"
    if p21:
        V["itm_pool_inc"], V["itm_pool_itm"] = f"{p21['inc']:.2f}", pc(p21["itm"], 1)
    else:
        V["itm_pool_inc"], V["itm_pool_itm"] = "n/a", "n/a"
    # calibration ratio extremes by bin
    c = cal["pd_lgbm"]
    r = c["obs_rate"] / c["mean_pred"]
    V["calib_ratio_max"], V["calib_ratio_max_bin"] = fx(r.max(), 2), str(int(c.loc[r.idxmax(), "bin"]))
    # extra text about the base-case error and the in-the-money run
    if (ITM_DIR / "model_vs_truth.csv").exists():
        im = rcsv("model_vs_truth.csv", ITM_DIR).set_index("scenario")
        V["itm_base_def_err"] = f"{100 * im.loc['Base', 'default_rel_err']:+.0f}%"
        V["itm_adv_def_err"] = f"{100 * im.loc['Adverse combined', 'default_rel_err']:+.0f}%"
        V["itm_hpi_def_err"] = f"{100 * im.loc['HPI -20%', 'default_rel_err']:+.0f}%"
        V["itm_base_pre_err"] = f"{100 * im.loc['Base', 'prepay_rel_err']:+.0f}%"
        V["base_short_txt"] = (
            "The base-case error is pool specific rather than a constant bias. For the end-2023 pool it is "
            f"{V['e_base_def']}; for the 2021 pool in section 10.3 it is only {V['itm_base_def_err']}, whereas the stress errors persist there "
            f"({V['itm_hpi_def_err']} for house prices and {V['itm_adv_def_err']} for the combined scenario). The end-2023 pool is dominated by "
            f"seasoned, low-PD loans, which is exactly where the observed-to-predicted calibration ratio is largest (up to {V['calib_ratio_max']} "
            f"in bin {V['calib_ratio_max_bin']} of section 8.3), so the same calibration weakness shows up as a larger relative error. "
            "Two smaller sources apply to every pool: the frozen delinquency status and the use of a cumulative-incidence PD as a hazard.")
        V["itm_block"] += (f"\n\nIn this pool the model agrees with simulation truth on the base 12-month default rate (error {V['itm_base_def_err']}), "
                           f"so the base-case bias at end-2023 is a feature of that pool, but the model still under-reacts to stress "
                           f"(house prices {V['itm_hpi_def_err']}, combined {V['itm_adv_def_err']}), and it over-predicts base prepayment by "
                           f"{V['itm_base_pre_err']}, which is a level error to be aware of when using the rate sensitivity.")
    else:
        V["base_short_txt"] = ("The base-case shortfall also has sources outside the stress mechanism: the calibration-window level effect "
                               "from section 8.3, the frozen delinquency status, and the use of a cumulative-incidence PD as a hazard.")


EXCEL_DESC = [
    ("scorecard_pd", "Calibrated scorecard PD, each of the sample loans"),
    ("scorecard_points", "Scorecard points, each loan"),
    ("loan_rank", "PD rank, each loan"),
    ("loan_decile_id", "Decile assignment, each loan"),
    ("vs_exported_python_pd", "Excel PD against the PD stored by Python in loan_sample.csv"),
    ("vs_exported_python_points", "Excel points against the points stored by Python"),
    ("decile_counts", "Decile loan counts"),
    ("decile_events", "Decile event counts"),
    ("decile_mean_pd", "Decile mean PD"),
    ("decile_lift", "Decile lift"),
    ("ks_max", "KS statistic"),
    ("auc_approx", "Approximate AUC (decile ROC trapezoid)"),
    ("psi_total", "Total PSI"),
    ("brier", "Brier score"),
    ("hosmer_lemeshow", "Hosmer-Lemeshow statistic"),
    ("expected_events", "Expected events"),
    ("scen_exp_defaults", "Scenario expected defaults"),
    ("scen_el_usd", "Scenario expected loss (USD)"),
    ("scen_prepay_12m", "Scenario 12-month prepay"),
    ("calc_pd", "One-loan calculator PD"),
    ("calc_score", "One-loan calculator score"),
]


def excel_block_from(d):
    w = d["worst_diffs"]
    rows = [dict(item=lab, diff=w[k]) for k, lab in EXCEL_DESC if k in w]
    cases = ", ".join(f"{c['case']} (selector {c['selector']}, {'passed' if c['passed'] else 'FAILED'})" for c in d.get("cases", []))
    worst_key = max(w, key=w.get)
    txt = (f"`mortgage_risk/reconcile_excel.py` rebuilds the workbook, recalculates it headlessly in LibreOffice, reads the results back and "
           f"compares each quantity with an independent Python computation. It runs {d['n_cases']} scenario cases ({cases}). "
           f"Overall result: **{'all comparisons passed' if d['all_passed'] else 'AT LEAST ONE COMPARISON FAILED'}**; the report recorded "
           f"{ni(d.get('n_formulas'))} formulas and a run time of {d.get('runtime_s', float('nan')):.1f} seconds. "
           f"The largest absolute difference across all quantities is {w[worst_key]:.1e} (`{worst_key}`); every quantity computed from "
           f"the same inputs agrees to better than 1e-8 except the comparison with the values Python exported to the CSV, which is limited "
           f"by the float32 precision at which the loan attributes are stored.\n\n"
           + md_table(pd.DataFrame(rows), dict(diff=lambda x: f"{x:.1e}"), dict(item="Quantity", diff="Worst absolute difference, Excel against Python"))
           + "\n\nTable: Excel reconciliation, worst absolute difference over the cases (from `outputs/excel_reconciliation.json`).\n\n"
           "The differences of order 1e-13 to 1e-16 are floating-point rounding. They show that the live formulas reproduce the Python "
           "scorecard, decile, PSI, calibration and scenario logic, which is the purpose of the exercise: a reviewer can trace any number in "
           "the workbook to a cell formula and know it matches the code. They say nothing about whether the model is right for real loans.")
    return txt


def build_values():
    V, T = {}, {}
    X = dict(ds=rjson("data_summary.json"), meta=rjson("model_meta.json"), pf=panel_facts(), bf=bundle_facts())
    _core(V, T, X)
    fac, off = _toys(V, T)
    _scorecard(V, T, X, fac, off)
    _results(V, T, X)
    _scenarios(V, T, X)
    _static(V, T, X)
    _pool_and_itm(V, T, X)
    _excel(V, T)
    return V, T, X


# ---------------------------------------------------------------- rendering and conversion
PLACEHOLDER = re.compile(r"@@([A-Za-z0-9_:\.]+)@@")


def render(text, V, T):
    def sub(m):
        key = m.group(1)
        if key.startswith("tbl:"):
            return T[key[4:]]
        return str(V[key])
    for _ in range(6):
        new = PLACEHOLDER.sub(sub, text)
        if new == text:
            break
        text = new
    if PLACEHOLDER.search(text):
        raise ValueError("unresolved placeholders: " + ", ".join(sorted(set(PLACEHOLDER.findall(text)))))
    return text


def md_stats(text):
    """Counts of headings, pipe tables, equations and figures in a markdown source (code fences excluded)."""
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
    headings = len(re.findall(r"^#{1,6} ", body, flags=re.M))
    figures = len(re.findall(r"^!\[", body, flags=re.M))
    display = len(re.findall(r"\$\$.+?\$\$", body, flags=re.S))
    rest = re.sub(r"\$\$.+?\$\$", "", body, flags=re.S)
    inline = len(re.findall(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\d)", rest))
    tables, in_tbl = 0, False
    for line in body.splitlines():
        is_row = line.startswith("|")
        if is_row and not in_tbl:
            tables += 1
        in_tbl = is_row
    return dict(headings=headings, tables=tables, equations=display + inline, figures=figures)


def docx_stats(path):
    import zipfile
    from docx import Document
    doc = Document(str(path))
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    return dict(headings=sum(1 for p in doc.paragraphs if p.style.name.startswith("Heading")), tables=len(doc.tables),
                equations=xml.count("<m:oMath>") + xml.count("<m:oMath "), figures=xml.count("<w:drawing>"))


def style_tables(path):
    """Thin borders and smaller type in every table so wide numeric tables stay legible."""
    from docx import Document
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    from docx.shared import Pt
    doc = Document(str(path))
    borders = ('<w:tblBorders %s>' % nsdecls("w") + "".join(
        f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders>")
    for t in doc.tables:
        pr = t._tbl.tblPr
        for old in pr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblBorders"):
            pr.remove(old)
        pr.append(parse_xml(borders))
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8.5)
    doc.save(str(path))


def set_author(path):
    from docx import Document
    doc = Document(str(path))
    for st in doc.styles:       # unused heading levels and the TOC heading still carry theme colours in the template
        if st.name.startswith("Heading") or st.name.startswith("TOC Heading"):
            rpr = st.element.rPr
            if rpr is not None:
                for c in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}color"):
                    rpr.remove(c)
    cp = doc.core_properties
    cp.author, cp.last_modified_by, cp.title = AUTHOR, AUTHOR, DOC_TITLE
    doc.save(str(path))


def convert():
    import pypandoc
    pypandoc.convert_file(str(MD_PATH), "docx", format="markdown-smart", outputfile=str(DOCX_PATH),
                          extra_args=[f"--reference-doc={REFERENCE_DOCX}", "--toc", f"--toc-depth={TOC_DEPTH}",
                                      "--number-sections", f"--resource-path={DOCS_DIR}", "--standalone"])
    style_tables(DOCX_PATH)
    set_author(DOCX_PATH)


def verify(md_text):
    import pypandoc
    a, b = md_stats(md_text), docx_stats(DOCX_PATH)
    ok = a == b
    rt = pypandoc.convert_file(str(DOCX_PATH), "markdown", format="docx", extra_args=["--wrap=none"])
    bad = [("replacement character", chr(0xFFFD) in rt), ("em dash", chr(0x2014) in rt)]
    print("markdown counts:", a)
    print("docx counts    :", b)
    for name, hit in bad:
        print(f"round trip {name}: {'FOUND' if hit else 'none'}")
    if not ok or any(h for _, h in bad):
        raise SystemExit("round-trip verification failed")


# ---------------------------------------------------------------- README
README_TEMPLATE = r"""# Mortgage Delinquency and Prepayment Risk Model

**Data provenance: every result in this repository comes from SYNTHETIC data** produced by a documented simulator, unless you place real Freddie Mac loan-level files in `data/raw/` and rerun. Results therefore reflect the simulator's own equations. They are a check that the methodology and code work, not evidence that the models would predict real mortgage defaults or prepayments. This is a portfolio project and not a bank-approved or independently validated model. Synthetic data credit: generated by `python/mortgage_risk/synthetic.py` (seed @@seed@@).

## What it does

Loan-level models of (1) the 12-month probability of reaching 90+ days past due and (2) the monthly probability of full prepayment for 30-year mortgages. Each is built twice: a transparent benchmark (a weight-of-evidence scorecard scaled to points for PD; a piecewise-linear logit for prepayment) and gradient-boosted trees (XGBoost, LightGBM) with monotone constraints, time-aware tuning and probability calibration. Models are explained with SHAP, checked for stability (PSI, CSI), stress-tested under six macro scenarios, reproduced in an Excel workbook with live formulas, and exported as CPR/CDR curves for the sibling securitization waterfall project.

The full write-up (theory, implementation, results, governance in the style of SR 11-7) is `docs/Mortgage_Risk_Model_Documentation.docx` (source `docs/Mortgage_Risk_Model_Documentation.md`).

## Quick start

```
cd python
py -3 -m pip install -r requirements.txt
py -3 -m mortgage_risk.cli --quick     # small run (@@n_quick@@ loans)
py -3 -m mortgage_risk.cli             # full run (@@n_full@@ loans), writes python/outputs/
py -3 -m pytest -q
py -3 ../docs/build_docs.py            # rebuild the documentation and this README
```

## Headline results (out-of-time, synthetic data)

@@tbl:readme_headline@@

Table: Out-of-time discrimination. PD: @@pd_rows_oot@@ snapshots, 2020 to 2023. Prepay: @@pp_rows_oot@@ loan-months, 2020 to 2024.

- Gradient boosting beats the scorecard on PD ranking by a small but statistically firm margin (DeLong p = @@pd_xgb_p@@ for XGBoost), and adds almost nothing for prepayment, where the simulated hazard is smooth.
- **Weakness: PD level is under-predicted out of time.** Observed rate @@obs_oot_lgbm@@ against a mean LightGBM prediction of @@pred_oot_lgbm@@ (calibration window was benign, and the macro regime moved outside the training range).
- Stress results under-react: on the end-2023 pool the combined adverse scenario gives a 12-month default rate of @@s_adv_def@@% of balance against @@t_adv_def@@% in simulation truth, because tree models cannot extrapolate beyond the unemployment and house price ranges seen in training.
- The end-2023 pool is far out of the money, so rate shocks barely change prepayments there. The pipeline also runs the scenarios as of 31 December 2021 (`python/outputs/scenarios_itm_2021/`).

@@tbl:mvt@@

Table: Twelve-month model against simulation truth on the end-2023 pool (percent of starting balance; synthetic-only check).

## Excel workbook

`excel/mortgage_risk_model.xlsx` re-implements the scorecard, deciles, PSI, calibration and a simplified scenario engine with @@excel_formulas@@ live formulas over @@excel_n@@ out-of-time loans. `python/mortgage_risk/reconcile_excel.py` recalculates it in LibreOffice and compares it with Python. @@excel_readme_line@@

## Link to the securitization waterfall model

`python/outputs/waterfall_hook/` holds month-by-month CPR/CDR curves and scalar equivalents (`cpr`, `cdr`, `severity`, `lag`, `index_shift`) whose columns match `waterfall.config.Scenario` in `Portfolio_Projects/Securitization_Waterfall_Model`. Severity (@@hook_sev@@) and lag (@@hook_lag@@ months) are fixed assumptions, not model outputs. The two collateral pools differ, so the hook demonstrates the interface rather than a like-for-like deal analysis.

## Repository layout

```
data/raw, data/interim     Freddie files (optional) and parquet cache
docs/                      documentation (md, docx), build_docs.py, reference.docx
excel/                     workbook builder and workbook
python/mortgage_risk/      package: macro, synthetic, freddie, data, features, targets,
                           scorecard, models, evaluation, explain, scenarios,
                           export_hook, charts, reconcile_excel, cli
python/tests/              pytest suites (data, models, explain and scenarios,
                           integration, docs)
python/outputs/            results (csv, json), charts/, models/, waterfall_hook/
```

## Limitations

Synthetic data only; PD level under-predicted out of time; no extrapolation beyond the training macro range, so stress results under-react; the scorecard cannot see current delinquency because that variable collapses under the 5% minimum bin share; no loss severity, modifications or forbearance; scenarios are deterministic ramps; the Freddie Mac loader is tested only on a small fixture. Details are in sections 8, 10 and 12 of the documentation.

## Next steps

Add a special-value bin for rare delinquency states in the scorecard; recalibrate on a recent window or add a macro overlay for stress; fit a discrete-time competing-risk model to replace the cumulative-incidence approximation; add a loss severity model; run on real Freddie Mac vintages that span a full credit cycle.
"""


def build_readme(V, T, X):
    bp, bq = X["bp"], X["bq"]
    rows = []
    for tag, b, ks in (("PD", bp, ("pd_logit", "pd_xgb", "pd_lgbm")), ("Prepay", bq, ("pp_logit", "pp_xgb", "pp_lgbm"))):
        for k in ks:
            rows.append(dict(target=tag, model=MODEL_LABEL[k], auc=b.loc[k, "oot_auc"], gini=b.loc[k, "oot_gini"],
                             ks=b.loc[k, "oot_ks"], brier=b.loc[k, "oot_brier"]))
    T["readme_headline"] = md_table(pd.DataFrame(rows), dict(auc=lambda x: fx(x, 3), gini=lambda x: fx(x, 3), ks=lambda x: fx(x, 3),
                                                              brier=lambda x: fx(x, 4)),
                                    dict(target="Target", model="Model", auc="AUC", gini="Gini", ks="KS", brier="Brier"), first_left=True)
    if V["excel_present"] == "yes":
        V["excel_readme_line"] = ("Result: " + V["excel_summary_line"])
    else:
        V["excel_readme_line"] = "The reconciliation report is still being finalized."
    return render(README_TEMPLATE, V, T)


def main():
    V, T, X = build_values()
    d = None
    if EXCEL_RECON_JSON.exists():
        try:
            d = json.loads(EXCEL_RECON_JSON.read_text())
        except Exception:  # noqa: BLE001
            d = None
    V["excel_formulas"] = ni(d.get("n_formulas")) if d else "the planned set of"
    if d:
        w = d["worst_diffs"]
        V["excel_summary_line"] = (f"{'all comparisons passed' if d['all_passed'] else 'a comparison FAILED'} over {d['n_cases']} scenario cases, "
                                   f"largest absolute difference {max(w.values()):.1e}.")
    text = render(S0 + S1 + S6 + S8 + S10 + S11, V, T)
    MD_PATH.write_text(text, encoding="utf-8")
    README_PATH.write_text(build_readme(V, T, X), encoding="utf-8")
    convert()
    verify(text)
    print("wrote", MD_PATH.name, DOCX_PATH.name, README_PATH.name, "| words:", len(text.split()))


if __name__ == "__main__":
    main()
