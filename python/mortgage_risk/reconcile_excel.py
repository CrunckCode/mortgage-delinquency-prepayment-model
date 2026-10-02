"""Independent Python recomputation of the Excel workbook plus LibreOffice recalculation and comparison.

Run: py -3 -m mortgage_risk.reconcile_excel [--quick]   (needs LibreOffice for the recalculation step)
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from . import config
from .columns import (BURNOUT, SEASONING_RAMP, AGE, DTI, FICO, INCENTIVE, LOAN_ID, MTM_LTV, ORIG_LTV, CUR_UPB, TARGET_PD12, UNEMP,
                      UNEMP_CHG_12M, FEATURES_PD)

# ===== CONFIG (user inputs) =====
EXCEL_DIR = config.ROOT / "excel"
BUILDER = EXCEL_DIR / "build_workbook.py"
XLSX = EXCEL_DIR / "mortgage_risk_model.xlsx"
META = XLSX.with_suffix(".meta.json")
REPORT = config.OUT_DIR / "excel_reconciliation.json"
SOFFICE_CANDIDATES = [r"C:\Program Files\LibreOffice\program\soffice.exe",
                      r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
AUTHOR = "Deepak Chaudhary"
TOL_PD, TOL_POINTS, TOL_USD, TOL_SMM, TOL_KS, TOL_PSI, TOL_MISC = 1e-9, 1e-6, 0.01, 1e-10, 1e-9, 1e-9, 1e-9
QUICK_SCENARIOS = (1, 6)
LO_TIMEOUT_S = 900
BURNOUT_SEARCH = (0.0, 600.0)      # sample has no burnout history; one fixed value is solved to match the Python GBM base prepay
BURNOUT_ROUND = 1
SEASONING_RAMP_MONTHS = 30.0       # models.features: min(AGE / 30, 1)
N_DECILES = 10
CALC_DEFAULTS = {FICO: 740.0, ORIG_LTV: 80.0, MTM_LTV: 70.0, DTI: 36.0, INCENTIVE: 0.0, AGE: 24.0,
                 UNEMP: 4.5, UNEMP_CHG_12M: 0.0}
SHOCK_VARS = (UNEMP, UNEMP_CHG_12M, MTM_LTV, INCENTIVE)     # scorecard variables a macro shock moves
# scorecard_table.csv and prepay_logit_coefs.csv headers
VAR_COL, LO_COL, HI_COL, WOE_COL, COEF_COL, POINTS_COL = "variable", "bin_lo", "bin_hi", "woe", "coef", "points"
TERM_COL, PP_COEF_COL, KNOT_COL = "term", "coef", "knot"
# psi_train_bins.csv and scenario_12m.csv headers
PSI_LO_COL, PSI_HI_COL, PSI_SHARE_COL = "lo", "hi", "train_share"
SCEN_COL, TARGET_COL = "scenario", "prepay_12m_pct_upb"
# ===== END CONFIG =====

SAMPLE_PATH = config.EXCEL_INPUT_DIR / "loan_sample.csv"


# ---------------------------------------------------------------- inputs
def load_tables():
    table = pd.read_csv(config.OUT_DIR / "scorecard_table.csv")
    sample = pd.read_csv(SAMPLE_PATH)
    # the sample was scored in float32; restore that precision so bin edges classify identically
    for c in sample.columns:
        if sample[c].dtype == np.float64 and c not in (CUR_UPB,):
            sample[c] = sample[c].astype(np.float32).astype(np.float64)
    coefs = pd.read_csv(config.OUT_DIR / "prepay_logit_coefs.csv")
    psi_bins = pd.read_csv(config.EXCEL_INPUT_DIR / "psi_train_bins.csv")
    return table, sample, coefs, psi_bins


def load_model_params() -> dict:
    """Scalars that are not in any csv (intercept, Platt calibrators); read from the saved bundles."""
    import joblib
    from . import evaluation, models
    pd_b = joblib.load(config.MODEL_DIR / "pd_logit.joblib")
    pp_b = joblib.load(config.MODEL_DIR / "pp_logit.joblib")
    for b in (pd_b, pp_b):
        if b.calibrator is None or b.calibrator.method != "platt":
            raise RuntimeError("workbook reproduces Platt calibration only; refit or extend the builder")
    sc = pd_b.scorecard
    p = dict(intercept=float(sc.intercept), factor=float(sc.factor), offset=float(sc.offset),
                pd_a=float(pd_b.calibrator.model.coef_[0, 0]), pd_b=float(pd_b.calibrator.model.intercept_[0]),
                pp_a=float(pp_b.calibrator.model.coef_[0, 0]), pp_b=float(pp_b.calibrator.model.intercept_[0]),
                prob_clip=float(models.PROB_CLIP), mtm_cap=float(models.MTM_LTV_CAP),
                psi_floor=float(evaluation.PSI_FLOOR), lgd=float(config.LGD_ASSUMPTION),
                season_months=SEASONING_RAMP_MONTHS, **{BURNOUT: 0.0})
    p[BURNOUT] = solve_burnout(p)
    return p


def solve_burnout(p) -> float:
    """Fixed BURNOUT (months) that makes the sample's base 12-month prepay equal the Python GBM base figure."""
    _, sample, coefs, _ = load_tables()
    target = float(pd.read_csv(config.OUT_DIR / "scenario_12m.csv").set_index(SCEN_COL).loc[config.SHOCKS[0].name, TARGET_COL]) / 100.0
    upb = sample[CUR_UPB].to_numpy(dtype=np.float64)
    lo, hi = BURNOUT_SEARCH
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        smm = prepay_smm(sample, coefs, dict(p, **{BURNOUT: mid}))
        cpr = 1.0 - (1.0 - np.sum(smm * upb) / upb.sum()) ** 12
        lo, hi = (mid, hi) if cpr > target else (lo, mid)
    return round(0.5 * (lo + hi), BURNOUT_ROUND)


# ---------------------------------------------------------------- scorecard maths
def make_bins(table):
    out = {}
    for v, g in table.groupby(VAR_COL, sort=False):
        real = g[g[LO_COL].notna()]
        miss = g[g[LO_COL].isna()]
        out[v] = dict(edges=real[HI_COL].to_numpy()[:-1], woe=real[WOE_COL].to_numpy(),
                      pts=real[POINTS_COL].to_numpy(), miss_woe=float(miss[WOE_COL].iloc[0]),
                      miss_pts=float(miss[POINTS_COL].iloc[0]), coef=float(real[COEF_COL].iloc[0]))
    return out


def woe_of(x, b):
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(b["edges"], x, side="left")
    return np.where(np.isnan(x), b["miss_woe"], b["woe"][idx])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def score(frame, bins, p):
    margin = np.full(len(frame), p["intercept"])
    woes = {}
    for v, b in bins.items():
        woes[v] = woe_of(frame[v].to_numpy(), b)
        margin = margin + b["coef"] * woes[v]
    pdv = np.clip(sigmoid(p["pd_a"] * margin + p["pd_b"]), p["prob_clip"], 1 - p["prob_clip"])
    return dict(woe=woes, margin=margin, pd=pdv, points=p["offset"] - p["factor"] * margin)


def stress(frame, shock):
    f = frame.copy()
    f[UNEMP] = f[UNEMP] + shock.unemp_pt
    f[UNEMP_CHG_12M] = f[UNEMP_CHG_12M] + shock.unemp_pt
    f[MTM_LTV] = f[MTM_LTV] / (1.0 + shock.hpi_pct / 100.0)
    f[INCENTIVE] = f[INCENTIVE] - shock.rate_bp / 100.0
    return f


def prepay_smm(frame, coefs, p):
    c = dict(zip(coefs[TERM_COL], coefs[PP_COEF_COL]))
    knots = dict(zip(coefs[TERM_COL], coefs[KNOT_COL]))
    x = frame[INCENTIVE].to_numpy(dtype=np.float64)
    z = np.full(len(frame), c["const"])
    for t, k in knots.items():
        if str(t).startswith("hinge_"):
            z = z + c[t] * np.maximum(x - k, 0.0)
    z = (z + c[BURNOUT] * p[BURNOUT]
         + c[SEASONING_RAMP] * np.minimum(frame[AGE].to_numpy(dtype=np.float64) / p["season_months"], 1.0)
         + c[MTM_LTV] * np.minimum(frame[MTM_LTV].to_numpy(dtype=np.float64), p["mtm_cap"])
         + c[FICO] * frame[FICO].to_numpy(dtype=np.float64))
    return np.clip(sigmoid(p["pp_a"] * z + p["pp_b"]), p["prob_clip"], 1 - p["prob_clip"])


def portfolio_summary(frame, bins, coefs, p, shock):
    f = stress(frame, shock)
    s = score(f, bins, p)
    smm = prepay_smm(f, coefs, p)
    upb = frame[CUR_UPB].to_numpy(dtype=np.float64)
    smm_w = float(np.sum(smm * upb) / upb.sum())
    return dict(exp_defaults=float(s["pd"].sum()), pd_w=float(np.sum(s["pd"] * upb) / upb.sum()),
                el_usd=float(np.sum(s["pd"] * p["lgd"] * upb)), el_pct=float(np.sum(s["pd"] * p["lgd"] * upb) / upb.sum()),
                smm_w=smm_w, prepay_12m=1.0 - (1.0 - smm_w) ** 12, pd=s["pd"], smm=smm, total_upb=float(upb.sum()))


# ---------------------------------------------------------------- decile / PSI / calibration
def decile_reference(y, pdv):
    """Same ranking as evaluation.decile_table (descending, stable ties, array_split groups)."""
    order = np.argsort(-pdv, kind="mergesort")
    rank = np.empty(len(pdv), dtype=int)
    rank[order] = np.arange(1, len(pdv) + 1)
    groups = np.array_split(np.arange(len(pdv)), N_DECILES)
    dec = np.empty(len(pdv), dtype=int)
    for d, g in enumerate(groups, start=1):
        dec[order[g]] = d
    n = np.array([len(g) for g in groups], float)
    ev = np.array([y[order[g]].sum() for g in groups], float)
    mean_pd = np.array([pdv[order[g]].mean() for g in groups])
    cum1, cum0 = np.cumsum(ev) / ev.sum(), np.cumsum(n - ev) / (n.sum() - ev.sum())
    ks = np.abs(cum1 - cum0)
    f_prev, t_prev, auc = 0.0, 0.0, 0.0
    for t, f in zip(cum1, cum0):
        auc += (f - f_prev) * (t + t_prev) / 2.0
        f_prev, t_prev = f, t
    return dict(rank=rank, decile=dec, n=n, events=ev, mean_pd=mean_pd, event_rate=ev / n, cum1=cum1, cum0=cum0,
                ks=ks, ks_max=float(ks.max()), auc_approx=float(auc), gini_approx=float(2 * auc - 1),
                lift=(ev / n) / (ev.sum() / n.sum()))


def psi_reference(pdv, psi_bins, floor):
    cuts = psi_bins[PSI_HI_COL].to_numpy()[:-1]
    idx = np.searchsorted(cuts, pdv, side="left")
    actual = np.bincount(idx, minlength=len(cuts) + 1) / len(pdv)
    expected = psi_bins[PSI_SHARE_COL].to_numpy()
    e, a = np.maximum(expected, floor), np.maximum(actual, floor)
    per_bin = (a - e) * np.log(a / e)
    return dict(actual=actual, expected=expected, per_bin=per_bin, total=float(per_bin.sum()))


def calibration_reference(y, pdv, dec):
    brier = float(np.mean((pdv - y) ** 2))
    base = float(y.mean())
    hl = 0.0
    for d in range(1, N_DECILES + 1):
        m = dec["decile"] == d
        n, o, pbar = m.sum(), y[m].sum(), pdv[m].mean()
        hl += (o - n * pbar) ** 2 / (n * pbar * (1 - pbar))
    return dict(brier=brier, brier_base=base * (1 - base), hl=float(hl), exp_events=float(pdv.sum()),
                events=float(y.sum()))


def calc_reference(bins, p, inputs=None):
    inputs = inputs or CALC_DEFAULTS
    margin, rows = p["intercept"], []
    for v, b in bins.items():
        x = np.array([inputs[v]], float)
        idx = int(np.searchsorted(b["edges"], x, side="left")[0])
        pts = float(b["pts"][idx])
        margin += b["coef"] * float(b["woe"][idx])
        rows.append((v, pts, float(b["pts"].max()) - pts))
    pdv = float(np.clip(sigmoid(p["pd_a"] * margin + p["pd_b"]), p["prob_clip"], 1 - p["prob_clip"]))
    top = sorted(range(len(rows)), key=lambda i: (-rows[i][2], i))[:3]
    return dict(pd=pdv, score=p["offset"] - p["factor"] * margin, margin=margin,
                top3=[(rows[i][0], rows[i][2]) for i in top])


def reference_all(table=None, sample=None, coefs=None, psi_bins=None, p=None):
    if table is None:
        table, sample, coefs, psi_bins = load_tables()
    p = p or load_model_params()
    bins = make_bins(table)
    y = sample[TARGET_PD12].to_numpy(dtype=np.float64)
    base = portfolio_summary(sample, bins, coefs, p, config.Shock("Base"))
    dec = decile_reference(y, base["pd"])
    out = dict(params=p, bins=bins, base=base, dec=dec, psi=psi_reference(base["pd"], psi_bins, p["psi_floor"]),
               cal=calibration_reference(y, base["pd"], dec), calc=calc_reference(bins, p),
               scen=[portfolio_summary(sample, bins, coefs, p, s) for s in config.SHOCKS])
    out["auc_exact"] = float(_auc(y, base["pd"]))
    return out


def _auc(y, p):
    from scipy import stats
    r = stats.rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (len(y) - n1))


# ---------------------------------------------------------------- LibreOffice
def find_soffice():
    for c in SOFFICE_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def load_builder():
    spec = importlib.util.spec_from_file_location("mortgage_build_workbook", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recalc(src: Path, outdir: Path, soffice: str) -> Path:
    shutil.rmtree(outdir, ignore_errors=True)
    subprocess.run([soffice, "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(outdir), str(src)],
                   check=True, capture_output=True, timeout=LO_TIMEOUT_S)
    return outdir / src.name


def patch_author(path: Path, author: str = AUTHOR):
    tmp = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                x = data.decode("utf-8")
                for tag in ("dc:creator", "cp:lastModifiedBy"):
                    if re.search(rf"<{tag}>.*?</{tag}>", x, flags=re.S):
                        x = re.sub(rf"<{tag}>.*?</{tag}>", f"<{tag}>{author}</{tag}>", x, flags=re.S)
                    elif re.search(rf"<{tag}\s*/>", x):
                        x = re.sub(rf"<{tag}\s*/>", f"<{tag}>{author}</{tag}>", x)
                    else:
                        x = x.replace("</cp:coreProperties>", f"<{tag}>{author}</{tag}></cp:coreProperties>")
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


# ---------------------------------------------------------------- comparison
def _col(ws, letter, first, last):
    return np.array([ws[f"{letter}{r}"].value for r in range(first, last + 1)], dtype=object)


def _num(a):
    return np.array([np.nan if v is None or isinstance(v, str) else float(v) for v in a], dtype=float)


def compare(label, sel, wbv, meta, ref, sample):
    a = meta["addr"]
    ls = wbv["Loan_Sample"]
    first, last = meta["first"], meta["last"]
    cols = meta["ls_cols"]
    issues, worst = [], {}

    def track(key, diff, tol):
        diff = float(diff)
        worst[key] = max(worst.get(key, 0.0), diff if np.isfinite(diff) else np.inf)
        if not diff <= tol:
            issues.append(f"{key}: diff {diff:.3e} > {tol:.0e}")

    def cell(name):
        sheet, ref_ = a[name].split("!")
        return wbv[sheet][ref_.replace("$", "")].value

    base = ref["base"]
    track("scorecard_pd", np.nanmax(np.abs(_num(_col(ls, cols["xl_pd"], first, last)) - base["pd"])), TOL_PD)
    track("scorecard_points",
          np.nanmax(np.abs(_num(_col(ls, cols["xl_points"], first, last)) - score(sample, ref["bins"], ref["params"])["points"])),
          TOL_POINTS)
    track("loan_decile_id", np.nanmax(np.abs(_num(_col(ls, cols["xl_decile"], first, last)) - ref["dec"]["decile"])), 0)
    track("loan_rank", np.nanmax(np.abs(_num(_col(ls, cols["xl_rank"], first, last)) - ref["dec"]["rank"])), 0)
    # export consistency (informational tolerance: csv is float32-derived)
    track("vs_exported_python_pd",
          np.nanmax(np.abs(_num(_col(ls, cols["xl_pd"], first, last)) - sample["scorecard_pd"].to_numpy())), 1e-6)
    track("vs_exported_python_points",
          np.nanmax(np.abs(_num(_col(ls, cols["xl_points"], first, last)) - sample["scorecard_points"].to_numpy())), 1e-3)

    d = wbv["Deciles"]
    dec = ref["dec"]
    rows = range(meta["deciles"]["first"], meta["deciles"]["first"] + N_DECILES)
    track("decile_counts", np.abs(_num([d[f"B{r}"].value for r in rows]) - dec["n"]).max(), 0)
    track("decile_events", np.abs(_num([d[f"C{r}"].value for r in rows]) - dec["events"]).max(), 0)
    track("decile_event_rate", np.abs(_num([d[f"D{r}"].value for r in rows]) - dec["event_rate"]).max(), TOL_MISC)
    track("decile_mean_pd", np.abs(_num([d[f"E{r}"].value for r in rows]) - dec["mean_pd"]).max(), TOL_PD)
    track("decile_lift", np.abs(_num([d[f"G{r}"].value for r in rows]) - dec["lift"]).max(), TOL_MISC)
    track("decile_ks", np.abs(_num([d[f"H{r}"].value for r in rows]) - dec["ks"]).max(), TOL_KS)
    track("ks_max", abs(cell("ks_max") - dec["ks_max"]), TOL_KS)
    track("auc_approx", abs(cell("auc_approx") - dec["auc_approx"]), TOL_MISC)
    track("gini_approx", abs(cell("gini_approx") - dec["gini_approx"]), TOL_MISC)

    ps = wbv["PSI"]
    prow = range(meta["psi"]["first"], meta["psi"]["first"] + N_DECILES)
    track("psi_actual_share", np.abs(_num([ps[f"E{r}"].value for r in prow]) - ref["psi"]["actual"]).max(), TOL_PSI)
    track("psi_total", abs(cell("psi_total") - ref["psi"]["total"]), TOL_PSI)

    cal = ref["cal"]
    track("brier", abs(cell("brier") - cal["brier"]), TOL_MISC)
    track("hosmer_lemeshow", abs(cell("hl") - cal["hl"]), 1e-7)
    track("expected_events", abs(cell("exp_events") - cal["exp_events"]), 1e-7)

    sc = ref["scen"][sel - 1]
    for key, pyv, tol in (("exp_defaults", sc["exp_defaults"], TOL_USD), ("el_usd", sc["el_usd"], TOL_USD),
                          ("smm_w", sc["smm_w"], TOL_SMM), ("prepay_12m", sc["prepay_12m"], TOL_SMM),
                          ("pd_w", sc["pd_w"], TOL_PD)):
        track("scen_" + key, abs(cell("sel_" + key) - pyv), tol)
    for key, pyv, tol in (("exp_defaults", base["pd"].sum(), TOL_USD), ("el_usd", base["el_usd"], TOL_USD),
                          ("smm_w", ref["scen"][0]["smm_w"], TOL_SMM)):
        track("base_" + key, abs(cell("base_" + key) - pyv), tol)
    sc_sheet = wbv["Scenario"]
    sf, sl = meta["scen_first"], meta["scen_last"]
    track("scen_loan_pd", np.nanmax(np.abs(_num(_col(sc_sheet, meta["scen_cols"]["pd_s"], sf, sl)) - sc["pd"])), TOL_PD)
    track("scen_loan_smm", np.nanmax(np.abs(_num(_col(sc_sheet, meta["scen_cols"]["smm_s"], sf, sl)) - sc["smm"])), TOL_SMM)

    cr = ref["calc"]
    track("calc_pd", abs(cell("calc_pd") - cr["pd"]), TOL_PD)
    track("calc_score", abs(cell("calc_score") - cr["score"]), TOL_POINTS)
    top = [(cell(f"calc_top{k}_var"), cell(f"calc_top{k}_gap")) for k in (1, 2, 3)]
    for (v, g), (rv, rg) in zip(top, cr["top3"]):
        if v != rv:
            issues.append(f"calc top variable {v} != {rv}")
        track("calc_top_gap", abs(g - rg), TOL_POINTS)
    if cell("checks_all") != 1:
        issues.append("workbook Checks sheet does not report all pass")
    return dict(case=label, selector=sel, passed=not issues, worst_diffs=worst, issues=issues)


# ---------------------------------------------------------------- driver
def build_and_reconcile(quick: bool = False) -> dict:
    t0 = time.time()
    soffice = find_soffice()
    if soffice is None:
        msg = "LibreOffice not found; install it (winget install TheDocumentFoundation.LibreOffice). Skipping recalculation."
        print(msg)
        return dict(all_passed=None, skipped=True, reason=msg)
    builder = load_builder()
    table, sample, coefs, psi_bins = load_tables()
    ref = reference_all(table, sample, coefs, psi_bins)
    scenarios = QUICK_SCENARIOS if quick else tuple(range(1, len(config.SHOCKS) + 1))
    work = Path(tempfile.mkdtemp(prefix="mortgage_xl_"))
    results, shipped = [], None
    for sel in scenarios:
        src = work / f"case{sel}.xlsx"
        meta = builder.build(src, selector=sel, write_meta=(sel == scenarios[0]))
        out = recalc(src, work / f"out{sel}", soffice)
        wbv = openpyxl.load_workbook(out, data_only=True)
        res = compare(config.SHOCKS[sel - 1].name, sel, wbv, meta, ref, sample)
        results.append(res)
        print(("PASS" if res["passed"] else "FAIL"), res["case"], res["issues"][:3])
        if sel == scenarios[0]:
            shipped = out
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(shipped, XLSX)
    patch_author(XLSX)
    worst = {}
    for r in results:
        for k, v in r["worst_diffs"].items():
            worst[k] = max(worst.get(k, 0.0), v)
    report = dict(data_source="synthetic", all_passed=all(r["passed"] for r in results), n_cases=len(results),
                  worst_diffs=worst, cases=results, runtime_s=round(time.time() - t0, 1),
                  n_formulas=meta.get("n_formulas"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=float))
    print("all passed:", report["all_passed"], "runtime", report["runtime_s"], "s")
    return report


def main():
    build_and_reconcile(quick="--quick" in sys.argv)


if __name__ == "__main__":
    main()
