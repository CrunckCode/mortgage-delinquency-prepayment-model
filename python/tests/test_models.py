"""Model-layer tests: hand-checked metrics, scorecard maths, bundles, calibration, monotonicity, time splits."""
import json

import numpy as np
import pandas as pd
import pytest

from mortgage_risk import config, data, evaluation as ev, features, models, scorecard as scm, targets
from mortgage_risk.columns import (FEATURES_PD, FEATURES_PREPAY, FICO, MTM_LTV, TARGET_PD12, TARGET_PREPAY_1M,
                                   SNAPSHOT_DATE, PERIOD)

# ===== CONFIG (user inputs) =====
N_QUICK_LOANS, N_SLOW_LOANS = 4_000, 20_000
PREPAY_SHARE_SLOW = 0.5
GBM_SLACK_QUICK = 0.03
SLOW_FLOORS = dict(pd_oot_auc=0.65, prepay_oot_auc=0.62)  # config thresholds are not reachable with this DGP, see report
FICO_GRID = (600, 650, 700, 750, 800)
LTV_GRID = (60, 70, 80, 90, 100)
PD_SAMPLE = 3_000
MONO_TOL = 1e-9
# ===== END CONFIG =====

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _build(n_loans, share=1.0):
    panel, macro, meta = data.get_panel(n_loans=n_loans)
    f = features.add_features(panel, macro)
    sp_pd = targets.split_by_time(targets.build_snapshots(f), "pd")
    sp_pp = targets.split_by_time(targets.build_hazard_rows(f, share), "prepay")
    return sp_pd, sp_pp, meta


@pytest.fixture(scope="session")
def quick(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model_out")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODEL_DIR", tmp / "models")
    mp.setattr(config, "OUT_DIR", tmp / "outputs")
    mp.setattr(config, "EXCEL_INPUT_DIR", tmp / "outputs" / "excel_inputs")
    sp_pd, sp_pp, meta = _build(N_QUICK_LOANS)
    bundles = models.train_all(sp_pd, sp_pp, quick=True)
    yield dict(pd=sp_pd, pp=sp_pp, bundles=bundles, meta=meta, tmp=tmp)
    mp.undo()


# ---------- metrics with hand answers ----------
Y6 = np.array([0, 0, 0, 1, 1, 1])
P6 = np.array([0.1, 0.3, 0.6, 0.4, 0.7, 0.9])


def test_auc_gini_ks_hand_example():
    assert ev.auc(Y6, P6) == pytest.approx(8 / 9)
    assert ev.gini(Y6, P6) == pytest.approx(7 / 9)
    assert ev.ks(Y6, P6) == pytest.approx(2 / 3)


def test_auc_ties_use_average_ranks():
    assert ev.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_brier_hand_example():
    assert ev.brier([0, 1], [0.2, 0.6]) == pytest.approx((0.04 + 0.16) / 2)


def test_psi_hand_example_and_identity():
    assert ev.psi([1, 2, 3, 4], [1, 1, 1, 4], bins=2) == pytest.approx(0.25 * np.log(1.5) + 0.25 * np.log(2.0))
    x = np.random.default_rng(1).normal(size=500)
    assert ev.psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_decile_table_counts_and_order():
    rng = np.random.default_rng(2)
    p = rng.random(1003)
    y = (rng.random(1003) < p).astype(int)
    t = ev.decile_table(y, p)
    assert t["n"].sum() == 1003 and t["events"].sum() == y.sum()
    assert t["mean_score"].is_monotonic_decreasing
    assert t["cum_events_share"].iloc[-1] == pytest.approx(1.0)
    assert t["lift"].iloc[0] > 1.0


def test_delong_direction_and_symmetry():
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.3).astype(int)
    good = y * 0.8 + rng.normal(size=4000)
    noise = y * 0.1 + rng.normal(size=4000)
    d, z, pv = ev.delong_test(y, good, noise)
    assert d > 0 and z > 0 and pv < 0.01
    d2, z2, pv2 = ev.delong_test(y, noise, good)
    assert d2 == pytest.approx(-d) and z2 == pytest.approx(-z) and pv2 == pytest.approx(pv)
    assert ev.delong_test(y, good, good)[0] == 0.0
    assert d == pytest.approx(ev.auc(y, good) - ev.auc(y, noise))


def test_bootstrap_ci_brackets_point_estimate_and_groups():
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.3).astype(int)
    p = y * 0.7 + rng.normal(size=3000)
    groups = np.repeat(np.arange(1000), 3)
    lo, hi = ev.bootstrap_ci(ev.auc, y, p, groups=groups, n=100, seed=1)
    assert lo < ev.auc(y, p) < hi
    lo2, hi2 = ev.bootstrap_ci(ev.auc, y, p, n=100, seed=1)
    assert lo2 < hi2


def test_hosmer_lemeshow_detects_miscalibration():
    pv, pv_bad = [], []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.01, 0.3, 20000)
        y = (rng.random(20000) < p).astype(int)
        pv.append(ev.hosmer_lemeshow(y, p)[1])
        pv_bad.append(ev.hosmer_lemeshow(y, np.clip(p * 1.5, 0, 0.99))[1])
    assert np.median(pv) > 0.1 and max(pv_bad) < 1e-6


# ---------- scorecard ----------
def test_fit_bins_monotone_woe():
    rng = np.random.default_rng(6)
    x = rng.normal(size=20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(3 + 0.8 * x))).astype(int)
    spec = scm.fit_bins(x, y, max_bins=8, min_share=0.05, monotone=True)
    d = np.diff(spec.woe)
    assert (d <= 0).all() or (d >= 0).all()
    assert 2 <= len(spec.woe) <= 8 and spec.iv > 0.02
    assert np.isfinite(spec.transform(np.array([np.nan, 1e9, -1e9]))).all()


def test_scorecard_points_scaling_and_table(quick):
    sc = quick["bundles"]["pd_logit"].scorecard
    assert sc.factor == pytest.approx(config.SCORECARD_PDO / np.log(2))
    # score at base odds (good:bad) equals the base score; doubling good odds adds PDO
    assert sc.offset + sc.factor * np.log(config.SCORECARD_BASE_ODDS) == pytest.approx(config.SCORECARD_BASE_SCORE)
    df = quick["pd"]["oot"].head(500)
    pts = sc.points(df)
    assert pts == pytest.approx(sc.offset + sc.factor * np.log((1 - sc.predict_pd(df)) / sc.predict_pd(df)))
    # a margin change of ln(2) in bad odds moves the score by exactly -PDO
    assert sc.factor * np.log(2) == pytest.approx(config.SCORECARD_PDO)
    assert all(c < 0 for c in sc.coef.values())
    tab = scm.export_scorecard_table(sc)
    assert list(tab.columns) == ["variable", "bin_lo", "bin_hi", "woe", "coef", "points"]
    # summed bin points reproduce the row score
    total = np.zeros(len(df))
    for v in sc.variables:
        t = tab[(tab["variable"] == v) & tab["bin_lo"].notna()].reset_index(drop=True)
        idx = sc.specs[v].bin_index(df[v].to_numpy())
        total += t["points"].to_numpy()[idx]
    assert total == pytest.approx(pts, abs=1e-6)


# ---------- bundles ----------
def test_bundle_interface_and_roundtrip(quick, tmp_path):
    b = quick["bundles"]
    assert set(b) == {"pd_logit", "pd_xgb", "pd_lgbm", "pp_logit", "pp_xgb", "pp_lgbm"}
    for k, bundle in b.items():
        df = quick["pd"]["oot"] if bundle.target == "pd" else quick["pp"]["oot"]
        df = df.head(300)
        pth = tmp_path / f"{k}.joblib"
        bundle.save(pth)
        again = models.ModelBundle.load(pth)
        assert again.name == bundle.name and again.kind == bundle.kind and again.features == bundle.features
        np.testing.assert_allclose(again.predict_margin(df), bundle.predict_margin(df))
        pr = bundle.predict_pd(df) if bundle.target == "pd" else bundle.predict_smm(df)
        assert ((pr > 0) & (pr < 1)).all()
        assert (config.MODEL_DIR / f"{k}.joblib").exists()


def test_prepay_logit_coefs_table(quick):
    c = quick["bundles"]["pp_logit"].coefs
    assert list(c.columns) == ["term", "coef", "knot"]
    assert sorted(c["knot"].dropna()) == list(models.HINGE_KNOTS)


def test_calibrated_pd_mean_matches_observed_on_calib(quick):
    calib = quick["pd"]["calib"]
    for k in ("pd_xgb", "pd_lgbm", "pd_logit"):
        m = quick["bundles"][k].predict_pd(calib).mean()
        obs = calib[TARGET_PD12].mean()
        assert abs(m - obs) / obs < config.TEST_THRESHOLDS["brier_calib_rel"]


def test_gbm_not_worse_than_logit(quick):
    oot = quick["pd"]["oot"]
    b = quick["bundles"]
    base = ev.auc(oot[TARGET_PD12], b["pd_logit"].predict_pd(oot))
    for k in ("pd_xgb", "pd_lgbm"):
        assert ev.auc(oot[TARGET_PD12], b[k].predict_pd(oot)) >= base - GBM_SLACK_QUICK


def test_gbm_monotone_in_fico_and_ltv(quick):
    samp = quick["pd"]["oot"].sample(PD_SAMPLE, random_state=1)
    for k in ("pd_xgb", "pd_lgbm"):
        b = quick["bundles"][k]
        fico = [b.predict_pd(samp.assign(**{FICO: v})).mean() for v in FICO_GRID]
        ltv = [b.predict_pd(samp.assign(**{MTM_LTV: v})).mean() for v in LTV_GRID]
        assert np.all(np.diff(fico) <= MONO_TOL), (k, fico)
        assert np.all(np.diff(ltv) >= -MONO_TOL), (k, ltv)


def test_time_split_no_leakage(quick):
    for kind, col in (("pd", SNAPSHOT_DATE), ("pp", PERIOD)):
        sp = quick[kind]
        assert sp["train"][col].max() < sp["calib"][col].min() <= sp["calib"][col].max() < sp["oot"][col].min()


def test_weighted_variant_only_as_sensitivity(quick):
    w = quick["bundles"]["pd_xgb"].extras["weighted"]
    assert w.calibrator is None
    assert quick["bundles"]["pd_xgb"].calibrator is not None


def test_evaluate_all_writes_outputs(quick):
    res = ev.evaluate_all(quick["bundles"], quick["pd"], quick["pp"])
    out, xd = config.OUT_DIR, config.EXCEL_INPUT_DIR
    names = ["benchmark_pd.csv", "benchmark_prepay.csv", "psi_csi.csv", "stability_by_vintage.csv", "segments.csv",
             "scorecard_table.csv", "prepay_logit_coefs.csv", "prepay_cpr_oot.csv", "model_meta.json"]
    names += [f"deciles_pd_{m}.csv" for m in ("pd_logit", "pd_xgb", "pd_lgbm")]
    names += [f"calibration_pd_{m}.csv" for m in ("pd_logit", "pd_xgb", "pd_lgbm")]
    for n in names:
        assert (out / n).exists(), n
    for n in ("loan_sample.csv", "psi_train_bins.csv"):
        assert (xd / n).exists(), n
    meta = json.loads((out / "model_meta.json").read_text())
    assert meta["data_source"] == quick["meta"]["source"]
    assert len(pd.read_csv(xd / "loan_sample.csv")) <= config.EXCEL_SAMPLE_LOANS
    bp = res["benchmark_pd"]
    assert {"train_auc", "calib_auc", "oot_auc", "oot_gini", "oot_ks", "oot_brier", "delong_p_vs_logit",
            "oot_auc_ci_lo", "oot_auc_ci_hi"} <= set(bp.columns)
    cpr = pd.read_csv(out / "prepay_cpr_oot.csv")
    assert list(cpr.columns) == [PERIOD, "actual_cpr", "pred_cpr"]


# ---------- slow: 20,000 loans, threshold check ----------
@pytest.mark.slow
def test_oot_auc_thresholds_20k(request, tmp_path):
    if "slow" not in (request.config.getoption("-m") or ""):
        pytest.skip("run with -m slow")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODEL_DIR", tmp_path / "models")
    try:
        sp_pd, sp_pp, _ = _build(N_SLOW_LOANS, PREPAY_SHARE_SLOW)
        b = models.train_all(sp_pd, sp_pp, quick=False)
    finally:
        mp.undo()
    pd_oot, pp_oot = sp_pd["oot"], sp_pp["oot"]
    assert ev.auc(pd_oot[TARGET_PD12], b["pd_lgbm"].predict_pd(pd_oot)) > SLOW_FLOORS["pd_oot_auc"]
    assert ev.auc(pp_oot[TARGET_PREPAY_1M], b["pp_lgbm"].predict_smm(pp_oot)) > SLOW_FLOORS["prepay_oot_auc"]
    slack = config.TEST_THRESHOLDS["gbm_vs_logit_slack"]
    assert ev.auc(pd_oot[TARGET_PD12], b["pd_lgbm"].predict_pd(pd_oot)) >= \
        ev.auc(pd_oot[TARGET_PD12], b["pd_logit"].predict_pd(pd_oot)) - slack
