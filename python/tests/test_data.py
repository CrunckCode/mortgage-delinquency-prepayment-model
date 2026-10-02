"""Data layer tests: synthetic DGP, features, targets, splits, Freddie loader, column-literal hygiene."""
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mortgage_risk import columns as C, config, data, features, freddie, macro, synthetic, targets

# ===== CONFIG (user inputs) =====
N_TEST_LOANS = 4_000
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_ORIG, FIXTURE_PERF = str(FIXTURES / "sample_orig_fixture.txt"), str(FIXTURES / "sample_svcg_fixture.txt")
NO_MACRO_GLOB = str(FIXTURES / "no_macro_here" / "*.csv")
PKG_DIR = Path(__file__).resolve().parents[1] / "mortgage_risk"
AST_EXEMPT = {"columns.py", "freddie.py", "__init__.py"}
# ===== END CONFIG =====

REQUIRED = [v for k, v in vars(C).items() if k.isupper() and isinstance(v, str)
            and k not in ("PRED_PD", "PRED_SMM", "TARGET_PD12", "TARGET_PREPAY_1M", "SNAPSHOT_DATE", "MTM_LTV",
                          "INCENTIVE", "BURNOUT", "SEASONING_RAMP", "HPI_CHG_12M", "UNEMP_CHG_12M",
                          "TIMES_30DPD_12M", "LOG_UPB", "FICO_BAND", "MONTH_OF_YEAR")]


@pytest.fixture(scope="session")
def panel():
    return synthetic.simulate_panel(N_TEST_LOANS)


@pytest.fixture(scope="session")
def pf(panel):
    return features.add_features(panel)


@pytest.fixture(scope="session")
def snaps(pf):
    return targets.build_snapshots(pf)


def test_schema(panel):
    assert set(REQUIRED) <= set(panel.columns)
    assert panel.groupby(C.LOAN_ID).size().min() >= 1
    ended = (panel[C.PREPAY_FLAG] + panel[C.DEFAULT_FLAG]).to_numpy()
    assert ended.max() == 1, "a row has both a prepay and a default flag"
    per_loan = panel.groupby(C.LOAN_ID)[[C.PREPAY_FLAG, C.DEFAULT_FLAG]].sum()
    assert per_loan.max().max() <= 1
    # no rows after termination: terminal flag may only sit on a loan's last row
    ended_s = pd.Series(ended, index=panel.index)
    prior_end = ended_s.groupby(panel[C.LOAN_ID]).cumsum() - ended_s
    assert (prior_end == 0).all()
    first_bad = panel[panel[C.DEFAULT_FLAG] == 1]
    assert (first_bad[C.DLQ_STATUS] == 3).all() and (first_bad[C.ZB_CODE] == 3).all()
    assert (panel[panel[C.PREPAY_FLAG] == 1][C.ZB_CODE] == 1).all()
    assert (panel[C.TRUE_HP].between(0, 1)).all() and (panel[C.TRUE_HD].between(0, 1)).all()
    assert panel[C.PERIOD].max() <= config.OBS_END
    assert (panel[C.AGE] >= 1).all()


def test_calibration(panel):
    rep = synthetic.calibration_report(panel)
    cal = config.SYNTH_CALIBRATION
    assert cal["pd12_train"][0] <= rep["pd12_train"] <= cal["pd12_train"][1]
    assert cal["pd12_oot_2020"][0] <= rep["pd12_oot_2020"] <= cal["pd12_oot_2020"][1]
    assert cal["cpr"][0] <= rep["cpr_mean"] <= cal["cpr"][1]
    by_year = rep["cpr_by_year"]
    assert min(by_year[2020], by_year[2021]) > max(by_year[2018], by_year[2023])


def test_macro_and_shock():
    m = macro.build_macro(pd.Timestamp("2019-01-31"), pd.Timestamp("2022-12-31"))
    assert set(m.columns) == {C.PERIOD, C.STATE, C.MKT_RATE, C.HPI, C.UNEMP}
    ca = m[m[C.STATE] == "CA"].set_index(C.PERIOD)
    assert ca.loc["2020-05-31", C.UNEMP] > 10 and ca.loc["2019-12-31", C.UNEMP] < 6
    shock = config.Shock("x", rate_bp=200, hpi_pct=-20, unemp_pt=4, ramp_months=12)
    s = macro.apply_shock(m, shock, pd.Timestamp("2021-01-31")).set_index([C.STATE, C.PERIOD])
    base = m.set_index([C.STATE, C.PERIOD])
    early, late = ("CA", pd.Timestamp("2020-12-31")), ("CA", pd.Timestamp("2022-12-31"))
    assert s.loc[early, C.MKT_RATE] == base.loc[early, C.MKT_RATE]
    assert s.loc[late, C.MKT_RATE] == pytest.approx(base.loc[late, C.MKT_RATE] + 2.0, abs=1e-4)
    assert s.loc[late, C.HPI] == pytest.approx(base.loc[late, C.HPI] * 0.8, rel=1e-4)
    assert s.loc[late, C.UNEMP] == pytest.approx(base.loc[late, C.UNEMP] + 4.0, abs=1e-4)


def test_feature_values(pf):
    assert (pf[C.BURNOUT].groupby(pf[C.LOAN_ID]).diff().dropna() >= 0).all()
    assert pf[C.TIMES_30DPD_12M].between(0, 12).all()
    assert (pf[C.INCENTIVE] == pf[C.CUR_RATE] - pf[C.MKT_RATE]).all()
    first = pf.groupby(C.LOAN_ID).head(1)
    assert first[C.MTM_LTV].between(30, 130).all()
    assert (pf[C.SEASONING_RAMP] <= 1).all()
    assert set(C.FEATURES_PD + C.FEATURES_PREPAY) <= set(pf.columns)


def test_features_backward_only(panel, pf):
    ids = panel.groupby(C.LOAN_ID).size()
    loan = ids[ids >= 40].index[0]
    rows = panel[C.LOAN_ID] == loan
    cut = panel.loc[rows, C.PERIOD].iloc[20]
    future = rows & (panel[C.PERIOD] > cut)
    pert = panel.copy()
    pert.loc[future, C.DLQ_STATUS] = 2
    pert.loc[future, C.MKT_RATE] += 3.0
    pert.loc[future, C.UNEMP] += 5.0
    pert.loc[future, C.HPI] *= 0.5
    pert.loc[future, C.CUR_UPB] *= 0.5
    pf2 = features.add_features(pert)
    cols = [c for c in pf.columns if c in pf2.columns]
    a = pf[(pf[C.LOAN_ID] == loan) & (pf[C.PERIOD] <= cut)][cols].reset_index(drop=True)
    b = pf2[(pf2[C.LOAN_ID] == loan) & (pf2[C.PERIOD] <= cut)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert not pf[(pf[C.LOAN_ID] == loan) & (pf[C.PERIOD] > cut)][C.INCENTIVE].equals(
        pf2[(pf2[C.LOAN_ID] == loan) & (pf2[C.PERIOD] > cut)][C.INCENTIVE])


def test_assert_no_leakage(pf):
    features.assert_no_leakage(pf, C.FEATURES_PD)
    features.assert_no_leakage(pf, C.FEATURES_PREPAY, asof_col=C.PERIOD)
    for bad in (*C.TRUE_COLUMNS, C.DEFAULT_FLAG, C.PREPAY_FLAG, C.ZB_CODE, C.TARGET_PD12, C.TARGET_PREPAY_1M):
        with pytest.raises(AssertionError):
            features.assert_no_leakage(pf, [*C.FEATURES_PD, bad])


def test_snapshots_and_pd_splits(snaps):
    assert len(snaps) > 5_000 and snaps[C.TARGET_PD12].between(0, 1).all()
    assert snaps[C.DLQ_STATUS].isin([0, 1]).all()
    assert snaps[C.SNAPSHOT_DATE].dt.month.isin(config.SNAPSHOT_MONTHS).all()
    horizon = pd.DateOffset(months=config.PD_HORIZON_MONTHS)
    assert (snaps[C.SNAPSHOT_DATE] + horizon <= config.OBS_END).all()
    sp = targets.split_by_time(snaps, "pd")
    assert set(sp) == {"train", "calib", "oot"} and all(len(v) > 0 for v in sp.values())
    assert sp["train"][C.SNAPSHOT_DATE].max() < sp["calib"][C.SNAPSHOT_DATE].min()
    assert sp["calib"][C.SNAPSHOT_DATE].max() < sp["oot"][C.SNAPSHOT_DATE].min()
    for k in ("train", "calib"):
        assert (sp[k][C.SNAPSHOT_DATE] + horizon <= config.DEV_CUTOFF).all()
    assert sum(len(v) for v in sp.values()) <= len(snaps)


def test_snapshot_label_matches_events(pf, snaps):
    s = snaps.sample(300, random_state=1)
    firsts = pf[pf[C.DEFAULT_FLAG] == 1].set_index(C.LOAN_ID)[C.PERIOD]
    for _, r in s.iterrows():
        d = firsts.get(r[C.LOAN_ID], pd.NaT)
        hit = pd.notna(d) and r[C.SNAPSHOT_DATE] < d <= r[C.SNAPSHOT_DATE] + pd.DateOffset(months=12)
        assert int(hit) == r[C.TARGET_PD12]


def test_hazard_rows_and_prepay_splits(pf):
    hz = targets.build_hazard_rows(pf, loan_share=0.5)
    assert hz[C.DLQ_STATUS].eq(0).all()
    assert hz[C.TARGET_PREPAY_1M].mean() > 0
    assert set(hz[C.LOAN_ID]) <= set(pf[C.LOAN_ID]) and hz[C.LOAN_ID].nunique() < pf[C.LOAN_ID].nunique()
    full = targets.build_hazard_rows(pf)
    # features come from t-1: AGE on a hazard row is one less than the panel AGE at t
    key = full[[C.LOAN_ID, C.PERIOD, C.AGE]].merge(pf[[C.LOAN_ID, C.PERIOD, C.AGE]], on=[C.LOAN_ID, C.PERIOD],
                                                   suffixes=("_h", "_t"))
    assert (key[C.AGE + "_t"] - key[C.AGE + "_h"] == 1).all()
    sp = targets.split_by_time(full, "prepay")
    assert sp["train"][C.PERIOD].max() < sp["calib"][C.PERIOD].min() <= sp["calib"][C.PERIOD].max() \
        < sp["oot"][C.PERIOD].min()
    assert sp["calib"][C.PERIOD].max() <= config.DEV_CUTOFF
    assert sp["oot"][C.TARGET_PREPAY_1M].mean() > 0
    assert sp["oot"][C.TRUE_HP].between(0, 1).all()


def test_freddie_loader_fixture(caplog):
    with caplog.at_level(logging.WARNING):
        df = freddie.load_freddie(FIXTURE_ORIG, FIXTURE_PERF, macro_glob=NO_MACRO_GLOB)
    assert any("synthetic macro" in r.message for r in caplog.records)
    assert df[C.LOAN_ID].nunique() == 5
    assert not set(C.TRUE_COLUMNS) & set(df.columns)
    assert set(REQUIRED) - set(C.TRUE_COLUMNS) <= set(df.columns)
    g = lambda lid: df[df[C.LOAN_ID] == lid].reset_index(drop=True)
    l1, l2, l3, l4, l5 = (g(f"F12Q100000{i}") for i in range(1, 6))
    assert l1[C.PREPAY_FLAG].tolist() == [0, 0, 0, 1] and l1[C.ZB_CODE].iloc[-1] == 1
    assert l1[C.CUR_UPB].iloc[-1] > 0 and l1[C.OCCUPANCY].iloc[0] == "O"
    assert l1[C.ORIG_DATE].iloc[0] == pd.Timestamp("2012-03-31") and l1[C.VINTAGE].iloc[0] == 2012
    assert l2[C.DLQ_STATUS].tolist() == [0, 1, 2, 3]            # status 4 row dropped after the default
    assert l2[C.DEFAULT_FLAG].tolist() == [0, 0, 0, 1]
    assert l3[C.DLQ_STATUS].tolist() == [0, 1, 3] and l3[C.DEFAULT_FLAG].sum() == 1   # 'RA' maps to 90+
    assert l4[C.ZB_CODE].iloc[-1] == 3 and l4[C.DEFAULT_FLAG].tolist() == [0, 0, 0, 1]
    assert l4[C.PREPAY_FLAG].sum() == 0
    assert l5[C.DEFAULT_FLAG].sum() == 0 and l5[C.PREPAY_FLAG].sum() == 0 and l5[C.FICO].iloc[0] > 0
    assert df[C.MKT_RATE].notna().all() and df[C.HPI_ORIG].notna().all()
    assert freddie.load_freddie(FIXTURE_ORIG, FIXTURE_PERF, max_loans=2, macro_glob=NO_MACRO_GLOB)[C.LOAN_ID].nunique() == 2
    ft = features.add_features(df)
    assert set(C.FEATURES_PD) <= set(ft.columns)
    assert len(targets.build_hazard_rows(ft)) > 0


def test_get_panel_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path)
    p1, m1, meta = data.get_panel("synthetic", n_loans=300)
    assert meta["source"] == "synthetic" and meta["n_loans"] <= 300 and meta["seed"] == config.SEED
    assert any(f.name.startswith("panel_synthetic_300_") for f in tmp_path.iterdir())
    p2, _, _ = data.get_panel("synthetic", n_loans=300)
    pd.testing.assert_frame_equal(p1, p2)
    assert set(m1.columns) == {C.PERIOD, C.STATE, C.MKT_RATE, C.HPI, C.UNEMP}


def _literal_strings(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for e in node.elts:
            yield from _literal_strings(e)


def test_no_column_string_literals():
    names = {v for k, v in vars(C).items() if k.isupper() and isinstance(v, str)}
    offenders = []
    for path in sorted(PKG_DIR.glob("*.py")):
        if path.name in AST_EXEMPT:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            hits = []
            if isinstance(node, ast.Subscript):
                hits = list(_literal_strings(node.slice))
            elif isinstance(node, ast.keyword):
                hits = list(_literal_strings(node.value)) + ([node.arg] if node.arg else [])
            offenders += [f"{path.name}:{getattr(node, 'lineno', '?')} {h!r}" for h in hits if h in names]
    assert not offenders, offenders


def test_no_em_dashes():
    bad = [p.name for p in list(PKG_DIR.glob("*.py")) + [Path(__file__)]
           if chr(0x2014) in p.read_text(encoding="utf-8")]
    assert not bad
