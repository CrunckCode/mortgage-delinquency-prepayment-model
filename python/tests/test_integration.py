"""Guards on the committed pipeline outputs and repo hygiene (reads outputs; never reruns the pipeline)."""
import json
from pathlib import Path
import pandas as pd
import pytest
from mortgage_risk import config

OUT = config.OUT_DIR
ROOT = config.ROOT
EM_DASH = chr(0x2014)
TEXT_SUFFIXES = (".py", ".md", ".csv", ".json", ".txt")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "models"}

pytestmark = pytest.mark.skipif(not (OUT / "benchmark_pd.csv").exists(), reason="pipeline outputs not generated")


def test_oot_auc_meets_thresholds():
    pdb = pd.read_csv(OUT / "benchmark_pd.csv").set_index("model")
    ppb = pd.read_csv(OUT / "benchmark_prepay.csv").set_index("model")
    t = config.TEST_THRESHOLDS
    assert pdb.loc["pd_lgbm", "oot_auc"] > t["pd_oot_auc"] and pdb.loc["pd_xgb", "oot_auc"] > t["pd_oot_auc"]
    assert ppb[["oot_auc"]].drop(index=[i for i in ppb.index if i.endswith("logit")]).min().iloc[0] > t["prepay_oot_auc"]
    assert pdb.loc["pd_xgb", "oot_auc"] >= pdb.loc["pd_logit", "oot_auc"] - t["gbm_vs_logit_slack"]


def test_dgp_alignment_all_passed():
    al = pd.read_csv(OUT / "dgp_alignment.csv")
    assert al["passed"].all(), al[~al["passed"]]


def test_in_the_money_rate_ordering():
    s = pd.read_csv(OUT / "scenarios_itm_2021" / "scenario_12m.csv").set_index("scenario")
    assert s.loc["Rates -200bp", "avg_cpr_12m"] > s.loc["Base", "avg_cpr_12m"] > s.loc["Rates +200bp", "avg_cpr_12m"]
    assert s.loc["Adverse combined", "default_12m_pct_upb"] >= s["default_12m_pct_upb"].drop("Adverse combined").max() - 1e-9


def test_curves_feed_waterfall_contract():
    cur = pd.read_csv(OUT / "waterfall_hook" / "cpr_cdr_curves.csv")
    assert {"scenario", "month", "smm", "mdr", "cpr_annual", "cdr_annual", "survival"} <= set(cur.columns)
    assert cur["month"].max() == config.HOOK_MONTHS
    assert cur[["smm", "mdr"]].min().min() >= 0 and cur[["smm", "mdr"]].max().max() <= 1


def test_outputs_declare_data_source():
    meta = json.loads((OUT / "model_meta.json").read_text())
    assert meta.get("data_source") in ("synthetic", "freddie")
    assert json.loads((OUT / "data_summary.json").read_text())["data_source"] in ("synthetic", "freddie")


def test_no_em_dash_in_repo_text_files():
    bad = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in TEXT_SUFFIXES and not (set(p.parts) & SKIP_DIRS) and "data" not in p.relative_to(ROOT).parts[:1]:
            if p.name == "Project_Notes.md":
                continue
            if EM_DASH in p.read_text(encoding="utf-8", errors="ignore"):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad


def test_gitignore_keeps_raw_data_and_models_out():
    text = (ROOT / ".gitignore").read_text()
    for needle in ("data/raw/*", "data/interim/", "python/outputs/models/", "Project_Notes.md"):
        assert needle in text
