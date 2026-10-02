"""End-to-end pipeline: py -3 -m mortgage_risk.cli [--quick] [--skip-excel]"""
import json
import logging
import sys
import time

import pandas as pd

from . import config, data, evaluation, explain, export_hook, features, models, scenarios, targets
from . import charts
from . import columns as C

# ===== CONFIG (user inputs) =====
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
ITM_ASOF = pd.Timestamp("2021-12-31")
ITM_DIR = "scenarios_itm_2021"
# ===== END CONFIG =====


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    t0 = time.time()
    step = lambda msg: logging.info("[%5.0fs] %s", time.time() - t0, msg)

    step("building panel")
    panel, macro, meta = data.get_panel(n_loans=config.quick_n(quick), refresh=True)
    feat = features.add_features(panel, macro)
    snaps = targets.build_snapshots(feat)
    hazard = targets.build_hazard_rows(feat, loan_share=config.PREPAY_LOAN_SAMPLE_SHARE)
    splits_pd, splits_pp = targets.split_by_time(snaps, "pd"), targets.split_by_time(hazard, "prepay")
    (config.OUT_DIR / "data_summary.json").write_text(json.dumps(dict(
        data_source=meta["source"], n_loans=meta["n_loans"], panel_rows=len(panel), snapshot_rows=len(snaps),
        hazard_rows=len(hazard), pd_event_rate=float(snaps[C.TARGET_PD12].mean()),
        split_rows_pd={k: len(v) for k, v in splits_pd.items()},
        split_rows_prepay={k: len(v) for k, v in splits_pp.items()}), indent=1))

    step("training models")
    bundles = models.train_all(splits_pd, splits_pp, quick=quick)
    step("evaluating")
    evaluation.evaluate_all(bundles, splits_pd, splits_pp, data_source=meta["source"])
    step("explaining (SHAP)")
    expl = explain.run_explain(bundles["pd_lgbm"], bundles["pp_xgb"], splits_pd["oot"], splits_pp["oot"])
    step("scenarios")
    scen = scenarios.run_scenarios(bundles["pd_lgbm"], bundles["pp_xgb"], feat, macro)
    export_hook.write_curves(scen["curves"])
    # in-the-money pool as-of date so rate shocks are informative (end-2023 pool is out of the money)
    scenarios.run_scenarios(bundles["pd_lgbm"], bundles["pp_xgb"], feat, macro, out_dir=config.OUT_DIR / ITM_DIR,
                            asof=ITM_ASOF)
    step("charts")
    charts.make_all_charts(bundles, splits_pd, splits_pp, explain_out=expl)
    if "--skip-excel" not in argv:
        step("excel workbook and reconciliation")
        from . import reconcile_excel
        reconcile_excel.build_and_reconcile()
    step("done")


if __name__ == "__main__":
    main()
