"""Global configuration shared by all modules. Change values here, not inside modules."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# ===== CONFIG (user inputs) =====
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR, INTERIM_DIR = ROOT / "data" / "raw", ROOT / "data" / "interim"
OUT_DIR = ROOT / "python" / "outputs"
CHART_DIR, MODEL_DIR = OUT_DIR / "charts", OUT_DIR / "models"
HOOK_DIR, EXCEL_INPUT_DIR = OUT_DIR / "waterfall_hook", OUT_DIR / "excel_inputs"

SEED = 20261001
N_LOANS_FULL, N_LOANS_QUICK = 60_000, 4_000
ORIG_START, ORIG_END = pd.Timestamp("2012-01-31"), pd.Timestamp("2021-12-31")
OBS_END = pd.Timestamp("2024-12-31")

# time split: every development label must be observable by DEV_CUTOFF
DEV_CUTOFF = pd.Timestamp("2019-12-31")
TRAIN_PD_SNAP = (pd.Timestamp("2013-01-31"), pd.Timestamp("2018-06-30"))
CALIB_PD_SNAP = (pd.Timestamp("2018-07-31"), pd.Timestamp("2018-12-31"))
OOT_PD_SNAP = (pd.Timestamp("2020-01-31"), pd.Timestamp("2023-12-31"))
TRAIN_PP = (pd.Timestamp("2013-01-31"), pd.Timestamp("2018-12-31"))
CALIB_PP = (pd.Timestamp("2019-01-31"), pd.Timestamp("2019-12-31"))
OOT_PP = (pd.Timestamp("2020-01-31"), pd.Timestamp("2024-12-31"))
SNAPSHOT_MONTHS = (1, 7)
PD_HORIZON_MONTHS = 12

STATES = ("CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "AZ", "NV")
STATE_WEIGHTS = (0.22, 0.14, 0.11, 0.09, 0.07, 0.07, 0.07, 0.09, 0.08, 0.06)
STATE_HPI_BETA = (1.25, 0.95, 1.20, 0.90, 0.80, 0.85, 0.70, 0.95, 1.30, 1.45)
STATE_UNEMP_OFFSET = (0.8, -0.3, 0.2, 0.4, 0.5, 0.0, 0.3, 0.1, 0.4, 1.2)

# true data generating process (synthetic mode); tests reference these
TRUE_DGP = dict(
    prepay=dict(a0=-5.6, a_ref=3.4, a_burn=0.03, a_seas=1.0, a_fico=0.15, a_size=0.25,
                incentive_mid=0.5, incentive_scale=0.25, ltv_cap=90.0, month_amp=0.12),
    to30=dict(b0=-5.6, b_fico=-0.99, b_ltv=0.81, b_dti=0.36, b_unemp=0.324, b_hpi=-0.54, b_age=0.63),
    roll30=dict(base=0.30, cure=0.55, fico_k=-0.36, ltv_k=0.27),
    roll60=dict(base=0.50, cure=0.20),
)
SYNTH_CALIBRATION = dict(pd12_train=(0.008, 0.025), pd12_oot_2020=(0.015, 0.045), cpr=(0.04, 0.40))

# model and scorecard settings
PREPAY_LOAN_SAMPLE_SHARE = 0.25
TUNE_ITER, TUNE_FOLDS, TUNE_GAP_MONTHS = 12, 3, 12
SCORECARD_PDO, SCORECARD_BASE_SCORE, SCORECARD_BASE_ODDS = 20, 600, 50
SCORECARD_IV_MIN = 0.02
PSI_BANDS = (0.10, 0.25)
BOOTSTRAP_N = 500
SHAP_SAMPLE = 5_000
EXCEL_SAMPLE_LOANS = 2_000

# scenarios
SEVERITY_ASSUMPTION, LAG_ASSUMPTION, LGD_ASSUMPTION = 0.35, 6, 0.35
PROJECTION_MONTHS = 60
HOOK_MONTHS = 120

TEST_THRESHOLDS = dict(pd_oot_auc=0.75, prepay_oot_auc=0.70, gbm_vs_logit_slack=0.01,
                       model_vs_truth_rel_err=0.25, brier_calib_rel=0.10)
# ===== END CONFIG =====


@dataclass(frozen=True)
class Shock:
    name: str
    rate_bp: float = 0.0
    hpi_pct: float = 0.0
    unemp_pt: float = 0.0
    ramp_months: int = 12


SHOCKS = (
    Shock("Base"), Shock("Rates +200bp", rate_bp=200), Shock("Rates -200bp", rate_bp=-200),
    Shock("HPI -20%", hpi_pct=-20), Shock("Unemployment +4pt", unemp_pt=4),
    Shock("Adverse combined", rate_bp=200, hpi_pct=-20, unemp_pt=4),
)

REASON_TEXT = {
    "fico": "Credit score lower than typical", "mtm_ltv": "High current loan-to-value",
    "dti": "High debt-to-income", "unemp": "Elevated local unemployment",
    "times_30dpd_12m": "Recent 30-day delinquencies", "dlq_status": "Currently delinquent",
    "hpi_chg_12m": "Falling home prices", "incentive": "Rate incentive", "age": "Loan age",
    "orig_ltv": "High original loan-to-value", "orig_rate": "High note rate",
}


def quick_n(quick: bool) -> int:
    return N_LOANS_QUICK if quick else N_LOANS_FULL
