"""Column-name constants. Every other module must import names from here; no string-literal column names elsewhere."""

# keys and time
LOAN_ID, PERIOD, ORIG_DATE, VINTAGE, AGE = "loan_id", "period", "orig_date", "vintage", "age"

# origination attributes
FICO, ORIG_LTV, ORIG_CLTV, DTI = "fico", "orig_ltv", "orig_cltv", "dti"
ORIG_RATE, ORIG_UPB, ORIG_TERM, STATE = "orig_rate", "orig_upb", "orig_term", "state"
PURPOSE, OCCUPANCY, PROP_TYPE = "purpose", "occupancy", "prop_type"
N_BORROWERS, FIRST_TIME_BUYER, CHANNEL, MI_PCT = "n_borrowers", "first_time_buyer", "channel", "mi_pct"

# monthly performance
CUR_UPB, CUR_RATE = "cur_upb", "cur_rate"
DLQ_STATUS, DLQ_BUCKET = "dlq_status", "dlq_bucket"      # status 0,1,2,3 (3 = 90+ DPD)
PREPAY_FLAG, DEFAULT_FLAG, ZB_CODE, IS_ACTIVE = "prepay_flag", "default_flag", "zb_code", "is_active"
DLQ_BUCKETS = ("C", "30", "60", "90+")

# macro (state-month)
MKT_RATE, HPI, HPI_ORIG, UNEMP = "mkt_rate", "hpi", "hpi_orig", "unemp"

# derived features
MTM_LTV, INCENTIVE, BURNOUT = "mtm_ltv", "incentive", "burnout"
SEASONING_RAMP, HPI_CHG_12M, UNEMP_CHG_12M = "seasoning_ramp", "hpi_chg_12m", "unemp_chg_12m"
TIMES_30DPD_12M, LOG_UPB, FICO_BAND, MONTH_OF_YEAR = "times_30dpd_12m", "log_upb", "fico_band", "month_of_year"

# targets and snapshots
TARGET_PD12, TARGET_PREPAY_1M, SNAPSHOT_DATE = "target_pd12", "target_prepay_1m", "snapshot_date"

# simulation truth, test use only; excluded from features by the leakage guard
TRUE_HP, TRUE_HD = "true_hp", "true_hd"
TRUE_COLUMNS = (TRUE_HP, TRUE_HD)

# scored outputs
PRED_PD, PRED_SMM = "pred_pd", "pred_smm"

# groupings
ORIG_FEATURES = [FICO, ORIG_LTV, ORIG_CLTV, DTI, ORIG_RATE, ORIG_UPB, STATE, PURPOSE, OCCUPANCY,
                 N_BORROWERS, FIRST_TIME_BUYER]
PERF_FEATURES = [CUR_UPB, CUR_RATE, DLQ_STATUS, AGE, TIMES_30DPD_12M]
MACRO_FEATURES = [MKT_RATE, HPI, UNEMP]

FEATURES_PD = [FICO, ORIG_LTV, MTM_LTV, DTI, ORIG_RATE, INCENTIVE, AGE, LOG_UPB, UNEMP, UNEMP_CHG_12M,
               HPI_CHG_12M, TIMES_30DPD_12M, DLQ_STATUS, PURPOSE, OCCUPANCY, N_BORROWERS, STATE]
FEATURES_PREPAY = [INCENTIVE, BURNOUT, AGE, SEASONING_RAMP, MTM_LTV, FICO, LOG_UPB, MONTH_OF_YEAR, PURPOSE, STATE, DTI]
CATEGORICAL_FEATURES = [PURPOSE, OCCUPANCY, STATE]
