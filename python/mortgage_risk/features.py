"""Backward-only feature engineering on a sorted loan-month panel."""
import numpy as np
import pandas as pd

from . import config
from .columns import (LOAN_ID, PERIOD, STATE, AGE, FICO, ORIG_LTV, ORIG_UPB, CUR_UPB, CUR_RATE, DLQ_STATUS,
                      MKT_RATE, HPI, HPI_ORIG, UNEMP, MTM_LTV, INCENTIVE, BURNOUT, SEASONING_RAMP,
                      HPI_CHG_12M, UNEMP_CHG_12M, TIMES_30DPD_12M, LOG_UPB, FICO_BAND, MONTH_OF_YEAR,
                      TRUE_COLUMNS, TARGET_PD12, TARGET_PREPAY_1M, ZB_CODE, DEFAULT_FLAG, PREPAY_FLAG,
                      SNAPSHOT_DATE, PRED_PD, PRED_SMM)

# ===== CONFIG (user inputs) =====
BURNOUT_THRESHOLD = 0.5
SEASONING_MONTHS = 30.0
LAG_MONTHS = 12
FICO_EDGES = (0, 620, 660, 700, 740, 780, 10_000)
FICO_LABELS = ("<620", "620-659", "660-699", "700-739", "740-779", "780+")
FORBIDDEN = (*TRUE_COLUMNS, TARGET_PD12, TARGET_PREPAY_1M, ZB_CODE, DEFAULT_FLAG, PREPAY_FLAG, SNAPSHOT_DATE,
             PRED_PD, PRED_SMM)
# ===== END CONFIG =====


def month_index(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return (ts.year * 12 + ts.month - 1).to_numpy().astype(np.int64)


def _state_lag(df, macro, cols):
    """State-level value of each column LAG_MONTHS earlier; NaN where the lag is not available."""
    src = macro if macro is not None else df
    tbl = src[[STATE, PERIOD, *cols]].drop_duplicates([STATE, PERIOD])
    states = pd.Index(pd.unique(tbl[STATE].astype(str)))
    key_t = pd.Categorical(tbl[STATE].astype(str), categories=states).codes.astype(np.int64) * 100_000 \
        + month_index(tbl[PERIOD])
    order = np.argsort(key_t)
    key_sorted = key_t[order]
    key_q = pd.Categorical(df[STATE].astype(str), categories=states).codes.astype(np.int64) * 100_000 \
        + month_index(df[PERIOD]) - LAG_MONTHS
    pos = np.clip(np.searchsorted(key_sorted, key_q), 0, len(key_sorted) - 1)
    hit = key_sorted[pos] == key_q
    res = {}
    for c in cols:
        vals = tbl[c].to_numpy(dtype=np.float64)[order][pos]
        res[c] = np.where(hit, vals, np.nan)
    return res


def add_features(df: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.sort_values([LOAN_ID, PERIOD], kind="stable").reset_index(drop=True)
    codes = pd.factorize(out[LOAN_ID])[0]

    inc = out[CUR_RATE].to_numpy(np.float64) - out[MKT_RATE].to_numpy(np.float64)
    out[INCENTIVE] = inc.astype(np.float32)

    hpi = out[HPI].to_numpy(np.float64)
    value = out[ORIG_UPB].to_numpy(np.float64) / out[ORIG_LTV].to_numpy(np.float64) * 100.0 \
        * hpi / out[HPI_ORIG].to_numpy(np.float64)
    out[MTM_LTV] = (out[CUR_UPB].to_numpy(np.float64) / value * 100.0).astype(np.float32)

    burn = pd.Series((inc > BURNOUT_THRESHOLD).astype(np.int32)).groupby(codes).cumsum()
    out[BURNOUT] = burn.to_numpy().astype(np.int16)
    out[SEASONING_RAMP] = np.minimum(out[AGE].to_numpy(np.float64) / SEASONING_MONTHS, 1.0).astype(np.float32)

    lag = _state_lag(out, macro, [HPI, UNEMP])
    out[HPI_CHG_12M] = np.nan_to_num((hpi / lag[HPI] - 1.0) * 100.0).astype(np.float32)
    out[UNEMP_CHG_12M] = np.nan_to_num(out[UNEMP].to_numpy(np.float64) - lag[UNEMP]).astype(np.float32)

    dq = pd.Series((out[DLQ_STATUS].to_numpy() >= 1).astype(np.int32)).groupby(codes)
    cs = dq.cumsum()
    lagged = cs.groupby(codes).shift(LAG_MONTHS).fillna(0)
    out[TIMES_30DPD_12M] = (cs - lagged).to_numpy().astype(np.int8)

    out[LOG_UPB] = np.log(np.maximum(out[CUR_UPB].to_numpy(np.float64), 1.0)).astype(np.float32)
    out[FICO_BAND] = pd.cut(out[FICO], bins=list(FICO_EDGES), labels=list(FICO_LABELS), right=False)
    out[MONTH_OF_YEAR] = out[PERIOD].dt.month.astype(np.int8)
    return out


def assert_no_leakage(df: pd.DataFrame, feature_cols, asof_col: str | None = None) -> None:
    feature_cols = list(feature_cols)
    bad = [c for c in feature_cols if c in FORBIDDEN]
    assert not bad, f"leaky columns in feature set: {bad}"
    missing = [c for c in feature_cols if c not in df.columns]
    assert not missing, f"feature columns missing from frame: {missing}"
    if asof_col is not None:
        assert asof_col in df.columns, f"as-of column {asof_col} missing"
        assert asof_col not in feature_cols, "as-of column must not be a model feature"
