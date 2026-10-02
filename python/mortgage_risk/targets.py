"""Snapshot (12-month PD) and monthly hazard (prepay) targets plus time-based splits."""
import numpy as np
import pandas as pd

from . import config
from .columns import (LOAN_ID, PERIOD, DLQ_STATUS, DEFAULT_FLAG, PREPAY_FLAG, IS_ACTIVE, TARGET_PD12,
                      TARGET_PREPAY_1M, SNAPSHOT_DATE, MONTH_OF_YEAR, ZB_CODE, TRUE_COLUMNS)
from .features import month_index

# ===== CONFIG (user inputs) =====
SNAPSHOT_STATUSES = (0, 1)
PD_KIND, PREPAY_KIND = "pd", "prepay"
# ===== END CONFIG =====


def _sorted(df):
    codes = pd.factorize(df[LOAN_ID])[0]
    mi = month_index(df[PERIOD])
    same = codes[1:] == codes[:-1]
    if not (np.all(same | (codes[1:] > codes[:-1])) and np.all(mi[1:][same] > mi[:-1][same])):
        df = df.sort_values([LOAN_ID, PERIOD], kind="stable").reset_index(drop=True)
        codes = pd.factorize(df[LOAN_ID])[0]
        mi = month_index(df[PERIOD])
    return df, codes, mi


def at_risk_rows(df) -> np.ndarray:
    """Positions of rows whose immediately preceding month exists with status current."""
    df, codes, mi = _sorted(df)
    ok = (codes[1:] == codes[:-1]) & (mi[1:] - mi[:-1] == 1) & (df[DLQ_STATUS].to_numpy()[:-1] == 0)
    return np.flatnonzero(ok) + 1


def build_snapshots(panel_feat: pd.DataFrame) -> pd.DataFrame:
    df, codes, mi = _sorted(panel_feat)
    cutoff = config.OBS_END - pd.DateOffset(months=config.PD_HORIZON_MONTHS)
    per = df[PERIOD]
    ok = per.dt.month.isin(config.SNAPSHOT_MONTHS) & (per <= cutoff) & df[DLQ_STATUS].isin(SNAPSHOT_STATUSES) \
        & (df[IS_ACTIVE] == 1) & (df[PREPAY_FLAG] == 0) & (df[DEFAULT_FLAG] == 0)
    first_default = pd.Series(mi[df[DEFAULT_FLAG].to_numpy() == 1], index=codes[df[DEFAULT_FLAG].to_numpy() == 1])
    first_default = first_default.groupby(level=0).min()
    snap = df[ok.to_numpy()].copy()
    dmi = pd.Series(codes[ok.to_numpy()]).map(first_default).to_numpy()
    gap = dmi - mi[ok.to_numpy()]
    snap[SNAPSHOT_DATE] = snap[PERIOD]
    snap[TARGET_PD12] = ((gap >= 1) & (gap <= config.PD_HORIZON_MONTHS)).astype(np.int8)
    return snap.reset_index(drop=True)


def build_hazard_rows(panel_feat: pd.DataFrame, loan_share: float = 1.0, seed: int = config.SEED) -> pd.DataFrame:
    df = panel_feat
    if loan_share < 1.0:
        ids = pd.unique(df[LOAN_ID])
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(ids), size=max(int(round(len(ids) * loan_share)), 1), replace=False)
        df = df[df[LOAN_ID].isin(ids[pick])]
    df, _, _ = _sorted(df)
    pos = at_risk_rows(df)
    prev = df.iloc[pos - 1].reset_index(drop=True)
    cur = df.iloc[pos].reset_index(drop=True)
    prev[PERIOD] = cur[PERIOD]
    prev[TARGET_PREPAY_1M] = cur[PREPAY_FLAG].to_numpy().astype(np.int8)
    for col in (PREPAY_FLAG, DEFAULT_FLAG, ZB_CODE, *[c for c in TRUE_COLUMNS if c in cur.columns]):
        prev[col] = cur[col].to_numpy()
    # calendar month of t is known in advance, so it is not leakage
    prev[MONTH_OF_YEAR] = cur[PERIOD].dt.month.astype(np.int8)
    return prev


def _window(series, win):
    return (series >= win[0]) & (series <= win[1])


def split_by_time(df: pd.DataFrame, kind: str) -> dict:
    if kind == PD_KIND:
        col, wins = SNAPSHOT_DATE, dict(train=config.TRAIN_PD_SNAP, calib=config.CALIB_PD_SNAP, oot=config.OOT_PD_SNAP)
        label_end = df[col] + pd.DateOffset(months=config.PD_HORIZON_MONTHS)
        dev_ok = label_end <= config.DEV_CUTOFF
    elif kind == PREPAY_KIND:
        col, wins = PERIOD, dict(train=config.TRAIN_PP, calib=config.CALIB_PP, oot=config.OOT_PP)
        dev_ok = df[col] <= config.DEV_CUTOFF
    else:
        raise ValueError(f"unknown kind {kind}")
    out = {}
    for name, win in wins.items():
        m = _window(df[col], win)
        if name != "oot":
            m = m & dev_ok
        out[name] = df[m].reset_index(drop=True)
    return out
