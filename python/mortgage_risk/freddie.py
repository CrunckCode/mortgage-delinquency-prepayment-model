"""Loader for Freddie Mac Single-Family Loan-Level Dataset (SFLLD) pipe-delimited files, mapped to the internal schema."""
import glob
import logging

import numpy as np
import pandas as pd

from . import config, macro as macro_mod
from .columns import (LOAN_ID, PERIOD, ORIG_DATE, VINTAGE, AGE, FICO, ORIG_LTV, ORIG_CLTV, DTI, ORIG_RATE, ORIG_UPB,
                      ORIG_TERM, STATE, PURPOSE, OCCUPANCY, PROP_TYPE, N_BORROWERS, FIRST_TIME_BUYER, CHANNEL,
                      MI_PCT, CUR_UPB, CUR_RATE, DLQ_STATUS, DLQ_BUCKET, DLQ_BUCKETS, PREPAY_FLAG, DEFAULT_FLAG,
                      ZB_CODE, IS_ACTIVE, MKT_RATE, HPI, HPI_ORIG, UNEMP)

# ===== CONFIG (user inputs) =====
FREDDIE_ORIG_LAYOUT = (
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date", "msa", "mi_pct", "num_units",
    "occupancy", "cltv", "dti", "orig_upb", "ltv", "orig_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_sequence", "loan_purpose", "orig_term",
    "num_borrowers", "seller_name", "servicer_name", "super_conforming", "pre_harp_sequence", "program_indicator",
    "harp_indicator", "valuation_method", "io_indicator", "mi_cancellation")
FREDDIE_PERF_LAYOUT = (
    "loan_sequence", "report_period", "current_upb", "delinquency_status", "loan_age", "remaining_months",
    "defect_settlement_date", "modification_flag", "zero_balance_code", "zero_balance_date", "current_rate",
    "deferred_upb", "ddlpi", "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "expenses", "legal_costs",
    "maintenance_costs", "taxes_insurance", "misc_expenses", "actual_loss", "cum_modification_cost",
    "step_modification_flag", "payment_deferral", "eltv", "zero_balance_removal_upb", "delinquent_accrued_interest",
    "delinquency_disaster", "borrower_assistance", "current_month_modification_cost", "interest_bearing_upb")

MISSING_FICO, MISSING_PCT = 9999, 999
PREPAY_ZB = "01"
CREDIT_ZB = ("02", "03", "09", "15", "96")
RA_STATUS = "RA"
DLQ_CAP = 3
OCCUPANCY_MAP = {"P": "O", "I": "I", "S": "S"}
MACRO_GLOB = str(config.RAW_DIR / "macro" / "*.csv")
MACRO_PAD_MONTHS = 14
# ===== END CONFIG =====

log = logging.getLogger(__name__)


def _read_pipe(path, layout, wanted=None, chunksize=None):
    """Read as strings, using only the leading layout fields so extra trailing columns are tolerated."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        n_fields = len(fh.readline().rstrip("\n").split("|"))
    n = min(len(layout), n_fields)
    return pd.read_csv(path, sep="|", header=None, names=list(layout[:n]), usecols=range(n), dtype=str,
                       chunksize=chunksize, index_col=False)


def _month_end(series):
    return pd.to_datetime(series, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _read_orig(orig_glob, max_loans):
    files = sorted(glob.glob(orig_glob))
    if not files:
        raise FileNotFoundError(f"no origination files match {orig_glob}")
    raw = pd.concat([_read_pipe(f, FREDDIE_ORIG_LAYOUT) for f in files], ignore_index=True)
    raw = raw.drop_duplicates("loan_sequence")
    if max_loans is not None:
        raw = raw.head(max_loans)
    fico = _num(raw["credit_score"]).where(lambda x: x != MISSING_FICO)
    ltv = _num(raw["ltv"]).where(lambda x: x < MISSING_PCT)
    cltv = _num(raw["cltv"]).where(lambda x: x < MISSING_PCT)
    dti = _num(raw["dti"]).where(lambda x: x < MISSING_PCT)
    ltv, cltv = ltv.fillna(cltv), cltv.fillna(ltv)
    first_pay = _month_end(raw["first_payment_date"])
    o = pd.DataFrame({
        LOAN_ID: raw["loan_sequence"], ORIG_DATE: first_pay - pd.offsets.MonthEnd(1),
        FICO: fico.fillna(fico.median()), ORIG_LTV: ltv.fillna(ltv.median()), ORIG_CLTV: cltv.fillna(cltv.median()),
        DTI: dti.fillna(dti.median()), ORIG_RATE: _num(raw["orig_rate"]), ORIG_UPB: _num(raw["orig_upb"]),
        ORIG_TERM: _num(raw["orig_term"]).fillna(360).astype(np.int16), STATE: raw["property_state"],
        PURPOSE: raw["loan_purpose"], OCCUPANCY: raw["occupancy"].map(OCCUPANCY_MAP).fillna(raw["occupancy"]),
        PROP_TYPE: raw["property_type"], N_BORROWERS: _num(raw["num_borrowers"]).fillna(1).astype(np.int8),
        FIRST_TIME_BUYER: (raw["first_time_homebuyer"] == "Y").astype(np.int8), CHANNEL: raw["channel"],
        MI_PCT: _num(raw["mi_pct"]).where(lambda x: x < MISSING_PCT).fillna(0.0)})
    o[VINTAGE] = o[ORIG_DATE].dt.year.astype(np.int16)
    return o.reset_index(drop=True)


def _read_perf(perf_glob, loan_ids):
    files = sorted(glob.glob(perf_glob))
    if not files:
        raise FileNotFoundError(f"no performance files match {perf_glob}")
    keep = set(loan_ids)
    parts = []
    for f in files:
        for chunk in _read_pipe(f, FREDDIE_PERF_LAYOUT, chunksize=2_000_000):
            parts.append(chunk[chunk["loan_sequence"].isin(keep)])
    raw = pd.concat(parts, ignore_index=True)
    status_raw = raw["delinquency_status"].astype(str).str.strip()
    status = _num(status_raw).fillna(pd.Series(np.where(status_raw == RA_STATUS, DLQ_CAP, 0), index=status_raw.index)).clip(0, DLQ_CAP)
    zb = raw["zero_balance_code"].where(raw["zero_balance_code"].isna(), raw["zero_balance_code"].str.zfill(2))
    p = pd.DataFrame({
        LOAN_ID: raw["loan_sequence"], PERIOD: _month_end(raw["report_period"]), CUR_UPB: _num(raw["current_upb"]),
        DLQ_STATUS: status.astype(np.int8), AGE: _num(raw["loan_age"]).fillna(0).astype(np.int16),
        CUR_RATE: _num(raw["current_rate"]), "_zb": zb})
    return p.sort_values([LOAN_ID, PERIOD], kind="stable").reset_index(drop=True)


def load_macro(states, start, end, macro_glob=MACRO_GLOB) -> pd.DataFrame:
    """Macro from CSVs with internal column names if present, else synthetic macro with a warning."""
    files = sorted(glob.glob(macro_glob))
    need = {PERIOD, STATE, MKT_RATE, HPI, UNEMP}
    if files:
        m = pd.concat([pd.read_csv(f, parse_dates=[PERIOD]) for f in files], ignore_index=True)
        if need.issubset(m.columns):
            m[PERIOD] = m[PERIOD] + pd.offsets.MonthEnd(0)
            return m[[PERIOD, STATE, MKT_RATE, HPI, UNEMP]]
        log.warning("macro CSVs lack columns %s; falling back to synthetic macro", sorted(need - set(m.columns)))
    else:
        log.warning("no macro CSVs at %s; using synthetic macro paths", macro_glob)
    return macro_mod.build_macro(start, end, states=sorted(set(states)))


def load_freddie(orig_glob, perf_glob, max_loans=None, macro_glob=MACRO_GLOB) -> pd.DataFrame:
    orig = _read_orig(orig_glob, max_loans)
    perf = _read_perf(perf_glob, orig[LOAN_ID])
    perf[CUR_UPB] = perf[CUR_UPB].astype(np.float64)
    zb = perf.pop("_zb")
    prev_upb = perf.groupby(LOAN_ID, sort=False)[CUR_UPB].shift(1)
    final = zb.notna()
    perf.loc[final & ~(perf[CUR_UPB] > 0), CUR_UPB] = prev_upb
    flag = (perf[DLQ_STATUS] >= DLQ_CAP) | zb.isin(CREDIT_ZB)
    prior = flag.astype(int).groupby(perf[LOAN_ID], sort=False).cumsum() - flag.astype(int)
    keep = prior == 0                      # stop each loan at its first default event
    first = flag & keep
    perf, zb, first = perf[keep].copy(), zb[keep], first[keep]
    perf[PREPAY_FLAG] = (zb == PREPAY_ZB).astype(np.int8)
    perf[DEFAULT_FLAG] = first.astype(np.int8)
    perf[ZB_CODE] = _num(zb).fillna(0).astype(np.int8)
    perf[IS_ACTIVE] = np.int8(1)
    df = perf.merge(orig, on=LOAN_ID, how="inner")
    df[DLQ_BUCKET] = pd.Categorical.from_codes(df[DLQ_STATUS].to_numpy(), categories=DLQ_BUCKETS)

    states = df[STATE].unique()
    mac = load_macro(states, df[ORIG_DATE].min() - pd.DateOffset(months=MACRO_PAD_MONTHS), df[PERIOD].max(),
                     macro_glob)
    cur = mac[[STATE, PERIOD, MKT_RATE, HPI, UNEMP]]
    df = df.merge(cur, on=[STATE, PERIOD], how="left")
    at_orig = mac[[STATE, PERIOD, HPI]].rename(columns={PERIOD: ORIG_DATE, HPI: HPI_ORIG})
    df = df.merge(at_orig, on=[STATE, ORIG_DATE], how="left")
    for c in (STATE, PURPOSE, OCCUPANCY, PROP_TYPE, CHANNEL):
        df[c] = df[c].astype("category")
    for c in (CUR_UPB, CUR_RATE, MKT_RATE, HPI, HPI_ORIG, UNEMP):
        df[c] = df[c].astype(np.float32)
    return df.sort_values([LOAN_ID, PERIOD], kind="stable").reset_index(drop=True)
