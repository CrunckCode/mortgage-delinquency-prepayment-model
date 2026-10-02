"""Synthetic loan-month panel from a known data generating process (hazards stored as TRUE_HP / TRUE_HD).

TRUE_HP is the monthly prepay hazard and TRUE_HD the monthly C->30 hazard, both evaluated for every active
loan-month as if the loan were current (they only act on loans in status C).
"""
import numpy as np
import pandas as pd

from . import config, macro as macro_mod, targets
from .columns import (LOAN_ID, PERIOD, ORIG_DATE, VINTAGE, AGE, FICO, ORIG_LTV, ORIG_CLTV, DTI, ORIG_RATE, ORIG_UPB,
                      ORIG_TERM, STATE, PURPOSE, OCCUPANCY, PROP_TYPE, N_BORROWERS, FIRST_TIME_BUYER, CHANNEL,
                      MI_PCT, CUR_UPB, CUR_RATE, DLQ_STATUS, DLQ_BUCKET, DLQ_BUCKETS, PREPAY_FLAG, DEFAULT_FLAG,
                      ZB_CODE, IS_ACTIVE, MKT_RATE, HPI, HPI_ORIG, UNEMP, TRUE_HP, TRUE_HD, TARGET_PD12,
                      SNAPSHOT_DATE)

# ===== CONFIG (user inputs) =====
FICO_MEAN, FICO_SD, FICO_LO, FICO_HI = 745.0, 45.0, 600.0, 830.0
LTV_MIX = (0.45, 0.30)                     # exactly 80, then U(60,80); remainder U(80,97)
DTI_MEAN, DTI_SD, DTI_LO, DTI_HI = 34.0, 9.0, 10.0, 50.0
UPB_MEDIAN, UPB_SIGMA, UPB_CAP, UPB_FLOOR = 230_000.0, 0.45, 766_550.0, 40_000.0
RATE_SPREAD, RATE_FICO_SLOPE, RATE_NOISE = 0.25, 0.05, 0.15     # percent; slope per 10 FICO points
TERM = 360
PURPOSES, PURPOSE_P = ("P", "C", "N"), (0.60, 0.25, 0.15)
OCCUPANCIES, OCCUPANCY_P = ("O", "I", "S"), (0.90, 0.06, 0.04)
PROP_TYPES, PROP_TYPE_P = ("SF", "PU", "CO", "CP"), (0.72, 0.16, 0.09, 0.03)
CHANNELS, CHANNEL_P = ("R", "B", "C"), (0.60, 0.25, 0.15)
BORROWER_P2, FTB_P_PURCHASE = 0.55, 0.30
MI_LTV_KNOTS, MI_PCT_KNOTS = (80, 85, 90, 95, 97), (6, 12, 25, 30, 35)
SEASONING_HUMP_MID, SEASONING_HUMP_WIDTH, SEASONING_HUMP_FLOOR = 30.0, 20.0, 0.3
BURNOUT_THRESHOLD = 0.5
PROB_CLIP = 0.9
OOT_YEAR = 2020
# ===== END CONFIG =====


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _draw_loans(n, rng, mk, periods):
    """Loan origination attributes; mk is the macro in array form."""
    d = {}
    orig_months = pd.date_range(config.ORIG_START, config.ORIG_END, freq="ME")
    first = int(np.searchsorted(periods.values, orig_months[0].to_datetime64()))
    o = first + rng.integers(0, len(orig_months), n)
    fico = np.round(np.clip(rng.normal(FICO_MEAN, FICO_SD, n), FICO_LO, FICO_HI))
    u = rng.random(n)
    ltv = np.where(u < LTV_MIX[0], 80.0,
                   np.where(u < sum(LTV_MIX), rng.uniform(60, 80, n), rng.uniform(80, 97, n)))
    dti = np.round(np.clip(rng.normal(DTI_MEAN, DTI_SD, n), DTI_LO, DTI_HI))
    upb = np.exp(np.log(UPB_MEDIAN) + UPB_SIGMA * rng.standard_normal(n))
    upb = np.round(np.clip(upb, UPB_FLOOR, UPB_CAP))
    st = rng.choice(len(config.STATES), size=n, p=np.asarray(config.STATE_WEIGHTS) / sum(config.STATE_WEIGHTS))
    rate = mk["nat_rate"][o] + RATE_SPREAD + RATE_FICO_SLOPE * (740 - fico) / 10.0 + RATE_NOISE * rng.standard_normal(n)
    purpose = rng.choice(len(PURPOSES), size=n, p=PURPOSE_P)
    d.update(x_o=o, x_fico=fico, x_ltv=ltv, x_dti=dti, x_upb=upb, x_st=st, x_rate=np.round(rate, 3),
             x_purpose=purpose, x_occ=rng.choice(len(OCCUPANCIES), size=n, p=OCCUPANCY_P),
             x_prop=rng.choice(len(PROP_TYPES), size=n, p=PROP_TYPE_P),
             x_nb=1 + (rng.random(n) < BORROWER_P2).astype(np.int8),
             x_ftb=((purpose == 0) & (rng.random(n) < FTB_P_PURCHASE)).astype(np.int8),
             x_chan=rng.choice(len(CHANNELS), size=n, p=CHANNEL_P),
             x_mi=np.where(ltv > 80, np.interp(ltv, MI_LTV_KNOTS, MI_PCT_KNOTS), 0.0))
    return d


def _macro_arrays(start, end):
    m = macro_mod.build_macro(start, end)
    periods = pd.DatetimeIndex(np.sort(m[PERIOD].unique()))
    piv = lambda col: m.pivot(index=STATE, columns=PERIOD, values=col).reindex(list(config.STATES))[periods]
    return dict(nat_rate=piv(MKT_RATE).to_numpy()[0].astype(np.float64), st_hpi=piv(HPI).to_numpy().astype(np.float64),
                st_unemp=piv(UNEMP).to_numpy().astype(np.float64)), periods


def simulate_panel(n_loans: int, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = config.ORIG_START - pd.offsets.MonthEnd(1)
    mk, periods = _macro_arrays(start, config.OBS_END)
    L = _draw_loans(n_loans, rng, mk, periods)
    P, T = config.TRUE_DGP["prepay"], config.TRUE_DGP["to30"]
    R30, R60 = config.TRUE_DGP["roll30"], config.TRUE_DGP["roll60"]

    o, st, fico, ltv, dti, upb0, rate = L["x_o"], L["x_st"], L["x_fico"], L["x_ltv"], L["x_dti"], L["x_upb"], L["x_rate"]
    r = rate / 1200.0
    growth_n = (1.0 + r) ** TERM
    hpi_orig = mk["st_hpi"][st, o]
    house0 = upb0 / ltv * 100.0
    status = np.zeros(n_loans, np.int8)
    burn = np.zeros(n_loans, np.int32)
    alive = np.ones(n_loans, bool)
    months_of_year = periods.month.to_numpy()
    rec = {k: [] for k in ("i", "m", "ag", "up", "s", "pp", "df", "zb", "hp", "hd")}

    for m in range(int(o.min()) + 1, len(periods)):
        idx = np.flatnonzero(alive & (o < m))
        if idx.size == 0:
            continue
        age = (m - o[idx]).astype(np.int16)
        s_idx = st[idx]
        hpi = mk["st_hpi"][s_idx, m]
        hpi_chg = (hpi / mk["st_hpi"][s_idx, m - 12] - 1.0) * 100.0 if m >= 12 else np.zeros(idx.size)
        unemp = mk["st_unemp"][s_idx, m]
        upb = upb0[idx] * (growth_n[idx] - (1.0 + r[idx]) ** age) / (growth_n[idx] - 1.0)
        mtm = upb / (house0[idx] * hpi / hpi_orig[idx]) * 100.0
        inc = rate[idx] - mk["nat_rate"][m]
        burn[idx] += inc > BURNOUT_THRESHOLD
        f40, f50 = (fico[idx] - 740.0) / 40.0, (fico[idx] - 740.0) / 50.0

        refi = _sigmoid((inc - P["incentive_mid"]) / P["incentive_scale"]) * np.exp(-P["a_burn"] * burn[idx]) \
            * (mtm < P["ltv_cap"])
        hp = _sigmoid(P["a0"] + P["a_ref"] * refi + P["a_seas"] * np.minimum(age / 30.0, 1.0) + P["a_fico"] * f40
                      + P["a_size"] * np.log(upb / 200_000.0)
                      + P["month_amp"] * np.cos(2 * np.pi * (months_of_year[m] - 6) / 12.0))
        hump = np.exp(-(((age - SEASONING_HUMP_MID) / SEASONING_HUMP_WIDTH) ** 2)) - SEASONING_HUMP_FLOOR
        hd = _sigmoid(T["b0"] + T["b_fico"] * f50 + T["b_ltv"] * np.maximum(mtm - 80.0, 0.0) / 10.0
                      + T["b_dti"] * (dti[idx] - 36.0) / 10.0 + T["b_unemp"] * (unemp - 5.0)
                      + T["b_hpi"] * np.minimum(hpi_chg, 0.0) / 10.0 + T["b_age"] * hump)

        tilt = np.exp(R30["fico_k"] * f50 + R30["ltv_k"] * np.maximum(mtm - 80.0, 0.0) / 10.0)
        p60 = np.clip(R30["base"] * tilt, 0.0, PROB_CLIP)
        p90 = np.clip(R60["base"] * tilt, 0.0, PROB_CLIP)
        u = rng.random(idx.size)
        s = status[idx]
        new = s.copy()
        pp = (s == 0) & (u < hp)
        new[(s == 0) & ~pp & (u < hp + hd)] = 1
        new[(s == 1) & (u < p60)] = 2
        new[(s == 1) & (u >= p60) & (u < p60 + R30["cure"])] = 0
        new[(s == 2) & (u < p90)] = 3
        new[(s == 2) & (u >= p90) & (u < p90 + R60["cure"])] = 0
        dflt = new == 3
        matured = age >= TERM
        status[idx] = new
        alive[idx[pp | dflt | matured]] = False
        zb = np.where(pp | matured, 1, np.where(dflt, 3, 0))
        for k, v in (("i", idx), ("m", np.full(idx.size, m, np.int16)), ("ag", age), ("up", upb), ("s", new),
                     ("pp", pp), ("df", dflt), ("zb", zb), ("hp", hp), ("hd", hd)):
            rec[k].append(v)

    r_ = {k: np.concatenate(v) for k, v in rec.items()}
    order = np.lexsort((r_["m"], r_["i"]))
    r_ = {k: v[order] for k, v in r_.items()}
    i, m_ = r_["i"], r_["m"].astype(np.int64)
    s_i = st[i]
    orig_dates = periods[o]
    f32 = np.float32
    panel = pd.DataFrame({
        LOAN_ID: i.astype(np.int32), PERIOD: periods[m_], ORIG_DATE: orig_dates[i],
        VINTAGE: orig_dates.year.to_numpy().astype(np.int16)[i], AGE: r_["ag"],
        FICO: fico[i].astype(f32), ORIG_LTV: ltv[i].astype(f32), ORIG_CLTV: ltv[i].astype(f32),
        DTI: dti[i].astype(f32), ORIG_RATE: rate[i].astype(f32), ORIG_UPB: upb0[i].astype(f32),
        ORIG_TERM: np.full(i.size, TERM, np.int16),
        STATE: pd.Categorical.from_codes(s_i, categories=config.STATES),
        PURPOSE: pd.Categorical.from_codes(L["x_purpose"][i], categories=PURPOSES),
        OCCUPANCY: pd.Categorical.from_codes(L["x_occ"][i], categories=OCCUPANCIES),
        PROP_TYPE: pd.Categorical.from_codes(L["x_prop"][i], categories=PROP_TYPES),
        N_BORROWERS: L["x_nb"][i].astype(np.int8), FIRST_TIME_BUYER: L["x_ftb"][i],
        CHANNEL: pd.Categorical.from_codes(L["x_chan"][i], categories=CHANNELS), MI_PCT: L["x_mi"][i].astype(f32),
        CUR_UPB: r_["up"].astype(f32), CUR_RATE: rate[i].astype(f32),
        DLQ_STATUS: r_["s"], DLQ_BUCKET: pd.Categorical.from_codes(r_["s"], categories=DLQ_BUCKETS),
        PREPAY_FLAG: r_["pp"].astype(np.int8), DEFAULT_FLAG: r_["df"].astype(np.int8),
        ZB_CODE: r_["zb"].astype(np.int8), IS_ACTIVE: np.ones(i.size, np.int8),
        MKT_RATE: mk["nat_rate"][m_].astype(f32), HPI: mk["st_hpi"][s_i, m_].astype(f32),
        HPI_ORIG: hpi_orig[i].astype(f32), UNEMP: mk["st_unemp"][s_i, m_].astype(f32),
        TRUE_HP: r_["hp"].astype(f32), TRUE_HD: r_["hd"].astype(f32)})
    return panel


def calibration_report(panel: pd.DataFrame) -> dict:
    snaps = targets.build_snapshots(panel)
    snap_dates = snaps[SNAPSHOT_DATE]
    train = snaps[(snap_dates >= config.TRAIN_PD_SNAP[0]) & (snap_dates <= config.TRAIN_PD_SNAP[1])]
    oot20 = snaps[snap_dates.dt.year == OOT_YEAR]
    panel_sorted = panel.sort_values([LOAN_ID, PERIOD], kind="stable").reset_index(drop=True)
    pos = targets.at_risk_rows(panel_sorted)
    risk = panel_sorted.iloc[pos]
    smm_by_year = risk.groupby(risk[PERIOD].dt.year)[PREPAY_FLAG].mean()
    cpr = lambda smm: 1.0 - (1.0 - smm) ** 12
    rep = dict(pd12_train=float(train[TARGET_PD12].mean()), pd12_oot_2020=float(oot20[TARGET_PD12].mean()),
               cpr_mean=float(cpr(risk[PREPAY_FLAG].mean())),
               cpr_by_year={int(y): float(cpr(v)) for y, v in smm_by_year.items()})
    print(f"calibration: pd12_train={rep['pd12_train']:.4f} pd12_oot_2020={rep['pd12_oot_2020']:.4f} "
          f"cpr_mean={rep['cpr_mean']:.4f}")
    print("cpr_by_year: " + ", ".join(f"{y}={v:.3f}" for y, v in rep["cpr_by_year"].items()))
    return rep
