"""Stylized monthly macro paths (rates, state HPI, state unemployment) and linear-ramp shocks."""
import numpy as np
import pandas as pd

from . import config
from .columns import PERIOD, STATE, MKT_RATE, HPI, UNEMP

# ===== CONFIG (user inputs) =====
GRID_START, GRID_END = pd.Timestamp("2000-01-31"), pd.Timestamp("2040-12-31")
HPI_BASE_MONTH = pd.Timestamp("2012-01-31")
HPI_GROWTH_ANNUAL = 0.05
HPI_FLAT_YEAR = 2020
RATE_PRE_BASE, RATE_WIGGLE_END = 4.0, pd.Timestamp("2019-12-31")
RATE_ANCHORS = (("2019-12-31", 4.0), ("2020-06-30", 3.3), ("2020-12-31", 2.9), ("2021-12-31", 3.1),
                ("2022-06-30", 5.6), ("2022-12-31", 6.4), ("2023-12-31", 6.8), ("2024-12-31", 6.5))
UNEMP_ANCHORS = (("2012-01-31", 8.2), ("2015-01-31", 5.8), ("2017-01-31", 4.7), ("2019-01-31", 3.9),
                 ("2020-02-29", 3.6), ("2020-03-31", 4.4), ("2020-04-30", 10.5), ("2020-05-31", 11.0),
                 ("2020-06-30", 9.5), ("2020-09-30", 7.5), ("2020-12-31", 6.5), ("2021-06-30", 5.5),
                 ("2021-12-31", 4.5), ("2022-12-31", 4.0), ("2024-12-31", 4.0))
UNEMP_FLOOR = 2.0
# ===== END CONFIG =====


def _mnum(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return (ts.year * 12 + ts.month - 1).to_numpy()


def _interp(anchors, grid_m):
    xs = _mnum(pd.to_datetime([a for a, _ in anchors]))
    return np.interp(grid_m, xs, [v for _, v in anchors])


def _national(grid: pd.DatetimeIndex):
    m = _mnum(grid)
    k = m - m[0]
    wiggle = 0.35 * np.sin(2 * np.pi * k / 30.0) + 0.12 * np.sin(2 * np.pi * k / 11.0)
    pre = RATE_PRE_BASE + wiggle
    post = _interp(RATE_ANCHORS, m)
    rate = np.where(grid <= RATE_WIGGLE_END, pre, post)
    g = np.where(grid.year == HPI_FLAT_YEAR, 0.0, np.log1p(HPI_GROWTH_ANNUAL) / 12.0)
    cum = np.cumsum(g)
    base = int(np.flatnonzero(grid == HPI_BASE_MONTH)[0])
    log_hpi = cum - cum[base]
    return rate, log_hpi, _interp(UNEMP_ANCHORS, m)


def build_macro(start, end, states=None) -> pd.DataFrame:
    """Month-end rows per state; national rate, state HPI (beta on log growth) and state unemployment."""
    states = tuple(states) if states is not None else config.STATES
    grid = pd.date_range(GRID_START, GRID_END, freq="ME")
    rate, log_hpi, unemp = _national(grid)
    keep = (grid >= pd.Timestamp(start)) & (grid <= pd.Timestamp(end))
    beta = dict(zip(config.STATES, config.STATE_HPI_BETA))
    off = dict(zip(config.STATES, config.STATE_UNEMP_OFFSET))
    frames = []
    for s in states:
        frames.append(pd.DataFrame({
            PERIOD: grid[keep], STATE: s,
            MKT_RATE: rate[keep].astype(np.float32),
            HPI: (100.0 * np.exp(beta.get(s, 1.0) * log_hpi[keep])).astype(np.float32),
            UNEMP: np.maximum(unemp[keep] + off.get(s, 0.0), UNEMP_FLOOR).astype(np.float32)}))
    return pd.concat(frames, ignore_index=True)


def apply_shock(macro: pd.DataFrame, shock: config.Shock, start: pd.Timestamp) -> pd.DataFrame:
    """Ramp rate (percent), HPI (multiplicative) and unemployment shocks linearly from start, then hold."""
    out = macro.copy()
    k = _mnum(out[PERIOD]) - int(_mnum([start])[0])
    frac = np.clip(k / max(shock.ramp_months, 1), 0.0, 1.0)
    out[MKT_RATE] = (out[MKT_RATE] + frac * shock.rate_bp / 100.0).astype(np.float32)
    out[HPI] = (out[HPI] * (1.0 + frac * shock.hpi_pct / 100.0)).astype(np.float32)
    out[UNEMP] = (out[UNEMP] + frac * shock.unemp_pt).astype(np.float32)
    return out
