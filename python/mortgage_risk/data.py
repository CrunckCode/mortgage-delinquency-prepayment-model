"""Panel access with a parquet cache; chooses real Freddie data when present, else the synthetic DGP."""
import glob

import pandas as pd

from . import config, freddie, macro as macro_mod, synthetic
from .columns import LOAN_ID, PERIOD, STATE, MKT_RATE, HPI, UNEMP

# ===== CONFIG (user inputs) =====
FREDDIE_ORIG_GLOB = str(config.RAW_DIR / "historical_data_[0-9]*.txt")
FREDDIE_PERF_GLOB = str(config.RAW_DIR / "historical_data_time_*.txt")
SOURCE_SYNTHETIC, SOURCE_FREDDIE, SOURCE_AUTO = "synthetic", "freddie", "auto"
# ===== END CONFIG =====


def _freddie_available() -> bool:
    return bool(glob.glob(FREDDIE_ORIG_GLOB)) and bool(glob.glob(FREDDIE_PERF_GLOB))


def get_panel(source: str = SOURCE_AUTO, n_loans: int | None = None, refresh: bool = False):
    if source == SOURCE_AUTO:
        source = SOURCE_FREDDIE if _freddie_available() else SOURCE_SYNTHETIC
    seed = config.SEED
    if source == SOURCE_SYNTHETIC:
        n = n_loans or config.N_LOANS_FULL
    elif source == SOURCE_FREDDIE:
        n = n_loans
    else:
        raise ValueError(f"unknown source {source}")
    tag = f"{source}_{n if n is not None else 'all'}_{seed}"
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    p_panel, p_macro = config.INTERIM_DIR / f"panel_{tag}.parquet", config.INTERIM_DIR / f"macro_{tag}.parquet"

    if p_panel.exists() and p_macro.exists() and not refresh:
        panel, macro = pd.read_parquet(p_panel), pd.read_parquet(p_macro)
    else:
        if source == SOURCE_SYNTHETIC:
            panel = synthetic.simulate_panel(n, seed)
            macro = macro_mod.build_macro(config.ORIG_START - pd.offsets.MonthEnd(1), config.OBS_END)
        else:
            panel = freddie.load_freddie(FREDDIE_ORIG_GLOB, FREDDIE_PERF_GLOB, max_loans=n)
            macro = freddie.load_macro(panel[STATE].astype(str).unique(), panel[PERIOD].min() - pd.DateOffset(months=14),
                                       panel[PERIOD].max())
        panel.to_parquet(p_panel, index=False)
        macro.to_parquet(p_macro, index=False)
    meta = {"source": source, "n_loans": int(panel[LOAN_ID].nunique()), "seed": seed}
    return panel, macro, meta
