"""Hand-off of projected CPR/CDR curves to the securitization waterfall project (assumptions, not modeled)."""
import numpy as np
import pandas as pd

from . import config

# ===== CONFIG (user inputs) =====
CURVES_FILE, SCALARS_FILE = "cpr_cdr_curves.csv", "scalar_equivalents.csv"
CURVE_OUT_COLUMNS = ["scenario", "month", "smm", "mdr", "cpr_annual", "cdr_annual", "survival"]
SCALAR_COLUMNS = ["scenario", "cpr", "cdr", "severity", "lag", "index_shift"]
# waterfall.config.Scenario fields, kept in sync by a test that reads that file
WATERFALL_SCENARIO_KEYS = ("name", "cpr", "cdr", "severity", "lag", "index_shift")
# ===== END CONFIG =====


def _extend(curve: pd.DataFrame, months: int) -> pd.DataFrame:
    """Hold the last modeled monthly rates flat out to `months`."""
    c = curve.sort_values("month").reset_index(drop=True)
    last = c.iloc[-1]
    smm, mdr, surv = float(last["smm"]), float(last["mdr"]), float(last["survival_upb"])
    rows = []
    for m in range(int(last["month"]) + 1, months + 1):
        surv *= (1.0 - mdr) * (1.0 - smm)
        rows.append({"month": m, "smm": smm, "mdr": mdr, "survival_upb": surv})
    ext = pd.concat([c[["month", "smm", "mdr", "survival_upb"]], pd.DataFrame(rows)], ignore_index=True)
    return ext.head(months)


def write_curves(curves_by_scenario: dict, out_dir=config.HOOK_DIR, months: int = config.HOOK_MONTHS,
                 shocks=config.SHOCKS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shock_by_name = {s.name: s for s in shocks}
    long, scalars = [], []
    for name, curve in curves_by_scenario.items():
        e = _extend(curve, months)
        e["cpr_annual"] = 1.0 - (1.0 - e["smm"]) ** 12
        e["cdr_annual"] = 1.0 - (1.0 - e["mdr"]) ** 12
        e["scenario"] = name
        e = e.rename(columns={"survival_upb": "survival"})
        long.append(e[CURVE_OUT_COLUMNS])
        # weights: surviving balance at the start of each month
        w = np.concatenate([[1.0], e["survival"].to_numpy()[:-1]])
        shock = shock_by_name.get(name)
        scalars.append({"scenario": name, "cpr": float(np.average(e["cpr_annual"], weights=w)),
                        "cdr": float(np.average(e["cdr_annual"], weights=w)),
                        "severity": config.SEVERITY_ASSUMPTION, "lag": config.LAG_ASSUMPTION,
                        "index_shift": (shock.rate_bp / 10_000.0) if shock is not None else 0.0})
    curves_path, scalars_path = out_dir / CURVES_FILE, out_dir / SCALARS_FILE
    pd.concat(long, ignore_index=True).to_csv(curves_path, index=False)
    pd.DataFrame(scalars)[SCALAR_COLUMNS].to_csv(scalars_path, index=False)
    return {"curves": curves_path, "scalars": scalars_path}


def to_waterfall_scenarios(path=None) -> list:
    """Read scalar_equivalents.csv into dicts keyed like waterfall.config.Scenario."""
    path = path if path is not None else config.HOOK_DIR / SCALARS_FILE
    df = pd.read_csv(path)
    return [{"name": r["scenario"], "cpr": float(r["cpr"]), "cdr": float(r["cdr"]), "severity": float(r["severity"]),
             "lag": int(r["lag"]), "index_shift": float(r["index_shift"])} for _, r in df.iterrows()]
