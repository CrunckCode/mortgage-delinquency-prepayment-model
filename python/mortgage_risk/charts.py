"""Plain matplotlib charts (Agg, 150 dpi). Each function takes data and an output path and returns the path."""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .columns import PERIOD, TARGET_PD12, TARGET_PREPAY_1M

log = logging.getLogger(__name__)

# ===== CONFIG (user inputs) =====
DPI = 150
FIG_WIDE, FIG_SQUARE = (10, 4.5), (6.5, 5.5)
LINE_STYLES = ("-", "--", "-.", ":")
GREY_LEVELS = ("black", "dimgray", "gray", "darkgray", "silver", "lightgray")
TOP_FEATURES = 15
BEESWARM_ROWS = 12
BEESWARM_MAX_POINTS = 1500
N_BINS = 10
SCENARIO_PANELS = (("cpr", "CPR (annualized)"), ("cdr", "CDR (annualized)"),
                   ("survival_upb", "Share of starting UPB still performing"), ("smm", "Monthly SMM"))
# ===== END CONFIG =====

plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _roc(y, p):
    order = np.argsort(-p, kind="stable")
    y = np.asarray(y)[order]
    tpr = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / max((1 - y).sum(), 1)])
    return fpr, tpr


def roc_ks(y, preds: dict, path, title="PD model discrimination, out-of-time"):
    """ROC curves for every model and the KS curve of the first (or best-labelled) model."""
    y = np.asarray(y)
    fig, ax = plt.subplots(1, 2, figsize=FIG_WIDE)
    for i, (name, p) in enumerate(preds.items()):
        p = np.asarray(p, dtype=np.float64)
        fpr, tpr = _roc(y, p)
        auc = np.trapezoid(tpr, fpr)
        ax[0].plot(fpr, tpr, LINE_STYLES[i % 4], color=GREY_LEVELS[i % 3], label=f"{name} (AUC {auc:.3f})")
    ax[0].plot([0, 1], [0, 1], color="lightgray")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax[0].legend(loc="lower right")
    name, p = list(preds.items())[-1]
    fpr, tpr = _roc(y, np.asarray(p, dtype=np.float64))
    k = int(np.argmax(tpr - fpr))
    ax[1].plot(np.linspace(0, 1, len(tpr)), tpr, color="black", label="Defaulted loans")
    ax[1].plot(np.linspace(0, 1, len(fpr)), fpr, color="gray", linestyle="--", label="Performing loans")
    ax[1].set(xlabel="Share of loans ranked from riskiest", ylabel="Cumulative share",
              title=f"KS curve, {name} (KS {tpr[k] - fpr[k]:.3f})")
    ax[1].legend(loc="lower right")
    return _save(fig, path)


def decile_lift(y, p, path, title="Decile lift, out-of-time"):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    groups = np.array_split(y[order], N_BINS)
    rate = np.array([g.mean() for g in groups])
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    ax.bar(np.arange(1, N_BINS + 1), rate / y.mean(), color="gray", edgecolor="black")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set(xlabel="Risk decile (1 = riskiest)", ylabel="Lift (decile default rate / overall rate)", title=title)
    ax.set_xticks(np.arange(1, N_BINS + 1))
    return _save(fig, path)


def calibration(y, preds: dict, path, title="Calibration, out-of-time"):
    y = np.asarray(y, dtype=np.float64)
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    top = 0.0
    for i, (name, p) in enumerate(preds.items()):
        p = np.asarray(p, dtype=np.float64)
        order = np.argsort(p, kind="stable")
        pm = [p[order][idx].mean() for idx in np.array_split(np.arange(len(p)), N_BINS)]
        ym = [y[order][idx].mean() for idx in np.array_split(np.arange(len(p)), N_BINS)]
        ax.plot(pm, ym, marker="o", linestyle=LINE_STYLES[i % 4], color=GREY_LEVELS[i % 3], label=name)
        top = max(top, max(pm), max(ym))
    ax.plot([0, top], [0, top], color="lightgray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title=title)
    ax.legend()
    return _save(fig, path)


def _pick(df, names, default=None):
    for n in df.columns:
        if str(n).lower() in names:
            return n
    return default


def psi_bars(psi_df: pd.DataFrame, path, title="Population stability index, out-of-time vs train"):
    """Bars of the PSI/CSI column against the standard 0.10 / 0.25 bands."""
    val = _pick(psi_df, ("psi", "csi", "value")) or psi_df.select_dtypes("number").columns[-1]
    lab = _pick(psi_df, ("feature", "variable", "name", "metric")) or psi_df.columns[0]
    d = psi_df.sort_values(val, ascending=True).tail(TOP_FEATURES)
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    ax.barh(d[lab].astype(str), d[val], color="gray", edgecolor="black")
    for b in config.PSI_BANDS:
        ax.axvline(b, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="PSI", title=title)
    return _save(fig, path)


def _shap_arrays(expl):
    return np.asarray(expl.values), list(expl.feature_names)


def shap_bar(importance: pd.DataFrame, path, title="Mean absolute SHAP value"):
    d = importance.head(TOP_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    ax.barh(d.iloc[:, 0].astype(str), d.iloc[:, 1], color="gray", edgecolor="black")
    ax.set(xlabel="Mean |SHAP| on the raw model margin (log-odds)", title=title)
    return _save(fig, path)


def _numeric(col: pd.Series) -> np.ndarray:
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy(np.float64)
    return pd.to_numeric(col, errors="coerce").to_numpy(np.float64)


def shap_beeswarm(expl, X: pd.DataFrame, path, title="SHAP summary (dark = high feature value)"):
    vals, names = _shap_arrays(expl)
    imp = np.abs(vals).mean(axis=0)
    top = np.argsort(-imp)[:BEESWARM_ROWS][::-1]
    rng = np.random.default_rng(config.SEED)
    keep = rng.choice(len(vals), size=min(BEESWARM_MAX_POINTS, len(vals)), replace=False)
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    for row, j in enumerate(top):
        v = _numeric(X[names[j]])[keep]
        lo, hi = np.nanpercentile(v, [2, 98])
        scaled = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(v)
        ax.scatter(vals[keep, j], row + rng.uniform(-0.3, 0.3, len(keep)), c=scaled, cmap="Greys", s=5,
                   edgecolors="none", vmin=-0.2, vmax=1.0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([names[j] for j in top])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="SHAP value (raw margin, log-odds)", title=title)
    return _save(fig, path)


def shap_dependence(expl, X: pd.DataFrame, feature: str, path):
    vals, names = _shap_arrays(expl)
    j = names.index(feature)
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    x = _numeric(X[feature])
    ax.scatter(x, vals[:, j], s=4, color="dimgray", alpha=0.5, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel=feature, ylabel="SHAP value (raw margin)", title=f"SHAP dependence: {feature}")
    return _save(fig, path)


def scenario_curves(curves: pd.DataFrame, path, scenario_col="scenario"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, label) in zip(axes.ravel(), SCENARIO_PANELS):
        for i, (s, g) in enumerate(curves.groupby(scenario_col, sort=False)):
            ax.plot(g["month"], g[col], linestyle=LINE_STYLES[i % 4], color=GREY_LEVELS[i % len(GREY_LEVELS)], label=s)
        ax.set(xlabel="Projection month", ylabel=label)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Portfolio projection by macro scenario")
    return _save(fig, path)


def prepay_cpr_oot(df: pd.DataFrame, path, title="Out-of-time CPR: actual vs predicted"):
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.plot(df[PERIOD], df["actual_cpr"], color="black", label="Actual")
    ax.plot(df[PERIOD], df["pred_cpr"], color="gray", linestyle="--", label="Predicted")
    ax.set(xlabel="Month", ylabel="CPR (annualized)", title=title)
    ax.legend()
    return _save(fig, path)


def stability_vintage(df: pd.DataFrame, path, title="Discrimination stability by vintage"):
    x = _pick(df, ("vintage", "segment", "group")) or df.columns[0]
    metric_cols = [c for c in df.columns if str(c).lower() in ("auc", "ks", "gini")] or \
        list(df.select_dtypes("number").columns.drop(x, errors="ignore")[:2])
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for i, c in enumerate(metric_cols):
        ax.plot(df[x].astype(str), df[c], marker="o", linestyle=LINE_STYLES[i % 4], color=GREY_LEVELS[i % 3], label=c)
    ax.set(xlabel="Origination vintage", ylabel="Metric value", title=title)
    ax.legend()
    return _save(fig, path)


def model_vs_truth(df: pd.DataFrame, path, title="Model vs simulation truth, 12-month projection (synthetic data only)"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    x = np.arange(len(df))
    for ax, (m, t, label) in zip(axes, (("model_default_12m", "truth_default_12m", "Expected 12m default (% of UPB)"),
                                        ("model_prepay_12m", "truth_prepay_12m", "Expected 12m prepay (% of UPB)"))):
        ax.bar(x - 0.2, df[m], 0.4, color="gray", edgecolor="black", label="Model")
        ax.bar(x + 0.2, df[t], 0.4, color="white", edgecolor="black", hatch="//", label="Simulation truth")
        ax.set_xticks(x)
        ax.set_xticklabels(df["scenario"], rotation=30, ha="right")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle(title)
    return _save(fig, path)


def make_all_charts(bundles: dict, splits_pd: dict, splits_pp: dict, explain_out: dict | None = None,
                    out_dir=config.OUT_DIR, chart_dir=config.CHART_DIR) -> list:
    """Write the full chart set from model bundles plus the CSVs already in out_dir; missing inputs are skipped."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    made = []

    def _try(name, fn):
        try:
            made.append(fn())
        except Exception as exc:  # one broken chart must not stop the rest
            log.warning("chart %s skipped: %s", name, exc)

    pd_oot, pp_oot = splits_pd["oot"], splits_pp["oot"]
    y_pd, y_pp = pd_oot[TARGET_PD12].to_numpy(), pp_oot[TARGET_PREPAY_1M].to_numpy()
    pd_names = [k for k in ("pd_logit", "pd_xgb", "pd_lgbm") if k in bundles]
    pp_names = [k for k in ("pp_logit", "pp_xgb", "pp_lgbm") if k in bundles]
    p_pd = {k: np.asarray(bundles[k].predict_pd(pd_oot)) for k in pd_names}
    p_pp = {k: np.asarray(bundles[k].predict_smm(pp_oot)) for k in pp_names}
    if p_pd:
        _try("roc_ks_pd", lambda: roc_ks(y_pd, p_pd, chart_dir / "roc_ks_pd.png"))
        _try("calibration_pd", lambda: calibration(y_pd, p_pd, chart_dir / "calibration_pd.png"))
        for k in (pd_names[0], pd_names[-1]):
            _try(f"decile_{k}", lambda k=k: decile_lift(y_pd, p_pd[k], chart_dir / f"decile_lift_{k}.png",
                                                        f"Decile lift, {k}, out-of-time"))
    if p_pp:
        _try("roc_ks_prepay", lambda: roc_ks(y_pp, p_pp, chart_dir / "roc_ks_prepay.png",
                                             "Prepay model discrimination, out-of-time"))
        _try("calibration_prepay", lambda: calibration(y_pp, p_pp, chart_dir / "calibration_prepay.png",
                                                       "Prepay calibration, out-of-time"))

    def _csv(name):
        p = out_dir / name
        if not p.exists():
            raise FileNotFoundError(name)
        return pd.read_csv(p, parse_dates=[PERIOD] if name == "prepay_cpr_oot.csv" else False)

    _try("psi", lambda: psi_bars(_csv("psi_csi.csv"), chart_dir / "psi_bars.png"))
    _try("prepay_cpr_oot", lambda: prepay_cpr_oot(_csv("prepay_cpr_oot.csv"), chart_dir / "prepay_cpr_oot.png"))
    _try("stability", lambda: stability_vintage(_csv("stability_by_vintage.csv"), chart_dir / "stability_vintage.png"))
    _try("scenario_curves", lambda: scenario_curves(_csv("scenario_curves.csv"), chart_dir / "scenario_curves.png"))
    _try("model_vs_truth", lambda: model_vs_truth(_csv("model_vs_truth.csv"), chart_dir / "model_vs_truth.png"))
    if explain_out:
        for tag, d in explain_out.items():
            _try(f"shap_bar_{tag}", lambda d=d, tag=tag: shap_bar(d["importance"], chart_dir / f"shap_bar_{tag}.png",
                                                                  f"Mean absolute SHAP, {tag} model"))
            _try(f"beeswarm_{tag}", lambda d=d, tag=tag: shap_beeswarm(d["expl"], d["X"],
                                                                       chart_dir / f"shap_beeswarm_{tag}.png",
                                                                       f"SHAP summary, {tag} model (dark = high value)"))
    return made
