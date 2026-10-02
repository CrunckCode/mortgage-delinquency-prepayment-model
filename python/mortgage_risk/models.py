"""Model bundles (scorecard logit, XGBoost, LightGBM), time-aware tuning, calibration and training driver."""
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config
from .columns import (FEATURES_PD, FEATURES_PREPAY, CATEGORICAL_FEATURES, TARGET_PD12, TARGET_PREPAY_1M,
                      SNAPSHOT_DATE, PERIOD, FICO, MTM_LTV, DTI, UNEMP, TIMES_30DPD_12M, INCENTIVE, BURNOUT,
                      SEASONING_RAMP)
from . import scorecard as sc_mod
from .features import assert_no_leakage

# ===== CONFIG (user inputs) =====
KIND_LOGIT, KIND_XGB, KIND_LGBM = "logit", "xgb", "lgbm"
TARGET_PD, TARGET_PREPAY = "pd", "prepay"
MONOTONE = {
    TARGET_PD: {FICO: -1, MTM_LTV: 1, DTI: 1, UNEMP: 1, TIMES_30DPD_12M: 1},
    TARGET_PREPAY: {INCENTIVE: 1, BURNOUT: -1},
}
HINGE_KNOTS = (0.0, 0.5, 1.0, 1.5)
MTM_LTV_CAP = 120.0
TUNE_MAX_ROWS = 200_000
TUNE_INITIAL_SHARE = 0.4
EARLY_STOP = 30
TUNE_GAP = {TARGET_PD: config.TUNE_GAP_MONTHS, TARGET_PREPAY: 1}
QUICK_TUNE_ITER, QUICK_MAX_TREES, FULL_MAX_TREES = 3, 120, 500
DEFAULT_PARAMS = dict(learning_rate=0.08, max_depth=4, num_leaves=15, min_child=50, subsample=0.8,
                      colsample=0.8, reg_lambda=5.0, n_estimators=250)
SEARCH = dict(learning_rate=(0.03, 0.15), max_depth=(3, 6), num_leaves=(8, 48), min_child=(30, 300),
              subsample=(0.6, 1.0), colsample=(0.6, 1.0), reg_lambda=(1.0, 30.0))
N_JOBS = max((os.cpu_count() or 2) - 1, 1)
CALIB_METHODS = ("isotonic", "platt")
PROB_CLIP = 1e-6
# ===== END CONFIG =====


warnings.filterwarnings("ignore", message=".*eval_set.*")  # lightgbm deprecation noise


def _ycol(target):
    return TARGET_PD12 if target == TARGET_PD else TARGET_PREPAY_1M


def _dcol(target):
    return SNAPSHOT_DATE if target == TARGET_PD else PERIOD


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class Calibrator:
    method: str
    model: object

    def transform(self, margin):
        if self.method == "isotonic":
            p = self.model.predict(_sigmoid(margin))
        else:
            p = self.model.predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def _fit_calibrator(method, margin, y):
    if method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(_sigmoid(margin), y)
    else:
        m = LogisticRegression(C=1e6, max_iter=500).fit(np.asarray(margin).reshape(-1, 1), y)
    return Calibrator(method, m)


@dataclass
class Encoder:
    """Categories learned on TRAIN so every later frame is encoded identically."""
    features: list
    categories: dict

    @classmethod
    def fit(cls, df, features):
        cats = {f: [str(c) for c in pd.Series(df[f]).astype(str).unique()] for f in features
                if f in CATEGORICAL_FEATURES}
        for f in cats:
            cats[f] = sorted(cats[f])
        return cls(list(features), cats)

    def category_frame(self, df):
        out = {}
        for f in self.features:
            if f in self.categories:
                out[f] = pd.Categorical(df[f].astype(str), categories=self.categories[f])
            else:
                out[f] = df[f].to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=df.index)

    def onehot_frame(self, df):
        out = {}
        for f in self.features:
            if f in self.categories:
                s = df[f].astype(str).to_numpy()
                for c in self.categories[f]:
                    out[f"{f}={c}"] = (s == c).astype(np.float32)
            else:
                out[f] = df[f].to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=df.index)


def _logit_prepay_frame(df):
    x = df[INCENTIVE].to_numpy(dtype=np.float64)
    cols = {f"hinge_{k}": np.maximum(x - k, 0.0) for k in HINGE_KNOTS}
    cols[BURNOUT] = df[BURNOUT].to_numpy(dtype=np.float64)
    cols[SEASONING_RAMP] = df[SEASONING_RAMP].to_numpy(dtype=np.float64)
    cols[MTM_LTV] = np.minimum(df[MTM_LTV].to_numpy(dtype=np.float64), MTM_LTV_CAP)
    cols[FICO] = df[FICO].to_numpy(dtype=np.float64)
    return pd.DataFrame(cols, index=df.index)


@dataclass
class ModelBundle:
    name: str
    kind: str
    target: str
    features: list
    estimator: object
    calibrator: object = None
    encoder: object = None
    coefs: object = None
    extras: dict = field(default_factory=dict)

    @property
    def scorecard(self):
        return self.estimator if isinstance(self.estimator, sc_mod.Scorecard) else None

    def design(self, df):
        """Matrix exactly as the estimator sees it (use for SHAP)."""
        if self.kind == KIND_LGBM:
            return self.encoder.category_frame(df)
        if self.kind == KIND_XGB:
            return self.encoder.onehot_frame(df)
        if self.scorecard is not None:
            return df
        return sm.add_constant(_logit_prepay_frame(df), has_constant="add")

    def predict_margin(self, df) -> np.ndarray:
        X = self.design(df)
        if self.kind == KIND_LGBM:
            return np.asarray(self.estimator.predict(X, raw_score=True), dtype=np.float64)
        if self.kind == KIND_XGB:
            return np.asarray(self.estimator.predict(X, output_margin=True), dtype=np.float64)
        if self.scorecard is not None:
            return self.scorecard.margin(df)
        return np.asarray(X.to_numpy() @ self.estimator.params.to_numpy(), dtype=np.float64)

    def predict_proba(self, df) -> np.ndarray:
        m = self.predict_margin(df)
        return _sigmoid(m) if self.calibrator is None else self.calibrator.transform(m)

    def predict_pd(self, df) -> np.ndarray:
        return self.predict_proba(df)

    def predict_smm(self, df) -> np.ndarray:
        return self.predict_proba(df)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# ---------- calibration ----------
def calibrate(bundle: ModelBundle, calib_df, method: str = "auto") -> ModelBundle:
    margin = bundle.predict_margin(calib_df)
    y = calib_df[_ycol(bundle.target)].to_numpy(dtype=np.float64)
    if method == "auto":
        # choose out-of-sample: fit on one date-ordered half, score Brier on the other
        order = np.argsort(calib_df[_dcol(bundle.target)].to_numpy(), kind="stable")
        halves = np.array_split(order, 2)
        score = {}
        for m in CALIB_METHODS:
            errs = []
            for a, b in ((0, 1), (1, 0)):
                c = _fit_calibrator(m, margin[halves[a]], y[halves[a]])
                errs.append(np.mean((c.transform(margin[halves[b]]) - y[halves[b]]) ** 2))
            score[m] = float(np.mean(errs))
        method = min(score, key=score.get)
    bundle.calibrator = _fit_calibrator(method, margin, y)
    return bundle


# ---------- GBMs ----------
def _constraints(design_cols, monotone):
    out = []
    for c in design_cols:
        out.append(monotone.get(c, 0))
    return out


def _lib_params(kind, p, max_trees=None):
    n_est = p["n_estimators"] if max_trees is None else max_trees
    if kind == KIND_XGB:
        return dict(learning_rate=p["learning_rate"], max_depth=int(p["max_depth"]),
                    min_child_weight=p["min_child"] / 10.0, subsample=p["subsample"],
                    colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"], n_estimators=int(n_est),
                    tree_method="hist", n_jobs=N_JOBS, eval_metric="logloss", random_state=config.SEED)
    return dict(learning_rate=p["learning_rate"], num_leaves=int(p["num_leaves"]),
                min_child_samples=int(p["min_child"]), subsample=p["subsample"], subsample_freq=1,
                colsample_bytree=p["colsample"], reg_lambda=p["reg_lambda"], n_estimators=int(n_est),
                n_jobs=N_JOBS, random_state=config.SEED, verbose=-1,
                monotone_constraints_method="intermediate")


def _make_estimator(kind, p, design_cols, monotone, max_trees=None, early_stop=None, pos_weight=None):
    mc = _constraints(design_cols, monotone)
    lp = _lib_params(kind, p, max_trees)
    if kind == KIND_XGB:
        import xgboost as xgb
        if pos_weight:
            lp["scale_pos_weight"] = pos_weight
        if early_stop:
            lp["early_stopping_rounds"] = early_stop
        return xgb.XGBClassifier(monotone_constraints=tuple(mc), **lp)
    import lightgbm as lgb
    if pos_weight:
        lp["scale_pos_weight"] = pos_weight
    return lgb.LGBMClassifier(monotone_constraints=mc, **lp)


def _design(kind, enc, df):
    return enc.category_frame(df) if kind == KIND_LGBM else enc.onehot_frame(df)


def _fit_est(kind, est, X, y, Xv=None, yv=None, early_stop=None):
    if Xv is None:
        est.fit(X, y)
    elif kind == KIND_XGB:
        est.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        import lightgbm as lgb
        est.fit(X, y, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(early_stop, verbose=False)])
    return est


def _best_iter(kind, est):
    if kind == KIND_XGB:
        return int(est.best_iteration) + 1
    return int(est.best_iteration_ or est.n_estimators)


def fit_gbm(kind, train_df, target, features, params=None, monotone=None, calib_df=None,
            pos_weight=None, name=None) -> ModelBundle:
    assert_no_leakage(train_df, features)
    p = {**DEFAULT_PARAMS, **(params or {})}
    mono = MONOTONE[target] if monotone is None else monotone
    enc = Encoder.fit(train_df, features)
    X = _design(kind, enc, train_df)
    y = train_df[_ycol(target)].to_numpy()
    est = _make_estimator(kind, p, list(X.columns), mono, pos_weight=pos_weight)
    _fit_est(kind, est, X, y)
    b = ModelBundle(name or f"{target}_{kind}", kind, target, list(features), est, None, enc)
    if calib_df is not None:
        calibrate(b, calib_df)
    return b


def _fold_masks(dates, gap, folds):
    mi = dates.astype("datetime64[M]").astype(np.int64)
    lo, hi = mi.min(), mi.max() + 1
    n0 = int((hi - lo) * TUNE_INITIAL_SHARE)
    step = max((hi - lo - n0) // folds, 1)
    out = []
    for k in range(folds):
        a = lo + n0 + k * step
        z = hi if k == folds - 1 else a + step
        tr, va = mi < a - gap, (mi >= a) & (mi < z)
        if tr.sum() > 0 and va.sum() > 0:
            out.append((tr, va))
    return out


def tune_time_cv(kind, train_df, target, features, n_iter=config.TUNE_ITER, folds=config.TUNE_FOLDS,
                 quick=False, seed=config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    df = train_df
    if len(df) > TUNE_MAX_ROWS:
        df = df.iloc[np.sort(rng.choice(len(df), TUNE_MAX_ROWS, replace=False))]
    enc = Encoder.fit(df, features)
    X = _design(kind, enc, df)
    y = df[_ycol(target)].to_numpy()
    masks = _fold_masks(df[_dcol(target)].to_numpy(), TUNE_GAP[target], folds)
    mono = MONOTONE[target]
    max_trees = QUICK_MAX_TREES if quick else FULL_MAX_TREES
    best, best_loss = dict(DEFAULT_PARAMS), np.inf
    for it in range(n_iter):
        cand = dict(
            learning_rate=float(np.exp(rng.uniform(np.log(SEARCH["learning_rate"][0]), np.log(SEARCH["learning_rate"][1])))),
            max_depth=int(rng.integers(SEARCH["max_depth"][0], SEARCH["max_depth"][1] + 1)),
            num_leaves=int(rng.integers(SEARCH["num_leaves"][0], SEARCH["num_leaves"][1] + 1)),
            min_child=int(rng.integers(SEARCH["min_child"][0], SEARCH["min_child"][1] + 1)),
            subsample=float(rng.uniform(*SEARCH["subsample"])), colsample=float(rng.uniform(*SEARCH["colsample"])),
            reg_lambda=float(np.exp(rng.uniform(np.log(SEARCH["reg_lambda"][0]), np.log(SEARCH["reg_lambda"][1])))))
        if quick:
            cand["max_depth"], cand["num_leaves"] = min(cand["max_depth"], 3), min(cand["num_leaves"], 8)
            cand["learning_rate"] = max(cand["learning_rate"], 0.1)
        losses, iters = [], []
        for tr, va in masks:
            est = _make_estimator(kind, cand, list(X.columns), mono, max_trees=max_trees,
                                  early_stop=EARLY_STOP if kind == KIND_XGB else None)
            _fit_est(kind, est, X[tr], y[tr], X[va], y[va], EARLY_STOP)
            pv = np.clip(est.predict_proba(X[va])[:, 1], PROB_CLIP, 1 - PROB_CLIP)
            yv = y[va]
            losses.append(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
            iters.append(_best_iter(kind, est))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best = {**cand, "n_estimators": int(max(np.median(iters) * 1.15, 20))}
    best["cv_logloss"] = best_loss
    return best


# ---------- logits ----------
def fit_logit_prepay(train_df, calib_df=None) -> ModelBundle:
    assert_no_leakage(train_df, FEATURES_PREPAY)
    X = sm.add_constant(_logit_prepay_frame(train_df), has_constant="add")
    y = train_df[TARGET_PREPAY_1M].to_numpy(dtype=np.float64)
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=100)
    except Exception:
        res = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=500)
    knots = {f"hinge_{k}": k for k in HINGE_KNOTS}
    coefs = pd.DataFrame(dict(term=list(res.params.index), coef=res.params.to_numpy(),
                              knot=[knots.get(t, np.nan) for t in res.params.index]))
    b = ModelBundle("pp_logit", KIND_LOGIT, TARGET_PREPAY, list(FEATURES_PREPAY), res, None, None, coefs)
    if calib_df is not None:
        calibrate(b, calib_df)
    return b


def fit_scorecard_pd(train_df, calib_df=None) -> ModelBundle:
    assert_no_leakage(train_df, FEATURES_PD)
    sc = sc_mod.fit_scorecard(train_df, TARGET_PD12, FEATURES_PD)
    b = ModelBundle("pd_logit", KIND_LOGIT, TARGET_PD, list(FEATURES_PD), sc)
    if calib_df is not None:
        calibrate(b, calib_df)
    return b


# ---------- driver ----------
def train_all(splits_pd, splits_pp, quick: bool = False) -> dict:
    n_iter = QUICK_TUNE_ITER if quick else config.TUNE_ITER
    folds = 2 if quick else config.TUNE_FOLDS
    out = {"pd_logit": fit_scorecard_pd(splits_pd["train"], splits_pd["calib"]),
           "pp_logit": fit_logit_prepay(splits_pp["train"], splits_pp["calib"])}
    specs = ((TARGET_PD, "pd", FEATURES_PD, splits_pd), (TARGET_PREPAY, "pp", FEATURES_PREPAY, splits_pp))
    for target, tag, feats, sp in specs:
        for kind in (KIND_XGB, KIND_LGBM):
            tuned = tune_time_cv(kind, sp["train"], target, feats, n_iter, folds, quick=quick)
            params = {k: v for k, v in tuned.items() if k in DEFAULT_PARAMS}
            if quick:
                params["n_estimators"] = min(params["n_estimators"], QUICK_MAX_TREES)
            b = fit_gbm(kind, sp["train"], target, feats, params, calib_df=sp["calib"], name=f"{tag}_{kind}")
            b.extras["tuned_params"] = params
            out[f"{tag}_{kind}"] = b
    # one class-weighted variant, reported only as a sensitivity row
    ytr = splits_pd["train"][TARGET_PD12].to_numpy()
    w = fit_gbm(KIND_XGB, splits_pd["train"], TARGET_PD, FEATURES_PD, out["pd_xgb"].extras["tuned_params"],
                pos_weight=float((1 - ytr.mean()) / max(ytr.mean(), 1e-9)), name="pd_xgb_weighted")
    out["pd_xgb"].extras["weighted"] = w
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, b in out.items():
        b.save(config.MODEL_DIR / f"{k}.joblib")
    return out
