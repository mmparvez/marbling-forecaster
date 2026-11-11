"""End-to-end: tune once -> walk-forward backtest -> score -> final forecast."""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from .backtest import baselines, walk_forward
from .config import Config
from .data import normalise_shares
from .features import build_features
from .foundation import TimesFM3Forecaster
from .metrics import score
from .models import reconcile, tune_all


@dataclass
class Result:
    actual: pd.DataFrame
    preds: dict            # model/baseline name -> DataFrame
    overall: pd.DataFrame  # per-model summary
    by_grade_mae: pd.DataFrame
    best: str
    best_scores: pd.DataFrame
    lo: pd.DataFrame
    hi: pd.DataFrame
    forecast: pd.DataFrame
    forecast_lo: pd.DataFrame
    forecast_hi: pd.DataFrame
    cv_scores: dict
    timings: dict
    tfm_coverage: float = None  # share of actuals inside TimesFM's native 10-90 % band


def _next_dates(last, n):
    return pd.date_range(last, periods=n + 1, freq="W-FRI")[1:]


def _forecast_ahead(est, y, cfg, h):
    """Recursive h-step forecast: predict, append, repeat (features are leak-free by construction)."""
    y = y.copy()
    out = []
    for d in _next_dates(y.index[-1], h):
        y.loc[d] = np.nan
        X, anchor, _ = build_features(y, cfg.lags, cfg.roll_windows, cfg.anchor_window)
        p = reconcile(est.predict(X.iloc[[-1]]) + anchor.iloc[[-1]].values)[0]
        y.loc[d] = p
        out.append(pd.Series(p, index=y.columns, name=d))
    return pd.DataFrame(out)


def run(raw: pd.DataFrame, cfg: Config = Config(), foundation=None) -> Result:
    """`foundation` may be any object with the TimesFM3Forecaster interface (defaults to TimesFM 3.0 when enabled)."""
    t0 = time.perf_counter()
    if foundation is None and cfg.use_timesfm:
        foundation = TimesFM3Forecaster(cfg.tfm_checkpoint, cfg.tfm_device, cfg.tfm_batch,
                                        cfg.tfm_max_context, cfg.tfm_multivariate)
    y_all = normalise_shares(raw)
    X, anchor, y = build_features(y_all, cfg.lags, cfg.roll_windows, cfg.anchor_window)
    resid = y - anchor

    t = time.perf_counter()
    models, cv_scores = tune_all(X.iloc[:cfg.min_train], resid.iloc[:cfg.min_train], cfg)
    t_tune = time.perf_counter() - t

    t = time.perf_counter()
    preds = walk_forward(models, X, anchor, y, cfg.min_train, cfg.refit_every, cfg.n_jobs)
    t_bt = time.perf_counter() - t
    idx = next(iter(preds.values())).index
    preds.update(baselines(X, y, idx))
    actual = y.loc[idx]

    t_tfm, tfm_cov = 0.0, None
    if foundation is not None:  # zero-shot: no fitting, one batched pass over all origins
        t = time.perf_counter()
        tf, q10, q90 = foundation.backtest(y_all, idx)
        preds["timesfm3"] = tf
        preds["ensemble_tfm"] = pd.DataFrame(reconcile((preds["ensemble"].values + tf.values) / 2),
                                             index=idx, columns=y.columns)
        tfm_cov = float(((actual >= q10) & (actual <= q90)).to_numpy().mean())
        t_tfm = time.perf_counter() - t

    tables = {n: score(actual, p) for n, p in preds.items()}
    overall = pd.DataFrame({n: s.loc["ALL"] for n, s in tables.items()}).T.sort_values("MAE")
    mae_grade = pd.DataFrame({n: s["MAE"] for n, s in tables.items()}).T
    candidates = list(models) + ["ensemble"] + (["timesfm3", "ensemble_tfm"] if foundation is not None else [])
    best = next(n for n in overall.index if n in candidates)

    err = preds[best] - actual
    a = (1 - cfg.ci) / 2
    q_lo, q_hi = err.quantile(a), err.quantile(1 - a)
    lo = pd.DataFrame(np.clip(preds[best].values - q_hi.values, 0, 1), index=idx, columns=y.columns)
    hi = pd.DataFrame(np.clip(preds[best].values - q_lo.values, 0, 1), index=idx, columns=y.columns)

    # final forecast: refit the chosen ML model(s) on ALL data and/or ask TimesFM, then average
    parts, dates = [], None
    if best != "timesfm3":
        names = list(models) if best in ("ensemble", "ensemble_tfm") else [best]
        fitted = [clone(models[n]).fit(X, resid) for n in names]

        class _Avg:
            def predict(self, Z):
                return np.mean([f.predict(Z) for f in fitted], axis=0)

        fc_ml = _forecast_ahead(_Avg(), y_all, cfg, cfg.horizon)
        parts.append(fc_ml.values)
        dates = fc_ml.index
    if best in ("timesfm3", "ensemble_tfm"):
        dates = _next_dates(y_all.index[-1], cfg.horizon)
        parts.append(foundation.forecast_future(y_all, cfg.horizon))
    fc = pd.DataFrame(reconcile(np.mean(parts, axis=0)), index=dates, columns=y.columns)
    fc_lo = pd.DataFrame(np.clip(fc.values - q_hi.values, 0, 1), index=fc.index, columns=fc.columns)
    fc_hi = pd.DataFrame(np.clip(fc.values - q_lo.values, 0, 1), index=fc.index, columns=fc.columns)

    return Result(actual, preds, overall, mae_grade, best, tables[best], lo, hi,
                  fc, fc_lo, fc_hi, cv_scores,
                  {"tune_s": t_tune, "backtest_s": t_bt, "timesfm_s": t_tfm, "total_s": time.perf_counter() - t0,
                   "n_origins": len(idx)}, tfm_cov)
