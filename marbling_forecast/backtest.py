"""Expanding-window walk-forward backtest with fixed (pre-tuned) hyper-parameters."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from .models import reconcile


def _step(models, Xtr, rtr, xte):
    out = {}
    for name, est in models.items():
        out[name] = clone(est).fit(Xtr, rtr).predict(xte)[0]
    return out


def walk_forward(models, X, anchor, y, min_train=60, refit_every=1, n_jobs=-1):
    """Predict each row i >= min_train using only rows < i.

    Models are refit every `refit_every` origins (cached in between) -> big speed knob.
    Returns dict name -> DataFrame of reconciled predictions, aligned with y.
    """
    resid = y - anchor
    origins = np.arange(min_train, len(X))

    def run_chunk(chunk):
        fitted = {n: clone(m).fit(X.iloc[:chunk[0]], resid.iloc[:chunk[0]]) for n, m in models.items()}
        rows = []
        for i in chunk:  # reuse the fit for refit_every consecutive origins
            rows.append({n: f.predict(X.iloc[[i]])[0] for n, f in fitted.items()})
        return rows

    chunks = [origins[k:k + refit_every] for k in range(0, len(origins), refit_every)]
    res = Parallel(n_jobs=n_jobs)(delayed(run_chunk)(c) for c in chunks)
    flat = [r for chunk in res for r in chunk]

    idx = y.index[origins]
    preds = {}
    for n in models:
        raw = np.vstack([r[n] for r in flat]) + anchor.iloc[origins].values
        preds[n] = pd.DataFrame(reconcile(raw), index=idx, columns=y.columns)
    preds["ensemble"] = pd.DataFrame(reconcile(np.mean([preds[n].values for n in models], axis=0)),
                                     index=idx, columns=y.columns)
    return preds


def baselines(X, y, origins_idx):
    """Naive (last value) and trailing-mean benchmarks the model must beat."""
    cols = list(y.columns)
    out = {"naive_last": X.loc[origins_idx, [f"{c}_lag1" for c in cols]].values}
    for w in (4, 8):
        out[f"moving_avg_{w}"] = X.loc[origins_idx, [f"{c}_ma{w}" for c in cols]].values
    return {k: pd.DataFrame(reconcile(v), index=origins_idx, columns=cols) for k, v in out.items()}
