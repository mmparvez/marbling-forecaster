"""Vectorised feature engineering (done once for the whole series, then sliced)."""
import numpy as np
import pandas as pd


def build_features(y: pd.DataFrame, lags=3, roll_windows=(4, 8), anchor_window=8):
    """Return X (info available *before* each row), anchor and y.

    Every feature is shifted by one row so row t only uses data up to t-1 -> no leakage,
    which means the feature matrix can be built once and sliced per walk-forward step.
    """
    past = y.shift(1)
    parts = [past.shift(k).add_suffix(f"_lag{k + 1}") if k else past.add_suffix("_lag1") for k in range(lags)]
    for w in roll_windows:
        parts.append(past.rolling(w, min_periods=2).mean().add_suffix(f"_ma{w}"))
    doy = y.index.dayofyear.values
    parts.append(pd.DataFrame({"sin_y": np.sin(2 * np.pi * doy / 365.25),
                               "cos_y": np.cos(2 * np.pi * doy / 365.25),
                               "trend": np.arange(len(y)) / len(y)}, index=y.index))
    X = pd.concat(parts, axis=1)
    anchor = past.rolling(anchor_window, min_periods=2).mean()
    valid = X.notna().all(axis=1) & anchor.notna().all(axis=1)
    return X[valid], anchor[valid], y[valid]
