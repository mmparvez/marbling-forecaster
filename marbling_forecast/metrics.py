import numpy as np
import pandas as pd


def score(actual: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Per-grade + overall MAE, RMSE, sMAPE, R2, bias."""
    rows = {}
    for c in actual.columns:
        a, p = actual[c].values, pred[c].values
        err = p - a
        ss_tot = ((a - a.mean()) ** 2).sum()
        rows[c] = {
            "MAE": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "sMAPE_%": 100 * np.mean(2 * np.abs(err) / np.maximum(np.abs(a) + np.abs(p), 1e-9)),
            "R2": 1 - (err ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
            "Bias": err.mean(),
        }
    out = pd.DataFrame(rows).T
    out.loc["ALL"] = out.mean()
    out.loc["ALL", "Bias"] = out["Bias"].abs().mean()  # signed biases cancel across shares, so use |bias|
    return out
