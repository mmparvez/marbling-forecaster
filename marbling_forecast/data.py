"""Loading, cleaning and demo-data generation."""
import numpy as np
import pandas as pd

from .config import DATE_COL, GRADES


def load_excel(path, sheet="Python"):
    """Read actual grade shares by *column name* (no hard-coded column letters)."""
    df = pd.read_excel(path, sheet_name=sheet)
    act_cols = [f"{g}_act%" for g in GRADES]
    df = df[[DATE_COL] + act_cols].dropna(subset=[DATE_COL] + act_cols)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df.columns = GRADES
    return df.astype(float)


def normalise_shares(df):
    """Make the four tracked grades sum to 1 (the MB_3 band is ~0 in recent data)."""
    return df.div(df.sum(axis=1), axis=0)


def make_demo_data(n=130, seed=0):
    """Synthetic data so the repo runs without the (private) plant data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-04", periods=n * 2)[::2][:n]
    t = np.arange(n)
    season = 0.03 * np.sin(2 * np.pi * t / 26)
    p = np.column_stack([
        0.40 - 0.0015 * t + season,
        0.38 + 0.0010 * t - season,
        0.20 + 0.0004 * t,
        0.02 + 0.0001 * t,
    ])
    p = np.clip(p + rng.normal(0, 0.04, p.shape), 0.001, None)
    return pd.DataFrame(p / p.sum(1, keepdims=True), index=pd.DatetimeIndex(dates, name=DATE_COL), columns=GRADES)

