from dataclasses import dataclass

GRADES = ["MB_4", "MB_6", "MB_8", "MB_9"]
DATE_COL = "Kill Date"


@dataclass
class Config:
    sheet: str = "Python"
    lags: int = 3                # autoregressive lags of every grade share
    roll_windows: tuple = (4, 8)  # trailing-mean windows used as features / baseline
    anchor_window: int = 8       # models learn the deviation from this trailing mean
    min_train: int = 60          # first walk-forward origin (rows of history)
    refit_every: int = 1         # refit cadence inside the backtest (hyper-params stay fixed)
    cv_splits: int = 4           # TimeSeriesSplit folds for the one-off tuning
    n_iter: int = 12             # random-search draws per model
    horizon: int = 1             # steps ahead to forecast
    ci: float = 0.80             # prediction-interval level (from backtest residuals)
    n_jobs: int = -1
    seed: int = 42

    # --- optional TimesFM 3.0 foundation model (off by default; see foundation.py for licence) ---
    use_timesfm: bool = False
    tfm_checkpoint: str = "google/timesfm-3.0-pytorch"
    tfm_device: str = "cpu"       # "cpu", "cuda" or "mps"
    tfm_batch: int = 16
    tfm_max_context: int = 1024   # most recent points handed to the model
    tfm_multivariate: bool = True  # forecast the four grades jointly (False = one series at a time)
