"""Optional zero-shot forecaster: Google TimesFM 3.0 (330M-parameter time-series foundation model).

TimesFM needs no training on your data. It reads the history before each date and forecasts ahead,
natively handling several related series at once, so the four grade shares are forecast jointly
(or one by one with ``multivariate=False``). It also returns 9 quantiles (10th-90th percentile).

LICENSE: the TimesFM 3.0 *pretrained weights* are under the TimesFM Non-Commercial License v1.0
(non-commercial, non-production use only). The code is Apache-2.0. Check the licence before using
this in a business workflow.
"""
import numpy as np
import pandas as pd

from .models import reconcile

LICENSE_NOTICE = (
    "TimesFM 3.0 weights are released under the TimesFM Non-Commercial License v1.0 "
    "(non-commercial, non-production use only). Make sure your use complies."
)
INSTALL_HINT = "TimesFM 3.0 is optional. Install it with:  pip install -r requirements-timesfm.txt  (Python >= 3.10)"
Q_LO, Q_HI = 0, 8   # positions of the 10th and 90th percentile in the 9 returned quantiles


class TimesFM3Forecaster:
    """Thin adapter that turns walk-forward histories into TimesFM 3.0 calls.

    Pass ``evaluator`` to inject any object exposing ``predict_batch`` (used for testing).
    """

    name = "timesfm3"

    def __init__(self, checkpoint="google/timesfm-3.0-pytorch", device="cpu", batch_size=16,
                 max_context=1024, multivariate=True, evaluator=None):
        self.checkpoint, self.device, self.batch_size = checkpoint, device, batch_size
        self.max_context, self.multivariate = max_context, multivariate
        self._ev = evaluator

    # ------------------------------------------------------------------ model handling
    def _load(self):
        if self._ev is None:
            try:
                from timesfm3 import ModelConfig, TimesFM3Evaluator
            except ImportError as e:  # pragma: no cover - depends on optional install
                raise ImportError(INSTALL_HINT) from e
            self._ev = TimesFM3Evaluator(ModelConfig(checkpoint_path=self.checkpoint,
                                                     per_core_batch_size=self.batch_size,
                                                     device=self.device))
        return self._ev

    def _call(self, contexts, horizon):
        ev, outs = self._load(), []
        for k in range(0, len(contexts), self.batch_size):  # keep memory flat for long backtests
            outs.extend(ev.predict_batch(contexts=contexts[k:k + self.batch_size], horizon=horizon,
                                         return_quantiles=True, use_symmetric_averaging=False))
        return outs

    # ------------------------------------------------------------------ core prediction
    def predict_histories(self, histories, horizon=1):
        """histories: list of DataFrames (rows = time, columns = grades).

        Returns point forecasts (n, horizon, n_grades) and quantiles (n, horizon, n_grades, 9).
        """
        g = histories[0].shape[1]
        if self.multivariate:
            ctx = [h.iloc[-self.max_context:].to_numpy(np.float32).T for h in histories]   # (grades, time)
            outs = self._call(ctx, horizon)
            fc = np.stack([np.asarray(o.forecast) for o in outs]).transpose(0, 2, 1)          # (n, H, G)
            q = np.stack([np.asarray(o.quantiles) for o in outs]).transpose(0, 2, 1, 3)      # (n, H, G, 9)
        else:
            ctx = [h[c].iloc[-self.max_context:].to_numpy(np.float32) for h in histories for c in h.columns]
            outs = self._call(ctx, horizon)
            fc = np.stack([np.asarray(o.forecast) for o in outs]).reshape(len(histories), g, horizon).transpose(0, 2, 1)
            q = np.stack([np.asarray(o.quantiles) for o in outs]).reshape(len(histories), g, horizon, -1).transpose(0, 2, 1, 3)
        return fc, q

    # ------------------------------------------------------------------ pipeline hooks
    def backtest(self, y_all: pd.DataFrame, origins: pd.DatetimeIndex):
        """One-step-ahead forecast for each origin date using only data strictly before it."""
        histories = [y_all[y_all.index < d] for d in origins]
        fc, q = self.predict_histories(histories, horizon=1)
        mk = lambda a: pd.DataFrame(a, index=origins, columns=y_all.columns)  # noqa: E731
        return mk(reconcile(fc[:, 0, :])), mk(q[:, 0, :, Q_LO]), mk(q[:, 0, :, Q_HI])

    def forecast_future(self, y_all: pd.DataFrame, horizon=1):
        """Multi-step forecast beyond the last observation -> array (horizon, grades)."""
        fc, _ = self.predict_histories([y_all], horizon=horizon)
        return reconcile(fc[0])
