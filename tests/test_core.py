import numpy as np
import pandas as pd

from marbling_forecast.config import Config
from marbling_forecast.data import make_demo_data
from marbling_forecast.features import build_features
from marbling_forecast.models import reconcile
from marbling_forecast.pipeline import run


def test_reconcile_sums_to_one_and_nonnegative():
    out = reconcile(np.array([[0.5, 0.4, -0.1, 0.3], [0, 0, 0, 0]]))
    assert np.allclose(out.sum(1), 1) and (out >= 0).all()


def test_features_have_no_lookahead():
    y = make_demo_data(80)
    X, anchor, yy = build_features(y)
    # lag1 of row t must equal y at t-1
    pos = y.index.get_loc(X.index[10])
    assert np.allclose(X.iloc[10][[f"{c}_lag1" for c in y.columns]].values, y.iloc[pos - 1].values)


def test_pipeline_runs_and_beats_nothing_silly():
    res = run(make_demo_data(100), Config(min_train=50, n_iter=3, cv_splits=3, n_jobs=1))
    assert np.allclose(res.forecast.sum(1), 1)
    assert res.overall["MAE"].min() < res.overall.loc["naive_last", "MAE"] * 1.2


class _FakeTimesFM:
    """Stand-in for TimesFM3Evaluator: repeats the last context value, so it must equal `naive_last`."""

    def predict_batch(self, contexts, horizon, return_quantiles=True, use_symmetric_averaging=False, **_):
        from types import SimpleNamespace
        for c in contexts:
            c = np.asarray(c)
            last = c[..., -1:]                                   # (G,1) multivariate or (1,) univariate
            f = np.repeat(last, horizon, axis=-1)                # (G,H) or (H,)
            q = np.stack([f + k * 0.01 for k in range(-4, 5)], axis=-1)   # (...,9)
            yield SimpleNamespace(forecast=f, quantiles=q)


def test_timesfm_adapter_is_leak_free_and_aligned():
    from marbling_forecast.foundation import TimesFM3Forecaster
    y = make_demo_data(90)
    origins = y.index[60:70]
    for multi in (True, False):
        tf, q10, q90 = TimesFM3Forecaster(evaluator=_FakeTimesFM(), multivariate=multi).backtest(y, origins)
        expect = y.shift(1).loc[origins]                          # last value strictly before each origin
        assert np.allclose(tf.values, (expect.div(expect.sum(1), axis=0)).values, atol=1e-6)
        assert (q10.values <= q90.values).all()


def test_pipeline_with_timesfm_standin():
    cfg = Config(min_train=50, n_iter=3, cv_splits=3, n_jobs=1)
    res = run(make_demo_data(100), cfg, foundation=_FakeTimesFM_adapter())
    assert {"timesfm3", "ensemble_tfm"} <= set(res.preds)
    assert np.allclose(res.forecast.sum(1), 1) and res.tfm_coverage is not None
    assert np.allclose(res.preds["timesfm3"].values, res.preds["naive_last"].values, atol=1e-6)


def _FakeTimesFM_adapter():
    from marbling_forecast.foundation import TimesFM3Forecaster
    return TimesFM3Forecaster(evaluator=_FakeTimesFM())


def test_final_forecast_paths_when_timesfm_wins():
    """An oracle stand-in (peeks at the truth) forces timesfm3 / ensemble_tfm to be 'best' -> exercises those branches."""
    from marbling_forecast.data import normalise_shares
    from marbling_forecast.foundation import TimesFM3Forecaster
    from marbling_forecast.models import reconcile

    class Oracle(TimesFM3Forecaster):
        def backtest(self, y_all, origins):
            truth = normalise_shares(y_all).loc[origins]
            return truth, truth - 0.01, truth + 0.01

    raw = make_demo_data(100)
    res = run(raw, Config(min_train=50, n_iter=3, cv_splits=3, n_jobs=1, horizon=2), foundation=Oracle(evaluator=_FakeTimesFM()))
    assert res.best == "timesfm3" and len(res.forecast) == 2 and np.allclose(res.forecast.sum(1), 1)
    assert res.tfm_coverage == 1.0


def test_final_forecast_path_for_ensemble_tfm():
    """TimesFM stand-in with independent noise -> averaging it with the ML ensemble should win."""
    from marbling_forecast.data import normalise_shares
    from marbling_forecast.foundation import TimesFM3Forecaster

    class Noisy(TimesFM3Forecaster):
        def backtest(self, y_all, origins):
            truth = normalise_shares(y_all).loc[origins]
            n = truth + np.random.default_rng(1).normal(0, 0.045, truth.shape)
            return n, n - 0.05, n + 0.05

    res = run(make_demo_data(100), Config(min_train=50, n_iter=3, cv_splits=3, n_jobs=1, horizon=3),
              foundation=Noisy(evaluator=_FakeTimesFM()))
    assert res.best == "ensemble_tfm", res.overall["MAE"].round(4).to_dict()
    assert res.forecast.shape == (3, 4) and np.allclose(res.forecast.sum(1), 1)
