"""Model zoo, one-off hyper-parameter tuning, and compositional reconciliation."""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def model_zoo(seed=42):
    """name -> (estimator, param distribution). All natively/explicitly multi-output."""
    sc = StandardScaler
    return {
        "ridge": (make_pipeline(sc(), Ridge()), {"ridge__alpha": loguniform(1, 1e3)}),
        "elasticnet": (make_pipeline(sc(), ElasticNet(max_iter=5000)),
                       {"elasticnet__alpha": loguniform(1e-4, 1e-1), "elasticnet__l1_ratio": uniform(0.1, 0.9)}),
        "knn": (make_pipeline(sc(), KNeighborsRegressor()),
                {"kneighborsregressor__n_neighbors": randint(3, 11),
                 "kneighborsregressor__weights": ["uniform", "distance"]}),
        "svr": (make_pipeline(sc(), MultiOutputRegressor(SVR())),
                {"multioutputregressor__estimator__C": loguniform(1e-2, 1e1),
                 "multioutputregressor__estimator__epsilon": loguniform(1e-3, 3e-2),
                 "multioutputregressor__estimator__gamma": loguniform(1e-3, 1e-1)}),
        "random_forest": (RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=1),
                          {"max_depth": randint(2, 6), "min_samples_leaf": randint(3, 12),
                           "max_features": uniform(0.3, 0.7)}),
        "extra_trees": (ExtraTreesRegressor(n_estimators=150, random_state=seed, n_jobs=1),
                        {"max_depth": randint(2, 6), "min_samples_leaf": randint(3, 12),
                         "max_features": uniform(0.3, 0.7)}),
    }


def tune_all(X, resid, cfg):
    """Tune every model once on the pre-backtest window; the backtest then reuses those hyper-parameters."""
    cv = TimeSeriesSplit(n_splits=cfg.cv_splits)
    tuned, cv_scores = {}, {}
    for name, (est, dist) in model_zoo(cfg.seed).items():
        search = RandomizedSearchCV(est, dist, n_iter=cfg.n_iter, cv=cv, random_state=cfg.seed,
                                    scoring="neg_mean_absolute_error", n_jobs=cfg.n_jobs, refit=False)
        search.fit(X, resid)
        best = search.best_params_
        tuned[name] = clone(est).set_params(**best)
        cv_scores[name] = -search.best_score_
    return tuned, cv_scores


def reconcile(pred):
    """Clip to [0, 1] and rescale so the grade shares sum to exactly 1.

    Closed-form, so shares are never negative and always total exactly 100 %.
    """
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    s = pred.sum(axis=-1, keepdims=True)
    return np.divide(pred, s, out=np.full_like(pred, 1 / pred.shape[-1]), where=s > 0)
