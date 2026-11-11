"""CLI:  python run_forecast.py --xlsx data.xlsx   |   python run_forecast.py --demo"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from marbling_forecast import plots
from marbling_forecast.config import Config
from marbling_forecast.data import load_excel, make_demo_data, normalise_shares
from marbling_forecast.excel_io import write_forecasts
from marbling_forecast.foundation import LICENSE_NOTICE
from marbling_forecast.pipeline import run


warnings.filterwarnings("ignore")  # sklearn convergence chatter on tiny CV folds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, help="workbook with <grade>_act%% columns")
    ap.add_argument("--sheet", default="Python")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--horizon", type=int, default=1, help="steps ahead (weeks)")
    ap.add_argument("--min-train", type=int, default=60)
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--write-excel", type=Path, help="write forecasts into a COPY of the workbook")
    ap.add_argument("--timesfm", action="store_true", help="add Google TimesFM 3.0 (optional install, see README)")
    ap.add_argument("--tfm-device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--tfm-checkpoint", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--tfm-univariate", action="store_true", help="forecast each grade separately instead of jointly")
    a = ap.parse_args()

    cfg = Config(sheet=a.sheet, horizon=a.horizon, min_train=a.min_train, refit_every=a.refit_every,
                 use_timesfm=a.timesfm, tfm_device=a.tfm_device, tfm_checkpoint=a.tfm_checkpoint,
                 tfm_multivariate=not a.tfm_univariate)
    if a.timesfm:
        print("NOTE:", LICENSE_NOTICE)
    raw = make_demo_data() if a.demo else load_excel(a.xlsx, a.sheet)
    try:
        res = run(raw, cfg)
    except ImportError as e:  # optional TimesFM install missing
        raise SystemExit(f"ERROR: {e}")

    a.out.mkdir(parents=True, exist_ok=True)
    pd.options.display.float_format = "{:.4f}".format
    print("\n=== Backtest leaderboard (lower MAE is better) ===\n", res.overall)
    print(f"\nBest: {res.best}\n", res.best_scores)
    print("\n=== Next forecast ===\n", res.forecast)
    if res.tfm_coverage is not None:
        print(f"\nTimesFM-3 native 10-90 % band covered {res.tfm_coverage:.0%} of actuals (nominal 80 %)")
    print(f"\nTimings: {res.timings}")

    res.overall.to_csv(a.out / "leaderboard.csv")
    res.best_scores.to_csv(a.out / "best_model_metrics.csv")
    res.forecast.to_csv(a.out / "next_forecast.csv")
    res.preds[res.best].to_csv(a.out / "backtest_predictions.csv")
    plots.plot_actual_vs_pred(res.actual, res.preds[res.best], res.lo, res.hi, a.out / "actual_vs_forecast.png",
                              f"Walk-forward backtest - {res.best} ({int(cfg.ci * 100)}% interval)")
    plots.plot_model_comparison(res.by_grade_mae.assign(ALL=res.overall["MAE"]), a.out / "model_comparison.png")
    plots.plot_residual_diagnostics(res.actual, res.preds[res.best], a.out / "residual_diagnostics.png")
    plots.plot_future(normalise_shares(raw), res.forecast, res.forecast_lo, res.forecast_hi, a.out / "next_forecast.png")
    if a.write_excel:
        write_forecasts(a.xlsx, a.write_excel, a.sheet, res.preds[res.best], res.forecast)
        print("Excel written ->", a.write_excel)


if __name__ == "__main__":
    main()
