# Workflow and Performance-Metrics Guide

This guide explains (1) what the script does from input to forecast and (2) how to read the performance matrix.

## 1. What is being forecast

Each kill date, the plant produces a distribution of carcasses across marbling-score bands: **MB_4, MB_6, MB_8 and MB_9**. Each band is a share of the day's kills, and the four shares add up to 100 %. The goal is to forecast that distribution for the next kill date.

Because the four shares move together (if one rises, another falls), the script forecasts them jointly and makes sure the output is a valid distribution.

## 2. Workflow at a glance

### Step by step

| # | Step | What happens | Code |
|---|---|---|---|
| 1 | **Load** | Reads `Kill Date` and the four `<grade>_act%` columns (`MB_4_act%`, `MB_6_act%`, `MB_8_act%`, `MB_9_act%`) by header name. Rows are sorted by date. | `data.py` |
| 2 | **Normalise** | Rescales the four shares so each row sums to 1 (the small MB_3 band is excluded). | `data.py` |
| 3 | **Features** | For each date, builds inputs from **earlier dates only**: the last 3 values of each grade, trailing 4- and 8-period averages, annual seasonality (sine/cosine of day of year) and a time trend. | `features.py` |
| 4 | **Target** | Models do not predict the share directly. They predict the **deviation from the trailing 8-period average**. With no real signal, a model therefore stays close to a sensible baseline. | `features.py`, `pipeline.py` |
| 5 | **Tune once** | Six models (Ridge, ElasticNet, KNN, SVR, Random Forest, Extra Trees) get a random hyper-parameter search using time-series cross-validation on the first 60 rows. Folds always train on the past and test on the future. | `models.py` |
| 6 | **Walk-forward backtest** | Starting at row 60, for each date: fit on everything before it, predict that date, move forward by one. Hyper-parameters stay fixed, so this is fast. | `backtest.py` |
| 7 | **Reconcile** | Negative predictions are set to 0 and the four shares are rescaled to total exactly 100 %. | `models.py` |
| 8 | **Baselines** | Computes naive (repeat last value) and 4- and 8-period moving-average forecasts on the same dates. | `backtest.py` |
| 9 | **Ensemble** | Averages all six models' predictions. | `backtest.py` |
| 9a | **TimesFM 3.0 (optional)** | With `--timesfm`, the pretrained foundation model forecasts each backtest date from the history before it (no fitting), all four grades jointly. Adds `timesfm3` and `ensemble_tfm` (average of the ML ensemble and TimesFM) to the comparison. | `foundation.py` |
| 10 | **Score** | Computes the metrics below per model and per grade. | `metrics.py` |
| 11 | **Intervals** | Takes the 10th and 90th percentile of the best model's backtest errors to build an 80 % prediction band. | `pipeline.py` |
| 12 | **Final forecast** | Refits the best model on all available data and forecasts the next date. For horizons above 1, it forecasts one step, appends it, and repeats. | `pipeline.py` |
| 13 | **Outputs** | Writes CSV tables and PNG charts to `outputs/`, and optionally the forecasts into a copy of the workbook. | `run_forecast.py`, `plots.py`, `excel_io.py` |

### Why it is designed this way

- **No look-ahead (leakage).** Every feature for a date is shifted so it only sees earlier data. A test in `tests/test_core.py` checks this.
- **Tune once, refit often.** Hyper-parameter search is the expensive part. It runs once; the backtest then only refits models with fixed settings. `--refit-every k` refits every k-th step for extra speed.
- **Honest benchmarking.** A model is only useful if it beats a trivial forecast, so baselines are scored on exactly the same dates.
- **Valid shares.** Forecasts are never negative and always sum to 100 %.

### Outputs

| File | Contents |
|---|---|
| `leaderboard.csv` | Overall metrics for every model and baseline, sorted by MAE |
| `best_model_metrics.csv` | Per-grade metrics for the best model |
| `backtest_predictions.csv` | The best model's forecast for every backtest date |
| `next_forecast.csv` | The forecast for the next date(s) |
| `actual_vs_forecast.png` | Backtest forecast vs actual, with interval |
| `model_comparison.png` | Heat-map and bar chart comparing models |
| `residual_diagnostics.png` | Four error-diagnostic plots |
| `next_forecast.png` | Next forecast with interval, after recent history |

### Configuration (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `lags` | 3 | Number of past values per grade used as features |
| `roll_windows` | (4, 8) | Trailing-average windows used as features |
| `anchor_window` | 8 | Trailing average the models correct |
| `min_train` | 60 | History rows before the first backtest forecast |
| `refit_every` | 1 | Refit cadence in the backtest |
| `cv_splits`, `n_iter` | 4, 12 | Cross-validation folds and search draws per model |
| `horizon` | 1 | Steps ahead to forecast |
| `ci` | 0.80 | Prediction-interval level |
| `use_timesfm` | False | Add TimesFM 3.0 (`--timesfm`) |
| `tfm_device` | cpu | `cpu`, `cuda` or `mps` |
| `tfm_multivariate` | True | Forecast the four grades jointly (False = one at a time) |
| `tfm_max_context` | 1024 | Most recent points handed to TimesFM |

## 3. Understanding the performance matrix

All metrics are computed on the **backtest**: forecasts the model made for dates it had not yet seen. Values below come from the sample data.

### The metrics

| Metric | Formula (per grade) | Plain meaning | Better is |
|---|---|---|---|
| **MAE** | mean of \|forecast - actual\| | Typical miss, in share units. An MAE of 0.03 means the forecast is off by about 3 percentage points on average. | Lower |
| **RMSE** | square root of mean of (forecast - actual)² | Like MAE but punishes large misses more. A big gap between RMSE and MAE means a few large errors. | Lower |
| **sMAPE %** | mean of 2·\|error\| / (\|actual\| + \|forecast\|) × 100 | Typical miss as a percentage of the size of the share. Useful for comparing a large band with a small one. | Lower |
| **R²** | 1 - (sum of squared errors / sum of squared deviations from the mean) | How much of the movement in the actual share the forecast explains. 1 is perfect, 0 is no better than always predicting the average, negative is worse than that. | Higher |
| **Bias** | mean of (forecast - actual) | Direction of systematic error. Positive means over-forecasting. In the **ALL** row the average of absolute biases is shown, because signed biases across shares cancel out. | Closer to 0 |

The **ALL** row is the simple average across the four grades.

### Example: best model (extra_trees)

| Grade | MAE | RMSE | sMAPE % | R² | Bias |
|---|---|---|---|---|---|
| MB_4 | 0.0525 | 0.0726 | 21.8 | 0.236 | +0.0054 |
| MB_6 | 0.0380 | 0.0519 | 7.5 | 0.083 | -0.0014 |
| MB_8 | 0.0312 | 0.0426 | 15.4 | 0.059 | -0.0029 |
| MB_9 | 0.0129 | 0.0173 | 39.7 | 0.013 | -0.0011 |

How to read it:

- **MAE is largest for MB_4** (about 5 points) because that band varies the most day to day.
- **MB_9 has the smallest MAE but the worst sMAPE.** The band is small (about 2-6 %), so a 1-point miss is a large relative error. Use MAE for absolute impact and sMAPE for relative accuracy.
- **R² is low everywhere** (best is 0.24 for MB_4). Most day-to-day movement is noise, not a predictable pattern. Do not expect the forecast to track single-day spikes.
- **Bias is near zero**, so the forecasts are not systematically high or low.

### Why baselines matter

A model's MAE only means something next to a trivial alternative:

| Baseline | Idea |
|---|---|
| `naive_last` | Tomorrow looks like the last kill date |
| `moving_avg_4` | Average of the last 4 kill dates |
| `moving_avg_8` | Average of the last 8 kill dates |

On the sample data, the models (MAE about 0.034) comfortably beat `naive_last` (0.044), because yesterday's value is a noisy guide. They are about equal to `moving_avg_8` (0.0335). In plain terms: **the models are reliable, but a smoothed average captures almost everything that is predictable.** If a model ever falls clearly behind `moving_avg_8`, it is adding complexity without value.

### How to read each chart

**1. `actual_vs_forecast.png`.** The black dots are actuals, the coloured line is the forecast, the shaded band is the 80 % interval.
- Good: the line follows the slow trend in the actuals, and roughly 8 in 10 dots fall inside the band (on the sample data, 78 % do).
- Warning: many dots outside the band on the same side means the model is biased there; a line that lags behind turning points means it reacts slowly.

**2. `model_comparison.png`.** Left: MAE heat-map by model and grade (green is better). Right: overall MAE bars, with the 8-period average as a dashed line.
- Look for models that are consistently green across all four columns, not just one.
- Models left of the dashed line beat the 8-period moving average; those to the right do not.

**3. `residual_diagnostics.png`.** Four panels that look at the errors:
- *Predicted vs actual*: points should sit near the dashed diagonal. A flat cloud means the model mostly predicts the average.
- *Error distribution by grade*: boxes centred on zero mean no bias; long whiskers or dots show occasional big misses.
- *Rolling 6-step absolute error*: a rising line means performance is drifting and the model may need retraining or more recent data.
- *Residual autocorrelation*: bars inside the grey bands mean errors are random, which is what you want. Persistent bars outside suggest a pattern the model is missing.

**4. `next_forecast.png`.** The next forecast for each grade (diamond) with its 80 % interval, after the last 40 observations. A wide interval means high uncertainty, so plan around the range, not just the point.

### Reading the TimesFM 3.0 rows (when enabled)

- **`timesfm3` vs the baselines.** Judge it exactly like any other model: it must beat `moving_avg_8` on the same dates to justify its extra weight (330M parameters, PyTorch install). Being zero-shot, it has no tuning step that could overfit, but it has never seen this plant's data either.
- **`ensemble_tfm`.** Averaging two different kinds of model often helps when their errors are not alike. Compare it to the better of its two parts.
- **Native-band coverage.** The run prints how often actuals fall inside TimesFM's own 10th-90th percentile band. Near 80 % means well calibrated; clearly lower means the bands are too narrow and the backtest-based interval from this project is the safer one to plan with.
- **Licence.** The TimesFM 3.0 pretrained weights are for non-commercial, non-production use only.

### Practical guidance

- **Use the interval, not just the point.** An MB_4 forecast of 17 % with an 80 % band of roughly 9-23 % is a more honest planning figure than "17 %".
- **Compare to the baseline before trusting a model.** The leaderboard shows this in one glance.
- **Treat near-ties as ties.** With 64 forecasts, differences of about 0.0005 MAE are within noise. The best model is selected on the same backtest it is scored on, so its score is slightly optimistic.
- **Re-run when new data arrives.** Hyper-parameters are tuned on the first 60 rows; for long-term use, re-tune periodically.

### Limitations

- About two years of history cannot confirm a yearly seasonal pattern with confidence.
- Intervals assume future errors look like past backtest errors.
- Forecasts for horizons above 1 feed on their own earlier predictions, so uncertainty grows with horizon and is not separately modelled.
- Single-day anomalies (for example a day with an unusually large MB_3 share) are not predicted.
