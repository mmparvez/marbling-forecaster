"""Performance-matrix visual checks."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAL = {"MB_4": "#4C78A8", "MB_6": "#F58518", "MB_8": "#54A24B", "MB_9": "#B279A2"}


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_pred(actual, pred, lo, hi, path, title):
    cols = list(actual.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.6 * len(cols)), sharex=True)
    for ax, c in zip(axes, cols):
        ax.fill_between(pred.index, lo[c], hi[c], color=PAL[c], alpha=.15, label="prediction interval")
        ax.plot(actual.index, actual[c], "o-", ms=3, color="#333", lw=1, label="actual")
        ax.plot(pred.index, pred[c], "-", color=PAL[c], lw=2, label="forecast")
        ax.set_ylabel(c)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(alpha=.25)
    axes[0].legend(ncol=3, loc="upper right", fontsize=8)
    fig.suptitle(title, y=0.92, fontsize=13)
    _save(fig, path)


def plot_model_comparison(table, path):
    """Heat-map of MAE per model x grade + overall bar chart (lower is better)."""
    t = table.sort_values("ALL")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]},
                                 layout="constrained")
    im = a1.imshow(t.values, cmap="RdYlGn_r", aspect="auto")
    a1.set_xticks(range(t.shape[1]), t.columns)
    a1.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            a1.text(j, i, f"{t.values[i, j]:.4f}", ha="center", va="center", fontsize=8)
    a1.set_title("MAE by model and grade (lower = better)")
    fig.colorbar(im, ax=a1, location="left", pad=0.12, fraction=.04)
    is_base = t.index.str.startswith(("moving_avg", "naive"))
    a2.barh(t.index[::-1], t["ALL"][::-1], color=["#BBBBBB" if b else "#4C78A8" for b in is_base[::-1]])
    if "moving_avg_8" in t.index:
        a2.axvline(t.loc["moving_avg_8", "ALL"], ls="--", color="k", lw=1)
    a2.set_xlim(t["ALL"].min() * 0.9, t["ALL"].max() * 1.03)
    a2.set_title("Overall MAE (blue = ML model, grey = baseline, dashed = 8-wk avg)", fontsize=10)
    a2.grid(axis="x", alpha=.25)
    _save(fig, path)


def plot_residual_diagnostics(actual, pred, path):
    cols = list(actual.columns)
    err = pred - actual
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].scatter(actual.values.ravel(), pred.values.ravel(), s=14, c=np.tile([PAL[c] for c in cols], len(actual)))
    lim = [0, max(actual.values.max(), pred.values.max()) * 1.05]
    ax[0, 0].plot(lim, lim, "k--", lw=1)
    ax[0, 0].set(title="Predicted vs actual", xlabel="actual share", ylabel="predicted share")
    ax[0, 1].boxplot([err[c] for c in cols], tick_labels=cols, patch_artist=True,
                     boxprops=dict(facecolor="#DDE6F2"))
    ax[0, 1].axhline(0, color="k", lw=1)
    ax[0, 1].set(title="Error distribution by grade (bias check)", ylabel="forecast - actual")
    for c in cols:
        ax[1, 0].plot(err.index, err[c].abs().rolling(6, min_periods=1).mean(), color=PAL[c], label=c)
    ax[1, 0].set(title="Rolling 6-step absolute error (drift check)")
    ax[1, 0].legend(ncol=4, fontsize=8)
    pd.plotting.autocorrelation_plot(err.mean(axis=1), ax=ax[1, 1])
    ax[1, 1].set_title("Residual autocorrelation (should hug zero)")
    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.tight_layout()
    _save(fig, path)


def plot_future(history, fc, lo, hi, path, tail=40):
    cols = list(history.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.4))
    for ax, c in zip(axes, cols):
        h = history[c].iloc[-tail:]
        ax.plot(h.index, h, "o-", ms=3, color="#333", lw=1)
        ax.errorbar(fc.index, fc[c], yerr=[fc[c] - lo[c], hi[c] - fc[c]], fmt="D", color=PAL[c], capsize=4, ms=7)
        ax.set_title(c)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(alpha=.25)
    fig.suptitle("Next-period forecast with prediction interval", y=1.03)
    _save(fig, path)

