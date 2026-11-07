"""Evaluate disaggregation quality and plot a sample day.

Metrics per appliance (test period):
- MAE / RMSE of the model vs a mean-predicting baseline
- energy error: |sum(pred) - sum(true)| / sum(true)  (billing accuracy)

Saves figures/sample_day.png: aggregate trace plus true-vs-predicted overlays
for fridge and HVAC on the first test day.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from disaggregate import APPLIANCES, TRAIN_DAYS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")


def appliance_metrics(true, pred, base):
    """Per-appliance error metrics for one true/predicted/baseline series.

    Returns a dict with:
    - mae: mean absolute error of the model
    - base_mae: mean absolute error of the mean-predicting baseline
    - rmse: root mean squared error of the model
    - energy_err: |sum(pred) - sum(true)| / sum(true) — billing accuracy
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    base = np.asarray(base, dtype=float)
    return {
        "mae": float(np.mean(np.abs(true - pred))),
        "base_mae": float(np.mean(np.abs(true - base))),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "energy_err": float(abs(pred.sum() - true.sum()) / (true.sum() + 1e-9)),
    }


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "predictions.csv"))
    preds["timestamp"] = pd.to_datetime(preds["timestamp"])

    print(f"{'appliance':10s} {'MAE':>8s} {'base MAE':>9s} {'RMSE':>8s} "
          f"{'energy err':>10s}")
    for app in APPLIANCES:
        m = appliance_metrics(preds[app], preds[f"{app}_pred"],
                              preds[f"{app}_base"])
        print(f"{app[:-2]:10s} {m['mae']:8.1f} {m['base_mae']:9.1f} "
              f"{m['rmse']:8.1f} {m['energy_err']:9.1%}")

    # plot the first test day
    day = preds["timestamp"].dt.date.min()
    sample = preds[preds["timestamp"].dt.date == day].iloc[::5]  # every 5 min
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(sample["timestamp"], sample["aggregate_w"], color="black",
                 linewidth=0.8)
    axes[0].set_title(f"Aggregate meter — {day}")
    axes[0].set_ylabel("Watts")
    for ax, app, color in zip(axes[1:], ["fridge_w", "hvac_w"],
                              ["tab:blue", "tab:red"]):
        ax.plot(sample["timestamp"], sample[app], label="true",
                color=color, linewidth=1.2)
        ax.plot(sample["timestamp"], sample[f"{app}_pred"], label="predicted",
                color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{app[:-2]}: true vs disaggregated")
        ax.set_ylabel("Watts")
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "sample_day.png"), dpi=120)
    print(f"\nplot saved -> {FIG_DIR}/sample_day.png")


if __name__ == "__main__":
    main()
