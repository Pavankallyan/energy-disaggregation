"""Disaggregate the meter reading: one regressor per appliance.

For each appliance, a RandomForestRegressor predicts its power draw from
aggregate-only features:

- aggregate_w, hour_sin, hour_cos, day_of_week
- rolling mean/std/max of the aggregate over 15-min and 60-min windows

Train on the first 10 days, test on the last 4. Predictions are clipped at 0.
A mean-predicting baseline is included for comparison.

Outputs: data/predictions.csv (test-period true vs predicted per appliance)
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")

APPLIANCES = ["fridge_w", "hvac_w", "washer_w", "tv_w"]
TRAIN_DAYS = 10


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow"] = df["timestamp"].dt.dayofweek
    agg = df["aggregate_w"]
    df["agg_roll_mean_15"] = agg.rolling(15, min_periods=1).mean()
    df["agg_roll_std_15"] = agg.rolling(15, min_periods=1).std().fillna(0)
    df["agg_roll_mean_60"] = agg.rolling(60, min_periods=1).mean()
    df["agg_roll_max_60"] = agg.rolling(60, min_periods=1).max()
    return df


FEATURE_COLS = ["aggregate_w", "hour_sin", "hour_cos", "dow",
                "agg_roll_mean_15", "agg_roll_std_15",
                "agg_roll_mean_60", "agg_roll_max_60"]


def disaggregate(df: pd.DataFrame):
    df = add_features(df)
    split = df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=TRAIN_DAYS)
    train, test = df[split], df[~split]

    preds = test[["timestamp", "aggregate_w"] + APPLIANCES].copy()
    baselines = {}
    for app in APPLIANCES:
        if app == "washer_w":
            # The washer runs ~1% of the time: per-minute regression can't see
            # it, so use event detection (on/off classifier) + typical power.
            ytr_on = (train[app] > 50).astype(int)
            clf = RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1,
                class_weight="balanced")
            clf.fit(train[FEATURE_COLS], ytr_on)
            on = clf.predict(test[FEATURE_COLS])
            typical = float(train.loc[ytr_on == 1, app].median())
            preds[f"{app}_pred"] = np.where(on, typical, 0.0)
        else:
            model = RandomForestRegressor(n_estimators=150, random_state=42,
                                          n_jobs=-1, min_samples_leaf=5)
            model.fit(train[FEATURE_COLS], train[app])
            preds[f"{app}_pred"] = np.clip(
                model.predict(test[FEATURE_COLS]), 0, None)
        baselines[app] = train[app].mean()
        preds[f"{app}_base"] = baselines[app]
    return preds, baselines


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "energy.csv"))
    preds, _ = disaggregate(df)
    preds.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)
    print(f"disaggregated {len(preds)} test readings -> {DATA_DIR}/predictions.csv")


if __name__ == "__main__":
    main()
