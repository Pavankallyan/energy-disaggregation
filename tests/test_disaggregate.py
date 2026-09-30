"""Unit tests for src/disaggregate.py feature engineering and pipeline."""
import os

import numpy as np
import pandas as pd
import pytest

import disaggregate
import generate_data

DATA_DIR = os.path.join(os.path.dirname(disaggregate.__file__), "..", "data")


def _minute_frame(hours):
    """Tiny aggregate frame with timestamps at the given hours."""
    ts = [pd.Timestamp("2026-03-02") + pd.Timedelta(hours=h) for h in hours]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "aggregate_w": [100.0] * len(hours),
            "fridge_w": [50.0] * len(hours),
            "hvac_w": [0.0] * len(hours),
            "washer_w": [0.0] * len(hours),
            "tv_w": [0.0] * len(hours),
        }
    )


def test_add_features_cyclic_hour():
    # hour=6 -> sin=1, cos=0 ; hour=12 -> sin=0, cos=-1
    df = disaggregate.add_features(_minute_frame([6.0, 12.0]))
    assert df["hour_sin"].iloc[0] == pytest.approx(1.0)
    assert df["hour_cos"].iloc[0] == pytest.approx(0.0)
    assert df["hour_sin"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert df["hour_cos"].iloc[1] == pytest.approx(-1.0)
    # encode/decode round-trips: angle preserved
    decoded = np.arctan2(df["hour_sin"], df["hour_cos"]) % (2 * np.pi)
    assert decoded.iloc[0] == pytest.approx(np.pi / 2)


def test_add_features_midnight_encoding():
    df = disaggregate.add_features(_minute_frame([0.0]))
    assert df["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["hour_cos"].iloc[0] == pytest.approx(1.0)


def test_add_features_day_of_week():
    # 2026-03-02 is a Monday (dow=0)
    df = disaggregate.add_features(_minute_frame([10.0]))
    assert df["dow"].iloc[0] == 0


def test_add_features_rolling_windows():
    ts = pd.Timestamp("2026-03-02") + pd.to_timedelta(np.arange(20), unit="m")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "aggregate_w": np.arange(20, dtype=float),
            "fridge_w": 0.0,
            "hvac_w": 0.0,
            "washer_w": 0.0,
            "tv_w": 0.0,
        }
    )
    out = disaggregate.add_features(df)
    for col in [
        "agg_roll_mean_15",
        "agg_roll_std_15",
        "agg_roll_mean_60",
        "agg_roll_max_60",
    ]:
        assert col in out.columns
    # min_periods=1: first row aggregates just itself
    assert out["agg_roll_mean_15"].iloc[0] == pytest.approx(0.0)
    assert out["agg_roll_std_15"].iloc[0] == pytest.approx(0.0)  # filled NaN
    # mean of values 0..14 at row 14
    assert out["agg_roll_mean_15"].iloc[14] == pytest.approx(
        np.arange(15).mean()
    )
    # 60-min window covers rows 0..19 at the last row
    assert out["agg_roll_max_60"].iloc[19] == pytest.approx(19.0)
    assert out["agg_roll_mean_60"].iloc[19] == pytest.approx(
        np.arange(20).mean()
    )


def test_add_features_preserves_input_columns():
    df = _minute_frame([3.0, 9.0])
    out = disaggregate.add_features(df)
    for col in df.columns:
        assert col in out.columns
    # original df untouched
    assert "hour_sin" not in df.columns


@pytest.fixture(scope="module")
def small_dataset():
    """3-day synthetic dataset + disaggregation with a 2-day train split."""
    monkey_TD = disaggregate.TRAIN_DAYS
    monkey_ND = generate_data.N_DAYS
    disaggregate.TRAIN_DAYS = 2
    generate_data.N_DAYS = 3
    try:
        generate_data.generate()
        df = pd.read_csv(os.path.join(DATA_DIR, "energy.csv"))
        yield df
    finally:
        disaggregate.TRAIN_DAYS = monkey_TD
        generate_data.N_DAYS = monkey_ND


@pytest.fixture(scope="module")
def disagg_result(small_dataset):
    return disaggregate.disaggregate(small_dataset)


def test_disaggregate_output_schema(disagg_result):
    preds, baselines = disagg_result
    expected = {"timestamp", "aggregate_w"} | set(disaggregate.APPLIANCES)
    expected |= {f"{a}_pred" for a in disaggregate.APPLIANCES}
    expected |= {f"{a}_base" for a in disaggregate.APPLIANCES}
    assert expected <= set(preds.columns)
    assert set(baselines) == set(disaggregate.APPLIANCES)


def test_disaggregate_train_test_split(disagg_result, small_dataset):
    preds, _ = disagg_result
    ts = pd.to_datetime(small_dataset["timestamp"])
    train_end = ts.min() + pd.Timedelta(days=2)
    expected_rows = (ts >= train_end).sum()
    assert len(preds) == expected_rows
    assert (pd.to_datetime(preds["timestamp"]) >= train_end).all()


def test_disaggregate_predictions_non_negative(disagg_result):
    preds, _ = disagg_result
    for app in disaggregate.APPLIANCES:
        assert (preds[f"{app}_pred"] >= 0).all()


def test_disaggregate_baselines_are_train_means(disagg_result, small_dataset):
    preds, baselines = disagg_result
    train = small_dataset[
        pd.to_datetime(small_dataset["timestamp"])
        < pd.to_datetime(small_dataset["timestamp"]).min()
        + pd.Timedelta(days=2)
    ]
    for app in disaggregate.APPLIANCES:
        assert baselines[app] == pytest.approx(train[app].mean())
        assert np.allclose(preds[f"{app}_base"], baselines[app], rtol=1e-9)


def test_disaggregate_washer_uses_event_detection(disagg_result, small_dataset):
    preds, _ = disagg_result
    # washer model predicts on/off * typical power: unique non-zero level
    nonzero = preds["washer_w_pred"].unique()
    nonzero = nonzero[nonzero > 0]
    assert len(nonzero) == 1
    # and it should have caught at least some of the real washer activity
    assert (preds["washer_w_pred"] > 0).sum() > 0
