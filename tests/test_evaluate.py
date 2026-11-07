"""Unit tests for src/evaluate.py — the per-appliance metric helper and
the evaluate script's end-to-end run on a small predictions file."""
import os

import numpy as np
import pandas as pd
import pytest

import evaluate
from disaggregate import APPLIANCES


def test_perfect_predictions_zero_errors():
    m = evaluate.appliance_metrics([100, 200, 150], [100, 200, 150],
                                   [120, 120, 120])
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["energy_err"] == 0.0


def test_constant_offset():
    # pred always 5 W high: MAE and RMSE both 5, energy error 5/mean(true)
    true = [100.0, 200.0, 300.0]
    m = evaluate.appliance_metrics(true, [105, 205, 305], [150, 150, 150])
    assert m["mae"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(5.0)
    assert m["energy_err"] == pytest.approx(15.0 / 600.0)


def test_rmse_penalizes_large_errors():
    # errors [3, 4] -> RMSE sqrt(12.5) ≈ 3.54, MAE 3.5: RMSE > MAE when
    # errors vary, and a single big error counts more than two small ones
    m = evaluate.appliance_metrics([0, 0], [3, 4], [0, 0])
    assert m["mae"] == pytest.approx(3.5)
    assert m["rmse"] == pytest.approx(3.5355339059327378)
    assert m["rmse"] > m["mae"]


def test_base_mae_uses_baseline():
    # baseline predicts the mean (150); model predicts the truth
    m = evaluate.appliance_metrics([100, 200], [100, 200], [150, 150])
    assert m["base_mae"] == pytest.approx(50.0)
    assert m["mae"] == 0.0


def test_energy_error_zero_when_sums_match():
    # errors cancel out (+10 / -10): pointwise MAE > 0, billing error 0
    m = evaluate.appliance_metrics([100, 100], [110, 90], [100, 100])
    assert m["mae"] == pytest.approx(10.0)
    assert m["energy_err"] == pytest.approx(0.0)


def test_energy_error_over_vs_under_prediction():
    over = evaluate.appliance_metrics([100, 100], [110, 110], [100, 100])
    under = evaluate.appliance_metrics([100, 100], [90, 90], [100, 100])
    assert over["energy_err"] == pytest.approx(0.10)
    assert under["energy_err"] == pytest.approx(0.10)


def test_zero_true_sum_is_finite():
    # all-zero appliance (e.g. washer never ran): no division blow-up
    m = evaluate.appliance_metrics([0, 0, 0], [0, 5, 0], [0, 0, 0])
    assert np.isfinite(m["energy_err"])
    assert m["mae"] == pytest.approx(5.0 / 3.0)


def test_accepts_lists_series_and_arrays():
    expected = evaluate.appliance_metrics([100, 200], [110, 190], [150, 150])
    for true, pred, base in (
        (pd.Series([100, 200]), pd.Series([110, 190]), pd.Series([150, 150])),
        (np.array([100, 200]), np.array([110, 190]), np.array([150, 150])),
    ):
        m = evaluate.appliance_metrics(true, pred, base)
        assert m == expected


def test_single_sample():
    m = evaluate.appliance_metrics([80], [100], [60])
    assert m["mae"] == pytest.approx(20.0)
    assert m["rmse"] == pytest.approx(20.0)
    assert m["base_mae"] == pytest.approx(20.0)
    assert m["energy_err"] == pytest.approx(0.25)


def test_returns_expected_keys_and_floats():
    m = evaluate.appliance_metrics([1, 2, 3], [1, 2, 4], [2, 2, 2])
    assert set(m) == {"mae", "base_mae", "rmse", "energy_err"}
    assert all(isinstance(v, float) for v in m.values())


def test_main_writes_plot_and_prints_metrics(tmp_path, monkeypatch, capsys):
    # small two-day predictions file, one row per hour
    ts = pd.date_range("2026-03-11", periods=48, freq="h")
    data = {"timestamp": ts, "aggregate_w": 500.0}
    for app in APPLIANCES:
        data[app] = 100.0
        data[f"{app}_pred"] = 95.0
        data[f"{app}_base"] = 80.0
    pd.DataFrame(data).to_csv(tmp_path / "predictions.csv", index=False)
    monkeypatch.setattr(evaluate, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(evaluate, "FIG_DIR", str(tmp_path / "figures"))

    evaluate.main()
    out = capsys.readouterr().out
    for app in APPLIANCES:
        assert app[:-2] in out  # fridge, hvac, washer, tv rows printed
    assert os.path.isfile(tmp_path / "figures" / "sample_day.png")
