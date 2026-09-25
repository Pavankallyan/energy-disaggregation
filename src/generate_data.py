"""Generate synthetic smart-meter data with appliance-level ground truth.

One-minute resolution for N_DAYS. Four appliances plus base load:

- fridge : compressor cycles ~35 min on / 25 min off at 150 W
- hvac   : thermostat-driven; runs when outdoor temp > 30 C (hysteresis),
           2000 W while running
- washer : 3 runs/week, 45-min 800 W blocks with a mid-cycle spin spike
- tv     : evening use (18:00-23:00), 120 W when on
- base   : 80 W always-on load

The meter only sees the aggregate; per-appliance columns are ground truth
for training/evaluation.

Output: data/energy.csv
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

N_DAYS = 14
FREQ_MIN = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


def generate():
    n = N_DAYS * 24 * 60
    t = np.arange(n)
    hour = (t / 60.0) % 24
    timestamps = pd.Timestamp("2026-03-01") + pd.to_timedelta(t, unit="m")

    # fridge: periodic compressor
    cycle = (t % 60) < 35
    fridge = np.where(cycle, 150.0 + RNG.normal(0, 5, n), 0.0)

    # hvac: thermostat on outdoor temperature with hysteresis
    outdoor = 28.0 + 6.0 * np.sin(2 * np.pi * (hour - 14) / 24.0) + RNG.normal(0, 0.8, n)
    hvac_on = np.zeros(n, dtype=bool)
    state = False
    for i in range(n):  # hysteresis loop
        if outdoor[i] > 30.0:
            state = True
        elif outdoor[i] < 29.0:
            state = False
        hvac_on[i] = state
    hvac = np.where(hvac_on, 2000.0 + RNG.normal(0, 30, n), 0.0)

    # washer: 3 random runs per week
    washer = np.zeros(n)
    for _ in range(6):
        start = int(RNG.integers(0, n - 60))
        washer[start:start + 45] = 800.0
        washer[start + 20:start + 25] = 1200.0  # spin spike
    washer = washer + RNG.normal(0, 10, n) * (washer > 0)

    # tv: evening viewing
    evening = (hour >= 18) & (hour < 23)
    tv_on = evening & (RNG.random(n) < 0.6)
    tv = np.where(tv_on, 120.0 + RNG.normal(0, 5, n), 0.0)

    base = 80.0 + RNG.normal(0, 3, n)
    aggregate = fridge + hvac + washer + tv + base + RNG.normal(0, 5, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "aggregate_w": np.round(np.clip(aggregate, 0, None), 1),
        "fridge_w": np.round(np.clip(fridge, 0, None), 1),
        "hvac_w": np.round(np.clip(hvac, 0, None), 1),
        "washer_w": np.round(np.clip(washer, 0, None), 1),
        "tv_w": np.round(np.clip(tv, 0, None), 1),
    })
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "energy.csv"), index=False)
    print(f"{len(df)} one-minute readings over {N_DAYS} days -> "
          f"{DATA_DIR}/energy.csv")


if __name__ == "__main__":
    generate()
