import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clim  # noqa: E402
from analysis import leave_out_bias  # noqa: E402
from stats import boot_ratio  # noqa: E402


def test_day_index_leap_calendar():
    t = pd.to_datetime(["2019-03-01", "2020-03-01", "2020-02-29", "2020-01-01", "2020-12-31"])
    assert list(clim.day_index(t)) == [60, 60, 59, 0, 365]


def test_tod_index():
    t = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 18:00"])
    assert list(clim.tod_index(t)) == [0, 1, 3]


def test_window_max_is_circular():
    a = np.zeros((4, 366, 1))
    a[0, 2, 0] = 5
    m = clim.window_max(a, 3)
    assert m[0, 365, 0] == 5 and m[0, 5, 0] == 5 and m[0, 6, 0] == 0
    assert m[1].max() == 0


def test_window_min():
    a = np.zeros((4, 366, 1))
    a[2, 100, 0] = -3
    assert clim.window_min(a, 1)[2, 101, 0] == -3


def test_mean_sd_matches_numpy():
    rng = np.random.default_rng(0)
    years = rng.normal(280, 3, size=(20, 4, 366, 2))
    s, ss = years.sum(0), (years**2).sum(0)
    n = np.full((4, 366), 20)
    mu, sd = clim.mean_sd(s, ss, n, 2)
    pooled = np.concatenate([np.roll(years, k, axis=2) for k in range(-2, 3)], axis=0)
    assert np.allclose(mu, pooled.mean(0), atol=1e-3)
    assert np.allclose(sd, pooled.std(0, ddof=1), atol=1e-3)


def test_leave_out_bias_recovers_constant_and_ignores_event_week():
    valid = pd.date_range("2020-01-01", periods=120, freq="12h")
    err = np.full((120, 1, 1), -1.0)
    spike = (valid >= "2020-02-01") & (valid < "2020-02-04")
    err[spike] = -20.0  # a forecast bust during the event itself
    b = leave_out_bias(err, valid)
    assert np.allclose(b[spike], -1.0)
    assert np.all(b[~spike] <= -1.0)


def test_boot_ratio_shape_and_point():
    num = np.array([[1.0, 2.0], [3.0, 4.0]])
    den = np.array([[2.0, 2.0], [2.0, 2.0]])
    idx = np.array([[0, 1], [1, 1], [0, 0]])
    pt, draws = boot_ratio(num, den, idx)
    assert np.allclose(pt, [1.0, 1.5])
    assert draws.shape == (3, 2)
    assert np.allclose(draws[1], [1.5, 2.0])


def test_control_forecasts():
    from analysis import control_forecast
    f, mu, edge = np.array([5.0, 1.0]), np.array([0.0, 0.0]), np.array([3.0, 3.0])
    assert np.allclose(control_forecast("HRES ceiling", f, mu, edge), [3.0, 1.0])
    assert np.allclose(control_forecast("HRES shrunk 20%", f, mu, edge), [4.0, 0.8])
