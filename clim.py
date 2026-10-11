"""Calendar indexing and windowed record/climatology helpers (pure numpy, tested)."""
import numpy as np
import pandas as pd

_CUM = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def day_index(times):
    """Day of a leap-year calendar, 0..365, so 1 March is always 60."""
    t = pd.DatetimeIndex(times)
    return (_CUM[t.month - 1] + t.day - 1).astype(int)


def tod_index(times):
    """Synoptic hour slot 0..3 for 00/06/12/18 UTC."""
    t = pd.DatetimeIndex(times)
    assert (t.hour % 6 == 0).all() and (t.minute == 0).all()
    return (t.hour // 6).astype(int)


def window_max(a, w):
    """Circular running max over axis 1 (calendar day), +-w days."""
    out = a.copy()
    for k in range(1, w + 1):
        np.maximum(out, np.roll(a, k, axis=1), out=out)
        np.maximum(out, np.roll(a, -k, axis=1), out=out)
    return out


def window_min(a, w):
    return -window_max(-a, w)


def window_sum(a, w):
    out = a.copy()
    for k in range(1, w + 1):
        out = out + np.roll(a, k, axis=1) + np.roll(a, -k, axis=1)
    return out


def mean_sd(s, ss, n, w):
    """Windowed mean and sd from per-(tod, day) sums. n has shape (4, 366)."""
    S, SS = window_sum(s, w), window_sum(ss, w)
    N = window_sum(n.astype(float), w).reshape(n.shape + (1,) * (s.ndim - 2))
    mu = S / N
    var = np.maximum(SS / N - mu**2, 1e-6) * N / (N - 1)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)
