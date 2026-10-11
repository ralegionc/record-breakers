"""Do ML models run cold inside long warm spells (and warm inside cold spells)?

Bias (raw forecast minus ERA5) binned by the 31-day background anomaly at the cell:
the mean standardized anomaly of the truth over valid date +-15 days. If ML models are
pulled toward their training climate on slow timescales, their bias relative to HRES
should slope downward with the background anomaly. Block bootstrap by week.
"""
import numpy as np
import pandas as pd

import clim
from analysis import load_stats
from config import BASELINE, DATA, LEADS_H, MODELS, TABLES

EDGES = np.array([-np.inf, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, np.inf])
HALF = 15 * 4  # 6-hourly steps in +-15 days
NBOOT = 1000


def background_anomaly(truth, times, mu, sd):
    tod, day = clim.tod_index(times), clim.day_index(times)
    z = (truth - mu[tod, day]) / sd[tod, day]
    cs = np.concatenate([np.zeros((1,) + z.shape[1:]), np.cumsum(z, 0, dtype=np.float64)])
    i = np.arange(len(times))
    a, b = np.clip(i - HALF, 0, len(times)), np.clip(i + HALF + 1, 0, len(times))
    return ((cs[b] - cs[a]) / (b - a)[:, None, None]).astype(np.float32)


def main():
    mu, sd, _, lat = load_stats()
    truth = np.load(DATA / "era5_truth.npy")
    tt = pd.DatetimeIndex(np.load(DATA / "era5_truth_times.npy"))
    bg = background_anomaly(truth, tt, mu, sd)
    tpos = {t: i for i, t in enumerate(tt)}
    names = list(MODELS)
    ftimes = {m: pd.DatetimeIndex(np.load(DATA / f"fc_{m}_times.npy")) for m in names}
    inits = pd.DatetimeIndex(sorted(set.intersection(*[set(t) for t in ftimes.values()])))
    w = np.clip(np.cos(np.deg2rad(lat)), 0, None)[None, None, :]
    rows = []
    rng = np.random.default_rng(0)
    for L in [72, 120]:
        li = LEADS_H.index(L)
        valid = inits + pd.Timedelta(hours=L)
        idx = [tpos[v] for v in valid]
        T, B = truth[idx], bg[idx]
        block = ((valid - pd.Timestamp("2020-01-01")).days // 7).to_numpy()
        nB = block.max() + 1
        k = np.digitize(B, EDGES[1:-1])
        W = np.broadcast_to(w, T.shape)
        key = (block[:, None, None] * (len(EDGES) - 1) + k).ravel()
        wsum = np.bincount(key, W.ravel(), nB * (len(EDGES) - 1)).reshape(nB, -1)
        esum = {}
        for m in names:
            F = np.load(DATA / f"fc_{m}.npy", mmap_mode="r")[ftimes[m].get_indexer(inits), li]
            esum[m] = np.bincount(key, (W * (F - T)).ravel(), wsum.size).reshape(nB, -1)
        boots = rng.integers(0, nB, size=(NBOOT, nB))
        C = np.stack([np.bincount(r, minlength=nB) for r in boots]).astype(float)
        wb = C @ wsum
        for m in names:
            pt = esum[m].sum(0) / wsum.sum(0)
            rel = pt - esum[BASELINE].sum(0) / wsum.sum(0)
            rb = (C @ esum[m]) / wb - (C @ esum[BASELINE]) / wb
            for j in range(len(EDGES) - 1):
                rows.append(dict(lead_h=L, model=m, bg_lo=EDGES[j], bg_hi=EDGES[j + 1],
                                 area_share=wsum[:, j].sum() / wsum.sum(), bias=pt[j], bias_minus_hres=rel[j],
                                 lo=np.percentile(rb[:, j], 2.5), hi=np.percentile(rb[:, j], 97.5)))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "regime.csv", index=False)
    print(df[df.model != BASELINE].pivot_table(index=["lead_h", "bg_lo"], columns="model",
                                                 values="bias_minus_hres").round(3).to_string())


if __name__ == "__main__":
    main()
