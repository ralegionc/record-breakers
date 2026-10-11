"""Turn forecasts + ERA5 truth into per-week sufficient statistics (data/sums.npz).

For every period definition of "record", side (hot/cold), lead, variant (raw/debiased) and model:
  freq   area-weighted counts of truth and forecast exceeding a ladder of thresholds
  strat  area-weighted error sums on extreme days, split by record status, within
         strata of standardized anomaly x latitude band (the matched comparison)
  exc    error sums on record days by how far past the old record the truth went
Weekly blocks let stats.py bootstrap over time while keeping heatwaves intact.
"""
import numpy as np
import pandas as pd

import clim
from config import BASELINE, DATA, LEADS_H, MODELS, PERIODS, WINDOW

SIDES = ["hot", "cold"]
VARIANTS = ["raw", "debiased"]
Z_LADDER = [1.0, 1.5, 2.0, 2.5, 3.0]
REC_LADDER = [0.0, 0.25, 0.5]
THRESHOLDS = [f"z>{k}" for k in Z_LADDER] + [f"record+{k}sd" for k in REC_LADDER]
Z_MIN, Z_STEP, N_ZBIN, N_LAT = 1.5, 0.1, 40, 6
EXC_EDGES = [0.0, 0.1, 0.25, 0.5, 1.0, np.inf]
DEBIAS_INNER, DEBIAS_OUTER = 7, 30  # days excluded / included around each valid date
# Synthetic controls built from HRES, to calibrate the tests:
#   ceiling: cannot exceed the old record (what a hard training-range limit looks like)
#   shrunk:  every anomaly damped by 20% (plain smoothing, no knowledge of the record)
CONTROLS = ["HRES ceiling", "HRES shrunk 20%"]
SHRINK = 0.8


def control_forecast(name, f, mu, edge):
    """f, mu, edge are oriented so that 'more extreme' is larger."""
    if name == "HRES ceiling":
        return np.minimum(f, edge)
    return mu + SHRINK * (f - mu)


def load_stats():
    st = np.load(DATA / "era5_stats.npz")
    mu, sd = clim.mean_sd(st["s"], st["ss"], st["n"], WINDOW)
    rec = {}
    for p, groups in PERIODS.items():
        mx = np.max([st[f"mx_{g}"] for g in groups], axis=0)
        mn = np.min([st[f"mn_{g}"] for g in groups], axis=0)
        rec[p] = (clim.window_max(mx, WINDOW), clim.window_min(mn, WINDOW))
    return mu, sd, rec, st["lat"]


def leave_out_bias(err, valid):
    """Mean error over days 8..30 either side of each valid date, same valid hour.
    Removes each model's ordinary seasonal/local bias without using the event week."""
    out = np.zeros_like(err)
    vt = pd.DatetimeIndex(valid)
    day = ((vt - pd.Timestamp("2019-01-01")) // pd.Timedelta("1D")).to_numpy()
    for h in np.unique(vt.hour):
        idx = np.where(vt.hour == h)[0]
        d = day[idx]
        full = np.zeros((d.max() + 1 + DEBIAS_OUTER + 1,) + err.shape[1:])
        cnt = np.zeros(full.shape[0])
        full[d] = err[idx]
        cnt[d] = 1
        cs = np.concatenate([np.zeros((1,) + full.shape[1:]), np.cumsum(full, 0)])
        cc = np.concatenate([[0], np.cumsum(cnt)])

        def rng(a, b):  # sum over days [a, b)
            a, b = np.clip(a, 0, len(cnt)), np.clip(b, 0, len(cnt))
            return cs[b] - cs[a], cc[b] - cc[a]

        s1, c1 = rng(d - DEBIAS_OUTER, d - DEBIAS_INNER)
        s2, c2 = rng(d + DEBIAS_INNER + 1, d + DEBIAS_OUTER + 1)
        c = (c1 + c2).reshape((-1,) + (1,) * (err.ndim - 1))
        out[idx] = (s1 + s2) / np.maximum(c, 1)
    return out


def main():
    mu, sd, rec, lat = load_stats()
    truth = np.load(DATA / "era5_truth.npy", mmap_mode="r")
    ttimes = pd.DatetimeIndex(np.load(DATA / "era5_truth_times.npy"))
    tpos = {t: i for i, t in enumerate(ttimes)}
    names = list(MODELS)
    fc = {m: np.load(DATA / f"fc_{m}.npy", mmap_mode="r") for m in names}
    ftimes = {m: pd.DatetimeIndex(np.load(DATA / f"fc_{m}_times.npy")) for m in names}
    common = sorted(set.intersection(*[set(t) for t in ftimes.values()]))
    inits = pd.DatetimeIndex(common)
    sel = {m: ftimes[m].get_indexer(inits) for m in names}
    print(f"{len(inits)} common inits {inits[0]} .. {inits[-1]}", flush=True)

    w_cell = np.broadcast_to(np.clip(np.cos(np.deg2rad(lat)), 0, None)[None, :], mu.shape[2:])
    latband = np.broadcast_to(np.clip(((lat + 90) // 30).astype(int), 0, N_LAT - 1)[None, :], mu.shape[2:])
    out_names = names + CONTROLS
    nP, nS, nL, nV, nM, nT = len(PERIODS), 2, len(LEADS_H), 2, len(out_names), len(THRESHOLDS)
    blocks_all = ((inits[-1] + pd.Timedelta(hours=max(LEADS_H)) - pd.Timestamp("2020-01-01")).days // 7) + 1
    nB, nZ, nE = blocks_all, N_ZBIN * N_LAT, len(EXC_EDGES) - 1
    R = dict(
        freq_obs=np.zeros((nP, nS, nL, nB, nT)), freq_fc=np.zeros((nP, nS, nL, nV, nM, nB, nT)),
        freq_hit=np.zeros((nP, nS, nL, nV, nM, nB, nT)),
        strat_w=np.zeros((nP, nS, nL, nB, nZ, 2)), strat_e=np.zeros((nP, nS, nL, nV, nM, nB, nZ, 2)),
        strat_es=np.zeros((nP, nS, nL, nV, nM, nB, nZ, 2)),
        exc_w=np.zeros((nP, nS, nL, nB, nE)), exc_e=np.zeros((nP, nS, nL, nV, nM, nB, nE)),
    )

    for li, L in enumerate(LEADS_H):
        valid = inits + pd.Timedelta(hours=L)
        T = np.asarray(truth[[tpos[v] for v in valid]])
        tod, day = clim.tod_index(valid), clim.day_index(valid)
        MU, SD = mu[tod, day], sd[tod, day]
        z = (T - MU) / SD
        block = np.broadcast_to(((valid - pd.Timestamp("2020-01-01")).days // 7).to_numpy()[:, None, None], T.shape)
        W = np.broadcast_to(w_cell, T.shape)
        F = {}
        for m in names:
            raw = np.asarray(fc[m][sel[m], li])
            F[(m, "raw")] = raw
            F[(m, "debiased")] = raw - leave_out_bias(raw - T, valid)
        for pi, p in enumerate(PERIODS):
            MXa, MNa = rec[p][0][tod, day], rec[p][1][tod, day]
            for si, side in enumerate(SIDES):
                sgn = 1.0 if side == "hot" else -1.0
                # orient so "beyond" always means larger: hot uses T, cold uses -T
                t, zz = sgn * T, sgn * z
                edge = MXa if side == "hot" else -MNa
                ex = (t - edge) / SD
                thr = [sgn * MU + k * SD for k in Z_LADDER] + [edge + k * SD for k in REC_LADDER]
                isrec = ex > 0
                obs = [t > th for th in thr]
                for ti, o in enumerate(obs):
                    R["freq_obs"][pi, si, li, :, ti] = np.bincount(block[o], W[o], nB)
                # matched strata among extreme days
                ext = zz >= Z_MIN
                zb = np.minimum(((zz[ext] - Z_MIN) / Z_STEP).astype(int), N_ZBIN - 1)
                lb = np.broadcast_to(latband, T.shape)[ext]
                key = (block[ext] * nZ + zb * N_LAT + lb) * 2 + isrec[ext]
                R["strat_w"][pi, si, li] = np.bincount(key, W[ext], nB * nZ * 2).reshape(nB, nZ, 2)
                eb = np.digitize(ex[isrec], EXC_EDGES[1:-1])
                ekey = block[isrec] * nE + eb
                R["exc_w"][pi, si, li] = np.bincount(ekey, W[isrec], nB * nE).reshape(nB, nE)
                for mi, m in enumerate(out_names):
                    for vi, v in enumerate(VARIANTS):
                        f = control_forecast(m, sgn * F[(BASELINE, v)], sgn * MU, edge) if m in CONTROLS else sgn * F[(m, v)]
                        err = f - t  # positive = forecast more extreme than truth
                        for ti, th in enumerate(thr):
                            fo = f > th
                            R["freq_fc"][pi, si, li, vi, mi, :, ti] = np.bincount(block[fo], W[fo], nB)
                            ho = fo & obs[ti]
                            R["freq_hit"][pi, si, li, vi, mi, :, ti] = np.bincount(block[ho], W[ho], nB)
                        R["strat_e"][pi, si, li, vi, mi] = np.bincount(
                            key, (W * err)[ext], nB * nZ * 2).reshape(nB, nZ, 2)
                        R["strat_es"][pi, si, li, vi, mi] = np.bincount(
                            key, (W * err / SD)[ext], nB * nZ * 2).reshape(nB, nZ, 2)
                        R["exc_e"][pi, si, li, vi, mi] = np.bincount(
                            ekey, (W * err)[isrec], nB * nE).reshape(nB, nE)
        print(f"lead {L}h done", flush=True)
    np.savez(DATA / "sums.npz", periods=list(PERIODS), sides=SIDES, leads=LEADS_H, variants=VARIANTS,
             models=out_names, thresholds=THRESHOLDS, exc_edges=EXC_EDGES, n_inits=len(inits), **R)
    print("saved sums.npz", flush=True)


if __name__ == "__main__":
    main()
