"""Point estimates and weekly-block bootstrap CIs from data/sums.npz -> tables/*.csv."""
import numpy as np
import pandas as pd

from config import BASELINE, TABLES, DATA

NBOOT = 2000


def boot_ratio(num, den, idx):
    """Ratio of block sums, plus bootstrap draws resampling blocks (axis 0) by idx (nboot, nB)."""
    pt = num.sum(0) / den.sum(0)
    draws = num[idx].sum(1) / den[idx].sum(1)
    return pt, draws


def ci(d):
    return np.nanpercentile(d, 2.5, axis=0), np.nanpercentile(d, 97.5, axis=0)


def resample(x, C):
    """Bootstrap sums: C (nboot, nB) block multiplicities, x (nB, ...)."""
    return np.tensordot(C, x, axes=(1, 0))


def matched_effect(E, W, Wshared):
    """Record minus matched non-record mean error, weighted by record area per stratum.
    E: (..., nZ, 2) error sums, W: (..., nZ, 2) weight sums."""
    ok = (Wshared[..., 0] > 0) & (Wshared[..., 1] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m0 = E[..., 1] / W[..., 1], E[..., 0] / W[..., 0]
    wt = np.where(ok, W[..., 1], 0.0)
    diff = np.where(ok, m1 - m0, 0.0)
    eff = (wt * diff).sum(-1) / wt.sum(-1)
    rec = (wt * np.where(ok, m1, 0)).sum(-1) / wt.sum(-1)
    non = (wt * np.where(ok, m0, 0)).sum(-1) / wt.sum(-1)
    return eff, rec, non


def main():
    TABLES.mkdir(exist_ok=True)
    S = np.load(DATA / "sums.npz")
    periods, sides, leads = list(S["periods"]), list(S["sides"]), list(S["leads"])
    variants, models, thresholds = list(S["variants"]), list(S["models"]), list(S["thresholds"])
    edges = S["exc_edges"]
    b = models.index(BASELINE)
    nB = S["freq_obs"].shape[3]
    rng = np.random.default_rng(42)
    idx = rng.integers(0, nB, size=(NBOOT, nB))
    C = np.stack([np.bincount(r, minlength=nB) for r in idx]).astype(float)

    freq_rows, did_rows, exc_rows, count_rows = [], [], [], []
    for pi, p in enumerate(periods):
        for si, sd in enumerate(sides):
            for li, L in enumerate(leads):
                obs = S["freq_obs"][pi, si, li]  # (nB, nT)
                obs_b = resample(obs, C)
                sw = S["strat_w"][pi, si, li]  # (nB, nZ, 2)
                sw_b = resample(sw, C)
                ew = S["exc_w"][pi, si, li]
                ew_b = resample(ew, C)
                count_rows.append(dict(period=p, side=sd, lead_h=L, record_area=sw[..., 1].sum(),
                                       extreme_nonrecord_area=sw[..., 0].sum()))
                for vi, v in enumerate(variants):
                    fcs = S["freq_fc"][pi, si, li, vi]  # (nM, nB, nT)
                    hit = S["freq_hit"][pi, si, li, vi]
                    fr = fcs.sum(1) / obs.sum(0)
                    fr_b = np.stack([resample(fcs[m], C) for m in range(len(models))], 1) / obs_b[:, None]
                    hr = hit.sum(1) / obs.sum(0)
                    hr_b = np.stack([resample(hit[m], C) for m in range(len(models))], 1) / obs_b[:, None]
                    mres = {}
                    for unit, key in [("K", "strat_e"), ("sd", "strat_es")]:
                        se = S[key][pi, si, li, vi]  # (nM, nB, nZ, 2)
                        eff, rec, non = matched_effect(se.sum(1), sw.sum(0)[None], sw.sum(0)[None])
                        se_b = np.stack([resample(se[m], C) for m in range(len(models))], 1)
                        eff_b, _, _ = matched_effect(se_b, sw_b[:, None], sw_b[:, None])
                        mres[unit] = (eff, rec, non, eff_b)
                    ee = S["exc_e"][pi, si, li, vi]
                    for mi, m in enumerate(models):
                        for ti, th in enumerate(thresholds):
                            r_b = fr_b[:, mi, ti] / fr_b[:, b, ti]
                            dh_b = hr_b[:, mi, ti] - hr_b[:, b, ti]
                            freq_rows.append(dict(
                                period=p, side=sd, lead_h=L, variant=v, model=m, threshold=th,
                                obs_area=obs[:, ti].sum(), freq_ratio=fr[mi, ti],
                                fr_lo=ci(fr_b[:, mi, ti])[0], fr_hi=ci(fr_b[:, mi, ti])[1],
                                fr_vs_hres=fr[mi, ti] / fr[b, ti], fvh_lo=ci(r_b)[0], fvh_hi=ci(r_b)[1],
                                hit_rate=hr[mi, ti], hr_lo=ci(hr_b[:, mi, ti])[0], hr_hi=ci(hr_b[:, mi, ti])[1],
                                hit_minus_hres=hr[mi, ti] - hr[b, ti], hmh_lo=ci(dh_b)[0], hmh_hi=ci(dh_b)[1]))
                        for unit, (eff, rec, non, eff_b) in mres.items():
                            d_b = eff_b[:, mi] - eff_b[:, b]
                            did_rows.append(dict(
                                period=p, side=sd, lead_h=L, variant=v, model=m, unit=unit,
                                bias_record=rec[mi], bias_matched_nonrecord=non[mi],
                                effect=eff[mi], eff_lo=ci(eff_b[:, mi])[0], eff_hi=ci(eff_b[:, mi])[1],
                                effect_minus_hres=eff[mi] - eff[b], emh_lo=ci(d_b)[0], emh_hi=ci(d_b)[1]))
                        e_b = resample(ee[mi], C) / ew_b
                        for k in range(len(edges) - 1):
                            exc_rows.append(dict(
                                period=p, side=sd, lead_h=L, variant=v, model=m,
                                exc_lo_sd=edges[k], exc_hi_sd=edges[k + 1], area=ew[:, k].sum(),
                                bias=ee[mi, :, k].sum() / ew[:, k].sum(), lo=ci(e_b[:, k])[0], hi=ci(e_b[:, k])[1]))
    pd.DataFrame(freq_rows).to_csv(TABLES / "freq.csv", index=False)
    pd.DataFrame(did_rows).to_csv(TABLES / "matched.csv", index=False)
    pd.DataFrame(exc_rows).to_csv(TABLES / "exceedance.csv", index=False)
    pd.DataFrame(count_rows).to_csv(TABLES / "counts.csv", index=False)
    print("tables written")


if __name__ == "__main__":
    main()
