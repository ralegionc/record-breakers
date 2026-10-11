"""Figures for the README (PNG, light background)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import clim  # noqa: E402
from analysis import load_stats  # noqa: E402
from config import BUCKET, DATA, ERA5, FIGS, LEADS_H, MODELS, PRIMARY, STORAGE, TABLES  # noqa: E402

COL = {"HRES": "#7d7c78", "GraphCast": "#2a78d6", "Pangu": "#eb6834", "FuXi": "#1baf7a",
       "HRES shrunk 20%": "#9a9993", "HRES ceiling": "#0b0b0b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False,
})
ML = ["GraphCast", "Pangu", "FuXi"]


def land_mask():
    path = DATA / "lsm.npy"
    if not path.exists():
        import xarray as xr
        ds = xr.open_zarr(BUCKET + ERA5, storage_options=STORAGE)
        np.save(path, ds["land_sea_mask"].transpose("longitude", "latitude").values)
    return np.load(path)


def lonlat_img(ax, field, lon, lat, **kw):
    order = np.argsort(np.where(lon > 180, lon - 360, lon))
    lon2 = np.where(lon > 180, lon - 360, lon)[order]
    im = ax.pcolormesh(lon2, lat, field[order].T, shading="nearest", **kw)
    lsm = land_mask()[order]
    ax.contour(lon2, lat, lsm.T, levels=[0.5], colors=INK, linewidths=0.4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.grid(False)
    return im


def fig_map(mu, sd, rec, lat):
    truth = np.load(DATA / "era5_truth.npy")
    times = pd.DatetimeIndex(np.load(DATA / "era5_truth_times.npy"))
    keep = times < "2021-01-01"
    truth, times = truth[keep], times[keep]
    tod, day = clim.tod_index(times), clim.day_index(times)
    mx, mn = rec[PRIMARY]
    hot = (truth > mx[tod, day]).sum(0) / 4  # days-equivalent of 6-hourly steps
    cold = (truth < mn[tod, day]).sum(0) / 4
    lon = np.load(DATA / "era5_stats.npz")["lon"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.6))
    for ax, f, cmap, title in [(axes[0], hot, "Reds", "Record-warm"), (axes[1], cold, "Blues", "Record-cold")]:
        im = lonlat_img(ax, np.where(f > 0, f, np.nan), lon, lat, cmap=cmap, vmin=0, vmax=np.nanpercentile(np.where(f > 0, f, np.nan), 98))
        ax.set_title(f"{title} 6-hourly steps in 2020 (in days), vs 1979-2019 ERA5 range +-15 days", loc="left", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.7, label="days")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_record_map.png", dpi=150)
    plt.close(fig)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :], hot.shape)
    ndays = len(times) / 4
    # area-weighted share of the globe-year spent beyond the old record
    return dict(hot_frac=float((hot * w).sum() / w.sum() / ndays), cold_frac=float((cold * w).sum() / w.sum() / ndays))


def fig_ladder(freq, variant, leads=(72, 120, 168)):
    """Forecast/observed exceedance frequency against how rare the threshold is.
    Lines: anomaly thresholds (z>1..3). Hollow markers: record thresholds.
    A training-range ceiling would show as record markers falling below the line."""
    d = freq[(freq.period == PRIMARY) & (freq.variant == variant)]
    n_obs = d[(d.model == "HRES")].groupby(["side", "lead_h", "threshold"]).obs_area.first()
    fig, axes = plt.subplots(2, len(leads), figsize=(13, 7.2), sharey=True)
    for r, side in enumerate(["hot", "cold"]):
        for c, L in enumerate(leads):
            ax = axes[r, c]
            tot = n_obs[(side, L, "z>1.0")]
            for m in list(MODELS) + ["HRES ceiling"]:
                q = d[(d.side == side) & (d.lead_h == L) & (d.model == m)].set_index("threshold")
                x = n_obs[side, L].reindex(q.index) / tot
                zq, rq = q[q.index.str.startswith("z")], q[q.index.str.startswith("record")]
                zx, rx = x[zq.index], x[rq.index]
                col = COL.get(m, INK)
                ls = ":" if m == "HRES ceiling" else "-"
                if m != "HRES ceiling":
                    ax.plot(zx, zq.freq_ratio, color=col, lw=2, ls=ls, marker="o", ms=5, label=m)
                ax.plot(rx, rq.freq_ratio, color=col, lw=1, ls=":", marker="o", ms=7, mfc="#fcfcfb", mew=2,
                        label="HRES with ceiling at old record" if m == "HRES ceiling" else None)
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.axhline(1, color=MUTED, lw=0.8)
            ax.set_title(f"{'Warm' if side == 'hot' else 'Cold'} extremes, day {L // 24}", loc="left", color=INK)
            if r == 1:
                ax.set_xlabel("threshold rarity (share of z>1 events), rarer to the right")
    axes[0, 0].set_ylabel("forecast / observed frequency")
    axes[1, 0].set_ylabel("forecast / observed frequency")
    axes[0, -1].legend(fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, "Filled: anomaly thresholds z>1 to z>3. Hollow: old record, +0.25 sd, +0.5 sd. "
             f"Errors {variant}.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGS / f"fig2_freq_rarity_{variant}.png", dpi=150)
    plt.close(fig)


def fig_matched(mat, variant, unit="sd"):
    d = mat[(mat.period == PRIMARY) & (mat.variant == variant) & (mat.unit == unit)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    show = ML + ["HRES shrunk 20%", "HRES ceiling"]
    for ax, side in zip(axes, ["hot", "cold"]):
        for k, m in enumerate(show):
            r = d[(d.side == side) & (d.model == m)].sort_values("lead_h")
            x = r.lead_h / 24 + (k - 2) * 0.14
            ctrl = m.startswith("HRES ")
            ax.errorbar(x, r.effect_minus_hres, yerr=[r.effect_minus_hres - r.emh_lo, r.emh_hi - r.effect_minus_hres],
                        color=COL.get(m, INK), marker="s" if ctrl else "o", ms=6, mfc="#fcfcfb" if ctrl else None,
                        lw=1 if ctrl else 2, ls=":" if ctrl else "-", capsize=3,
                        label={"HRES ceiling": "control: HRES capped at old record",
                               "HRES shrunk 20%": "control: HRES anomalies shrunk 20%"}.get(m, m))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks([L // 24 for L in LEADS_H])
        ax.set_xlabel("lead (days)")
        ax.set_title("Record-warm days" if side == "hot" else "Record-cold days", loc="left", color=INK)
    axes[0].set_ylabel(f"extra error on records vs matched non-records,\nminus HRES ({unit}; negative = undershoot)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / f"fig3_matched_{variant}.png", dpi=150)
    plt.close(fig)


def fig_exceedance(exc, variant, lead=72):
    d = exc[(exc.period == PRIMARY) & (exc.variant == variant) & (exc.lead_h == lead)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, side in zip(axes, ["hot", "cold"]):
        for k, m in enumerate(MODELS):
            r = d[(d.side == side) & (d.model == m)]
            x = np.arange(len(r)) + (k - 1.5) * 0.12
            ax.errorbar(x, r.bias, yerr=[r.bias - r.lo, r.hi - r.bias], color=COL[m], marker="o", ms=6, lw=2, capsize=3, label=m)
        ax.set_xticks(np.arange(len(r)), [f"{a:g} to {b:g}" for a, b in zip(r.exc_lo_sd, r.exc_hi_sd)])
        ax.set_xlabel("how far past the old record (sd)")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(("Record-warm" if side == "hot" else "Record-cold") + f", day {lead // 24} forecasts", loc="left", color=INK)
    axes[0].set_ylabel("mean error toward the extreme (K)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / f"fig4_exceedance_{variant}.png", dpi=150)
    plt.close(fig)


def fig_case(rec, lat, lead=72, box=(60, 72, 100, 160), start="2020-05-25", end="2020-07-15"):
    """Siberian heatwave. Forecasts exist only at 00 and 12 UTC valid times, so truth and the
    old record are shown at those hours too. The cell is the land cell in the box with the
    most record-breaking 00/12 UTC steps in the window."""
    lon = np.load(DATA / "era5_stats.npz")["lon"]
    truth = np.load(DATA / "era5_truth.npy", mmap_mode="r")
    tt = pd.DatetimeIndex(np.load(DATA / "era5_truth_times.npy"))
    m = (tt >= start) & (tt <= end) & np.isin(tt.hour, [0, 12])
    tod, day = clim.tod_index(tt[m]), clim.day_index(tt[m])
    T = np.asarray(truth[m])
    MX = rec[PRIMARY][0][tod, day]
    n_rec = (T > MX).sum(0)
    lo, la = np.meshgrid(lon, lat, indexing="ij")
    inbox = (la >= box[0]) & (la <= box[1]) & (lo >= box[2]) & (lo <= box[3]) & (land_mask() > 0.5)
    i, j = np.unravel_index(np.argmax(np.where(inbox, n_rec, -1)), n_rec.shape)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tt[m], T[:, i, j] - 273.15, color=INK, lw=2, marker="o", ms=3, label="ERA5 (truth)")
    ax.plot(tt[m], MX[:, i, j] - 273.15, color=MUTED, lw=1, ls="--", label="1979-2019 max")
    li = LEADS_H.index(lead)
    for mdl in MODELS:
        ft = pd.DatetimeIndex(np.load(DATA / f"fc_{mdl}_times.npy")) + pd.Timedelta(hours=lead)
        k = (ft >= start) & (ft <= end)
        f = np.load(DATA / f"fc_{mdl}.npy", mmap_mode="r")[np.where(k)[0], li, i, j] - 273.15
        ax.plot(ft[k], f, color=COL[mdl], lw=1.5, marker="o", ms=3, label=f"{mdl} day {lead // 24}")
    ax.set_ylabel("2 m temperature (C), 00 and 12 UTC")
    ax.set_title(f"Siberian heatwave 2020, grid cell {lat[j]:.1f}N {lon[i]:.1f}E: {n_rec[i, j]} record-breaking steps",
                 loc="left", color=INK)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_siberia_case.png", dpi=150)
    plt.close(fig)
    return dict(lat=float(lat[j]), lon=float(lon[i]), n_rec=int(n_rec[i, j]))


def fig_regime(lead=72):
    d = pd.read_csv(TABLES / "regime.csv")
    d = d[(d.lead_h == lead) & (d.model != "HRES")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, m in enumerate(ML):
        r = d[d.model == m]
        x = np.arange(len(r)) + (k - 1) * 0.12
        ax.errorbar(x, r.bias_minus_hres, yerr=[r.bias_minus_hres - r.lo, r.hi - r.bias_minus_hres],
                    color=COL[m], marker="o", ms=6, lw=2, capsize=3, label=m)
    labels = [f"{a:g} to {b:g}".replace("-inf to ", "< ").replace(" to inf", "+") for a, b in zip(r.bg_lo, r.bg_hi)]
    ax.set_xticks(np.arange(len(r)), labels, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("31-day background anomaly at the cell (sd), cold spells left, warm spells right")
    ax.set_ylabel("bias minus HRES bias (K)")
    ax.set_title(f"Relative to HRES, ML models run colder inside persistent warm spells (day {lead // 24}, raw)", loc="left", color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_regime.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    mu, sd, rec, lat = load_stats()
    frac = fig_map(mu, sd, rec, lat)
    pd.Series(frac).to_csv(TABLES / "record_area_fraction.csv")
    freq = pd.read_csv(TABLES / "freq.csv")
    mat = pd.read_csv(TABLES / "matched.csv")
    exc = pd.read_csv(TABLES / "exceedance.csv")
    for v in ["raw", "debiased"]:
        fig_ladder(freq, v)
        fig_matched(mat, v)
        fig_exceedance(exc, v)
    pd.Series(fig_case(rec, lat)).to_csv(TABLES / "case_cell.csv")
    fig_regime()
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
