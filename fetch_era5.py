"""Stream ERA5 2 m temperature 1959-2019 into per-(time-of-day, calendar day) max/min/sums,
and save the 2020 verification truth. About 10 minutes, ~11 GB read, ~1.5 GB written."""
import os
import time

import numpy as np
import xarray as xr

import clim
from config import BUCKET, CLIM_GROUPS, DATA, ERA5, GROUPS, STORAGE, TRUTH_END, TRUTH_START, VAR


def open_era5():
    return xr.open_zarr(BUCKET + ERA5, storage_options=STORAGE)[VAR]


def read(da, start, end, tries=5):
    for i in range(tries):
        try:
            sub = da.sel(time=slice(start, end))
            return sub.time.values, sub.transpose("time", "longitude", "latitude").values.astype(np.float32)
        except Exception as e:  # transient GCS errors
            print("  retry", i, e, flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed {start}..{end}")


def main():
    DATA.mkdir(exist_ok=True)
    da = open_era5()
    nlon, nlat = da.sizes["longitude"], da.sizes["latitude"]
    shape = (4, 366, nlon, nlat)
    acc = {g: {"mx": np.full(shape, -np.inf, np.float32), "mn": np.full(shape, np.inf, np.float32)} for g in GROUPS}
    s = np.zeros(shape)
    ss = np.zeros(shape)
    n = np.zeros((4, 366), np.int64)
    for g, (y0, y1) in GROUPS.items():
        for y in range(y0, y1 + 1):
            t0 = time.time()
            times, x = read(da, f"{y}-01-01", f"{y}-12-31T18")
            tod, day = clim.tod_index(times), clim.day_index(times)
            assert len(set(zip(tod, day))) == len(times)  # each slot once per year
            a = acc[g]
            a["mx"][tod, day] = np.maximum(a["mx"][tod, day], x)
            a["mn"][tod, day] = np.minimum(a["mn"][tod, day], x)
            if g in CLIM_GROUPS:
                s[tod, day] += x
                ss[tod, day] += x.astype(np.float64) ** 2
                n[tod, day] += 1
            print(f"{y}: {len(times)} steps, {time.time() - t0:.1f}s", flush=True)
    out = {f"{k}_{g}": v for g, a in acc.items() for k, v in a.items()}
    np.savez(DATA / "era5_stats.npz", s=s, ss=ss, n=n, lon=da.longitude.values, lat=da.latitude.values, **out)
    times, x = read(da, TRUTH_START, TRUTH_END)
    np.save(DATA / "era5_truth.npy", x)
    np.save(DATA / "era5_truth_times.npy", times)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # gcsfs threads can hang interpreter shutdown
