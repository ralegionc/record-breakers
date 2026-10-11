"""Download 2020 2 m temperature forecasts at the chosen lead times for every model.
Saves data/fc_<model>.npy with shape (init, lead, lon, lat) plus init times."""
import os
import sys
import time

import numpy as np
import xarray as xr

from config import BUCKET, DATA, EVAL_END, EVAL_START, LEADS_H, MODELS, STORAGE, VAR


def fetch(name, batch=20):
    out = DATA / f"fc_{name}.npy"
    if out.exists():
        print(name, "cached", flush=True)
        return
    ds = xr.open_zarr(BUCKET + MODELS[name], storage_options=STORAGE, decode_timedelta=True)
    da = ds[VAR].sel(time=slice(EVAL_START, EVAL_END), prediction_timedelta=[np.timedelta64(h, "h") for h in LEADS_H])
    da = da.sel(time=da.time.dt.hour.isin([0, 12])).transpose("time", "prediction_timedelta", "longitude", "latitude")
    times = da.time.values
    arr = np.lib.format.open_memmap(DATA / f"fc_{name}.part.npy", "w+", np.float32, da.shape)
    t0 = time.time()
    for i in range(0, len(times), batch):
        for k in range(5):
            try:
                arr[i:i + batch] = da.isel(time=slice(i, i + batch)).values
                break
            except Exception as e:
                print("  retry", k, e, flush=True)
                time.sleep(5 * (k + 1))
        else:
            raise RuntimeError(f"{name} batch {i} failed")
        print(f"{name} {min(i + batch, len(times))}/{len(times)} {time.time() - t0:.0f}s", flush=True)
    arr.flush()
    del arr
    os.replace(DATA / f"fc_{name}.part.npy", out)
    np.save(DATA / f"fc_{name}_times.npy", times)
    np.savez(DATA / f"fc_{name}_coords.npz", lon=ds.longitude.values, lat=ds.latitude.values)


if __name__ == "__main__":
    for m in sys.argv[1:] or MODELS:
        fetch(m)
    os._exit(0)
