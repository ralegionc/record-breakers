# Data

This repository does not carry its raw inputs. The files listed below were present locally when the results in the README were produced, and were deleted on 2026-10-12 to reclaim **1.68 GB** of disk. Every one of them is a download from a public upstream source, so nothing unique was lost.

This file exists so that a re-download can be *checked* rather than assumed to match. It records the exact inventory, not just the source.

Upstream sources, with links and licences, are in the README under **Data**.

## What was removed

1.68 GB in 15 files.

| Path | Size | Files |
|---|---|---|
| `data` | 1.68 GB | 15 |

## Restoring it

All of these were read anonymously from `gs://weatherbench2`: the 2020 HRES, GraphCast, Pangu-Weather and FuXi forecast archives at 1.5 degrees, plus the ERA5 truth field and land-sea mask. Re-fetch with `fetch_fc` and `fetch_era5`. **`data/era5_stats.npz` and `data/sums.npz` were kept**, since those are the 1979 to 2019 climatology and the accumulated statistics this project computed, not a download.

## Inventory

`data_manifest.csv` carries every file with its exact byte length and a fingerprint. Files of 32 MiB or less have a full `sha256`. Larger files have `headtail256`: a sha256 over the first mebibyte, the last mebibyte and the exact byte length, which catches a truncated or altered download without rereading gigabytes.

| File | Size | Fingerprint |
|---|---|---|
| `data/fc_FuXi.npy` | 388.83 MB | `headtail256:4eda2d6265bc782f7094f4cb982f00378be2579c77ee1ab71d62691a3421de5d` |
| `data/fc_GraphCast.npy` | 388.83 MB | `headtail256:f92f8225b2a2fee6e09b1e2dc18b40e0b143701bda1af7ce4610c08d89bdaaa9` |
| `data/fc_HRES.npy` | 388.83 MB | `headtail256:31fd336e75d9274ec7c24ef3ddeec263501dd8edb940303ae95053b10bddd4d8` |
| `data/fc_Pangu.npy` | 388.83 MB | `headtail256:97d5338de84902c66c89db57289b85c2876a660bd08d62376150e392010fc9cc` |
| `data/era5_truth.npy` | 162.29 MB | `headtail256:fb0205b7342cd164db42a4ca8fa65347dc21a45d3925083b7b93856e952af6af` |
| `data/lsm.npy` | 113.56 KB | `sha256:9075f8c50a50ccf3d44e05e262037c17b74cf463399b5a287370be789d7ce898` |
| `data/era5_truth_times.npy` | 11.57 KB | `sha256:be4ff8947510e7c08647dba045708a19825d91a441df9a33ec351cca7f93883c` |
| `data/fc_FuXi_times.npy` | 5.61 KB | `sha256:345f2824165953501ada09a20bfe8c212ac2b92a0521823ee2f5be1d4ea11eac` |
| `data/fc_GraphCast_times.npy` | 5.61 KB | `sha256:345f2824165953501ada09a20bfe8c212ac2b92a0521823ee2f5be1d4ea11eac` |
| `data/fc_HRES_times.npy` | 5.61 KB | `sha256:345f2824165953501ada09a20bfe8c212ac2b92a0521823ee2f5be1d4ea11eac` |
| `data/fc_Pangu_times.npy` | 5.61 KB | `sha256:345f2824165953501ada09a20bfe8c212ac2b92a0521823ee2f5be1d4ea11eac` |
| `data/fc_FuXi_coords.npz` | 3.31 KB | `sha256:382148f7df93de4eb5ae920a79aa52b15f7d725efe004352e9c28b9b7d5b8ce6` |
| `data/fc_GraphCast_coords.npz` | 3.31 KB | `sha256:382148f7df93de4eb5ae920a79aa52b15f7d725efe004352e9c28b9b7d5b8ce6` |
| `data/fc_HRES_coords.npz` | 3.31 KB | `sha256:382148f7df93de4eb5ae920a79aa52b15f7d725efe004352e9c28b9b7d5b8ce6` |
| `data/fc_Pangu_coords.npz` | 3.31 KB | `sha256:382148f7df93de4eb5ae920a79aa52b15f7d725efe004352e9c28b9b7d5b8ce6` |

A source that is live rather than a fixed release will not reproduce these fingerprints. Where that applies it is said above.

