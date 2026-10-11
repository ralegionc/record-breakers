"""Shared settings: data sources, models, lead times, record periods."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"

BUCKET = "gs://weatherbench2/datasets/"
STORAGE = {"token": "anon"}
GRID = "240x121_equiangular_with_poles_conservative"  # 1.5 degree, WB2 headline grid

ERA5 = f"era5/1959-2023_01_10-6h-{GRID}.zarr"
MODELS = {
    "HRES": f"hres/2016-2022-0012-{GRID}.zarr",
    "GraphCast": f"graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-{GRID}.zarr",
    "Pangu": f"pangu/2018-2022_0012_{GRID}.zarr",
    "FuXi": f"fuxi/2020-{GRID}.zarr",
}
ML_MODELS = ["GraphCast", "Pangu", "FuXi"]
BASELINE = "HRES"
# Training data per WeatherBench 2 paper and model papers (validation years count as seen).
TRAINED_THROUGH = {"GraphCast": 2019, "Pangu": 2019, "FuXi": 2017}

VAR = "2m_temperature"
LEADS_H = [24, 72, 120, 168, 240]
EVAL_START, EVAL_END = "2020-01-01", "2020-12-16T12"  # FuXi archive ends 2020-12-16
TRUTH_START, TRUTH_END = "2020-01-01", "2021-01-01T00"

WINDOW = 15  # +-days around the calendar day pooled for records and climatology

# ERA5 years are accumulated in disjoint groups, then combined into periods.
GROUPS = {"1959-1978": (1959, 1978), "1979-2017": (1979, 2017), "2018-2019": (2018, 2019)}
PERIODS = {
    "1979-2019": ["1979-2017", "2018-2019"],  # primary: beyond anything any model trained on
    "1979-2017": ["1979-2017"],
    "1959-2019": ["1959-1978", "1979-2017", "2018-2019"],
}
PRIMARY = "1979-2019"
CLIM_GROUPS = ["1979-2017", "2018-2019"]  # mean and sd for anomalies
