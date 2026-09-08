# Data schema

Every experiment script in this repository expects one CSV file per meter/substation, with the
following columns. This is the schema of the real EVN SCADA telemetry used in the paper; the
bundled `demo_data/*.csv` files are synthetic data generated to the same schema (see
`demo_data/generate_synthetic_demo.py`).

| Column      | Type              | Units | Description                                              |
|-------------|-------------------|-------|------------------------------------------------------------|
| `pc_time`   | timestamp         | --    | Reading timestamp (parsed with `pandas.to_datetime`). Real telemetry is sampled at approximately 30-minute cadence. |
| `pt_kw`     | float             | kW    | Active power (target 1).                                   |
| `qt_kvar`   | float             | kVAR  | Reactive power (target 2).                                  |
| `ua_v`      | float, optional   | V     | Phase A voltage.                                             |
| `ub_v`      | float, optional   | V     | Phase B voltage.                                             |
| `uc_v`      | float, optional   | V     | Phase C voltage.                                             |
| `cos_phi`   | float, optional   | --    | Power factor.                                                 |
| `freq_hz`   | float, optional   | Hz    | Grid frequency.                                                |

Columns marked "optional" are used if present (`features.features.add_calendar_features` /
`add_weather_features` do not require them), but `pt_kw` and `qt_kvar` are always required since
they are the two joint forecasting targets (Eq. 1 in the paper).

## Calendar features

`features.features.add_calendar_features` derives eight deterministic calendar/holiday columns
from `pc_time` alone (time-of-day, day-of-week, weekend flag, Vietnamese public holiday flag,
month-of-year encodings). These require no external data and are always safe to compute.

## Weather features (optional)

`features.features.add_weather_features` joins a separate per-meter weather CSV (nearest
timestamp) with ERA5 reanalysis-style columns (temperature, humidity, cloud cover, shortwave
radiation, wind, precipitation, apparent temperature). See `features/fetch_weather.py` for the
Open-Meteo fetch pipeline used in the paper (requires internet access and station coordinates; not
run automatically since it depends on external, rate-limited APIs). `experiments/weather_ablation/`
expects these files at `<weather-dir>/<CODE>.csv`.

## Meter naming convention

Meter codes are treated as opaque identifiers. The `results/` CSVs in this repository use
anonymized labels (`M01`-`M33`) rather than the original EVN meter codes; `demo_data/` uses its
own synthetic codes (e.g. `DEMO0001`). This repository does not publish any mapping from these
labels to a real-world meter code, substation name, or location, consistent with EVN's
data-access restrictions on identifiable operational infrastructure (see the top-level README's
"What is and is not included" section).
