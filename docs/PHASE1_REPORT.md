# EV Energy & Thermal Monitoring: Phase 1 Report


## Overview

Phase 1 is complete: the raw TUM telemetry (98.1M readings) is now a clean 10-second dataset of 911,145 rows and a table of 2,389 trips, ready for clustering and forecasting.

The project (CIS 5450) builds an EV energy and thermal monitoring system: driving-regime clustering, battery temperature forecasting, trip energy forecasting, and thermal-demand classification. Phase 1 covered data loading, cleaning and EDA, all run on the real dataset.

| Item | Value |
| --- | --- |
| Dataset | [TUMFTM Electric Vehicle UDS Dataset](https://github.com/TUMFTM/electric-vehicle-uds-dataset) (Zenodo record 17532823, CC BY-NC 4.0) |
| Vehicles | 7: CUP1–CUP5 (CUPRA Born), ID1–ID2 (VW ID.3), 58 kWh pack each |
| Raw readings | 98,061,214 (long format: vehicle_id, time, value_id, value) |
| Signals | 29 defined, 15 with data |
| Clean 10 s grid | 911,145 rows (290,380 while driving) |
| Trips | 2,389 trips, 43,662 km |
| Tools | DuckDB (raw queries), pandas, matplotlib/seaborn, Jupyter |

## What I did

The pipeline runs in three notebooks, in order, and every step was run and checked on the real data.

1. **Profiled the raw data (notebook 01).** DuckDB queried all 7 Parquet files without loading them into memory. We checked row counts, signal coverage per vehicle, value ranges and sampling intervals, and confirmed 15 of 29 signals contain data.
2. **Fixed timestamps.** 812 CUP1 readings were stamped in the year 2087 (a logger clock glitch) and were dropped by keeping 2020–2024 only.
3. **Mapped signals to names.** Each raw `value_id` was matched against `value_overview.csv` and given a readable column name, saved in `config/signal_map.json` (for example `4` → `speed`, `900` → `soc`, `1209` → `bat_temp`).
4. **Built the 10-second grid (notebook 02).** DuckDB pivoted the long table to one column per signal and averaged readings in 10 s windows, giving one row per vehicle per 10 s.
5. **Cleaned values.**
    - Placeholder values (exact 0.00 and the 1023.98 sentinel) in coolant and pack temperatures were set to missing.
    - Physically impossible values were removed (speed above 200 km/h, pack voltage above 500 V, and similar).
    - A rolling-median filter removed about 19,000 one-reading spikes in the coolant temperatures.
6. **Filled gaps carefully.** Slow signals (SOC, temperatures) were carried forward for at most 60 s. Pack max temperature, which is logged in bursts, was interpolated across gaps of up to 10 minutes, with a `bat_temp_observed` flag marking real readings. Speed was never filled.
7. **Derived motion.** Distance = speed × time, plus acceleration between consecutive 10 s rows.
8. **Split into trips.** A trip is a run of moving rows (speed above 2 km/h); stops shorter than 5 minutes stay inside the trip. Trips under 2 minutes or 0.5 km were dropped. Each row is labelled `driving`, `charging` (parked with SOC rising by at least 1 point) or `stationary`.
9. **Built the trip table.** One row per trip with distance, duration, speed, stop share, acceleration, battery % used per 100 km, temperature rise and outside temperature.
10. **Validated.** No duplicate rows, every trip has moving rows, no trip spans two vehicles, and 97% of trips show an SOC decrease.
11. **Explored the data (notebook 03).** EDA charts and an auto-generated summary of key numbers, saved in `reports/`.

## Key findings

The dataset has no battery current, motor RPM or odometer, so parts of the proposal must change; outside temperature, not speed, is the main driver of energy use.

**Data facts that change the proposal**

- **No HV battery current.** Energy is measured as battery % used per 100 km (SOC %/100 km), as the proposal anticipated. Energy in Wh is not possible.
- **No motor RPM.** Replace RPM features with speed and acceleration in clustering and forecasting.
- **No odometer.** Distance comes from integrating speed. Speed is present in 99.9% of trip rows, so this is reliable.
- **Battery temperature is sparse.** Pack max temperature is logged in bursts and covers 43% of driving rows after interpolation. The coolant outlet temperature covers 91% and tracks it closely (r = 0.88), so it is a usable backup target.
- **Logging covers 43.7k km**, not the 72k km in the dataset README; the logger did not capture every drive.

**Energy use**

- Median consumption is 33.8 SOC %/100 km, about 20 kWh/100 km, which is realistic for these cars.
- Outside temperature has the strongest link to consumption (Spearman −0.56). Winter months run about 36–45 %/100 km, summer months about 26–30.
- Mean trip speed has almost no link to consumption (Spearman −0.02).
- 1,374 trips of 5 km or more, and 420 trips of 20 km or more, have plausible energy values (5–100 %/100 km).

**Battery thermal behaviour**

- While driving, battery temperature has p90 = 30.3 °C and p95 = 32.0 °C.
- The 5-minute temperature rise has p95 = 1.1 °C.
- Candidate thermal-demand label: temperature 5 minutes ahead at or above 30.3 °C, or a 5-minute rise of at least 1.1 °C. That makes 16% of cases positive, so the classes are imbalanced.

**Uneven data across vehicles**

| Vehicle | Model | Logged | Trips | Distance (km) | Trips ≥ 20 km |
| --- | --- | --- | --- | --- | --- |
| ID2 | VW ID.3 | Jul 2021 – Oct 2023 | 836 | 18,089 | 132 |
| ID1 | VW ID.3 | Feb 2021 – Jun 2023 | 511 | 15,448 | 184 |
| CUP1 | CUPRA Born | Nov 2022 – Apr 2023 | 395 | 4,049 | 53 |
| CUP4 | CUPRA Born | Nov 2022 – Apr 2023 | 226 | 2,333 | 25 |
| CUP5 | CUPRA Born | Nov 2022 – Apr 2023 | 188 | 1,343 | 9 |
| CUP2 | CUPRA Born | Nov 2022 – May 2023 | 187 | 1,859 | 17 |
| CUP3 | CUPRA Born | Nov 2022 – Apr 2023 | 46 | 541 | 4 |

The CUPRAs were logged only in winter and spring, so any vehicle comparison is mixed with season.

## Repository files

The GitHub repo holds the code, the cleaned data (about 35 MB) and the reports; the 1.2 GB raw download is not in the repo and is fetched from Zenodo by notebook 01.

| Path | What it is | Who needs it |
| --- | --- | --- |
| `README.md` | Folder guide, how to run, data findings, rules for Phase 2 | Everyone |
| `requirements.txt` | Python packages (duckdb, pyarrow, pandas, seaborn, scikit-learn, jupyterlab) | Everyone |
| `src/ev_pipeline.py` | Shared functions: download, DuckDB SQL builders, cleaning, gap filling, trip splitting, trip table | Anyone re-running or extending the pipeline |
| `notebooks/01_data_acquisition_and_profiling.ipynb` | Downloads the data and profiles it with DuckDB: coverage, ranges, bad timestamps, signal map | Reference |
| `notebooks/02_cleaning_and_trip_segmentation.ipynb` | Builds the 10 s grid, cleans it, splits trips, writes `data/processed/`. Ends with the column dictionary | Reference; read the column dictionary |
| `notebooks/03_eda.ipynb` | All EDA charts and the key-number summary | Everyone |
| `config/signal_map.json` | Raw `value_id` → column name, plus raw schema settings | Pipeline |
| `data/processed/telemetry_10s.parquet` | **Main dataset.** One row per vehicle per 10 s: all signals, `trip_id`, `trip_elapsed_s`, `trip_dist_km`, `state`, `dist_km`, `accel_ms2`, `bat_temp_observed` | Temperature forecasting, thermal classification, early-trip energy features |
| `data/processed/trips.parquet` / `trips.csv` | One row per trip: start/end, duration, distance, speed stats, stop share, acceleration, `soc_used`, `soc_per_100km`, battery temperature start/end/max/rise, outside temperature, aux power, validity flags `energy_valid_5km` / `energy_valid_20km` | Clustering, energy analysis, hypothesis tests |
| `data/processed/pipeline_params.json` | Settings used (10 s window, 5 min stop rule, …), removed-glitch counts, validation results | Reproducibility |
| `data/processed/data_quality_report.csv` | Per signal: invalid, placeholder and missing counts | Data quality section of the report |
| `reports/eda_summary.md` | Key numbers, auto-generated by notebook 03 | Everyone |
| `reports/figures/*.png` | 16 charts: coverage, fleet, distributions, correlations, energy, thermal, seasonality | Slides and the final report |

Not in the repo: `data/raw/` (re-downloadable) and `data/interim/` (regenerated by notebooks 01–02).

## Getting started for Phase 2

Clone the repo, install the requirements, and load the two processed files; there's no need to re-run the pipeline.

```bash
git clone https://github.com/sumit-kolpe/EV-Energy-and-Thermal-Monitoring.git
cd EV-Energy-and-Thermal-Monitoring
pip install -r requirements.txt
```

```python
import sys; sys.path.insert(0, "src")
import pandas as pd, ev_pipeline as ev
tel   = pd.read_parquet("data/processed/telemetry_10s.parquet")  # vehicle x 10 s rows
trips = pd.read_parquet("data/processed/trips.parquet")          # one row per trip
```

**Rules that keep results honest**

- Split train and test **by trip or by vehicle** (GroupKFold), never by random rows; neighbouring 10 s rows are nearly identical.
- For "rest of the trip" and "5 minutes ahead" models, build features only from rows up to the prediction time (`trip_elapsed_s`). Whole-trip columns in `trips.parquet`, including any driving regime clustered on whole trips, leak the future.
- Use `energy_valid_5km` or `energy_valid_20km` to pick trips for energy analysis.
- Permutation tests should shuffle within each vehicle, as the proposal planned, because trip counts range from 46 to 836.

**Decisions for the team**

- [ ] Thermal target for RQ2/RQ4: pack max `bat_temp` (accurate, 43% coverage) or coolant `bat_outlet_temp` (91% coverage, r = 0.88)?
- [ ] Confirm the thermal-demand label rule (≥ 30.3 °C in 5 min, or a ≥ 1.1 °C rise).
- [ ] Minimum trip distance for the energy model: 5 km (1,374 trips) or 20 km (420 trips)?
- [ ] Replace motor RPM in the proposal's feature lists with speed and acceleration.

## Sources

- [TUMFTM Electric Vehicle UDS Dataset on GitHub](https://github.com/TUMFTM/electric-vehicle-uds-dataset)
- [Dataset archive on Zenodo (record 17532823)](https://zenodo.org/records/17532823)
- Schreiber et al., "Lab-to-Field Gap in Battery Aging Studies", eTransportation 27 (2026) 100518, [doi:10.1016/j.etran.2025.100518](https://doi.org/10.1016/j.etran.2025.100518)
