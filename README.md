# EV Energy and Thermal Monitoring

**CIS 5450 – Big Data Analytics · University of Pennsylvania · Fall 2026**

We are building a context-aware EV energy and thermal monitoring system. It learns how electric vehicles behave from about two years of real-world telemetry (7 vehicles), groups trips into driving regimes, forecasts battery temperature and trip energy use, and tests which conditions drive them.

The full Phase 1 write-up (what was done, findings, what every file is) is in **[docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md)**.

## Team

| Member | Main responsibility |
| --- | --- |
| Sumit Kolpe | Data loading, cleaning, EDA; supervised learning: trip energy use |
| Dee Xie | Supervised learning: battery temperature forecasting; data cleaning, EDA |
| Hari Raman | Hypothesis testing and bootstrapping; supervised learning: battery temperature forecasting |
| Aditya Bajoria | Unsupervised learning; hypothesis testing |
| Aditya Muppavarapu | Supervised learning: trip energy use; unsupervised learning |

## Project phases

| Phase | What it covers | Research question | Owners | Status |
| --- | --- | --- | --- | --- |
| 0. Proposal | Problem, dataset choice, modelling and testing plan | – | All | Done |
| 1. Data pipeline & EDA | Load 98M raw readings with DuckDB, clean, build 10 s grid, split trips, EDA | – | Sumit, Dee | **Done** |
| 2. Driving regimes | K-Means on trip features, silhouette score, PCA plots; regime label per trip | RQ1 | Aditya B., Aditya M. | Next |
| 3a. Battery temperature forecasting | Predict battery temperature 5 min ahead; persistence/ARIMA baseline vs Random Forest, XGBoost, LSTM; RMSE, R² | RQ2 | Dee, Hari | Not started |
| 3b. Trip energy forecasting | Predict rest-of-trip SOC %/100 km from the first minutes; Lasso baseline vs Random Forest/XGBoost (LSTM optional) | RQ3 | Aditya M., Sumit | Not started |
| 3c. Thermal-demand classification (extension) | Will the battery become thermally demanding in the next 5 min? Logistic regression vs Random Forest/XGBoost; precision, recall, F1, ROC-AUC | RQ4 | Dee, Hari | Not started |
| 4. Hypothesis testing | Permutation tests (10,000 shuffles, within vehicle, trip as unit) for RQ1–RQ4 | RQ1–RQ4 | Hari, Aditya B. | Not started |
| 5. Usage patterns & battery health (extension) | Link long-term usage to battery State of Health using the TUM SOH dataset; exploratory with 7 vehicles | RQ5 | Optional | Not started |
| 6. Final report & presentation | Results, figures, limitations, slides | – | All | Not started |

Research questions (from the proposal):
- **RQ1:** Does energy consumption differ between driving regimes?
- **RQ2:** Do recent operating conditions predict battery temperature 5 minutes ahead, beyond the current temperature?
- **RQ3:** Does early-trip driving behaviour predict energy use for the rest of the trip?
- **RQ4:** Does recent history predict thermal demand 5 minutes ahead?
- **RQ5:** Do long-term driving patterns predict battery State of Health?

## Repository layout

```
EV-Energy-and-Thermal-Monitoring/
├── README.md
├── requirements.txt
├── CIS_5450_-_Project_Proposal.pdf
├── docs/
│   └── PHASE1_REPORT.md            # Phase 1 report: steps, findings, file guide
├── src/
│   └── ev_pipeline.py              # shared functions: download, DuckDB SQL, cleaning, trips
├── notebooks/
│   ├── 01_data_acquisition_and_profiling.ipynb   # Phase 1: download + DuckDB profiling
│   ├── 02_cleaning_and_trip_segmentation.ipynb   # Phase 1: 10 s grid, cleaning, trips
│   └── 03_eda.ipynb                              # Phase 1: exploratory analysis
├── config/
│   └── signal_map.json             # raw value_id -> column names
├── data/
│   ├── raw/                        # original download (~1.2 GB, not in git)
│   ├── interim/                    # intermediate files (not in git)
│   └── processed/                  # cleaned datasets used by all later phases
└── reports/
    ├── eda_summary.md              # key EDA numbers
    └── figures/                    # all charts (PNG)
```

New work goes into numbered notebooks so the order stays clear:
`04_driving_regimes.ipynb` (Phase 2), `05_battery_temp_forecasting.ipynb` (3a), `06_trip_energy_forecasting.ipynb` (3b), `07_thermal_demand_classification.ipynb` (3c), `08_hypothesis_tests.ipynb` (Phase 4), `09_soh_extension.ipynb` (Phase 5).

## Getting started

```bash
git clone https://github.com/sumit-kolpe/EV-Energy-and-Thermal-Monitoring.git
cd EV-Energy-and-Thermal-Monitoring
pip install -r requirements.txt
```

The cleaned data is already in `data/processed/`, so you don't need to re-run Phase 1 to start modelling:

```python
import sys; sys.path.insert(0, "../src")      # from inside notebooks/
import pandas as pd, ev_pipeline as ev

tel   = pd.read_parquet("../data/processed/telemetry_10s.parquet")   # one row per vehicle per 10 s
trips = pd.read_parquet("../data/processed/trips.parquet")           # one row per trip
```

Column definitions are at the end of notebook 02. Pipeline settings (window size, trip rules, cleaning counts) are in `data/processed/pipeline_params.json`.

### Re-running Phase 1 from scratch (optional)

Notebook 01 downloads the dataset from Zenodo (~620 MB zip) if `data/raw/` is empty. Run the notebooks in order:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_acquisition_and_profiling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_cleaning_and_trip_segmentation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
```

To keep the raw data somewhere else (for example outside OneDrive), set `EV_RAW_DIR` before starting Jupyter: `$env:EV_RAW_DIR="C:\ev_raw"` in PowerShell or `export EV_RAW_DIR=~/ev_raw` on macOS/Linux.

## Phase 1 results at a glance

- 98,061,214 raw readings from 7 vehicles (CUP1–CUP5 = CUPRA Born, ID1–ID2 = VW ID.3); 29 signals defined, 15 contain data.
- Cleaned output: 911,145 rows on the 10 s grid (290,380 while driving), 2,389 trips, 43,662 km.
- **The dataset has no HV battery current, no motor RPM and no odometer.** Energy is measured as SOC %/100 km, distance comes from integrating speed, and speed/acceleration replace RPM as features.
- Outside temperature is the strongest driver of consumption (Spearman −0.56); average trip speed has almost no effect.
- Battery pack max temperature (`bat_temp`) is logged in bursts (43% of driving rows after interpolation). Coolant outlet temperature (`bat_outlet_temp`, 91% coverage, r = 0.88) is a backup thermal target.
- Data is very uneven: 46 trips for CUP3 vs 836 for ID2, and the CUPRAs were logged only from November to May.

Details, tables and figures are in [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) and `reports/`.

## Ground rules for Phases 2–5

- Split train and test **by trip or by vehicle** (GroupKFold), never by random rows.
- For "rest of the trip" and "5 minutes ahead" models, build features only from rows up to the prediction time (`trip_elapsed_s`). Whole-trip features, including a regime clustered on whole trips, leak the future.
- For energy work, filter trips with `energy_valid_5km` or `energy_valid_20km`.
- Permutation tests shuffle within each vehicle.

### Open decisions

- Thermal target for RQ2/RQ4: `bat_temp` (pack max, 43% coverage) or `bat_outlet_temp` (91% coverage)?
- Thermal-demand label: temperature ≥ 30.3 °C in 5 min, or a rise of ≥ 1.1 °C (16% positive)?
- Minimum trip distance for the energy model: 5 km (1,374 trips) or 20 km (420 trips)?

## Data source and licence

TUMFTM Electric Vehicle UDS Dataset ([GitHub](https://github.com/TUMFTM/electric-vehicle-uds-dataset), [Zenodo record 17532823](https://zenodo.org/records/17532823)), licensed CC BY-NC 4.0.

M. Schreiber et al., "Lab-to-Field Gap in Battery Aging Studies: Mismatch of Operating Conditions between Laboratory Environments and Real-world Automotive Applications", *eTransportation* 27 (2026) 100518, [doi:10.1016/j.etran.2025.100518](https://doi.org/10.1016/j.etran.2025.100518).
