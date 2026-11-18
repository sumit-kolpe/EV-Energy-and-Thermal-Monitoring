# EV Energy and Thermal Monitoring

CIS 5450 final project. We use real-world telemetry from 7 electric vehicles (about two years, 98M sensor readings) to group trips into driving regimes, forecast battery temperature and trip energy use, and test which conditions drive them.

**Everything is in one notebook: [`EV_Energy_and_Thermal_Monitoring.ipynb`](EV_Energy_and_Thermal_Monitoring.ipynb).**

## Repository layout

```
EV-Energy-and-Thermal-Monitoring/
├── EV_Energy_and_Thermal_Monitoring.ipynb   # the project: download, cleaning, EDA, models, hypothesis tests
├── proposal/                                # project proposal (.pdf and LaTeX source)
├── requirements.txt
├── README.md
└── data/                                    # created by the notebook, never committed
    ├── raw/                                 #   original parquet files from Zenodo (~1.2 GB)
    └── processed/                           #   cleaned data (~30 MB)
        ├── telemetry_10s.parquet            #     one row per vehicle per 10 seconds
        ├── trips.parquet / trips.csv        #     one row per trip
        ├── data_quality_report.csv          #     cleaning counts per signal
        └── pipeline_params.json             #     row counts and spikes removed
```

## Running it

```bash
git clone https://github.com/sumit-kolpe/EV-Energy-and-Thermal-Monitoring.git
cd EV-Energy-and-Thermal-Monitoring
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Open `EV_Energy_and_Thermal_Monitoring.ipynb`, pick the `.venv` kernel and run all cells. The first run downloads the raw dataset from Zenodo (about 620 MB) and builds `data/processed/`, which takes a few minutes. Later runs reuse the processed files. To rebuild them, set `REBUILD_DATA = True` in section 2.2.

The `data/` folder is in `.gitignore`. To keep the raw download somewhere else (for example outside OneDrive), set `EV_RAW_DIR` before starting Jupyter: `$env:EV_RAW_DIR="C:\ev_raw"` in PowerShell, or `export EV_RAW_DIR=~/ev_raw` on macOS/Linux.

**Windows note:** if an import fails with "An Application Control policy has blocked this file", Windows Smart App Control is blocking a brand-new package build. Install the previous release instead, e.g. `pip install "pandas<3.1"`.

## Research questions

| RQ | Question | Result |
| --- | --- | --- |
| RQ1 | Can trips be grouped into driving regimes, and does energy use differ between them? | 4 K-Means regimes (silhouette 0.27 vs 0.13 for speed bins); energy differs by regime after adjusting for temperature (p = 0.0002) |
| RQ2 | Can battery temperature be predicted 5 minutes ahead? | XGBoost MAE 0.31 °C, 43% better than persistence (p = 0.0001) |
| RQ3 | Can the first 5 minutes of a trip predict energy use for the rest of it? | XGBoost 20% better than the mean, R² = 0.32 (p = 0.0001) |
| RQ4 | Do cold trips use more battery per 100 km? | Yes, 7.8 SOC %/100 km more below 10 °C (p = 0.0001) |
| RQ5 (optional) | Do usage patterns relate to battery health? | Not started |

Changes from the proposal are explained at the top of the notebook.

## Team

| Member | Main responsibility |
| --- | --- |
| Sumit Kolpe | Data loading, cleaning, EDA; trip energy models |
| Dee Xie | Battery temperature forecasting; data cleaning, EDA |
| Hari Raman | Hypothesis testing; battery temperature forecasting |
| Aditya Bajoria | Driving regimes; hypothesis testing |
| Aditya Muppavarapu | Trip energy models; driving regimes |

## Data source and licence

TUMFTM Electric Vehicle UDS Dataset ([GitHub](https://github.com/TUMFTM/electric-vehicle-uds-dataset), [Zenodo record 17532823](https://zenodo.org/records/17532823)), licensed CC BY-NC 4.0.

M. Schreiber et al., "Lab-to-Field Gap in Battery Aging Studies: Mismatch of Operating Conditions between Laboratory Environments and Real-world Automotive Applications", *eTransportation* 27 (2026) 100518, [doi:10.1016/j.etran.2025.100518](https://doi.org/10.1016/j.etran.2025.100518).
