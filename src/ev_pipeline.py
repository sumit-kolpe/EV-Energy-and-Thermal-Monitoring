"""
ev_pipeline.py - shared helpers for the CIS 5450 EV Energy & Thermal Monitoring project.

Phase 1 (data loading, cleaning, trip segmentation). Used by the notebooks in /notebooks,
and importable by teammates for later phases:

    import sys; sys.path.insert(0, "../src")
    import ev_pipeline as ev

Dataset: TUMFTM "Electric Vehicle UDS Dataset" (7 VW-group BEVs: 2x ID.3, 5x CUPRA Born),
long format parquet: one row = one sensor reading (vehicle_id, time, value_id, value).
Licence CC BY-NC 4.0 - cite Schreiber et al., eTransportation 27 (2026) 100518.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths / download
# --------------------------------------------------------------------------------------
ZENODO_URL = ("https://zenodo.org/records/17532823/files/"
              "electric-vehicle-uds-dataset-public.zip?download=1")
ZENODO_MD5 = "d1f35f4a4a27db7e8c31ccef4600dbdb"
GITHUB_ZIP_URL = ("https://github.com/TUMFTM/electric-vehicle-uds-dataset/"
                  "archive/refs/heads/main.zip")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until we find the folder holding src/ev_pipeline.py."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "src" / "ev_pipeline.py").exists():
            return cand
    raise FileNotFoundError("Could not find project root (folder containing src/ev_pipeline.py)")


def project_paths(root: Path) -> dict:
    """Standard folder layout. Creates folders if missing."""
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "interim": root / "data" / "interim",
        "processed": root / "data" / "processed",
        "config": root / "config",
        "figures": root / "reports" / "figures",
        "reports": root / "reports",
    }
    for k, v in paths.items():
        if k != "root":
            v.mkdir(parents=True, exist_ok=True)
    return paths


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> Path:
    """Stream a download with a simple progress print. Skips if dest already exists."""
    import urllib.request
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "cis5450-ev-project"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            pct = int(100 * done / total) if total else -1
            if pct != last and pct % 5 == 0:
                print(f"  {done/1e6:8.1f} MB" + (f"  ({pct}%)" if total else ""), flush=True)
                last = pct
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def extract_needed(zip_path: Path, out_dir: Path) -> dict:
    """Extract only value_overview.csv and uds_data/*.parquet from the dataset zip.
    Returns {'overview': Path, 'parquet': [Path, ...]}."""
    import zipfile
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if m.endswith("value_overview.csv")
                   or (m.endswith(".parquet") and "uds_data" in m)]
        if not members:
            raise RuntimeError("Zip does not contain value_overview.csv / uds_data/*.parquet")
        for m in members:
            target = out_dir / Path(m).name
            if target.exists() and target.stat().st_size == z.getinfo(m).file_size:
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    dst.write(b)
    return locate_raw_files(out_dir)


def locate_raw_files(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir)
    overview = next(iter(sorted(raw_dir.rglob("value_overview.csv"))), None)
    parquet = sorted(p for p in raw_dir.rglob("*.parquet"))
    # Git-LFS pointer files are tiny text files, not real parquet
    lfs = [p for p in parquet if p.stat().st_size < 2000]
    if lfs:
        raise RuntimeError(
            f"{len(lfs)} parquet files look like Git-LFS pointers (tiny files). "
            "Download the Zenodo zip instead of the GitHub zip.")
    return {"overview": overview, "parquet": parquet}


# --------------------------------------------------------------------------------------
# Signal dictionary
# --------------------------------------------------------------------------------------
# Canonical column names used by every later notebook. Keywords are matched against the
# text columns of value_overview.csv (English + German, since the source is TUM).
# Short keywords (<=4 chars) match as whole words, longer ones as substrings.
CANONICAL_SIGNALS = {
    "speed":        {"include": ["vehicle speed", "speed", "velocity", "geschwindigkeit", "v_veh"],
                     "exclude": ["rpm", "motor", "rotation", "drehzahl", "fan", "pump", "wheel", "limit"],
                     "required": True,  "unit": "km/h"},
    "soc":          {"include": ["soc", "state of charge", "ladezustand"],
                     "exclude": ["max", "min", "target", "limit", "soh", "health"],
                     "required": True,  "unit": "%"},
    "bat_temp":     {"include": ["battery temp", "battery temperature", "hv battery temp", "bat temp",
                                 "cell temp", "module temp", "batterietemp", "temperature battery",
                                 "temp hv", "hv temp", "pack temp"],
                     "exclude": ["ambient", "outside", "coolant", "motor", "inverter", "max", "min"],
                     "required": True,  "unit": "degC"},
    "bat_temp_max": {"include": ["max cell temp", "max temp", "temperature max", "temp max", "maximum temp"],
                     "exclude": ["ambient", "outside", "coolant", "motor", "inverter"],
                     "required": False, "unit": "degC"},
    "bat_temp_min": {"include": ["min cell temp", "min temp", "temperature min", "temp min", "minimum temp"],
                     "exclude": ["ambient", "outside", "coolant", "motor", "inverter"],
                     "required": False, "unit": "degC"},
    "hv_voltage":   {"include": ["hv voltage", "battery voltage", "pack voltage", "voltage", "spannung"],
                     "exclude": ["cell", "zell", "12v", "lv", "min", "max", "limit"],
                     "required": False, "unit": "V"},
    "hv_current":   {"include": ["hv current", "battery current", "pack current", "current", "strom"],
                     "exclude": ["cell", "12v", "lv", "limit", "max", "min"],
                     "required": False, "unit": "A"},
    "motor_rpm":    {"include": ["rpm", "motor speed", "rotational speed", "drehzahl", "rotation"],
                     "exclude": ["fan", "pump", "compressor"],
                     "required": False, "unit": "1/min"},
    "ambient_temp": {"include": ["ambient", "outside temp", "outdoor", "aussentemp", "außentemp",
                                 "environment temp", "outside air"],
                     "exclude": [],
                     "required": False, "unit": "degC"},
    "odometer":     {"include": ["odometer", "mileage", "kilometerstand", "km stand", "odo", "total distance"],
                     "exclude": ["range", "trip"],
                     "required": False, "unit": "km"},
    "coolant_temp": {"include": ["coolant", "kühlmittel", "kuehlmittel"],
                     "exclude": [], "required": False, "unit": "degC"},
    "motor_temp":   {"include": ["motor temp", "stator temp", "rotor temp", "e-machine temp", "motortemp"],
                     "exclude": [], "required": False, "unit": "degC"},
    "range_est":    {"include": ["range", "reichweite"],
                     "exclude": [], "required": False, "unit": "km"},
}

# Verified mapping for the TUMFTM EV UDS dataset (value_overview.csv, Nov 2025 release).
# Note: this dataset has NO HV pack current, NO motor RPM and NO odometer signal.
TUM_SIGNAL_MAP = {
    "speed": "4",              # vehicle speed, km/h, ~200 ms
    "soc": "900",              # HV SOC, %, ~5 s
    "bat_temp": "1209",        # HV battery temp MAX (hottest sensor) - thermal target
    "bat_temp_min": "1208",    # HV battery temp min
    "bat_inlet_temp": "1272",  # battery pack coolant inlet temp
    "bat_outlet_temp": "1273", # battery pack coolant outlet temp
    "hv_voltage": "1200",      # pack voltage, V
    "ambient_temp": "15",      # ambient air temp, degC, ~10 s
    "interior_temp": "43",     # cabin temp
    "hv_aux_power": "56",      # HV auxiliary power (heating/AC), W
    "coolant_temp": "1269",    # inverter coolant inlet temp
    "motor_temp": "961",       # rear motor stator temp
    "rotor_temp": "1265",      # rear motor rotor temp
    "cell_voltage_max": "1293",
    "cell_voltage_min": "1294",
}
for _k in TUM_SIGNAL_MAP:
    CANONICAL_SIGNALS.setdefault(_k, {"include": [], "exclude": [], "required": False, "unit": ""})

# Physically plausible ranges; values outside are set to NaN during cleaning.
VALID_RANGES = {
    "speed": (0, 200), "soc": (0, 100), "bat_temp": (-40, 80), "bat_temp_max": (-40, 80),
    "bat_temp_min": (-40, 80), "hv_voltage": (200, 500), "hv_current": (-700, 700),
    "motor_rpm": (-20000, 20000), "ambient_temp": (-40, 60), "odometer": (0, 1_000_000),
    "coolant_temp": (-40, 120), "motor_temp": (-40, 200), "range_est": (0, 1000),
    "bat_inlet_temp": (-40, 80), "bat_outlet_temp": (-40, 80), "interior_temp": (-40, 80),
    "hv_aux_power": (-20000, 20000), "rotor_temp": (-40, 250),
    "cell_voltage_max": (2.5, 4.5), "cell_voltage_min": (2.5, 4.5),
}


def norm_id(x) -> str:
    """Normalise a value_id so 12, 12.0 and '12' all become '12'."""
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    s = str(x).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def sanitize(name: str, maxlen: int = 30) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s[:maxlen] or "x"


def read_value_overview(path: Path) -> pd.DataFrame:
    """Read value_overview.csv (separator auto-detected). Adds 'value_id' (str) and 'label'
    (all text columns joined) so we can keyword-match signals."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    id_col = next((c for c in df.columns if c.lower() == "value_id"), None) \
        or next((c for c in df.columns if "id" in c.lower()), df.columns[0])
    df = df.rename(columns={id_col: "value_id"})
    df["value_id"] = df["value_id"].map(norm_id)
    text_cols = [c for c in df.columns if c != "value_id"
                 and not pd.api.types.is_numeric_dtype(df[c])]
    df["label"] = df[text_cols].fillna("").astype(str).agg(" | ".join, axis=1) if text_cols \
        else df["value_id"]
    return df


def _kw_match(kw: str, text: str) -> bool:
    if len(kw) <= 4:
        return re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", text) is not None
    return kw in text


def propose_signal_map(overview: pd.DataFrame, coverage: pd.DataFrame | None = None) -> dict:
    """Keyword-based proposal {canonical_name: value_id or None}.
    `coverage` (optional) has columns value_id, n_valid; signals with no data are skipped.
    ALWAYS review the result in config/signal_map.json - it is a proposal, not ground truth."""
    ov = overview.copy()
    ov["text"] = ov["label"].str.lower()
    if coverage is not None:
        ov = ov.merge(coverage[["value_id", "n_valid"]], on="value_id", how="left")
        ov["n_valid"] = ov["n_valid"].fillna(0)
        ov = ov[ov["n_valid"] > 0]
    else:
        ov["n_valid"] = 1
    taken, out = set(), {}
    for name, spec in CANONICAL_SIGNALS.items():
        best, best_key = None, None
        for _, r in ov.iterrows():
            if r["value_id"] in taken:
                continue
            t = r["text"]
            if any(_kw_match(k, t) for k in spec["exclude"]):
                continue
            hits = [k for k in spec["include"] if _kw_match(k, t)]
            if not hits:
                continue
            # earlier keywords in the list are more specific -> higher priority
            key = (-min(spec["include"].index(h) for h in hits), len(hits), r["n_valid"])
            if best_key is None or key > best_key:
                best, best_key = r["value_id"], key
        out[name] = best
        if best is not None:
            taken.add(best)
    return out


def build_column_names(signal_map: dict, overview: pd.DataFrame, keep_ids: list[str]) -> dict:
    """value_id -> output column name. Mapped signals get canonical names, the rest
    'sig_<id>_<label>' so nothing useful is thrown away."""
    inv = {v: k for k, v in signal_map.items() if v is not None}
    labels = dict(zip(overview["value_id"], overview["label"]))
    cols = {}
    for vid in keep_ids:
        cols[vid] = inv.get(vid) or f"sig_{sanitize(vid, 10)}_{sanitize(labels.get(vid, ''), 25)}"
    return cols


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# DuckDB SQL builders
# --------------------------------------------------------------------------------------
NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                 "UINTEGER", "UBIGINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL")


def _q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def infer_time_unit(max_val: float) -> str:
    """Guess epoch unit from magnitude of the largest timestamp value."""
    if max_val > 1e17:
        return "ns"
    if max_val > 1e14:
        return "us"
    if max_val > 1e11:
        return "ms"
    return "s"


def build_ts_expr(col: str, col_type: str, unit: str | None = None) -> str:
    """SQL expression converting the raw time column into a DuckDB TIMESTAMP."""
    t = col_type.upper()
    if "TIMESTAMP" in t or t == "DATE":
        return f"CAST({_q(col)} AS TIMESTAMP)"
    if t.startswith(NUMERIC_TYPES):
        to_ms = {"s": 1000.0, "ms": 1.0, "us": 1e-3, "ns": 1e-6}[unit or "s"]
        return f"epoch_ms(CAST(round({_q(col)} * {to_ms}) AS BIGINT))"
    return f"TRY_CAST({_q(col)} AS TIMESTAMP)"


def build_id_expr(col: str, col_type: str) -> str:
    t = col_type.upper()
    if t.startswith(("FLOAT", "DOUBLE", "REAL", "DECIMAL")):
        return f"CAST(CAST({_q(col)} AS BIGINT) AS VARCHAR)"
    return f"trim(CAST({_q(col)} AS VARCHAR))"


def configure_duckdb(con, tmp_dir: Path, memory_limit: str = "4GB"):
    """Let DuckDB spill to disk instead of failing with out-of-memory on laptops."""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    con.sql(f"SET temp_directory = '{Path(tmp_dir).as_posix()}'")
    con.sql(f"SET memory_limit = '{memory_limit}'")
    con.sql("SET preserve_insertion_order = false")


def build_base_view_sql(files: dict, cfg: dict, view_name: str = "base",
                        apply_time_window: bool = True) -> str:
    """files: {vehicle_name_from_file: path}. cfg: schema info saved by notebook 01
    (time_col, time_type, time_unit, id_col, id_type, value_col, vehicle_col or None,
    optional ts_min / ts_max to drop corrupt timestamps).
    Produces view `<view_name>(vehicle_id, ts, vid, val)`."""
    parts = []
    for veh, path in files.items():
        p = Path(path).as_posix().replace("'", "''")
        veh_expr = (f"CAST({_q(cfg['vehicle_col'])} AS VARCHAR)" if cfg.get("vehicle_col")
                    else f"'{veh}'")
        parts.append(
            f"SELECT {veh_expr} AS vehicle_id, "
            f"{build_ts_expr(cfg['time_col'], cfg['time_type'], cfg.get('time_unit'))} AS ts, "
            f"{build_id_expr(cfg['id_col'], cfg['id_type'])} AS vid, "
            f"TRY_CAST({_q(cfg['value_col'])} AS DOUBLE) AS val "
            f"FROM read_parquet('{p}')")
    union = "\nUNION ALL\n".join(parts)
    if apply_time_window and cfg.get("ts_min") and cfg.get("ts_max"):
        union = (f"SELECT * FROM (\n{union}\n) WHERE ts BETWEEN TIMESTAMP '{cfg['ts_min']}' "
                 f"AND TIMESTAMP '{cfg['ts_max']}'")
    return f"CREATE OR REPLACE VIEW {view_name} AS\n" + union


def build_grid_sql(vid_to_col: dict, bucket_s: int = 10, source: str = "base") -> str:
    """Pivot long -> wide and average readings in fixed `bucket_s`-second windows."""
    aggs = ",\n  ".join(
        f"avg(val) FILTER (WHERE vid = '{vid}') AS {_q(col)}" for vid, col in vid_to_col.items())
    ids = ", ".join(f"'{v}'" for v in vid_to_col)
    return (
        f"SELECT vehicle_id,\n  time_bucket(INTERVAL '{int(bucket_s)} seconds', ts) AS ts,\n"
        f"  count(*) AS n_readings,\n  {aggs}\n"
        f"FROM {source}\nWHERE vid IN ({ids}) AND ts IS NOT NULL AND val IS NOT NULL\n"
        f"GROUP BY 1, 2\nORDER BY 1, 2")


# --------------------------------------------------------------------------------------
# Cleaning on the 10-second grid (pandas; ~300k rows so this is fast)
# --------------------------------------------------------------------------------------
def normalize_units(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fix obvious scale issues. Returns (df, notes)."""
    df, notes = df.copy(), []
    if "soc" in df and df["soc"].dropna().quantile(0.99) <= 1.0:
        df["soc"] *= 100
        notes.append("soc was 0-1 scaled -> converted to %")
    if "speed" in df and df["speed"].dropna().quantile(0.99) < 50 and df["speed"].max() < 70:
        notes.append("WARNING: speed looks like m/s (p99 < 50). Check units; multiplying by 3.6.")
        df["speed"] *= 3.6
    for c in [c for c in df.columns if c.endswith("temp") or "temp_" in c]:
        if df[c].dropna().median() > 200:  # Kelvin
            df[c] -= 273.15
            notes.append(f"{c} looked like Kelvin -> converted to degC")
    return df, notes


# Signals where an exact 0.0 is a logger placeholder rather than a measurement
# (found in notebook 01: spikes of exact zeros next to 1023.98 sentinels).
ZERO_IS_MISSING = ["bat_temp", "bat_inlet_temp", "bat_outlet_temp", "bat_temp_min", "hv_voltage", "soc"]


def apply_valid_ranges(df: pd.DataFrame, ranges: dict = VALID_RANGES,
                       zero_is_missing: list = ZERO_IS_MISSING) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set physically implausible values (and placeholder zeros) to NaN. Returns (df, report)."""
    df, rows = df.copy(), []
    for c, (lo, hi) in ranges.items():
        if c not in df:
            continue
        n_non_null = int(df[c].notna().sum())
        zero = df[c].eq(0) if c in zero_is_missing else pd.Series(False, index=df.index)
        df.loc[zero, c] = np.nan
        bad = df[c].notna() & ~df[c].between(lo, hi)
        rows.append({"column": c, "valid_min": lo, "valid_max": hi, "n_non_null": n_non_null,
                     "n_placeholder_zero": int(zero.sum()), "n_out_of_range": int(bad.sum())})
        df.loc[bad, c] = np.nan
    return df, pd.DataFrame(rows)


DESPIKE_COLS = {"bat_inlet_temp": 4.0, "bat_outlet_temp": 4.0, "coolant_temp": 4.0,
                "motor_temp": 6.0, "rotor_temp": 8.0, "ambient_temp": 4.0}


def despike(df: pd.DataFrame, cols: dict = DESPIKE_COLS, window: int = 7) -> tuple[pd.DataFrame, dict]:
    """Remove single-sample glitches: a value deviating more than `max_dev` from the centred
    rolling median (window buckets, per vehicle) is set to NaN. Coolant temperatures cannot
    jump 15-30 degC within 10 s, but the raw logs contain such one-sample spikes.
    Returns (df, {col: n_removed})."""
    df = df.sort_values(["vehicle_id", "ts"]).copy()
    removed = {}
    for c, max_dev in cols.items():
        if c not in df:
            continue
        med = df.groupby("vehicle_id")[c].transform(
            lambda s: s.rolling(window, center=True, min_periods=3).median())
        bad = df[c].notna() & (df[c] - med).abs().gt(max_dev)
        removed[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    return df, removed


def interpolate_slow(df: pd.DataFrame, cols: list[str], max_gap_s: float = 600) -> pd.DataFrame:
    """Time-based linear interpolation for slowly-changing, burst-sampled signals (battery temp).
    Only fills between two real observations of the same vehicle that are <= max_gap_s apart.
    Adds '<col>_observed' (True where the bucket had a real reading)."""
    df = df.sort_values(["vehicle_id", "ts"]).copy()
    for c in cols:
        if c not in df:
            continue
        df[f"{c}_observed"] = df[c].notna()
        out = []
        for _, g in df.groupby("vehicle_id", sort=False):
            s = pd.Series(g[c].values, index=pd.DatetimeIndex(g["ts"].values))
            obs_t = pd.Series(np.where(s.notna(), g["ts"].values, np.datetime64("NaT")), index=s.index)
            prev_t, next_t = obs_t.ffill(), obs_t.bfill()
            gap = (next_t - prev_t).dt.total_seconds()
            interp = s.interpolate(method="time", limit_area="inside")
            out.append(pd.Series(np.where(gap.values <= max_gap_s, interp.values, np.nan), index=g.index))
        df[c] = pd.concat(out)
    return df


def fill_short_gaps(df: pd.DataFrame, cols: list[str], max_gap_s: float = 60) -> pd.DataFrame:
    """Forward-fill each column within a vehicle, but only if the last real observation
    is at most `max_gap_s` seconds old (slow signals like SOC / temperature are polled
    less often than speed)."""
    df = df.sort_values(["vehicle_id", "ts"]).copy()
    for c in cols:
        if c not in df:
            continue
        last_obs_ts = df["ts"].where(df[c].notna())
        last_obs_ts = last_obs_ts.groupby(df["vehicle_id"]).ffill()
        filled = df[c].groupby(df["vehicle_id"]).ffill()
        age = (df["ts"] - last_obs_ts).dt.total_seconds()
        df[c] = filled.where(age <= max_gap_s)
    return df


def add_kinematics(df: pd.DataFrame, bucket_s: int = 10) -> pd.DataFrame:
    """dt_s, contiguous flag, distance per bucket (km) and acceleration (m/s^2)."""
    df = df.sort_values(["vehicle_id", "ts"]).copy()
    dt = df.groupby("vehicle_id")["ts"].diff().dt.total_seconds()
    df["dt_s"] = dt
    df["contiguous"] = dt.le(bucket_s * 1.5)
    df["dist_km"] = df["speed"].fillna(0) * bucket_s / 3600.0
    dv = df.groupby("vehicle_id")["speed"].diff() / 3.6
    df["accel_ms2"] = (dv / dt).where(df["contiguous"])
    return df


def segment_trips(df: pd.DataFrame, moving_kmh: float = 2.0, max_stop_s: float = 300,
                  min_duration_s: float = 120, min_distance_km: float = 0.5,
                  bucket_s: int = 10) -> pd.DataFrame:
    """Trip = run of 'moving' buckets (speed > moving_kmh) where consecutive moving buckets
    are at most `max_stop_s` apart (so traffic lights / short stops do not split a trip).
    All buckets between a trip's first and last moving bucket belong to the trip.
    Trips shorter than min_duration_s or min_distance_km are dropped (trip_id = NaN).
    Adds trip_id ('<vehicle>_<nnnn>'), trip_elapsed_s and trip_dist_km (cumulative)."""
    df = df.sort_values(["vehicle_id", "ts"]).reset_index(drop=True).copy()
    trip_id = pd.Series(pd.NA, index=df.index, dtype="object")
    for veh, g in df.groupby("vehicle_id", sort=False):
        moving = g["speed"] > moving_kmh
        mts = g.loc[moving, "ts"]
        if mts.empty:
            continue
        new = mts.diff().dt.total_seconds().gt(max_stop_s)
        new.iloc[0] = True
        tno = new.cumsum()
        starts = mts.groupby(tno).min().values
        ends = mts.groupby(tno).max().values
        tsv = g["ts"].values
        idx = np.searchsorted(starts, tsv, side="right") - 1
        ok = (idx >= 0) & (tsv <= ends[np.clip(idx, 0, None)])
        raw_no = pd.Series(np.where(ok, idx, -1), index=g.index)
        # filter short trips
        dur = (ends - starts) / np.timedelta64(1, "s") + bucket_s
        dist = g["dist_km"].groupby(raw_no).sum()
        keep = [k for k in range(len(starts))
                if dur[k] >= min_duration_s and dist.get(k, 0) >= min_distance_km]
        renum = {k: f"{veh}_{i+1:04d}" for i, k in enumerate(keep)}
        trip_id.loc[g.index] = raw_no.map(renum).values
    df["trip_id"] = trip_id
    in_trip = df["trip_id"].notna()
    grp = df.loc[in_trip].groupby("trip_id")
    df.loc[in_trip, "trip_elapsed_s"] = (df.loc[in_trip, "ts"] - grp["ts"].transform("min")).dt.total_seconds()
    df.loc[in_trip, "trip_dist_km"] = grp["dist_km"].cumsum()
    return df


def label_states(df: pd.DataFrame, max_gap_s: float = 300, charge_soc_gain: float = 1.0) -> pd.DataFrame:
    """state = 'driving' (inside a trip), 'charging' (stationary session where SOC rises by
    >= charge_soc_gain %-points), otherwise 'stationary'."""
    df = df.sort_values(["vehicle_id", "ts"]).copy()
    df["state"] = np.where(df["trip_id"].notna(), "driving", "stationary")
    stat = df["state"].eq("stationary")
    # session id: new session when state changes or a time gap > max_gap_s
    gap = df.groupby("vehicle_id")["ts"].diff().dt.total_seconds().gt(max_gap_s)
    change = df["state"].ne(df.groupby("vehicle_id")["state"].shift())
    df["session_no"] = (gap | change).cumsum()
    if "soc" in df:
        s = df.loc[stat].groupby("session_no")["soc"]
        gain = s.last() - s.first()  # first/last skip NaN
        charging_sessions = gain[gain >= charge_soc_gain].index
        df.loc[stat & df["session_no"].isin(charging_sessions), "state"] = "charging"
    return df


def add_power(df: pd.DataFrame, bucket_s: int = 10) -> tuple[pd.DataFrame, str]:
    """If HV voltage & current exist: power_kw (discharge positive) and energy_kwh per bucket."""
    if not {"hv_voltage", "hv_current"} <= set(df.columns) or df["hv_current"].notna().sum() == 0:
        return df, "No HV current signal -> energy must be measured as SOC%/km (as in the proposal)."
    df = df.copy()
    p = df["hv_voltage"] * df["hv_current"] / 1000.0
    cruising = df["speed"].gt(30) & p.notna()
    sign = 1.0
    if cruising.any() and p[cruising].median() < 0:
        sign = -1.0
    df["power_kw"] = sign * p
    df["energy_kwh"] = df["power_kw"] * bucket_s / 3600.0
    note = "HV current found -> power_kw/energy_kwh computed" + \
        (" (sign flipped so discharge is positive)" if sign < 0 else "")
    return df, note


def build_trip_table(df: pd.DataFrame, moving_kmh: float = 2.0) -> pd.DataFrame:
    """One row per trip with features useful for clustering (3.1), energy (3.3) and tests (4.x)."""
    t = df[df["trip_id"].notna()].copy()
    t["is_stopped"] = t["speed"].le(moving_kmh)
    g = t.groupby("trip_id", sort=False)
    out = pd.DataFrame({
        "vehicle_id": g["vehicle_id"].first(),
        "start_ts": g["ts"].min(),
        "end_ts": g["ts"].max(),
        "n_buckets": g.size(),
        "distance_km": g["dist_km"].sum(),
        "speed_mean_moving": t[~t["is_stopped"]].groupby("trip_id")["speed"].mean(),
        "speed_mean": g["speed"].mean(),
        "speed_max": g["speed"].max(),
        "speed_std": g["speed"].std(),
        "stop_fraction": g["is_stopped"].mean(),
        "accel_abs_mean": g["accel_ms2"].apply(lambda s: s.abs().mean()),
        "accel_p95": g["accel_ms2"].quantile(0.95),
        "decel_p05": g["accel_ms2"].quantile(0.05),
    })
    out["duration_min"] = (out["end_ts"] - out["start_ts"]).dt.total_seconds() / 60 + 10 / 60
    if "motor_rpm" in t:
        out["rpm_abs_mean"] = g["motor_rpm"].apply(lambda s: s.abs().mean())
        out["rpm_abs_max"] = g["motor_rpm"].apply(lambda s: s.abs().max())
    if "soc" in t:
        out["soc_start"] = g["soc"].first()
        out["soc_end"] = g["soc"].last()
        out["soc_used"] = out["soc_start"] - out["soc_end"]
        out["soc_per_100km"] = 100 * out["soc_used"] / out["distance_km"].where(out["distance_km"] > 0)
    for c in ["bat_temp", "bat_outlet_temp"]:
        if c in t:
            out[f"{c}_start"] = g[c].first()
            out[f"{c}_end"] = g[c].last()
            out[f"{c}_max"] = g[c].max()
            out[f"{c}_rise"] = out[f"{c}_end"] - out[f"{c}_start"]
    for c in ["ambient_temp", "hv_voltage", "coolant_temp", "motor_temp", "interior_temp",
              "bat_inlet_temp", "hv_aux_power"]:
        if c in t:
            out[f"{c}_mean"] = g[c].mean()
    if "hv_voltage" in t:
        out["hv_voltage_min"] = g["hv_voltage"].min()
    if "odometer" in t:
        out["odo_distance_km"] = g["odometer"].last() - g["odometer"].first()
    if "energy_kwh" in t:
        out["energy_kwh"] = g["energy_kwh"].sum()
        out["kwh_per_100km"] = 100 * out["energy_kwh"] / out["distance_km"].where(out["distance_km"] > 0)
    out["start_hour"] = out["start_ts"].dt.hour
    out["start_month"] = out["start_ts"].dt.month
    out["weekday"] = out["start_ts"].dt.dayofweek
    if "soc_used" in out:
        # plausible consumption only: 58 kWh pack -> 5-100 %/100 km = 3-58 kWh/100 km
        plausible = (out["soc_used"] > 0) & out["soc_per_100km"].between(5, 100)
        out["energy_valid_5km"] = (out["distance_km"] >= 5) & plausible
        out["energy_valid_20km"] = (out["distance_km"] >= 20) & plausible
    return out.reset_index()
