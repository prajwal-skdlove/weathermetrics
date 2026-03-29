import os
import re
import math
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import cudf
except ImportError:
    cudf = None

import pandas as pd

# logging utility
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# geospatial constants
EARTH_RADIUS_MILES = 3958.8

DIRECTION_DELTAS = {
    "N": (1, 0),
    "S": (-1, 0),
    "E": (0, 1),
    "W": (0, -1),
    "NE": (1, 1),
    "NW": (1, -1),
    "SE": (-1, 1),
    "SW": (-1, -1),
}


@dataclass
class PipelineConfig:
    stations_csv: str = "src/weathermertricsdata/data/stations.csv"
    station_id_col: str = "stid"
    latitude_col: str = "latitude"
    longitude_col: str = "longitude"
    weather_dir: str = "src/weathermertricsdata/data"
    n_miles: float = 0.0
    direction: Optional[str] = None
    max_stations: Optional[int] = 10
    extra_station_ids: List[str] = field(default_factory=list)
    chunk_size: int = 200_000
    engine: str = "polars"  # supported: polars, pandas, cudf
    n_workers: int = 4
    time_args: Dict[str, str] = field(default_factory=lambda: {"date_col":"date","time_col":"hour","datetime_col":"datetime"})    
    agg_map: Optional[Dict[str, List[str]]] = None
    period_freq: str = "1d"
    lookback_days: int = 7
    bins: List[float] = field(default_factory=lambda: [-9999, 1, 5, 20, 99999])
    labels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    years_back: Optional[float] = None
    indep_vars: Optional[List[str]] = None
    calculate_aux_features: bool = False
    aux_resample_apply_to_primary: bool = False
    aux_agg_map: Optional[Dict[str, List[str]]] = None
    aux_rolling_windows: Optional[List[int]] = None
    aux_output_mode: str = 'both'
    drop_date_stationid: bool = True
    output_dir: str = "src/weathermetricsdata/output"
    output_prefix: str = "pipeline"
    output_formats: List[str] = field(default_factory=lambda: ["csv", "parquet"])



def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_MILES * c


def _direction_filter(df: pd.DataFrame, lat: float, lon: float, direction: Optional[str]) -> pd.DataFrame:
    if not direction:
        return df
    direction = direction.upper().strip()
    if direction not in DIRECTION_DELTAS:
        logger.warning("Unknown direction %s. Ignoring direction filter.", direction)
        return df
    lat_sign, lon_sign = DIRECTION_DELTAS[direction]
    cond = pd.Series([True] * len(df))
    if lat_sign > 0:
        cond &= df["latitude"] >= lat
    elif lat_sign < 0:
        cond &= df["latitude"] <= lat
    if lon_sign > 0:
        cond &= df["longitude"] >= lon
    elif lon_sign < 0:
        cond &= df["longitude"] <= lon
    return df[cond]


def load_stations(stations_csv: str, colmap: Dict[str, str] = None, engine: str = "polars") -> pd.DataFrame:
    colmap = colmap or {}
    default_map = {"stid": "stid", "latitude": "latitude", "longitude": "longitude"}
    merged = {**default_map, **colmap}

    if engine == "cudf" and cudf is not None:
        stations = cudf.read_csv(stations_csv)
        stations = stations.rename(columns={merged["stid"]: "stid", merged["latitude"]: "latitude", merged["longitude"]: "longitude"})
        stations = stations.to_pandas()
    elif engine == "polars" and pl is not None:
        stations = pl.read_csv(stations_csv, infer_schema_length=100000).to_pandas()
    else:
        stations = pd.read_csv(stations_csv)

    required_cols = {"stid": merged["stid"], "latitude": merged["latitude"], "longitude": merged["longitude"]}

    # Handle missing required columns gracefully and keep all other columns
    for std_name, raw_name in required_cols.items():
        if raw_name not in stations.columns:
            logger.warning("Station metadata column '%s' not found in %s; imputing with null for all rows", raw_name, stations_csv)
            stations[raw_name] = pd.NA

    # Rename required columns, leaving extras untouched
    rename_map = {raw_name: std_name for std_name, raw_name in required_cols.items()}
    stations = stations.rename(columns=rename_map)

    # Create extra_data column with all non-required metadata
    std_required = ["stid", "latitude", "longitude"]
    extra_columns = [c for c in stations.columns if c not in std_required]
    if extra_columns:
        stations["extra_data"] = stations[extra_columns].apply(lambda row: {col: row[col] for col in extra_columns if pd.notna(row[col])}, axis=1)

        # Warn per station per missing extra column values
        if "stid" in stations.columns:
            station_id_col = "stid"
        else:
            station_id_col = None

        for col in extra_columns:
            missing_rows = stations[stations[col].isna()]
            for idx, row in missing_rows.iterrows():
                sid = row[station_id_col] if station_id_col in stations.columns else f"index {idx}"
                logger.warning("Station %s is missing value for extra column '%s'; setting null", sid, col)

    else:
        stations["extra_data"] = [{} for _ in range(len(stations))]

    # Final station list with required and extra_data columns
    stations = stations[["stid", "latitude", "longitude", "extra_data"]].drop_duplicates(subset=["stid"])
    return stations


def find_nearest(
    stations: pd.DataFrame,
    station_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    n_miles: float = 0.0,
    max_stations: Optional[int] = None,
    direction: Optional[str] = None,
) -> pd.DataFrame:
    if station_id is not None:
        logging.info("Finding nearest stations to station_id=%s with n_miles=%s, direction=%s, limited to %s stations", station_id, n_miles, direction, max_stations)
        target = stations[stations["stid"].astype(str) == str(station_id)]
        if target.empty:
            logger.warning("Station id %s not found in station metadata.", station_id)
            return stations.iloc[0:0]
        target = target.iloc[0]
        latitude, longitude = float(target["latitude"]), float(target["longitude"])
        if n_miles == 0:            
            logging.info("Returning target station: %s", target['stid'])             
            return target.to_frame().T
    if latitude is None or longitude is None:        
        logging.info("Finding nearest stations to latitude: %s, longitude: %s, n_miles: %s, limited to %s, stations", station_id, latitude, longitude, n_miles, max_stations)
        raise ValueError("latitude/longitude must be provided if station_id not found or n_miles > 0")

    filtered = stations.copy()
    filtered["distance_miles"] = filtered.apply(lambda row: haversine_miles(latitude, longitude, float(row.latitude), float(row.longitude)), axis=1)
    if n_miles > 0:
        filtered = filtered[filtered["distance_miles"] <= n_miles]
    if direction:
        filtered = _direction_filter(filtered, latitude, longitude, direction)

    filtered = filtered.sort_values("distance_miles")
    if max_stations is not None:
        filtered = filtered.head(max_stations)
    logging.info("Found %d stations matching criteria", len(filtered))                

    return filtered


def list_weather_files(data_dir: str, station_ids: List[str], extra_station_ids: List[str] = None) -> List[str]:
    extra_station_ids = extra_station_ids or []
    allowed_ids = set(str(x) for x in station_ids) | set(str(x) for x in extra_station_ids)
    accepted = []
    pattern = re.compile(r"^(\d+)_.*\.csv$")

    for root, _, files in os.walk(data_dir):
        for fname in files:
            m = pattern.match(fname)
            if m:
                sid = m.group(1)
                if sid in allowed_ids:
                    accepted.append(os.path.join(root, fname))

    # if there are extra station ids with no pattern file, log and ignore
    missing = [sid for sid in allowed_ids if not any(os.path.basename(p).startswith(str(sid) + "_") for p in accepted)]
    if missing:
        logger.warning("No weather CSV found for station IDs: %s", missing)

    logger.info("Weather files matching station IDs: %s", [os.path.basename(p) for p in accepted] )
    return accepted

def _ensure_datetime_col(
    df: Any,
    date_col: str = "date",
    time_col: str = "hour",
    datetime_col: str = "datetime",
) -> Any:
    if pl is not None and isinstance(df, pl.DataFrame):
        if datetime_col in df.columns:
            if df.schema[datetime_col].is_temporal():
                return df.with_columns(pl.col(datetime_col).cast(pl.Datetime))
            return df.with_columns(pl.col(datetime_col).str.strptime(pl.Datetime))
        if date_col in df.columns and time_col in df.columns:
            if df.schema[date_col].is_temporal():
                d = pl.col(date_col).cast(pl.Datetime)
            else:
                d = pl.col(date_col).str.strptime(pl.Date, format=None).cast(pl.Datetime)
            h = pl.col(time_col).cast(pl.Int64)
            return df.with_columns((d + pl.duration(hours=h)).alias(datetime_col))
        raise ValueError("No datetime source column found in polars DataFrame")
    # cudf/pandas path
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    elif date_col in df.columns and time_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[time_col].astype(float), unit="h")
    else:
        raise ValueError("No date/time source found")
    return df


def _parse_datetime_from_df(df: Any, engine: str) -> Any:
    if engine == "cudf" and cudf is not None:
        if "date" in df.columns and "hour" in df.columns:
            df["datetime"] = cudf.to_datetime(df["date"] + " " + df["hour"].astype(str) + ":00:00")
        elif "datetime" in df.columns:
            df["datetime"] = cudf.to_datetime(df["datetime"])
        return df

    if engine == "polars" and pl is not None and isinstance(df, pl.DataFrame):
        if "date" in df.columns and "hour" in df.columns:
            # sometimes hour is int or string
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").alias("__d")
            )
            df = df.with_columns(pl.col("hour").cast(pl.Int64).alias("__h"))
            df = df.with_columns(
                (pl.col("__d").cast(pl.Datetime) + pl.duration(hours=pl.col("__h"))).alias("datetime")
            )
            df = df.drop(["__d", "__h"])
        elif "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.strptime(pl.Datetime, fmt=None))
        return df

    # fallback pandas
    if "date" in df.columns and "hour" in df.columns:
        # try numeric or string hour
        df["datetime"] = pd.to_datetime(df["date"].astype(str), errors="coerce") + pd.to_timedelta(df["hour"].astype(float), unit="h")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def read_weather_csv(file_path: str, station_id: str, chunk_size: int = 200_000, engine: str = "polars") -> Any:
    if engine == "cudf" and cudf is not None:
        df = cudf.read_csv(file_path)
        df = _parse_datetime_from_df(df, engine)
        df["stationid"] = station_id
        return df

    if engine == "polars" and pl is not None:
        try:
            df = pl.read_csv(file_path, low_memory=True)
            df = _parse_datetime_from_df(df, engine)
            df = df.with_columns(pl.lit(station_id).alias("stationid"))
            return df
        except Exception as e:
            logger.warning("Polars failed for %s with %s, falling back to pandas", file_path, e)

    # fallback pandas
    usecols = None
    try:
        if chunk_size is not None and chunk_size > 0:
            dfs = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk = _parse_datetime_from_df(chunk, "pandas")
                chunk["stationid"] = station_id
                dfs.append(chunk)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
            df = _parse_datetime_from_df(df, "pandas")
            df["stationid"] = station_id
        return df
    except Exception as e:
        logger.exception("Failed reading weather file %s: %s", file_path, e)
        raise


def load_weather_files(
    station_ids: List[str],
    data_dir: str,
    chunk_size: int = 200_000,
    engine: str = "polars",
    extra_station_ids: List[str] = None,
    n_workers: int = 4,
) -> Any:
    extra_station_ids = extra_station_ids or []
    logger.info("Loading weather files for station IDs: %s with extra_station_ids: %s using engine: %s", station_ids, extra_station_ids, engine)
    files = list_weather_files(data_dir, station_ids, extra_station_ids)
    logger.info("Found %d weather files", len(files))
    logger.info("Weather files: %s", [os.path.basename(f) for f in files])
    if not files:
        logger.warning("No weather files discovered for station set %s", station_ids)
        if engine == "polars" and pl is not None:
            return pl.DataFrame()
        elif engine == "cudf" and cudf is not None:
            return cudf.DataFrame()
        else:
            return pd.DataFrame()

    def _read(path: str) -> Any:
        m = re.match(r"^(\d+)_", os.path.basename(path))
        stationid = m.group(1) if m else "unknown"
        return read_weather_csv(path, stationid, chunk_size, engine)

    records = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_path = {executor.submit(_read, p): p for p in files}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                df = future.result()
                records.append(df)
                logger.info("Loaded %s rows from %s", len(df), p)
            except Exception:
                logger.exception("Failed to read weather file %s", p)

    if engine == "polars" and pl is not None:
        return pl.concat(records, how="vertical") if records else pl.DataFrame()
    elif engine == "cudf" and cudf is not None:
        return cudf.concat(records, axis=0) if records else cudf.DataFrame()
    else:
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

def resample_summary(
    df: Any,
    period: str = "1h",
    agg: List[str] = None,
    agg_vars: Optional[List[str]] = None,
    agg_map: Optional[Dict[str, List[str]]] = None,
    date_col: str = "date",
    time_col: str = "hour",
    datetime_col: str = "datetime",
    group_col: Optional[str] = "stationid",
) -> Any:
    agg = agg or ["sum", "min", "max", "median", "mean", "std"]

    if df is None or len(df) == 0:
        logger.warning("resample_summary called with empty dataset")
        return df
    if group_col is None:
        group_col = "stationid"

    df = _ensure_datetime_col(df, date_col=date_col, time_col=time_col, datetime_col=datetime_col)
    # logger.info("Datetime Normalized Schema: %s", df.schema)
    # logger.info("Datetime Normalized Head: %s", df.head())

    if pl is not None and isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.col(datetime_col).cast(pl.Datetime))

        if agg_map is None:
            if agg_vars is None:
                agg_vars = [c for c in df.columns if c not in [group_col, datetime_col]]
            agg_map = {c: agg for c in agg_vars}

        exprs = []
        for col, ops in agg_map.items():
            if col in [group_col, datetime_col]:
                continue
            for op in ops:
                if not hasattr(pl, op):
                    raise ValueError(f"Unknown polars aggregation '{op}'")
                exprs.append(getattr(pl, op)(col).alias(f"{col}_{op}"))

        logger.info("Polars resample with agg_map %s", agg_map)
        
        # return df.sort(datetime_col).group_by(datetime_col, every=period, group_by=group_col).agg(exprs)
        return df.sort(datetime_col).group_by_dynamic(datetime_col, every=period, by=group_col).agg(exprs)

    # pandas/cudf fallback
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' is missing")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' is missing")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.set_index(datetime_col)

    if agg_map is None:
        if agg_vars is None:
            agg_vars = [c for c in df.columns if c not in [group_col]]
        agg_map = {c: agg for c in agg_vars}

    logger.info("Success: Dataset resampled with period %s and the mapping %s", period, agg_map)
    result = df.groupby(group_col).resample(period).agg(agg_map).reset_index()

    # Flatten MultiIndex columns if they exist (common in Pandas resample/agg)
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in result.columns]

    return result

def resample_auxiliary_summary(
    df: Any,
    aux_agg_map: Dict[str, List[str]],
    aux_rolling_windows: Optional[List[int]] = None,
    target_stations: Optional[List[str]] = None,
    datetime_col: str = "datetime",
    group_col: str = "stationid",
) -> Any:
    """
    Resample auxiliary station data with historical-only aggregates to prevent data leakage.
    Rolling windows and aggregates exclude the current period (uses shift(1)).
    Returns consolidated DataFrame with aux_* columns (single value per datetime across all aux stations).
    """
    logger.info("Computing auxiliary resampling for stations: %s", target_stations)

    if len(target_stations) > 1:
        var_prefix = "multistations"
    else:
        var_prefix = f"{target_stations[0]}" if target_stations else "aux"    

    if df is None or len(df) == 0:
        logger.warning("resample_auxiliary_summary called with empty dataset")
        if pl is not None:
            return pl.DataFrame()
        elif cudf is not None:
            return cudf.DataFrame()
        else:
            return pd.DataFrame()
    
    # Convert to Polars if possible
    if pl is not None:
        df_pl = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        if hasattr(df_pl, 'to_pandas'):
            df_pl = pl.from_pandas(df_pl.to_pandas())
    elif cudf is not None and isinstance(df, cudf.DataFrame):
        df_pl = df
    else:
        df_pl = df
    
    if datetime_col not in df_pl.columns or group_col not in df_pl.columns:
        raise ValueError(f"Missing {datetime_col} or {group_col} columns")
    
    df_pl = df_pl.with_columns(pl.col(datetime_col).cast(pl.Datetime))
    
    # Filter to target stations if provided
    if target_stations:
        target_stations_str = [str(s) for s in target_stations]
        df_pl = df_pl.filter(pl.col(group_col).cast(pl.Utf8).is_in(target_stations_str))
    
    if len(df_pl) == 0:
        logger.warning("No data for target stations after filtering")
        if pl is not None:
            return pl.DataFrame()
        elif cudf is not None:
            return cudf.DataFrame()
        else:
            return pd.DataFrame()
    
    # Sort
    df_pl = df_pl.sort([group_col, datetime_col])
    
    result_dfs = []
    
    for var, methods in aux_agg_map.items():
        if var not in df_pl.columns:
            logger.warning(f"Variable {var} not found in auxiliary data, skipping")
            continue
        
        for method in methods:
            if aux_rolling_windows:
                for window_days in aux_rolling_windows:
                    rolling_col_name = f"{var_prefix}_{var}_{method}_{window_days}d"
                    
                    # Compute rolling per station using Polars rolling
                    if method == 'mean':
                        rolling_expr = pl.col(var).shift(1).rolling_mean(window_size=window_days)
                    elif method == 'min':
                        rolling_expr = pl.col(var).shift(1).rolling_min(window_size=window_days)
                    elif method == 'max':
                        rolling_expr = pl.col(var).shift(1).rolling_max(window_size=window_days)
                    elif method == 'sum':
                        rolling_expr = pl.col(var).shift(1).rolling_sum(window_size=window_days)
                    else:
                        # For other methods, fallback to pandas
                        logger.warning(f"Method {method} not directly supported in Polars rolling, falling back to pandas")
                        df_pd = df_pl.to_pandas()
                        def compute_rolling(group_df):
                            shifted = group_df[var].shift(1)
                            rolling = shifted.rolling(window=window_days, closed='left').agg(method)
                            return pd.DataFrame({datetime_col: group_df[datetime_col].values, 'rolling_val': rolling.values})
                        rolling_by_station = df_pd.groupby(group_col).apply(compute_rolling, include_groups=True).reset_index(level=0, drop=True)
                        rolling_consolidated = rolling_by_station.groupby(datetime_col)['rolling_val'].agg(method).reset_index()
                        rolling_consolidated.columns = [datetime_col, rolling_col_name]
                        result_dfs.append(pl.from_pandas(rolling_consolidated) if pl is not None else rolling_consolidated)
                        continue
                    
                    df_with_rolling = df_pl.with_columns(rolling_expr.over(group_col).alias("rolling_val"))
                    rolling_by_station = df_with_rolling.select([datetime_col, "rolling_val"])
                    
                    # Consolidate across stations
                    if method == 'mean':
                        consolidated = rolling_by_station.group_by(datetime_col).agg(pl.col("rolling_val").mean().alias(rolling_col_name))
                    elif method == 'min':
                        consolidated = rolling_by_station.group_by(datetime_col).agg(pl.col("rolling_val").min().alias(rolling_col_name))
                    elif method == 'max':
                        consolidated = rolling_by_station.group_by(datetime_col).agg(pl.col("rolling_val").max().alias(rolling_col_name))
                    elif method == 'sum':
                        consolidated = rolling_by_station.group_by(datetime_col).agg(pl.col("rolling_val").sum().alias(rolling_col_name))
                    else:
                        consolidated = rolling_by_station.group_by(datetime_col).agg(pl.col("rolling_val").agg(method).alias(rolling_col_name))
                    
                    result_dfs.append(consolidated)
    
    if not result_dfs:
        logger.warning("No auxiliary aggregates computed")
        if pl is not None:
            return pl.DataFrame()
        elif cudf is not None:
            return cudf.DataFrame()
        else:
            return pd.DataFrame()
    
    # Join all results on datetime
    result = result_dfs[0]
    for rdf in result_dfs[1:]:
        result = result.join(rdf, on=datetime_col, how='full', coalesce=True)
    
    result = result.sort(datetime_col)
    logger.info("Auxiliary resampling complete with %d columns", len(result.columns))
    return result

def create_dep_var(
    df: Any,
    target_var: str,
    lookback_days: int = 7,
    freq: str = "1h",
    date_col: str = "datetime",
    station_col: str = "stationid",
    indep_vars: Optional[List[str]] = None,
    include_auxiliary: bool = False,
    aux_stations: Optional[List[str]] = None,
    aux_dfs: Optional[List[pl.DataFrame]] = None,  # Updated: List of optional Polars DataFrames for future multiple aux datasets
    aux_output_mode: str = 'primary',  # Updated: Default to 'primary' for backward compatibility
    years_back: Optional[float] = None,
    drop_date_stationid: bool = True,
    bins: Optional[List[float]] = None,
    labels: Optional[List[int]] = None,
) -> Any:  # Updated: Returns Polars or Pandas DataFrame
    """
    Create dependent variable dataset with lagged features for modeling.
    
    Args:
        df: Input DataFrame (Polars, Pandas, or CuDF).
        target_var: Name of the target variable column.
        lookback_days: Number of days for lag features.
        freq: Frequency for resampling (e.g., '1h').
        date_col: Name of the datetime column.
        station_col: Name of the station ID column.
        indep_vars: List of independent variable columns to lag.
        include_auxiliary: Whether to include auxiliary station data.
        aux_stations: List of auxiliary station IDs.
        aux_dfs: Optional list of Polars DataFrames with datetime, stationid, and pre-computed rolling aggregates.
                 Each DataFrame's aggregates are lagged and merged based on aux_output_mode.
        aux_output_mode: 'both' (primary + lagged aux + y), 'primary' (only primary lags + y), 
                         'aux' (only lagged aux + y, drops primary lags).
        years_back: Restrict to last N years.
        drop_date_stationid: Whether to drop date and station columns in output.
        bins: Bins for target variable discretization.
        labels: Labels for binned target variable.
    
    Returns:
        DataFrame with y and lagged X variables, based on aux_output_mode.
    """
    if df is None or len(df) == 0:
        logger.warning("create_dep_var called with empty dataset")
        return pl.DataFrame() if pl is not None else pd.DataFrame()

    # Convert to Pandas for initial processing (existing logic)
    if pl is not None and isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    elif cudf is not None and isinstance(df, cudf.DataFrame):
        df = df.to_pandas()
    else:
        df = df.copy()

    # Validate required columns
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' missing")
    if station_col not in df.columns:
        raise ValueError(f"Station column '{station_col}' missing")
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' missing")
    
    indep_vars = indep_vars or []
    for var in indep_vars:
        if var not in df.columns:
            raise ValueError(f"Independent variable '{var}' missing")

    # Handle aux_dfs: Validate and combine multiple aux DataFrames
    combined_aux_df = None
    if aux_dfs:
        valid_aux_dfs = []
        for aux_df in aux_dfs:
            if not isinstance(aux_df, pl.DataFrame):
                if hasattr(aux_df, 'to_pandas'):
                    aux_df = pl.from_pandas(aux_df.to_pandas())
                else:
                    aux_df = pl.from_pandas(aux_df)
            if date_col not in aux_df.columns or station_col not in aux_df.columns:
                logger.warning("An aux_df missing required columns; skipping")
                continue
            valid_aux_dfs.append(aux_df)
        
        if valid_aux_dfs:
            # Combine all valid aux_dfs by joining on datetime (outer join to include all dates)
            combined_aux_df = valid_aux_dfs[0]
            for aux_df in valid_aux_dfs[1:]:
                combined_aux_df = combined_aux_df.join(aux_df, on=date_col, how='outer', suffix='_aux')
            # Drop duplicate columns if any (e.g., stationid_aux)
            columns_to_drop = [c for c in combined_aux_df.columns if c.endswith('_aux') and c != f"{station_col}_aux"]
            if columns_to_drop:
                combined_aux_df = combined_aux_df.drop(columns_to_drop)
            logger.info("Combined %d aux_dfs into one DataFrame", len(valid_aux_dfs))

    # Existing preprocessing
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = None
    if years_back is not None:
        if years_back <= 0:
            raise ValueError("years_back must be positive")
        cutoff = df[date_col].max() - pd.DateOffset(months=int(years_back * 12))
        df = df[df[date_col] >= cutoff]
        logger.info("Primary Dataset restricted to last %.2f years, cutoff >=%s", years_back, cutoff)
    df = df.sort_values([station_col, date_col])

    # Restrict combined_aux_df to the same timeframe
    if combined_aux_df is not None and cutoff is not None:
        combined_aux_df = combined_aux_df.filter(pl.col(date_col) >= pl.lit(cutoff))
        logger.info("Restricted Auxillary Dataset to cutoff >=%s", cutoff)

    freq_td = pd.Timedelta(freq)
    total_lags = int((lookback_days * 24 * 3600) / freq_td.total_seconds())
    if total_lags < 1:
        total_lags = 1

    # Generate primary lagged DataFrame (existing logic)
    station_frames = {}
    for station, g in df.groupby(station_col):
        numeric_data = g.set_index(date_col).asfreq(freq).infer_objects().select_dtypes(exclude=['object', 'str'])
        string_data = g.set_index(date_col).asfreq(freq).select_dtypes(include=['object', 'str'])

        g2 = (pd.concat([
            numeric_data.interpolate(method='time').ffill().bfill(),
            string_data.ffill().bfill()
        ], axis=1))
        # g2 = (g.set_index(date_col).asfreq(freq).infer_objects(copy=False).interpolate(method='time').ffill().bfill())
        
        lag_cols = []
        # Lags for target_var
        for lag in range(1, total_lags + 1):
            col_name = f"{station}_{target_var}_lag_{lag}"
            g2[col_name] = g2[target_var].shift(lag)
            lag_cols.append(col_name)
        
        # Lags for indep_vars
        for var in indep_vars:
            for lag in range(1, total_lags + 1):
                col_name = f"{station}_{var}_lag_{lag}"
                g2[col_name] = g2[var].shift(lag)
                lag_cols.append(col_name)
        
        g2["y"] = g2[target_var]
        base_cols = [station_col, date_col, "y"] + lag_cols
        g2 = g2.reset_index()[base_cols]
        g2 = g2.iloc[total_lags:]  # Drop rows with NaN lags
        g2[station_col] = station
        station_frames[station] = g2

    # Handle auxiliary stations (existing logic, but adapted)
    if include_auxiliary and aux_stations:
        aux_stations = [str(s) for s in aux_stations]
        # Identify primary stations as those NOT designated as auxiliary
        primary_stations = [s for s in station_frames.keys() if str(s) not in aux_stations]
        if not primary_stations:
            primary_stations = list(station_frames.keys())
            logger.warning("No primary stations found after excluding aux_stations %s; using all stations", aux_stations)

        # Start the result with primary stations; their 'y' column is the prediction target
        result = pd.concat([station_frames[s] for s in primary_stations], ignore_index=True) if primary_stations else pd.DataFrame(columns=[station_col, date_col, "y"])

        for aux in aux_stations:
            if aux not in station_frames:
                logger.warning("Auxiliary station %s not found in data", aux)
                continue
            aux_g = station_frames[aux].copy()
            # For auxiliary stations, we only want their lagged features as independent variables.
            # We explicitly exclude their 'y' (target) and 'stationid' columns to avoid conflict.
            merge_cols = [date_col] + [c for c in aux_g.columns if c not in [station_col, date_col, "y"]]
            aux_g = aux_g[merge_cols]
            result = result.merge(aux_g, on=date_col, how="left")
    else:
        result = pd.concat(station_frames.values(), ignore_index=True) if station_frames else pd.DataFrame(columns=[station_col, date_col, "y"])

    # Handle combined_aux_df based on aux_output_mode
    if combined_aux_df is not None and len(combined_aux_df) > 0:
        logger.info("Processing combined_aux_df with mode: %s", aux_output_mode)
        # Convert result to Polars for merging
        result_pl = pl.from_pandas(result) if not isinstance(result, pl.DataFrame) else result
        combined_aux_df = combined_aux_df.sort(date_col)  # Ensure sorted

        # Apply lags to combined_aux_df aggregated columns (exclude datetime, stationid)
        aux_lagged_cols = []
        agg_cols = [c for c in combined_aux_df.columns if c not in [date_col, station_col]]
        for col in agg_cols:
            for lag in range(1, total_lags + 1):
                lagged_col = f"{col}_lag_{lag}"
                combined_aux_df = combined_aux_df.with_columns(pl.col(col).shift(lag).alias(lagged_col))
                aux_lagged_cols.append(lagged_col)

        # Merge on datetime (left join to preserve primary data)
        result_pl = result_pl.join(combined_aux_df.select([date_col] + aux_lagged_cols), on=date_col, how="left")
        logger.info("Merged combined_aux_df; result has %d columns", len(result_pl.columns))

        # Select columns based on aux_output_mode
        primary_lags = [c for c in result_pl.columns if c.endswith(tuple(f'_lag_{i}' for i in range(1, total_lags + 1))) and not any(c.startswith(agg) for agg in agg_cols)]  # Existing lags
        aux_lags = aux_lagged_cols
        base_cols = [date_col, station_col, "y"] # if not drop_date_stationid else ["y"]

        if aux_output_mode == 'primary':
            selected_cols = base_cols + primary_lags
        elif aux_output_mode == 'both':
            selected_cols = base_cols + primary_lags + aux_lags
        elif aux_output_mode == 'aux':
            selected_cols = base_cols + aux_lags
        else:
            logger.warning("Unknown aux_output_mode '%s'; defaulting to 'primary'", aux_output_mode)
            selected_cols = base_cols + primary_lags

        # logger.info("Selected columns for output based on aux_output_mode '%s': %s", aux_output_mode, selected_cols)
        result_pl = result_pl.select(selected_cols)
        result = result_pl.to_pandas()  # Convert back for consistency with existing binning
    else:
        # No aux_dfs: Default to 'primary' mode
        if aux_output_mode != 'primary':
            logger.info("aux_dfs not provided; defaulting to 'primary' mode")

    # Sort and bin (existing logic)
    # logger.info("Result class before binning: %s with columns: %s", type(result), result.columns.tolist())    
    result = result.sort_values(by=date_col, ascending=False).reset_index(drop=True)
    if bins is not None and labels is not None:
        result["y"] = pd.cut(result["y"], bins=bins, labels=labels, right=False)
        logger.info("Target variable binned with bins %s and labels %s", bins, labels)

    logger.info("Model Final Dataset created for Station ID: %s with Dependent variable: %s & Independent variables: %s with lookback of %d days", 
                result[station_col].unique().tolist() if station_col in result.columns else [], target_var, indep_vars, lookback_days)
    if drop_date_stationid:
        result = result.drop(columns=[date_col, station_col], errors='ignore')
        logger.info("Removing StationID and Date Column from the Final dataset")

    # Return as Polars if possible
    if pl is not None:
        # Polars requires unique string column names.
        # 1. Force all column names to strings
        result.columns = [str(c) for c in result.columns]
        # 2. De-duplicate column names by keeping only the first occurrence
        if not result.columns.is_unique:
            logger.warning("Non-unique columns detected in create_dep_var; deduplicating before Polars conversion.")
            result = result.loc[:, ~result.columns.duplicated()]

        result = pl.from_pandas(result)
        
    return result


def pipeline_run(config: PipelineConfig, station_id: Optional[str] = None, latitude: Optional[float] = None, longitude: Optional[float] = None, target_var: str = 'precip_depth_sum_mm_sum') -> Dict[str, Any]:
    stations = load_stations(config.stations_csv, {
        'stid': config.station_id_col,
        'latitude': config.latitude_col,
        'longitude': config.longitude_col,
    }, engine=config.engine)

    nearest = find_nearest(
        stations,
        station_id=station_id,
        latitude=latitude,
        longitude=longitude,
        n_miles=config.n_miles,
        max_stations=config.max_stations,
        direction=config.direction,
    )    
    
    if nearest.empty:
        logger.warning("No stations matched query. Returning empty result.")
        return {'stations': nearest, 'weather_df': None, 'resampled': None, 'dep_var': None}
    else:        
        logger.info("Station IDs matched: %s", nearest['stid'].tolist())

    matches = nearest['stid'].tolist()
    weather_df = load_weather_files(matches, 
                                    config.weather_dir, 
                                    chunk_size=config.chunk_size, 
                                    engine=config.engine, 
                                    extra_station_ids=config.extra_station_ids, 
                                    n_workers=config.n_workers)
    # logger.info("Weather df.schema: %s", weather_df.schema if weather_df is not None else "N/A")
    # logger.info("Weather df.head(): %s", weather_df.head() if weather_df is not None else "N/A")

    # adjust matches to only those that were actually loaded
    if isinstance(weather_df, pd.DataFrame):
        loaded_matches = weather_df['stationid'].astype(str).unique().tolist() if 'stationid' in weather_df.columns else []
    elif pl is not None and isinstance(weather_df, pl.DataFrame):
        loaded_matches = weather_df.select(pl.col('stationid').cast(pl.Utf8)).unique().to_series().to_list() if 'stationid' in weather_df.columns else []
    elif cudf is not None and isinstance(weather_df, cudf.DataFrame):
        loaded_matches = weather_df['stationid'].astype(str).unique().to_pandas().tolist() if 'stationid' in weather_df.columns else []
    else:
        loaded_matches = []

    if set(loaded_matches) != set(matches):
        missing = [s for s in matches if str(s) not in set(loaded_matches)]
        logger.warning("Dropping missing stations with no weather data csv: %s", missing)
        matches = loaded_matches

    time_args = config.time_args if hasattr(config, 'time_args') else {}
    agg_map = config.agg_map if hasattr(config, 'agg_map') else {target_var: ["sum"]}
    period_freq = config.period_freq if hasattr(config, 'period_freq') else "1d"
    lookback_days = config.lookback_days if hasattr(config, 'lookback_days') else 7
    years_back = config.years_back if hasattr(config, 'years_back') else None
    indep_vars = config.indep_vars if hasattr(config, 'indep_vars') else None
    drop_date_stationid = config.drop_date_stationid if hasattr(config, 'drop_date_stationid') else False
    bins = config.bins if hasattr(config, 'bins') else [-9999, 1, 5, 20, 99999]
    labels = config.labels if hasattr(config, 'labels') else [0, 1, 2, 3]
    aux_stations = list(set([s for s in matches if s != station_id] + [sid for sid in config.extra_station_ids if sid != station_id]))
    include_auxiliary = aux_stations is not None and len(aux_stations) > 0 # bool(aux_stations)
    # logging.info("aux stations type: %s, value: %s", type(aux_stations), aux_stations)
    logging.info("Auxiliary stations for dependent variable creation: %s & include auxiliary: %s", aux_stations, include_auxiliary)

    resampled = resample_summary(
                                weather_df,
                                period=period_freq,
                                agg_map=agg_map,
                                group_col="stationid",
                                **time_args
                            )
    # logger.info("Resampled df.schema: %s", resampled.schema if resampled is not None else "N/A")
    # logger.info("Resampled Data Sample: %s", resampled.head() if resampled is not None else "N/A")

    # Process auxiliary resampling if configured
    if hasattr(config, 'calculate_aux_features') and config.calculate_aux_features and hasattr(config, 'aux_agg_map') and config.aux_agg_map:  
        # Compute auxiliary aggregates for all neighboring stations together, then merge back to main resampled df
        
        neighborstation_aux_resampled = resample_auxiliary_summary(
            resampled,
            aux_agg_map=config.aux_agg_map,
            aux_rolling_windows=config.aux_rolling_windows if hasattr(config, 'aux_rolling_windows') else None,
            target_stations=aux_stations,
            datetime_col='datetime',
            group_col='stationid'
        )        

        if config.aux_resample_apply_to_primary:
            # Compute auxiliary aggregates for the target station then merge back to main resampled df
            targetstation_aux_resampled = resample_auxiliary_summary(
                resampled,
                aux_agg_map=config.aux_agg_map,
                aux_rolling_windows=config.aux_rolling_windows if hasattr(config, 'aux_rolling_windows') else None,
                target_stations=[station_id],
                datetime_col='datetime',
                group_col='stationid'
            )
        
            # Combine them together
            aux_resampled = targetstation_aux_resampled.join(neighborstation_aux_resampled, on='datetime', how='left', suffix="_target")
        else:
            aux_resampled = neighborstation_aux_resampled
        
        aux_resampled = aux_resampled.with_columns(pl.lit(station_id).alias("stationid"))  # add stationid back for merging with main df later if needed

        # logger.info("Resampled Auxiliary schema: %s", aux_resampled.columns if aux_resampled is not None else "N/A")
        # logger.info("Resampled Auxiliary Sample: %s", aux_resampled.head() if aux_resampled is not None else "N/A")
        logger.info("Auxiliary resampling merged; final shape: %s", aux_resampled.shape if hasattr(aux_resampled, 'shape') else len(aux_resampled))

        # Merge based on output mode
        output_mode = config.aux_output_mode if hasattr(config, 'aux_output_mode') else 'both'
        logger.info("Auxiliary output mode: %s", output_mode)       

    dep_var = create_dep_var(resampled, 
                             target_var=target_var, 
                             lookback_days=lookback_days, 
                             freq= period_freq , 
                             years_back= years_back,
                             indep_vars = indep_vars,
                             include_auxiliary = include_auxiliary,
                             aux_stations = aux_stations,
                             date_col='datetime', 
                             station_col='stationid',
                             aux_dfs= [aux_resampled] if hasattr(config, 'calculate_aux_features') and config.calculate_aux_features and hasattr(config, 'aux_agg_map') and config.aux_agg_map else None,
                             aux_output_mode=config.aux_output_mode if hasattr(config, 'aux_output_mode') else 'both',                             
                             drop_date_stationid = drop_date_stationid,
                             bins = bins,
                             labels = labels)    
    # logger.info("Dependent variable df.schema: %s", dep_var.schema if dep_var is not None else "N/A")
    # logger.info("Dependent variable Sample: %s", dep_var.head() if dep_var is not None else "N/A")

    return(dep_var)


def load_config(path: str) -> PipelineConfig:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return PipelineConfig(**payload)


if __name__ == '__main__':
    config = PipelineConfig()
    logger.info("Pipeline ready. Use pipeline_run(config, station_id=..., latitude=..., longitude=...)")