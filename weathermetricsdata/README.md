# Weather Metrics Data Cleaning Pipeline

High-performance Polars-driven pipeline for station-based weather time-series cleaning and model-ready independent/dependent feature framing.

## Features
- Station metadata loader with column overrides
- spatial query by station ID or latitude/longitude
- radius + directional filters + max station count
- efficient weather file discovery by `^<stationid>_.*\.csv$`
- chunked / parallel read + datetime parsing
- resample aggregation (hour, daily, weekly, monthly, etc.) with custom aggs
- dependent-independent variable window framing (lookback)
- Historical-only auxiliary feature calculation with data leakage prevention (shift-rolling logic)
- optional GPU mode via `cudf` / `cupy`

## Dependencies
- Python 3.10+
- pandas
- polars (recommended for speed)
- numba
- pyarrow
- cudf (optional for GPU mode)
- uv (package manager, optional)
- pytest (for unit tests)

### Install
Using pip:

```bash
python -m pip install pandas polars numba
# Optional GPU:
# python -m pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
```

or using uv (as requested):

```bash
uv install pandas polars numba
uv install cudf  # optional
uv install astral
```

## Quickstart

1. Put station metadata in `data/stations.csv` with columns `stid,latitude,longitude`.
2. Put weather files in `data/` as `72206013889_model_data_combined.csv` etc.
3. Adjust `pipeline_config.json` for your paths and params.

Run:

```bash
python run_pipeline.py --config pipeline_config.json --station-id 72206013889
```

or query by coordinates:

```bash
python run_pipeline.py --config pipeline_config.json --lat 40.0 --lon -105.0
```

## Code usage

```python
from pipeline import PipelineConfig, pipeline_run

config = PipelineConfig(
    stations_csv='data/stations.csv',
    weather_dir='data',
    n_miles=50,
    max_stations=5,
    engine='polars',
)

output = pipeline_run(config, station_id='72206013889')
print(output['weather_df'].head())
print(output['dep_var'].head())
```

## Performance tips
- Warm OS file caches by sequential pre-read of files.
- Set `engine='polars'` and keep column datatypes explicit.
- Ensure each CSV is pre-sorted by datetime.
- Keep `chunk_size` tuned to memory constraints.
- Use `n_workers=4` or 8 on multi-core machines.

## Testing

```bash
python -m pytest tests/test_pipeline.py
```
