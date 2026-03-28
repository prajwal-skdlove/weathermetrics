import argparse
import logging
from pathlib import Path
from turtle import write
from .pipeline import pipeline_run, PipelineConfig, load_config
import pandas as pd
import polars as pl
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def format_runtime(seconds):
    """Convert seconds to appropriate time unit."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    elif seconds < 604800:
        return f"{seconds / 86400:.2f} days"
    else:
        return f"{seconds / 604800:.2f} weeks"

def main():    
    start_time = time.perf_counter()
    logger.info("Pipeline execution started.")
    parser = argparse.ArgumentParser(description='Weather data cleaning pipeline runner')
    parser.add_argument('--config', type=str, default='pipeline_config.json', help='Path to config JSON')
    parser.add_argument('--station-id', type=str, default=None, help='Optional station id to query')
    parser.add_argument('--lat', type=float, default=None, help='Optional lat coordinate for query')
    parser.add_argument('--lon', type=float, default=None, help='Optional lon coordinate for query')
    parser.add_argument('--target-var', type=str, default='precip_depth_sum_mm_sum', help='Target variable name for dependent variable framing')
    args = parser.parse_args()

    logger.info("Starting pipeline with config: %s", args.config)
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_config(args.config)
    logger.info("Config loaded: %s", config)

    result = pipeline_run(config, station_id=args.station_id, latitude=args.lat, longitude=args.lon, target_var=args.target_var)            
    # resampled.to_csv(f'{config.output_dir}/{args.station_id}_resampled_{config.output_prefix}.csv')
    # resampled.write_csv(f'{config.output_dir}/{args.station_id}_resampled_{config.output_prefix}.csv')
    # aux_resampled.write_csv(f'{config.output_dir}/{args.station_id}_aux_resampled_{config.output_prefix}.csv')    
    # result.write_csv(f'{config.output_dir}/{args.station_id}_{config.output_prefix}.csv')
    result.write_parquet(f'{config.output_dir}/{args.station_id}_{config.output_prefix}.parquet')
    # result.to_parquet(f'{config.output_dir}/{args.station_id}_{config.output_prefix}.parquet')
    logger.info(f"Parquet file output at {config.output_dir}/{args.station_id}_{config.output_prefix}.parquet with shape {result.shape}") #and columns {result.columns}")
    logger.info(f"Pipeline run for {args.station_id} completed with {args.target_var} as target variable.")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info("Pipeline execution completed.")
    logger.info(f"Total pipeline execution time: {format_runtime(elapsed_time)}")               


if __name__ == '__main__':
    main()
