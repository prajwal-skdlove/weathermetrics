import os
import tempfile
import logging
import pandas as pd
import polars as pl
import pytest
from weathermetricsdata.pipeline import (
    load_stations, find_nearest, list_weather_files, load_weather_files, 
    create_dep_var, PipelineConfig, pipeline_run, resample_auxiliary_summary,
    haversine_miles, _direction_filter, resample_summary
)

logger = logging.getLogger(__name__)

def test_haversine_miles():
    # Basic distance check: NYC to London is approx 3470 miles
    nyc = (40.7128, -74.0060)
    london = (51.5074, -0.1278)
    dist = haversine_miles(nyc[0], nyc[1], london[0], london[1])
    assert 3400 < dist < 3500

def test_direction_filter():
    df = pd.DataFrame({
        'latitude': [10, -10, 10, -10],
        'longitude': [10, 10, -10, -10]
    })
    # Test North-East (lat > 0, lon > 0)
    filtered = _direction_filter(df, 0, 0, "NE")
    assert len(filtered) == 1
    assert filtered.iloc[0]['latitude'] == 10 and filtered.iloc[0]['longitude'] == 10
    # Test South (lat < 0)
    filtered_s = _direction_filter(df, 0, 0, "S")
    assert len(filtered_s) == 2
    assert (filtered_s['latitude'] <= 0).all()


def test_find_nearest_no_station():
    with tempfile.TemporaryDirectory() as tmp:
        stations = pd.DataFrame({
            'stid': ['1', '2'],
            'latitude': [10.0, 20.0],
            'longitude': [10.0, 20.0],
        })
        file = os.path.join(tmp, 'stations.csv')
        stations.to_csv(file, index=False)
        s = load_stations(file, engine='pandas')
        result = find_nearest(s, station_id='999', n_miles=0)
        assert result.empty

def test_load_stations_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        stations_df = pd.DataFrame({
            'ID': ['S1'],
            'LAT': [45.0],
            'LON': [-90.0],
            'elev': [100]
        })
        file = os.path.join(tmp, 'stations.csv')
        stations_df.to_csv(file, index=False)
        colmap = {'stid': 'ID', 'latitude': 'LAT', 'longitude': 'LON'}
        s = load_stations(file, colmap=colmap, engine='pandas')
        assert 'stid' in s.columns
        assert s.iloc[0]['stid'] == 'S1'
        assert s.iloc[0]['extra_data'] == {'elev': 100}

def test_find_nearest_radius_zero():
    with tempfile.TemporaryDirectory() as tmp:
        stations = pd.DataFrame({
            'stid': ['A', 'B'],
            'latitude': [30.0, 30.1],
            'longitude': [50.0, 50.1],
        })
        file = os.path.join(tmp, 'stations.csv')
        stations.to_csv(file, index=False)
        s = load_stations(file, engine='pandas')
        result = find_nearest(s, station_id='A', n_miles=0)
        assert len(result) == 1
        assert result.iloc[0]['stid'] == 'A'


def test_list_weather_files_custom_station():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = tmp
        fA = os.path.join(data_dir, '100_data.csv')
        fB = os.path.join(data_dir, '200_data.csv')
        pd.DataFrame({'date':['2020-01-01'], 'hour':[0]}).to_csv(fA, index=False)
        pd.DataFrame({'date':['2020-01-01'], 'hour':[0]}).to_csv(fB, index=False)

        files = list_weather_files(data_dir, station_ids=['100'], extra_station_ids=['200'])
        assert os.path.basename(fA) in [os.path.basename(x) for x in files]
        assert os.path.basename(fB) in [os.path.basename(x) for x in files]

def test_resample_summary_logic():
    df = pd.DataFrame({
        'stationid': ['1', '1', '1', '1'],
        'datetime': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 01:00', '2020-01-02 00:00', '2020-01-03 00:00']),
        'temp': [10, 20, 30, 40]
    })
    # Daily mean
    res = resample_summary(df, period='1d', agg_map={'temp': ['mean']})
    # 2020-01-01 should have mean(10,20) = 15
    row1 = res[res['datetime'] == '2020-01-01']
    assert row1['temp_mean'][0] == 15
    # 2020-01-02 should have mean(30) = 30
    row2 = res[res['datetime'] == '2020-01-02']
    assert row2['temp_mean'][1] == 30

def test_pipeline_run_minimal():
    with tempfile.TemporaryDirectory() as tmp:
        stations = pd.DataFrame({'stid':['72206013889'], 'latitude':[40.0], 'longitude':[-105.0]})
        stations.to_csv(os.path.join(tmp, 'stations.csv'), index=False)
        weather = pd.DataFrame({
            'date': ['2020-01-01','2020-01-02','2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06'],
            'hour':[1,1,1,1,1,1],
            'stid': ['72206013889']*6,
            'air_temp':[10,11,12,13,14,15],
        })
        weather.to_csv(os.path.join(tmp, '72206013889_data.csv'), index=False)

        config = PipelineConfig(stations_csv = "src/weathermetricsdata/data/stations.csv",
                                station_id_col = "stid",
                                latitude_col = "latitude",
                                longitude_col = "longitude",
                                weather_dir = tmp,
                                n_miles = 0,
                                direction = None,
                                max_stations = 10,
                                extra_station_ids = [],
                                indep_vars = ["air_temp_mean"],
                                time_args = {
                                    "date_col": "date",
                                    "time_col": "hour",
                                    "datetime_col": "datetime" },
                                agg_map = {
                                    "air_temp": ["mean"]},    
                                period_freq = "1d",
                                lookback_days = 1,
                                years_back = 2,
                                bins = [-9999,1,5,20,99999],
                                labels = [0,1,2,3],
                                drop_date_stationid = True                                
                                )
        result = pipeline_run(config, station_id='72206013889', target_var='air_temp_mean')
        assert isinstance(result, (pl.DataFrame, pd.DataFrame)), f"Got {type(result)}"        
        assert 'y' in result.columns
        assert len(result['y']) >= 0


def test_create_dep_var_lookback():
    df = pd.DataFrame({
        'stationid': ['1']*3,
        'datetime': pd.date_range('2020-01-01', periods=3, freq='h'),
        'air_temp':[10, 12, 14],
    })
    out = create_dep_var(df, target_var='air_temp', lookback_days=1, freq='1h')
    assert '1_air_temp_lag_1' in out.columns

def test_create_dep_var_multi_indep_vars_station_prefix():
    df = pd.DataFrame({
    'stationid': ['72206013889']*48 + ['72502014734']*48,
    'datetime': pd.date_range('2020-01-01', periods=48, freq='h').tolist() * 2,
    'air_temp': [10, 12, 14, 16, 6, 8, 3, 5, 7, 8, 1, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 30, 35, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92] * 2,
    'humidity': [30, 35, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 10, 12, 14, 16, 6, 8, 3, 5, 7, 8, 1, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46] * 2
    })
    out = create_dep_var(
        df,
        target_var='air_temp',
        lookback_days=1,
        freq='1h',
        indep_vars=['humidity']
    )
    assert '72206013889_air_temp_lag_1' in out.columns
    assert '72206013889_humidity_lag_1' in out.columns
    assert '72502014734_air_temp_lag_1' in out.columns
    assert '72502014734_humidity_lag_1' in out.columns
    assert out[0, 'y'] == 92


def test_create_dep_var_auxiliary_aligns_as_columns():
    df = pd.DataFrame({
            'stationid': ['1']*48 + ['2']*48 + ['3']*48,
            'datetime': pd.date_range('2020-01-01', periods=48, freq='h').tolist() * 3,
            'air_temp': [10, 12, 14, 16] * 12 * 3,
            'humidity': [30, 32, 34, 36] * 12 * 3,
         })

    out = create_dep_var(
        df,
        target_var='air_temp',
        lookback_days=1,
        freq='1h',
        indep_vars=['humidity'],
        include_auxiliary=True,
        aux_stations=['2', '3'],
        date_col='datetime',
        station_col='stationid',
        drop_date_stationid=False,
    )

    assert '1_air_temp_lag_1' in out.columns
    assert '2_air_temp_lag_1' in out.columns
    assert '2_humidity_lag_1' in out.columns
    assert '3_air_temp_lag_1' in out.columns
    assert '3_humidity_lag_1' in out.columns

    assert set(out['stationid'].unique()) == {'1'}
    assert len(out) > 0


def test_resample_auxiliary_summary_basic():
    """Test basic auxiliary resampling with snapshot aggregates."""
    df = pd.DataFrame({
        'stationid': ['2']*10 + ['3']*10,
        'datetime': pd.date_range('2020-01-01', periods=10, freq='h').tolist() * 2,
        'precip': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] * 2,
        'temp': [10, 12, 14, 16, 18, 10, 12, 14, 16, 18] * 2,
    })
    
    result = resample_auxiliary_summary(
        df,
        aux_agg_map={'precip': ['mean', 'min'], 'temp': ['max']},
        aux_rolling_windows=[1],
        target_stations=['2', '3'],
        datetime_col='datetime',
        group_col='stationid'
    )    

    # Check columns exist
    assert 'multistations_precip_mean_1d' in result.columns
    assert 'multistations_precip_min_1d' in result.columns
    assert 'multistations_temp_max_1d' in result.columns
    
    # Check result has correct number of rows (one per datetime)
    assert len(result) == 10
    
    # Check values (mean of precip across 2 stations at lag should be 4.0)
    assert result['multistations_precip_mean_1d'][4] == 4.0


def test_resample_auxiliary_summary_rolling():
    """Test rolling window aggregates with data leakage prevention."""
    df = pd.DataFrame({
        'stationid': ['2']*5,
        'datetime': pd.date_range('2020-01-01', periods=5, freq='h'),
        'precip': [1, 2, 3, 4, 5],
    })
    
    result = resample_auxiliary_summary(
        df,
        aux_agg_map={'precip': ['mean']},
        aux_rolling_windows=[2],
        target_stations=['2'],
        datetime_col='datetime',
        group_col='stationid'
    )
    
    # Check rolling column exists
    assert '2_precip_mean_2d' in result.columns
    
    # First row should be NaN (no prior history)
    assert pd.isna(result['2_precip_mean_2d'][0])
    
    # Second row should be NaN (shift(1) * rolling(2, closed='left') = only 1 value)
    assert pd.isna(result['2_precip_mean_2d'][1])

    # Third value (index 2) should be mean of [1, 2] = 1.5
    if not pd.isna(pd.isna(result['2_precip_mean_2d'][2])):
        assert result['2_precip_mean_2d'][2] == 1.5

def test_create_dep_var_with_aux_dfs():
    """Test create_dep_var using the pre-computed aux_dfs and output modes."""
    primary_df = pd.DataFrame({
        'stationid': ['P1']*5,
        'datetime': pd.date_range('2020-01-01', periods=5, freq='1h'),
        'target': [10, 20, 30, 40, 50]
    })
    
    aux_df = pl.DataFrame({
        'stationid': ['P1']*5, # must match primary for join in current logic
        'datetime': pd.date_range('2020-01-01', periods=5, freq='1h'),
        'aux_val': [1, 2, 3, 4, 5]
    })

    # Test 'both' mode
    out_both = create_dep_var(
        primary_df, target_var='target', lookback_days=0.1, freq='1h',
        aux_dfs=[aux_df], aux_output_mode='both', drop_date_stationid=False
    )
    # Should have primary lags AND aux lags
    assert any('target_lag' in c for c in out_both.columns)
    assert any('aux_val_lag' in c for c in out_both.columns)

    # Test 'aux' mode
    out_aux_only = create_dep_var(
        primary_df, target_var='target', lookback_days=0.1, freq='1h',
        aux_dfs=[aux_df], aux_output_mode='aux', drop_date_stationid=False
    )
    assert not any('target_lag' in c for c in out_aux_only.columns)
    assert any('aux_val_lag' in c for c in out_aux_only.columns)

def test_create_dep_var_binning():
    df = pd.DataFrame({
        'stationid': ['1']*5,
        'datetime': pd.date_range('2020-01-01', periods=5, freq='1h'),
        'air_temp': [5, 15, 25, 35, 45]
    })
    bins = [0, 20, 40, 60]
    labels = [0, 1, 2]
    out = create_dep_var(df, target_var='air_temp', lookback_days=0.1, bins=bins, labels=labels)
    # Values [5, 15, 25, 35, 45] (after lag drop and descending becomes) [45, 35, 25] -> bins [2, 1, 1]
    assert out['y'][0] == 2


def test_resample_auxiliary_summary_empty():
    """Test with empty dataset."""
    df = pd.DataFrame(columns=['stationid', 'datetime', 'precip'])
    
    result = resample_auxiliary_summary(
        df,
        aux_agg_map={'precip': ['mean']},
        target_stations=['2'],
        datetime_col='datetime',
        group_col='stationid'
    )
    
    # Should return empty DataFrame
    assert len(result) == 0
