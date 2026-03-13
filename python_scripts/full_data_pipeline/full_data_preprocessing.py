import pandas as pd
import numpy as np


def full_data_preproc(df):
    weather_cols = [
        'temperature_2m_c','wind_speed_100m_ms','wind_gusts_10m_ms',
        'cloud_cover_pct','shortwave_radiation_wm2','direct_radiation_wm2',
        'diffuse_radiation_wm2','pressure_msl_hpa'
        ]

    df[weather_cols] = df[weather_cols].interpolate(method='linear')


    gen_cols = [
    'Biomass','Fossil Gas','Fossil Hard coal','Fossil Oil',
    'Hydro Pumped Storage','Hydro Run-of-river and poundage',
    'Nuclear','Other','Solar','Wind Offshore','Wind Onshore'
    ]

    df[gen_cols] = df[gen_cols].interpolate()
    df['TotalOutput-MW'] = df[gen_cols].sum(axis=1)


    # precipitation is good enough and Fossil Oil is mostly irrelevant
    df = df.drop(columns=['rain_mm','snowfall_cm', 'Fossil Oil', 'status'])

    # Create Time Features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # cyclical encoding
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['doy_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
    df['doy_cos'] = np.cos(2*np.pi*df['day_of_year']/365)

    df = df.drop(columns=['hour','day_of_week', 'day_of_year'])

    lags = [48, 336, 17520]   # 1 day, 1 week, 1 year

    for lag in lags:
        df[f'carbon_lag_{lag}'] = df['carbon_intensity_gCO2_kWh'].shift(lag)

    # using same naming convention for all columns
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    # custom imputing carbon intensity values
    # Recalculate Carbon Intensity From Generation Mix using approximate typical emission factors
    emissions = (
        df['biomass'] * 230 +
        df['fossil_gas'] * 490 +
        df['fossil_hard_coal'] * 820 +
        df['nuclear'] * 12 +
        df['solar'] * 45 +
        df['wind_onshore'] * 11 +
        df['wind_offshore'] * 11 +
        df['hydro_run_of_river_and_poundage'] * 24
    )

    df['carbon_estimate'] = emissions / df['totaloutput_mw']
    df['carbon_intensity_gco2_kwh'] = df['carbon_intensity_gco2_kwh'].fillna(df['carbon_estimate'])
    df = df.drop(columns='carbon_estimate')

    return df


from google.cloud import bigquery

PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "test_merge_2017_onward_raw"

query = f"""
    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
"""

client = bigquery.Client('gridzero-489711')
query_job = client.query(query)
result = query_job.result()
df = result.to_dataframe()

df = full_data_preproc(df)

#Upload to bq
PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "full_feature_engineered_data_test"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df, table, job_config=job_config)
result = job.result()
