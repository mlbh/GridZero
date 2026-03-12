import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from google.cloud import bigquery
from carbon_pipeline import load_carbon_intensity_data, preprocess_carbon_intensity_data
from exelonAGBT_fetch_preproc import fetch_exelon, exelon_preproc
from weather_fetch_preproc import fetch_weather, weather_preproc

    #Merge


def API_and_merge(start_date: str, end_date: str):
    # Load and process Exelon data
    exelon_df = fetch_exelon(start_date, end_date)
    exelon_df = exelon_preproc(exelon_df)

    # Load and process carbon intensity
    carbon_df = load_carbon_intensity_data(start_date, end_date)
    carbon_df = preprocess_carbon_intensity_data(carbon_df)

    # Load and process weather data
    weather_df = fetch_weather(start_date, end_date, latitude=51.5, longitude=-0.1)
    weather_df = weather_preproc(weather_df)

    df = (
        weather_df
        .merge(exelon_df, left_on='datetime', right_index=True, how='left')
        .merge(carbon_df, left_on='datetime', right_on='timestamp', how='left')
        .drop(columns=['timestamp'], errors='ignore')
        .set_index('datetime')
        .sort_index()
    )

    return df

    # weather_AGBT_df = weather_df.merge(
    #     exelon_df, left_on='datetime', right_on='startTime', how='left'
    # )
    # fully_merged_data = weather_AGBT_df.merge(carbon_df, left_on='datetime', right_on='timestamp', how='left')

    # fully_merged_data.drop(columns=['startTime', 'timestamp'], inplace=True, errors='ignore')

    # df = fully_merged_data.set_index('datetime').sort_index()


def impute_values(df):
    df = df.copy()

    df['hour'] = df.index.hour
    df['month'] = df.index.month

    # Most sources
    linear_sources = ['Biomass', 'Fossil Gas', 'Nuclear', 'Other',
                    'Fossil Hard coal', 'Fossil Oil'
                        ]

    df[linear_sources] = df[linear_sources].interpolate(method='linear')
    #hydro sources to be moved out once precipitation becomes a feature

    # Weather-dependent
    solar_features = ['Solar', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
                    'diffuse_radiation_wm2', 'temperature_2m_c', 'hour', 'month']

    wind_features = ['Wind Offshore', 'Wind Onshore',
                    'wind_speed_100m_ms', 'wind_gusts_10m_ms', 'hour', 'month']

    #FOR WHEN PRECIPITATION IS A FEATURE
    hydro_features = ['Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                      'month', 'Other', 'precipitation_mm']
    imputer = KNNImputer(n_neighbors=5)
    df[hydro_features] = imputer.fit_transform(df[hydro_features])


    for features in [solar_features, wind_features]:
        imputer = KNNImputer(n_neighbors=5)
        df[features] = imputer.fit_transform(df[features])

    # Recalculate TotalOutput and carbon_intensity from imputed values
    # rather than imputing them directly
    energy_cols = ['Biomass', 'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil',
                'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
                'Nuclear', 'Other', 'Solar', 'Wind Offshore', 'Wind Onshore']
    df['TotalOutput-MW'] = df[energy_cols].sum(axis=1)

    return df




#TEST
df = API_and_merge('2017-09-12','2026-03-12')
# df1 = impute_values(df)


#Upload to bq
PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "test_merge_2017_onward_raw"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df, table, job_config=job_config)
result = job.result()
