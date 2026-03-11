from google.cloud import bigquery

#need to pip install db_dtypes into the env

#generation
PROJECT = "gridzero-489711"
DATASET = "Exelon_Generation_Mix"
TABLE = "Exelon_Generation_AGBT_2025"

query = f"""
    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
"""

client = bigquery.Client('gridzero-489711')
query_job = client.query(query)
result = query_job.result()
AGBT_df = result.to_dataframe()


#carbon
PROJECT = "gridzero-489711"
DATASET = "gridzero"
TABLE = "carbon_intensity_2025"

query = f"""
    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
"""

client = bigquery.Client('gridzero-489711')
query_job = client.query(query)
result = query_job.result()
carbon_intensity_df = result.to_dataframe()


#Weather
PROJECT = "gridzero-489711"
DATASET = "historical_weather"
TABLE = "05_halfhourly_select_params_weather_2025"

query = f"""
    SELECT *
    FROM {PROJECT}.{DATASET}.{TABLE}
"""

client = bigquery.Client('gridzero-489711')
query_job = client.query(query)
result = query_job.result()
weather_data_df = result.to_dataframe()

#Merge
weather_AGBT_df = weather_data_df.merge(AGBT_df, left_on='time', right_on='StartTime', how='left')
fully_merged_data = weather_AGBT_df.merge(carbon_intensity_df, left_on='time', right_on='timestamp', how='left')

#Drop extra time columns - (think this could be done in the merge though..?)
fully_merged_data.drop(columns='timestamp', inplace=True)
fully_merged_data.drop(columns='StartTime', inplace=True)



#Imputing strategy:
from sklearn.impute import KNNImputer

merged = fully_merged_data.set_index('time').sort_index()

# Most sources
linear_sources = ['Biomass', 'Fossil Gas', 'Nuclear', 'Other',
                  'Hydro Run-of-river and poundage', 'Hydro Pumped Storage',
                  'Fossil Hard coal', 'Fossil Oil']
merged[linear_sources] = merged[linear_sources].interpolate(method='linear')
#hydro sources to be moved out once precipitation becomes a feature

# Weather-dependent
solar_features = ['Solar', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
                  'diffuse_radiation_wm2', 'temperature_2m_c', 'hour', 'month']

wind_features = ['Wind Offshore', 'Wind Onshore',
                 'wind_speed_100m_ms', 'wind_gusts_10m_ms', 'hour', 'month']

#FOR WHEN PRECIPITATION IS A FEATURE
# hydro_features = ['Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
#                   'month', *'Other'*, *'precipitation_mm'*]
# imputer = KNNImputer(n_neighbors=5)
# merged[hydro_features] = imputer.fit_transform(merged[hydro_features])


merged['hour'] = merged.index.hour
merged['month'] = merged.index.month

for features in [solar_features, wind_features]:
    imputer = KNNImputer(n_neighbors=5)
    merged[features] = imputer.fit_transform(merged[features])

# Recalculate TotalOutput and carbon_intensity from imputed values
# rather than imputing them directly
energy_cols = ['Biomass', 'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil',
               'Hydro Pumped Storage', 'Hydro Run-of-river and poundage',
               'Nuclear', 'Other', 'Solar', 'Wind Offshore', 'Wind Onshore']
merged['TotalOutput-MW'] = merged[energy_cols].sum(axis=1)




#Upload to bq
PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "Fully_merged_dataset_2025"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(merged, table, job_config=job_config)
result = job.result()
