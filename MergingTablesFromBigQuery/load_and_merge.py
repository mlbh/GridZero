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

#Upload to bq
PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "Fully_merged_dataset_2025"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(fully_merged_data, table, job_config=job_config)
result = job.result()
