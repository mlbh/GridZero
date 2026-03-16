from exelon_fetch_preproc import fetch_exelon, exelon_preproc
from weather_fetch_preproc import fetch_weather, weather_preproc

from google.cloud import bigquery



def merge_2015(start_date: str, end_date: str):
    # Load and process Exelon data
    exelon_df = fetch_exelon(start_date, end_date)
    exelon_df = exelon_preproc(exelon_df)

    # Load and process weather data
    weather_df = fetch_weather(start_date, end_date, latitude=51.5, longitude=-0.1)
    weather_df = weather_preproc(weather_df)

    df = (weather_df
        .merge(exelon_df, left_on="datetime", right_on="startTime", how="left")
        .drop(columns="startTime"))

    return df




# fetching data
df = merge_2015('2015-01-01','2026-03-11')

# loading data to BQ
PROJECT = "gridzero-489711"
DATASET = "merged_set"
TABLE = "test_2015_exelon_weather_raw"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df, table, job_config=job_config)
result = job.result()
