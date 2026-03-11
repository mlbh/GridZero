# Code to load a df into big query

PROJECT = 'gridzero-489711'
DATASET ='historical_weather'
TABLE = '02_full_params_weather_2025'

table = f'{PROJECT}.{DATASET}.{TABLE}'

client = bigquery.Client(project=PROJECT)

write_mode = 'WRITE_TRUNCATE'
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(weather_a_df, table, job_config=job_config)
result = job.result()


def upload_to_bigquery(df: pd.DataFrame,PROJECT: str,DATASET: str,TABLE: str, write_mode: str = "WRITE_TRUNCATE"):
    '''
    Upload a DataFrame to a BigQuery table.
    Function arguments:
        DF: DataFrame to upload
        PROJECT: GCP project ID
        DATASET: BigQuery dataset name
        TABLE: BigQuery table name
        write_mode: Write disposition (default set to WRITE_TRUNCATE)
    '''
    table_ref = f'{PROJECT}.{DATASET}.{TABLE}'

    client = bigquery.Client(project=PROJECT)
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

    rows, cols = df.shape
    return f'{TABLE}, {len(df):,} rows to {table_ref}'


# Example
# upload_to_bigquery(
#     df=weather_a_df,
#     project='gridzero-489711',
#     dataset='historical_weather',
#     table='02_full_params_weather_2025'
# )
# 
