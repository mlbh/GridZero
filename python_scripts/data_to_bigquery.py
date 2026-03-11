# Contents:
# 0 - imports needed
# 1 - function to upload a df to big query
# 2 - fucntion to load df from big query

from google.cloud import bigquery
import pandas as pd

def upload_to_bigquery(df: pd.DataFrame,PROJECT: str,DATASET: str,TABLE: str, write_mode: str = "WRITE_TRUNCATE"):
    '''
    Upload a pd DataFrame to a BigQuery table.
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


def load_from_bigquery(PROJECT: str, DATASET: str, TABLE: str):
    ''' Load data from BigQuery into a pd DataFrame.
    Arguments:
        PROJECT: GCP project ID
        DATASET: BigQuery dataset name
        TABLE: BigQuery table name
    Returns: pandas DataFrame '''
    
    client = bigquery.Client(project=PROJECT)
    query = f'''
    SELECT *
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    '''
    df = client.query(query).to_dataframe()
    rows, cols = df.shape
    print(f"Loaded {rows:,} rows and {cols} columns from {PROJECT}.{DATASET}.{TABLE}")

    return df
