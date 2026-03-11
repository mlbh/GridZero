# Contents:
# 1 - function to upload a df to big query
# 2 - fucntion to load df from big query

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


def load_from_bigquery():
