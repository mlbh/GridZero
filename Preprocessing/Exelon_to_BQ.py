import pandas as pd
from datetime import datetime, timedelta
import requests
from google.cloud import bigquery


url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST"

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

all_data = []

current_start = start_date

while current_start <= end_date:
    current_end = min(current_start + timedelta(days=6), end_date)

    params = {
        "publishDateTimeFrom": current_start.strftime("%Y-%m-%dT00:00:00Z"),
        "publishDateTimeTo": current_end.strftime("%Y-%m-%dT23:59:59Z"),
        "format": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    if "data" in data:
        all_data.extend(data["data"])

    current_start = current_end + timedelta(days=1)

df = pd.DataFrame(all_data)



PROJECT = "gridzero-489711"
DATASET = "Exelon_Generation_Mix"
TABLE = "Exelon_Generation_Mix_2025"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df, table, job_config=job_config)
result = job.result()
