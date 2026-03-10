import pandas as pd
import requests
from google.cloud import bigquery

#url for the energy output per energy source
file_path = "ActualAggregatedGenerationByType-2025-01-01T13_30_00.000Z-2026-01-01T13_30_00.000Z.csv"

#json to DataFrame for API calls
df = pd.read_csv(file_path)

#Turning the 'StartTime' column into a datetime format (removed timezone)
df['StartTime'] = pd.to_datetime(df['StartTime']).dt.tz_localize(None)

#take each individual energy type and turn them into columns
df_pivot = df.pivot_table(
    index='StartTime',
    columns='PsrType',
    values='Quantity',
    aggfunc='sum'
)

#Make a total output column for each 30 minute timeslot
df_pivot['TotalOutput-MW'] = df_pivot.sum(axis=1)

PROJECT = "gridzero-489711"
DATASET = "Exelon_Generation_Mix"
TABLE = "Exelon_Generation_AGBT_2025"

table = f"{PROJECT}.{DATASET}.{TABLE}"

client = bigquery.Client()

write_mode = "WRITE_TRUNCATE" # or "WRITE_APPEND"
job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

job = client.load_table_from_dataframe(df_pivot, table, job_config=job_config)
result = job.result()





#FOR API CALLS
# import pandas as pd
# import requests

# #url for the energy output per energy source
# url = "ActualAggregatedGenerationByType-2025-01-01T13_30_00.000Z-2026-01-01T13_30_00.000Z.csv"

# #json to DataFrame for API calls
# response = requests.get(url)
# json_data = response.json()

# df = pd.json_normalize(
#     json_data['data'],
#     record_path='data',
#     meta=['startTime', 'settlementPeriod']
# )

# #Turning the 'StartTime' column into a datetime format (removed timezone)
# df['startTime'] = pd.to_datetime(df['startTime']).dt.tz_localize(None)

# #take each individual energy type and turn them into columns
# df_pivot = df.pivot_table(
#     index='startTime',
#     columns='psrType',
#     values='quantity',
#     aggfunc='sum'
# )

# #Make a total output column for each 30 minute timeslot
# df_pivot['TotalOutput(MW)'] = df_pivot.sum(axis=1)
