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


AGBT_weather_merged = AGBT_df.merge(weather_data_df, left_on='StartTime', right_on='time', how='inner')
fully_merged_data = AGBT_weather_merged.merge(carbon_intensity_df, left_on='time', right_on='timestamp', how='inner')

#MAKING CORRELATION HEATMAP
import seaborn as sns
import matplotlib.pyplot as plt
ordered_data = fully_merged_data.sort_values('time')
ordered_data.describe()
correlation_matrix = fully_merged_data.corr()
# Plot the heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of the Dataset')
plt.show()
