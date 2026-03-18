from google.cloud import bigquery
import pandas as pd


#THIS DOESN'T WORK AT THE MOMENT AS THE BQ ISNT UP TO DATE - NEED TO FIND OUT HOW CLOSE TO THE MOMENT WE CAN GET IT AND ADJUST
def fetch_historical_gen(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)

    # Query the last 7 days of generation data
    query = f"""
        SELECT
            timestamp as time,
            biomass, fossil_gas, fossil_hard_coal, hydro_pumped_storage,
            hydro_run_of_river_and_poundage, nuclear, other, solar,
            wind_offshore, wind_onshore
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY time ASC
    """

    df_gen = client.query(query).to_dataframe()

    # Resample to 30min and forward-fill to match weather prep
    df_gen['time'] = pd.to_datetime(df_gen['time'])
    df_gen = df_gen.set_index('time').resample('30min').ffill().reset_index()

    return df_gen
    
#EXELON DATA FETCHER
import requests
import pandas as pd

def fetch_elexon_history(api_key):
    # This endpoint gets Actual Generation by Fuel Type
    # Note: Check Elexon Insights for the exact latest endpoint (B1620 is common)
    url = "https://api.bmreports.com/BMRS/B1620/v1"
    
    params = {
        'APIKey': api_key,
        'SettlementDate': (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        'Period': '*',
        'ServiceType': 'json'
    }
    
    # Logic: Loop through the last 7 days and collect the data
    # Elexon provides data in 30-minute chunks (Settlement Periods 1-48)
    # which aligns PERFECTLY with your 30-min training data.
    
    # ... code to parse Elexon's nested JSON into a DataFrame ...
    # Ensure columns match your model: 'solar', 'wind_onshore', etc.
    columns = cols_from_weather #copy paste
    #MERGE HERE?

