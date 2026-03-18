import pandas as pd
from datetime import datetime, timedelta
import requests


def fetch_exelon(start_date, end_date):
    """
    fetch data within specifite dates from Exelon AGBT API and returns it as a pandas dataframe
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dfs = []

    while start <= end:

        chunk_end = min(start + timedelta(days=6), end)

        params = {
            "from": start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "settlementPeriodFrom": 1,
            "settlementPeriodTo": 50,
            "format": "json"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json().get("data", [])

        if data:
            df_raw = pd.DataFrame(data)
            df_exploded = df_raw.explode('data').reset_index(drop=True)
            df_normalized = pd.json_normalize(df_exploded['data'])
            df_normalized['startTime'] = df_exploded['startTime'].values
            dfs.append(df_normalized)

        start = chunk_end + timedelta(days=1)

    if not dfs:
        return pd.DataFrame(columns=["startTime", "psrType", "quantity"])

    return pd.concat(dfs, ignore_index=True)


def exelon_preproc(df):
    """
    preprocessing exelon dataframe:
    convert StartTime column from object into datetime, pivot PsrType (fuel type) column
    into their own columns with their individual generation quantities,
    """
    df['startTime'] = pd.to_datetime(df['startTime']).dt.tz_convert(None)
    df_pivot = df.pivot_table(
        index='startTime',
        columns='psrType',
        values='quantity',
        aggfunc='sum'
    )

    # df_pivot['TotalOutput-MW'] = df_pivot.sum(axis=1)

    return df_pivot



#EXELON DATA FETCHER
import requests
import pandas as pd

def fetch_elexon_history(api_key):
    # This endpoint gets Actual Generation by Fuel Type
    # Note: Check Elexon Insights for the exact latest endpoint (B1620 is common)
    url = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"

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
