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

        response = requests.get(url, params=params)
        data = response.json()["data"]

        df = pd.DataFrame(data)
        dfs.append(df)

        start = chunk_end + timedelta(days=1)

    final_df = pd.concat(dfs, ignore_index=True)

    return final_df


def exelon_preproc(df):
    """
    preprocessing exelon dataframe:
    convert StartTime column from object into datetime, pivot PsrType (fuel type) column
    into their own columns with their individual generation quantities,
    """
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df_pivot = df.pivot_table(
        index='StartTime',
        columns='PsrType',
        values='Quantity',
        aggfunc='sum'
    )

    df_pivot['TotalOutput-MW'] = df_pivot.sum(axis=1)
