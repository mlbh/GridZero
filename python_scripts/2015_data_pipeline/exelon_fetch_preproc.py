import pandas as pd
from datetime import datetime, timedelta
import requests


def fetch_exelon(start_date, end_date):
    """
    Fetch generation data from Elexon BMRS API for a given date range.
    Loops through the range in 7-day chunks and returns a combined DataFrame.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date:   End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame with columns: startTime, settlementPeriod, businessType, psrType, quantity
    """
    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_records = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=7), end)

        params = {
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to": chunk_end.strftime("%Y-%m-%d"),
            "settlementPeriodFrom": 1,
            "settlementPeriodTo": 50,
            "format": "json"
        }

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        json_data = response.json()

        for period in json_data["data"]:
            for item in period["data"]:
                all_records.append({
                    "startTime":        period["startTime"],
                    "settlementPeriod": period["settlementPeriod"],
                    "businessType":     item["businessType"],
                    "psrType":          item["psrType"],
                    "quantity":         item["quantity"]
                })

        print(f"Fetched: {params['from']} → {params['to']}")
        chunk_start = chunk_end + timedelta(days=1)

    df = pd.DataFrame(all_records)
    df["startTime"] = pd.to_datetime(df["startTime"])
    df = df.sort_values("startTime").reset_index(drop=True)

    return df



def exelon_preproc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw generation DataFrame:
    - Converts startTime to datetime
    - Pivots psrType into individual columns
    - Creates a total_output column
    - Drops redundant columns

    Args:
        df: Raw DataFrame from get_generation_data()

    Returns:
        Cleaned and pivoted pd.DataFrame
    """
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True).dt.tz_localize(None)

    df_pivot = df.pivot_table(
        index="startTime",
        columns="psrType",
        values="quantity",
        aggfunc="sum"
    ).reset_index()

    # Flatten column names
    df_pivot.columns.name = None

    # Total output across all fuel types
    fuel_cols = [col for col in df_pivot.columns if col != "startTime"]
    df_pivot["total_output_MW"] = df_pivot[fuel_cols].sum(axis=1)

    return df_pivot
