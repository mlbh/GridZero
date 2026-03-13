import requests
import pandas as pd
from datetime import timedelta


def load_carbon_intensity_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load Carbon Intensity API data for any specified date range.
    Handles the API 31-day limit by fetching data in chunks.
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if start >= end:
        raise ValueError("end_date must be after start_date")

    dfs = []
    current = start

    while current < end:
        next_date = min(current + timedelta(days=30), end)

        url = (
            f"https://api.carbonintensity.org.uk/intensity/"
            f"{current.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            print(f"Warning: Carbon API returned non-JSON response for {current} - {next_date}")
            payload = {}

        data = payload.get("data", []) if isinstance(payload, dict) else []

        if data:
            dfs.append(pd.json_normalize(data))

        current = next_date

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)



def preprocess_carbon_intensity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess Carbon Intensity API dataframe.
    """

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "carbon_intensity_gCO2_kWh"])

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["from"], utc=True).dt.tz_convert(None)
    df = df.rename(columns={
        "intensity.actual": "actual",
        "intensity.forecast": "forecast"
    })
    df["carbon_intensity_gCO2_kWh"] = pd.to_numeric(df["actual"], errors='coerce')
    df["carbon_intensity_gCO2_kWh"] = df["carbon_intensity_gCO2_kWh"].fillna(
        pd.to_numeric(df["forecast"], errors='coerce')
    )
    df = df[["timestamp", "carbon_intensity_gCO2_kWh"]]

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    return df
