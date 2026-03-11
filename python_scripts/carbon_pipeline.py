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

        data = response.json().get("data", [])

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
        return pd.DataFrame(columns=["datetime", "carbon_intensity/gCO2/kWh"])

    df = df.copy()

    df = df.rename(columns={
        "intensity.actual": "actual",
        "intensity.forecast": "forecast"
    })

    df["datetime"] = pd.to_datetime(df["from"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Remove rows where datetime parsing failed
    df = df.dropna(subset=["datetime"])

    # Check that a usable intensity column exists
    if "actual" not in df.columns and "forecast" not in df.columns:
        raise ValueError(
            "API response missing both 'intensity.actual' and 'intensity.forecast'"
        )

    # Carbon Intensity API may return null actual values, so fallback to forecast
    if "actual" in df.columns:
        df["carbon_intensity/gCO2/kWh"] = df["actual"]

        if "forecast" in df.columns:
            df["carbon_intensity/gCO2/kWh"] = df["carbon_intensity/gCO2/kWh"].fillna(df["forecast"])
    else:
        df["carbon_intensity/gCO2/kWh"] = df["forecast"]

    df = df[["datetime", "carbon_intensity/gCO2/kWh"]]

    df = (
        df
        .sort_values("datetime")
        .drop_duplicates(subset="datetime")
        .reset_index(drop=True)
    )

    return df
