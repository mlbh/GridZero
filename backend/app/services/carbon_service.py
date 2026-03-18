import requests
import pandas as pd

CARBON_API = "https://api.carbonintensity.org.uk"


def fetch_carbon_history(forecast_timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Fetches real historical carbon intensity for lag feature lookups.

    Fetches two windows:
    - 24 hours ago  (for carbon_lag_48)
    - 1 year ago    (for carbon_lag_17520)

    forecast_timestamps: datetime index of the forecast window
    Returns: datetime-indexed Series of carbon intensity (gCO2/kWh)
    """

    # calculate the two lag windows we need
    lag_48_times    = forecast_timestamps - pd.Timedelta(hours=24)
    lag_17520_times = forecast_timestamps - pd.Timedelta(days=365)

    # fetch one window covering all the times we need
    # min = earliest lag_17520 time, max = latest lag_48 time
    start = lag_17520_times.min().strftime("%Y-%m-%dT%H:%MZ")
    end   = lag_48_times.max().strftime("%Y-%m-%dT%H:%MZ")

    url = f"{CARBON_API}/intensity/{start}/{end}"

    response = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=30
    )
    response.raise_for_status()

    records = []
    for entry in response.json()["data"]:
        # use actual if available, fall back to forecast
        intensity = entry["intensity"]["actual"] or entry["intensity"]["forecast"]
        records.append({
            "datetime": pd.to_datetime(entry["from"]).tz_localize(None),
            "carbon_intensity": intensity
        })

    if not records:
        raise ValueError("No carbon intensity data returned from API")

    series = (
        pd.DataFrame(records)
        .drop_duplicates("datetime")
        .set_index("datetime")["carbon_intensity"]
        .sort_index()
    )

    return series
