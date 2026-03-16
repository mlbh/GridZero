import requests
import pandas as pd


def fetch_forecast(latitude=51.5, longitude=-0.1, forecast_days=1):

    url = "https://api.open-meteo.com/v1/forecast"

    hourly_vars = [
        "temperature_2m",
        "wind_gusts_10m",
        "cloud_cover",
        "direct_radiation",
        "diffuse_radiation",
        "shortwave_radiation",
        "wind_speed_120m",
        "wind_speed_80m",
        "pressure_msl",
        "precipitation",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "timezone": "GMT",
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Unexpected API response: {data}")

    df = pd.DataFrame(data["hourly"])

    return df


def weather_preproc(df):

    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    df['wind_speed_100m'] =  (df['wind_speed_120m'] + df['wind_speed_80m'])/2

    rename_map = {
        "temperature_2m": "temperature_2m_c",
        "wind_speed_120m": "wind_speed_120m_ms",
        "wind_speed_80m": "wind_speed_80m_ms",
        "wind_gusts_10m": "wind_gusts_10m_ms",
        "cloud_cover": "cloud_cover_pct",
        "shortwave_radiation": "shortwave_radiation_wm2",
        "direct_radiation": "direct_radiation_wm2",
        "diffuse_radiation": "diffuse_radiation_wm2",
        "pressure_msl": "pressure_msl_hpa",
        "precipitation": "precipitation_mm",
    }

    df = df.rename(columns=rename_map)

    return df
