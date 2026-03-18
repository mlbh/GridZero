import requests
import pandas as pd
import numpy as np


def fetch_forecast(latitude=51.5, longitude=-0.1):#target_date

    url = "https://api.open-meteo.com/v1/forecast"

    #start_str = target_date.strftime("%Y-%m-%d") if hasattr(target_date, 'strftime') else target_date

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
        "latitude": 51.5,
        "longitude": -0.1,
        "hourly": ",".join(hourly_vars),
        "timezone": "GMT",
        "past_days": 7, #HISTORIC DATA - MAYBE USE THE TRAINED SET ALTERNATIVELY?
        "forecast_days": 14,
        #"start_date": start_str,
        #"end_date": start_str,
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

    if not isinstance(df.index, pd.DatetimeIndex):
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    df = df.resample('30min').ffill()

    cols = df.columns

    if 'wind_speed_120m' in cols and 'wind_speed_80m' in cols:
            df['wind_speed_100m'] = (df['wind_speed_120m'] + df['wind_speed_80m']) / 2
    else:
            # If they are missing, lets see why
            raise KeyError(f"Calculated columns missing. Available: {list(cols)}")

    rename_map = {
        "temperature_2m": "temperature_2m_c",
        "wind_speed_100m": "wind_speed_100m_ms",
        "wind_gusts_10m": "wind_gusts_10m_ms",
        "cloud_cover": "cloud_cover_pct",
        "shortwave_radiation": "shortwave_radiation_wm2",
        "direct_radiation": "direct_radiation_wm2",
        "diffuse_radiation": "diffuse_radiation_wm2",
        "pressure_msl": "pressure_msl_hpa",
        "precipitation": "precipitation_mm",
    }
    df = df.rename(columns=rename_map)

    # Hour
    #MAYBE DELETE
    total_hours = df.index.hour + df.index.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * total_hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * total_hours / 24)
    # Day of Week
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    # Day of Year
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    #Grid Generation Features
    # CRITICAL: If you don't have a live feed for these, you must
    # fill them with 0.0 or a 'typical' value so the shape matches.
    gen_columns = [
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore'
    ]
    for col in gen_columns:
        if col not in df.columns: #THIS CANNOT BE GOOD FOR PREDICTIONS
            df[col] = 0.0

    feature_order = [
        "temperature_2m_c",
        "wind_speed_100m_ms",
        "wind_gusts_10m_ms",
        "cloud_cover_pct",
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
        "pressure_msl_hpa",
        "precipitation_mm",
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore'
    ]

    return df[feature_order]




#############################NEW-SYSTEM###############################
from app.fast_api_functions import get_aligned_weather_elexon_fill,merge_weather_elexon,preproc, get_london_forecast_step_halfhour_all

def get_master_context():
    """
    Fetches 7 days of HISTORY (Weather + Elexon)
    AND 14 days of FORECAST (Weather).
    Returns a unified DataFrame.
    """
    # 1. Get History (Weather & Elexon)
    weather_hist_df, elexon_hist_df = get_aligned_weather_elexon_fill()
    history_df = merge_weather_elexon(weather_hist_df, elexon_hist_df)

    # 2. Get Forecast (Next 14 Days Weather)
    forecast_df = get_london_forecast_step_halfhour_all()

    # 3. Combine them
    # Forecast starts where history ends
    master_df = pd.concat([history_df, forecast_df], ignore_index=True)

    # 4. Final Preprocess (Cyclical features, renaming)
    # Note: Gen columns in the 'future' part of master_df will be NaN initially
    return preproc(master_df)
