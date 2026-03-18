
import requests
import pandas as pd
import numpy as np



OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"


def get_aligned_weather_elexon_fill():
    # -----------------------------
    # 1) Weather
    # -----------------------------
    weather_params = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hourly": [
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
        ],
        "timezone": "GMT",
        "forecast_days": 14,
        "past_days": 7,
        "wind_speed_unit": "ms",
        "current": "temperature_2m",
    }

    weather_res = requests.get(OPEN_METEO_URL, params=weather_params, timeout=30)
    weather_res.raise_for_status()
    weather_data = weather_res.json()

    weather_df = pd.DataFrame(weather_data["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    weather_df["wind_speed_100m"] = (
        weather_df["wind_speed_120m"] + weather_df["wind_speed_80m"]
    ) / 2
    weather_df = weather_df.drop(columns=["wind_speed_120m", "wind_speed_80m"])

    forecast_start = pd.to_datetime(weather_data["current"]["time"])

    # hourly -> half-hourly
    weather_df = (
        weather_df.set_index("time")
        .resample("30min")
        .interpolate(method="time")
        .reset_index()
    )

    # keep exactly the same historical weather window as before
    forecast_idx = weather_df[weather_df["time"] >= forecast_start].index[0]
    start_idx = max(0, forecast_idx - 336)
    weather_hist_df = weather_df.iloc[start_idx:forecast_idx].reset_index(drop=True)

    weather_index = weather_hist_df["time"]

    # -----------------------------
    # 2) Elexon
    # -----------------------------
    elexon_params = {
        "from": weather_index.min().strftime("%Y-%m-%dT%H:%MZ"),
        "to": weather_index.max().strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }

    elexon_res = requests.get(ELEXON_URL, params=elexon_params, timeout=30)
    elexon_res.raise_for_status()
    elexon_json = elexon_res.json()

    records = []
    for period in elexon_json["data"]:
        for item in period["data"]:
            records.append({
                "startTime": period["startTime"],
                "psrType": item["psrType"],
                "quantity": item["quantity"],
            })

    elexon_raw = pd.DataFrame(records)

    if elexon_raw.empty:
        raise ValueError("No Elexon data returned for requested range.")

    elexon_raw["startTime"] = pd.to_datetime(
        elexon_raw["startTime"], utc=True
    ).dt.tz_localize(None)

    elexon_df = (
        elexon_raw.pivot_table(
            index="startTime",
            columns="psrType",
            values="quantity",
            aggfunc="sum",
        )
        .sort_index()
        .reset_index()
    )

    elexon_df.columns.name = None

    fuel_cols = [c for c in elexon_df.columns if c != "startTime"]
    elexon_df["total_output_MW"] = elexon_df[fuel_cols].sum(axis=1)

    # -----------------------------
    # 3) Reindex Elexon to weather timestamps
    # -----------------------------
    elexon_hist_df = (
        elexon_df.set_index("startTime")
        .reindex(weather_index)
        .rename_axis("startTime")
        .reset_index()
    )

    # -----------------------------
    # 4) Fill missing Elexon values
    # -----------------------------
    gen_cols = [c for c in elexon_hist_df.columns if c != "startTime"]

    # interpolate interior gaps
    elexon_hist_df[gen_cols] = elexon_hist_df[gen_cols].interpolate(
        method="linear",
        limit_direction="both"
    )

    # forward fill any remaining trailing gaps
    elexon_hist_df[gen_cols] = elexon_hist_df[gen_cols].ffill()

    # backfill any remaining leading gaps
    elexon_hist_df[gen_cols] = elexon_hist_df[gen_cols].bfill()

    return weather_hist_df, elexon_hist_df




def merge_weather_elexon(weather_df, elexon_df):
    df = weather_df.merge(
        elexon_df,
        left_on="time",
        right_on="startTime",
        how="outer"
    )

    # Drop duplicate time column from Elexon
    df = df.drop(columns=["startTime"])

    return df



def preproc(df):
    if "time" in df.columns:
        df["time"] = df["time"].astype("datetime64[us]",  errors="ignore")

        df = df.rename(columns={'time': 'datetime'},  errors="ignore")

    weather_cols = [
        'temperature_2m_c','wind_speed_100m_ms','wind_gusts_10m_ms',
        'cloud_cover_pct','shortwave_radiation_wm2','direct_radiation_wm2',
        'diffuse_radiation_wm2','pressure_msl_hpa'
        ]

    gen_cols = [
    'Biomass','Fossil Gas','Fossil Hard coal','Fossil Oil',
    'Hydro Pumped Storage','Hydro Run-of-river and poundage',
    'Nuclear','Other','Solar','Wind Offshore','Wind Onshore'
    ]
    #Fossil oil might need to come back for XGboost
    df = df.drop(columns=['Fossil Oil', 'total_output_MW'], errors="ignore")

    # Create Time Features
    if "datetime" in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear

    # cyclical encoding
        df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['doy_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
        df['doy_cos'] = np.cos(2*np.pi*df['day_of_year']/365)

    df = df.drop(columns=['hour','day_of_week', 'day_of_year', 'datetime'], errors="ignore")

    # using same naming convention for all columns
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    df = df.rename(columns={
        "temperature_2m": "temperature_2m_c",
        "wind_speed_100m": "wind_speed_100m_ms",
        "wind_gusts_10m": "wind_gusts_10m_ms",
        "cloud_cover": "cloud_cover_pct",
        "shortwave_radiation": "shortwave_radiation_wm2",
        "direct_radiation": "direct_radiation_wm2",
        "diffuse_radiation": "diffuse_radiation_wm2",
        "pressure_msl": "pressure_msl_hpa",
        "precipitation": "precipitation_mm",
    })

    return df

def preproc(df):
    df = df.copy()

    # 1. Handle Time Column
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    # 2. Cleanup Generation Columns
    # LSTM was trained on lowercase_with_underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    # 3. Rename Map (to match LSTM feature_order)
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

    # Create Time Features
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

    df['dow_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 7)

    df['doy_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear / 365.25)

    # IMPORTANT: We do NOT drop 'datetime' here.
    # We need it to index the DataFrame in the route.
    return df

def make_lstm_input(df):
    """
    Convert a preprocessed dataframe into one LSTM input sample.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.
    feature_columns : list[str]
        Exact feature order used during model training.
    scaler : fitted scaler, optional
        The same scaler used during training.
    lookback : int
        Number of timesteps expected by the model.

    Returns
    -------
    X : np.ndarray
        Shape (1, lookback, n_features)
    """
    df = df.copy()

    # Ensure required columns exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep exact training column order
    df = df[feature_cols]

    # Handle missing values if needed
    df = df.ffill().bfill()

    if len(df) < 336:
        raise ValueError(f"Need at least {336} rows, got {len(df)}")

    # Take latest 336 rows
    df_window = df.iloc[-336:]

    # Apply training scaler if provided
    values = df_window.values

    values = X_scaler.transform(df_window)

    # Convert to LSTM shape: (1, 336, n_features)
    X = np.expand_dims(values.astype(np.float32), axis=0)

    return X



def get_london_forecast_step_halfhour_all():
    """
    Return a single-row DataFrame for London's forecast at a given half-hour step.

    Parameters
    ----------
    step : int, default=0
        Half-hour step from the most recent forecast boundary.
        0 = most recent forecast rounded down to nearest half hour
        1 = next half-hour forecast
        2 = half-hour after that
        ...
        Up to 14 days ahead.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing one forecast record.
    """
    # if step < 0:
    #     raise ValueError("step must be >= 0")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "hourly": [
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
        ],
        "timezone": "GMT",
        "forecast_days": 16,
        "wind_speed_unit": "ms",
        "current": "temperature_2m",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    # Create wind_speed_100m and drop originals
    df["wind_speed_100m"] = (df["wind_speed_120m"] + df["wind_speed_80m"]) / 2
    df = df.drop(columns=["wind_speed_120m", "wind_speed_80m"])

    # Convert hourly data to half-hourly
    df = (
        df.set_index("time")
        .resample("30min")
        .interpolate(method="time")
        .reset_index()
    )

    # Most recent forecast boundary rounded down to nearest half hour
    forecast_start = pd.to_datetime(data["current"]["time"]).floor("30min")

    # Keep only forecast rows from that point onward
    forecast_df = df[df["time"] >= forecast_start].reset_index(drop=True)

    # max_step = len(forecast_df) - 1
    # if step > max_step:
    #     raise IndexError(f"step must be between 0 and {max_step}")

    # Return a single-record DataFrame
    return forecast_df
