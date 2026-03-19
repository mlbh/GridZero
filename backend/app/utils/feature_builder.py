import numpy as np
import pandas as pd
#this needs editing probably

LSTM_FEATURES = [
    "temperature_2m_c",
    "wind_speed_100m_ms",
    "wind_gusts_10m_ms",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "pressure_msl_hpa",
    "precipitation_mm",


]

GENERATION_COLS = [
    "biomass",
    "fossil_gas",
    "fossil_hard_coal",
    "hydro_pumped_storage",
    "hydro_run_of_river_and_poundage",
    "nuclear",
    "other",
    "solar",
    "wind_offshore",
    "wind_onshore",
    "totaloutput_mw"
]

XGB_FEATURES = [
    "temperature_2m_c",
    "wind_speed_100m_ms",
    "wind_gusts_10m_ms",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "pressure_msl_hpa",
    "precipitation_mm",
    "biomass",
    "fossil_gas",
    "fossil_hard_coal",
    "hydro_pumped_storage",
    "hydro_run_of_river_and_poundage",
    "nuclear",
    "other",
    "solar",
    "wind_offshore",
    "wind_onshore",
    "totaloutput_mw",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "carbon_lag_48",
    "carbon_lag_336",
    "carbon_lag_17520"
]

def build_lstm_features(weather_df: pd.DataFrame):

    df = weather_df[LSTM_FEATURES].copy()

    X = df.values

    # reshape for LSTM
    X = X.reshape(1, X.shape[0], X.shape[1])

    return X

def build_xgb_features(
    weather_df: pd.DataFrame,
    generation_prediction: np.ndarray,
    carbon_history: pd.Series
) -> pd.DataFrame:
    """
    weather_df: 48 rows, datetime-indexed
    generation_prediction: shape (48, 10) from LSTM 10 generation types, no total yet needs calc
    carbon_history: datetime indexed series from carbon_service.py

    Returns DataFrame of shape (48, 28) matching XGB_FEATURES exactly.
    """

    # weather feats
    weather_features = weather_df[LSTM_FEATURES].reset_index(drop=True)

    # generation features (raw MW matching training)
    gen_df = pd.DataFrame(
        generation_prediction,
        columns=GENERATION_COLS
    )

    # calc: totaloutput_mw by summing all generation types
    gen_df["totaloutput_mw"] = gen_df[GENERATION_COLS].sum(axis=1)

    # cyclical features from datetime index
    # replicates exactly what full_data_preproc func does
    timestamps = weather_df.index

    cyclical_df = pd.DataFrame(index=range(len(timestamps)))
    hour = timestamps.hour
    dow  = timestamps.dayofweek
    doy  = timestamps.dayofyear

    cyclical_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cyclical_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    cyclical_df["dow_sin"]  = np.sin(2 * np.pi * dow  / 7)
    cyclical_df["dow_cos"]  = np.cos(2 * np.pi * dow  / 7)
    cyclical_df["doy_sin"]  = np.sin(2 * np.pi * doy  / 365)
    cyclical_df["doy_cos"]  = np.cos(2 * np.pi * doy  / 365)

    # carbon lag from historical API
    lag_df = pd.DataFrame(index=range(len(timestamps)))
    lag_df["carbon_lag_48"]    = carbon_history["yesterday"][:48]
    lag_df["carbon_lag_336"]   = 0 # placeholder to remove
    lag_df["carbon_lag_17520"] = carbon_history["year_ago"][:48]

    # note - fallback if any historical values missing suggestion to impute tomo
    lag_df = lag_df.ffill().bfill()

    # combine in exact training order
    combined = pd.concat([
        weather_features,
        gen_df.reset_index(drop=True),
        cyclical_df,
        lag_df
    ], axis=1)

    # enforce exact column order to match model training
    combined = combined[XGB_FEATURES]

    return combined  # shape (48, 28)
