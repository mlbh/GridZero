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


def build_lstm_features(weather_df: pd.DataFrame):

    df = weather_df[LSTM_FEATURES].copy()

    X = df.values

    # reshape for LSTM
    X = X.reshape(1, X.shape[0], X.shape[1])

    return X


def build_xgb_features(weather_df: pd.DataFrame, generation_prediction):

    # features = {}

    # features["temp_mean"] = weather_df["temperature_2m_c"].mean()
    # features["wind_mean"] = weather_df["wind_speed_100m_ms"].mean()
    # features["wind_max"] = weather_df["wind_speed_100m_ms"].max()
    # features["solar_total"] = weather_df["shortwave_radiation_wm2"].sum()
    # features["precip_total"] = weather_df["precipitation_mm"].sum()
    # features["cloud_mean"] = weather_df["cloud_cover_pct"].mean()

    # features["generation_mw"] = generation_prediction[0]
    # features["demand_mw"] = generation_prediction[1]

    return pd.DataFrame([features])
