import numpy as np
import pandas as pd
#this needs editing probably

LSTM_FEATURES = [
    "temperature_2m_c",
    "wind_speed_120m_ms",
    "wind_speed_80m_ms",
    "wind_gusts_10m_ms",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "pressure_msl_hpa",
    "precipitation_mm"
]

GENERATION_COLS = [
    "biomass",
    "fossil_oil"
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

def build_lstm_features(weather_df: pd.DataFrame):

    df = weather_df[LSTM_FEATURES].copy()

    X = df.values

    # reshape for LSTM
    X = X.reshape(1, X.shape[0], X.shape[1])

    return X


def build_xgb_features(weather_df: pd.DataFrame, generation_prediction: np.ndarray):
    """ weather_df coming in of nx48
    generation_prediction: shape (48, 11)
    XGBoost will predict nx48carbon intensity value per row.
    """
    # pred generation features
    gen_pred_df = pd.DataFrame(generation_prediction, columns=GENERATION_COLS)

    total_mw = gen_pred_df["totaloutput_mw"].replace(0, 1e-8)

    for col in GENERATION_COLS:
        gen_pred_df[f"gen_mw_{col}"] = gen_pred_df[col]
        if col != "totaloutput_mw":
            gen_pred_df[f"gen_prop_{col}"] = gen_pred_df[col] / total_mw

    gen_pred_df = gen_pred_df.drop(columns=GENERATION_COLS)  # drop raw cols, keep mw_ and prop_ versions

    # wather features
    weather_features = weather_df[LSTM_FEATURES].reset_index(drop=True)

    # combine rows
    combined = pd.concat([weather_features, gen_pred_df], axis=1)

    return combined  # shape (48, n_features)
