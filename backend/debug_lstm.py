import pandas as pd
import numpy as np
from app.cloud.gcs_loader import GCSModelLoader
from app.config import GCS_BUCKET, XGB_MODEL_PATH, LSTM_MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH
import joblib
import xgboost as xgb
from tensorflow import keras

class ModelStore:
    def __init__(self):
        self.lstm = None
        self.xgb = None
        self.x_scaler = None
        self.y_scaler = None

    def load_models(self):
        gcs = GCSModelLoader(GCS_BUCKET)
        lstm_path = gcs.download_file(LSTM_MODEL_PATH)
        xgb_path = gcs.download_file(XGB_MODEL_PATH)
        x_scaler_local = gcs.download_file(X_SCALER_PATH)
        y_scaler_local = gcs.download_file(Y_SCALER_PATH)
        self.lstm = keras.models.load_model(lstm_path)
        self.xgb = xgb.Booster()
        self.xgb.load_model(str(xgb_path))
        self.x_scaler = joblib.load(x_scaler_local)
        self.y_scaler = joblib.load(y_scaler_local)

class LSTMPredictor:
    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_order = [
            'temperature_2m_c', 'wind_speed_100m_ms', 'wind_gusts_10m_ms',
            'cloud_cover_pct', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
            'diffuse_radiation_wm2', 'pressure_msl_hpa', 'precipitation_mm',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
            'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
            'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
            'wind_offshore', 'wind_onshore'
        ]

    def predict(self, df_input):
        data_to_scale = df_input.drop(columns=['time', 'datetime'], errors='ignore')
        ordered_df = data_to_scale[self.feature_order]
        scaled_input = self.x_scaler.transform(ordered_df)
        lstm_in = scaled_input.reshape(1, 336, 25)
        scaled_pred = self.model.predict(lstm_in, verbose=0)
        real_pred = self.y_scaler.inverse_transform(scaled_pred)
        real_pred = np.maximum(real_pred, 0)
        return real_pred

if __name__ == "__main__":
    # ── 1. Load models ────────────────────────────────────────────────────────
    model_store = ModelStore()
    model_store.load_models()

    # ── 2. Load data ──────────────────────────────────────────────────────────
    master_df = pd.read_csv("master_df_debug1.csv", parse_dates=["time"])

    # ── 3. Instantiate predictor ──────────────────────────────────────────────
    predictor = LSTMPredictor(model_store.lstm, model_store.x_scaler, model_store.y_scaler)

    # ── 4. Debug loop ─────────────────────────────────────────────────────────
    target_dt = pd.to_datetime("2026-03-28").replace(hour=0, minute=0, second=0)
    time_col = 'datetime' if 'datetime' in master_df.columns else 'time'
    target_idx = master_df.index[master_df[time_col] == target_dt][0]

    current_window = master_df.iloc[target_idx - 336 : target_idx].copy()

    gen_features = [
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore'
    ]

    for i in range(5):
        print(f"--- Step {i} ---")
        print(f"Window start: {current_window.iloc[0][time_col]}")
        print(f"Window end:   {current_window.iloc[-1][time_col]}")

        next_step_row = master_df.iloc[target_idx + i].copy()
        print(f"Next row time: {next_step_row[time_col]}, gas: {next_step_row['fossil_gas']}")

        pred = predictor.predict(current_window)
        print(f"Predicted gas: {pred[0][1]}")

        for idx, col in enumerate(gen_features):
            next_step_row[col] = float(pred[0][idx])

        new_row_df = pd.DataFrame([next_step_row])
        current_window = pd.concat([current_window.iloc[1:], new_row_df], ignore_index=True)
