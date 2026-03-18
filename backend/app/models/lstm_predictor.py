from tensorflow import keras
import numpy as np


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
        # df_input should be your full_forecast DataFrame (336 rows)
        ordered_df = df_input[self.feature_order]
        # 1. SCALE THE INPUTS
        # This turns raw weather/time data into the 0-1 range the LSTM expects
        scaled_input = self.x_scaler.transform(ordered_df)

        # 2. RESHAPE FOR LSTM
        # Shape: (1 sample, 336 timesteps, 25 features)
        lstm_in = scaled_input.reshape(1, 336, 25)

#LOOK AT THIS - MAKE THIS WORK
        # Inside LSTMPredictor.predict(self, df_input):
        # Drop 'time' if it exists before scaling
        # data_to_scale = df_input.drop(columns=['time'], errors='ignore')
        # scaled_data = self.x_scaler.transform(data_to_scale)



        # 3. GET SCALED PREDICTION
        # The model outputs 0-1 decimals
        scaled_pred = self.model.predict(lstm_in)

        # 4. INVERSE TRANSFORM THE OUTPUT
        # This turns 0.12 back into actual Megawatts/Units
        real_pred = self.y_scaler.inverse_transform(scaled_pred)



        return real_pred
