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

        data_to_scale = df_input.drop(columns=['time'], errors='ignore')


        ordered_df = data_to_scale[self.feature_order]
        # SCALE INPUTS
        #raw weather/time data into the 0-1 range the LSTM expects
        scaled_input = self.x_scaler.transform(ordered_df)

        # RESHAPE FOR LSTM
        # Shape: (1 sample, 336 timesteps, 25 features)
        lstm_in = scaled_input.reshape(1, 336, -1)

        # GET SCALED PREDICTION
        # The model outputs 0-1 decimals
        scaled_pred = self.model.predict(lstm_in)

        # INVERSE TRANSFORM OUTPUT
        # This turns 0.12 back into actual Megawatts/Units
        real_pred = self.y_scaler.inverse_transform(scaled_pred)

        real_pred = np.maximum(real_pred, 0)

        return real_pred

    import numpy as np
import pandas as pd
#to be integrated into a different file for iterative looping
def predict_24h_generation(target_date, full_df, lstm_predictor):
    # 1. Find the starting point (the 7 days leading up to the target date)
    target_dt = pd.to_datetime(target_date)
    target_idx = full_df.index[full_df['time'] == target_dt][0]

    # Initial window: The 7 days of real history before the day starts
    current_window = full_df.iloc[target_idx - 336 : target_idx].copy()

    predictions = []

    # 2. Iterative Loop: 48 half-hour slots in a day
    for i in range(48):
        # Predict the next slot
        # (This uses the predict method we built earlier)
        single_pred = lstm_predictor.predict(current_window)
        val = single_pred[0][0] # Get the scalar value
        predictions.append(val)

        # 3. Update the window for the next iteration
        # Get the weather for the next slot from your forecast
        next_weather_row = full_df.iloc[target_idx + i].copy()

        # Manually set the generation columns in this weather row
        # to the prediction we just made (this 'feeds' the model its own output)
        gen_cols = ['biomass', 'fossil_gas', 'fossil_hard_coal', 'solar', 'wind_onshore', ...] # etc
        for col in gen_cols:
            next_weather_row[col] = val # Or use specific logic if model predicts multiple types

            # Slide the window: Drop the oldest row, add the new 'future' row
            current_window = pd.concat([current_window.iloc[1:], next_weather_row.to_frame().T])

        # 4. Total Output
        # Since these are MW (power), and each slot is 30 mins (0.5 hours),
        # Total Energy (MWh) = Sum of (MW * 0.5)
        total_mwh = sum(predictions) * 0.5

        return predictions, total_mwh


