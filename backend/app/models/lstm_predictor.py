from tensorflow import keras
import numpy as np


import numpy as np
import pandas as pd

class LSTMPredictor:

    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        # This MUST match the exact order your model saw during training
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
        """
        Takes a 336-row DataFrame and returns a prediction array for the next timestep.
        """
        # DROP NON-NUMERIC COLUMNS
        # We drop 'time' and 'datetime' because the Scaler only speaks 'float'
        data_to_scale = df_input.drop(columns=['time', 'datetime'], errors='ignore')

        # ENFORCE FEATURE ORDER
        # This ensures columns like 'solar' are in the same 'slot' as training
        ordered_df = data_to_scale[self.feature_order]

        # SCALE INPUTS
        # Transforms raw weather/gen data into the 0-1 range
        scaled_input = self.x_scaler.transform(ordered_df)

        # RESHAPE FOR LSTM
        # Shape: (1 sample, 336 timesteps, 25 features)
        lstm_in = scaled_input.reshape(1, 336, 25)

        # GET SCALED PREDICTION
        # Result is a 2D array, e.g., [[0.1, 0.5, ...]]
        scaled_pred = self.model.predict(lstm_in, verbose=0)

        # INVERSE TRANSFORM & CLEAN
        # Convert 0-1 decimals back into actual Megawatts
        real_pred = self.y_scaler.inverse_transform(scaled_pred)

        real_pred = np.maximum(real_pred, 0)

        # Returns a 2D numpy array (1, 10)
        return real_pred



#to be integrated into a different file for iterative looping
def predict_24h_generation(target_date, full_df, lstm_predictor):
    # 1. Setup
    target_dt = pd.to_datetime(target_date)
    # Use datetime instead of time if that's what preproc returns
    time_col = 'datetime' if 'datetime' in full_df.columns else 'time'
    target_idx = full_df.index[full_df[time_col] == target_dt][0]

    # Initial window: 7 days of history
    current_window = full_df.iloc[target_idx - 336 : target_idx].copy()

    predictions = []

    # Define generation columns based on your feature_order
    # We exclude weather and time features
    gen_features = [
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore'
    ]

    # 2. Iterative Loop: 48 half-hour slots
    for i in range(48):
        # Predict 1 step ahead
        single_pred = lstm_predictor.predict(current_window)

        # single_pred shape is likely (1, 10) if you predict all fuel types
        # or (1, 1) if just total. Assuming it returns the fuel type array:
        pred_values = single_pred[0]
        predictions.append(pred_values)

        # 3. Update Window
        # Get the forecast weather for the slot we just predicted
        next_step_row = full_df.iloc[target_idx + i].copy()

        # Update the generation columns with our new predictions
        # This aligns the predicted values to the correct feature names
        for idx, col in enumerate(gen_features):
            next_step_row[col] = pred_values[idx]

        # Slide the window: Remove oldest, append newest
        # Convert row to DataFrame to match current_window structure
        new_row_df = pd.DataFrame([next_step_row])
        current_window = pd.concat([current_window.iloc[1:], new_row_df], ignore_index=True)

    # 4. Total Output (Move OUTSIDE the loop)
    # If pred_values was the whole array, sum them all for total power
    daily_preds = np.array(predictions) # Shape (48, 10)

    # Total MWh = (Sum of all predicted units across 48 periods) * 0.5 hours
    total_mwh = np.sum(daily_preds) * 0.5

    return daily_preds, total_mwh
