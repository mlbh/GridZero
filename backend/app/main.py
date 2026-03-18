import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

# Internal app imports
from app.models.model_loader import model_store
from app.models.lstm_predictor import LSTMPredictor, predict_24h_generation
from app.models.xgb_predictor import XGBPredictor
from app.schemas import PredictionRequest
from app.services.weather_service import (
    get_aligned_weather_elexon_fill,
    merge_weather_elexon,
    get_london_forecast_step_halfhour_all,
    preproc
)
from app.services.carbon_service import fetch_carbon_history

STATION_LAT = 51.5
STATION_LON = -0.1


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and scalers...")
    model_store.load_models()

    print("Fetching historical weather and Elexon generation...")
    weather_hist, elexon_hist = get_aligned_weather_elexon_fill()
    history_df = merge_weather_elexon(weather_hist, elexon_hist)

    print("Fetching 14-day weather forecast...")
    forecast_df = get_london_forecast_step_halfhour_all()

    # create 'Master Ribbon' (history + forecast)
    # This stitches -7 days and +14 days into one continuous timeline
    master_df = pd.concat([history_df, forecast_df], ignore_index=True)

    print("Cleaning and feature engineering...")
    app.state.master_df = preproc(master_df)

    app.state.lstm_predictor = LSTMPredictor(
        model_store.lstm,
        x_scaler=model_store.x_scaler,
        y_scaler=model_store.y_scaler
    )
    app.state.xgb_predictor = XGBPredictor(model_store.xgb)

    # print("Pre-loading carbon history for XGBoost lags...")
    carbon_data = fetch_carbon_history()
    print("Startup complete. Master context is ready for predictions.")

    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "GridZero API is running"}

@app.post("/predict")
def predict(data: PredictionRequest, request: Request):
    # 1. Pull assets from app state
    master_df = request.app.state.master_df
    lstm_predictor = request.app.state.lstm_predictor
    xgb_predictor = request.app.state.xgb_predictor

    # 2. Align the Target Date
    # Ensure we are looking for the exact start of the day (00:00)
    target_dt = pd.to_datetime(data.target_date).replace(hour=0, minute=0, second=0)

    # Identify the time column name used in your preproc
    time_col = 'datetime' if 'datetime' in master_df.columns else 'time'

    try:
        # Find the integer index for the start of the requested day
        target_idx = master_df.index[master_df[time_col] == target_dt][0]

        # Check if we have the 7-day lookback (336 slots) available
        if target_idx < 336:
            raise HTTPException(
                status_code=400,
                detail=f"Target date {data.target_date} is too early. Need 7 days of history."
            )

    except IndexError:
        raise HTTPException(
            status_code=400,
            detail=f"Target date {data.target_date} not found in the 21-day master context."
        )

    # 3. Run the Iterative Generation Forecast (48 slots)
    # This uses the helper function we refined in the previous step
    try:
        # daily_preds shape: (48, 10) -> 48 time slots, 10 fuel types
        daily_preds, total_mwh = predict_24h_generation(
            target_date=target_dt,
            full_df=master_df,
            lstm_predictor=lstm_predictor
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM Iteration Error: {str(e)}")

    # 4. Prepare features for XGBoost (Carbon Intensity)
    # We take the weather/time features for that specific day
    day_features = master_df.iloc[target_idx : target_idx + 48].copy()

    # Add the LSTM predictions into the feature set for XGBoost to use
    # Note: Ensure these column names match what your XGBoost was trained on
    gen_names = [
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore'
    ]

    for i, col in enumerate(gen_names):
        day_features[col] = daily_preds[:, i]



    #making lags
    lags_48 = []
    lags_336 = []
    lags_17520 = []

    for i in range(48):
        current_step_idx = target_idx + i

        #48
        val_48 = master_df.loc[current_step_idx - 48, 'carbon_intensity_gco2_kwh']
        lags_48.append(val_48)

        val_336 = master_df.loc[current_step_idx - 336, 'carbon_intensity_gco2_kwh']
        lags_336.append(val_336)

        val_17520 = master_df.loc[current_step_idx - 17520, 'carbon_intensity_gco2_kwh']
        lags_17520.append(val_17520)

    day_features['carbon_lag_48'] = lags_48
    day_features['carbon_lag_336'] = lags_336
    day_features['carbon_lag_17520'] = lags_17520

    day_features['totaloutput_mw'] = daily_preds.sum(axis=1)
    #THERE IS NO CARBON DATA TO LAG

    xgb_required_features = [
        'temperature_2m_c', 'wind_speed_100m_ms', 'wind_gusts_10m_ms',
        'cloud_cover_pct', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
        'diffuse_radiation_wm2', 'pressure_msl_hpa', 'precipitation_mm',
        'biomass', 'fossil_gas', 'fossil_hard_coal', 'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage', 'nuclear', 'other', 'solar',
        'wind_offshore', 'wind_onshore', 'totaloutput_mw', 'hour_sin',
        'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
        'carbon_lag_48', 'carbon_lag_336', 'carbon_lag_17520'
    ]

    day_features_final = day_features[xgb_required_features]

    carbon_intensities = xgb_predictor.predict(day_features_final)

    # 6. Construct the JSON Response
    return {
        "target_date": data.target_date,
        "summary": {
            "total_generation_mwh": round(float(total_mwh), 2),
            "avg_carbon_intensity": round(float(np.mean(carbon_intensities)), 2)
        },
        "forecast": {
            "times": master_df[time_col].iloc[target_idx : target_idx + 48].dt.strftime('%H:%M').tolist(),
            "generation_mix_mw": daily_preds.tolist(), # List of 48 lists (each containing 10 fuel values)
            "carbon_intensity_gco2_kwh": carbon_intensities.flatten().tolist()
        }
    }
