import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

# Internal app imports
from app.models.model_loader import model_store
from app.models.lstm_predictor import LSTMPredictor, predict_24h_generation
from app.models.xgb_predictor import XGBPredictor
from app.schemas import PredictionRequest
from app.utils.utils import get_day_from_forecast
import pandas as pd

import datetime
import joblib
from tensorflow import keras
from app.fast_api_functions import get_aligned_weather_elexon_fill, merge_weather_elexon, preproc, make_lstm_input, get_london_forecast_step_halfhour_all
# app = FastAPI()
# uvicorn fast:app --reload
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
    processed_df = preproc(master_df)
    app.state.master_df = processed_df
    processed_df.to_csv("master_df_debug1.csv", index=False)
    app.state.lstm_predictor = LSTMPredictor(
        model_store.lstm,
        x_scaler=model_store.x_scaler,
        y_scaler=model_store.y_scaler
    )
    app.state.xgb_predictor = XGBPredictor(model_store.xgb)


    carbon_data = fetch_carbon_history()
    print("Startup complete. Master context is ready for predictions.")

    app.state.carbon_yesterday = carbon_data["yesterday"]
    app.state.carbon_year_ago_ribbon = carbon_data["year_ago"]

    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "GridZero API is running"}



@app.get("/predict_lstm")
# JUST LSTM
def predict_lstm(days = 14):

    feature_cols = [
        # weather
        'temperature_2m_c',
        'wind_speed_100m_ms',
        'wind_gusts_10m_ms',
        'cloud_cover_pct',
        'shortwave_radiation_wm2',
        'direct_radiation_wm2',
        'diffuse_radiation_wm2',
        'pressure_msl_hpa',
        'precipitation_mm',

        # time
        'hour_sin','hour_cos',
        'dow_sin','dow_cos',
        'doy_sin','doy_cos',

        # past generation (important)
        'biomass',
        'fossil_gas',
        'fossil_hard_coal',
        'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage',
        'nuclear',
        'other',
        'solar',
        'wind_offshore',
        'wind_onshore'
    ]

    target_cols = [
        'biomass',
        'fossil_gas',
        'fossil_hard_coal',
        'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage',
        'nuclear',
        'other',
        'solar',
        'wind_offshore',
        'wind_onshore'
    ]

    model = keras.models.load_model("gs://grid_zero_bucket/lstm_model1.keras")


    y_scaler = model_store.y_scaler

    weather_df, elexon_df = get_aligned_weather_elexon_fill()
    weather_forecast = get_london_forecast_step_halfhour_all()
    if weather_forecast.loc[0, 'time'] == weather_df.loc[335, 'time']:
        weather_concat = pd.concat((weather_df, weather_forecast[1:])).reset_index(drop=True)
    else:
        weather_concat = pd.concat((weather_df, weather_forecast)).reset_index(drop=True)


    for i in range((days*48)+1):

        weather_seq = weather_concat[0+i:336+i].reset_index(drop=True)

        seq = merge_weather_elexon(weather_seq, elexon_df[-336:].reset_index(drop=True))
        # print(f'sending data with starttime {weather_df.loc[335+i, 'time']}| to model')
        result = y_scaler.inverse_transform(model.predict(make_lstm_input(preproc(seq))))

        result_df = pd.DataFrame(result, columns=['Biomass', 'Fossil Gas', 'Fossil Hard coal',
            'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Nuclear',
            'Other', 'Solar', 'Wind Offshore', 'Wind Onshore'])
        result_df['total_output_MW'] = float(result.sum())
        result_df['startTime'] = elexon_df.loc[335+i, 'startTime'] + datetime.timedelta(minutes=30)
        result_df['Fossil Oil'] = 0
        elexon_df = pd.concat((elexon_df, result_df)).reset_index(drop=True)

    elexon_df[['Biomass', 'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil',
                       'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Nuclear',
                       'Other', 'Solar', 'Wind Offshore', 'Wind Onshore', 'total_output_MW']] = elexon_df[['Biomass', 'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil',
                       'Hydro Pumped Storage', 'Hydro Run-of-river and poundage', 'Nuclear',
                       'Other', 'Solar', 'Wind Offshore', 'Wind Onshore', 'total_output_MW']].clip(lower=0)

    predictions = elexon_df[336:]
    results = pd.concat((weather_concat[336:], predictions), axis=1).dropna()
    print(len(results))
    return results




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



    # 1. Access the pre-loaded API data from app state
    # These are the lists you fetched in lifespan
    carbon_yesterday = request.app.state.carbon_yesterday # 48 items
    carbon_year_ago = request.app.state.carbon_year_ago_ribbon # 672 items (14 days)

    # 2. Get the "Year Ago" slice for the specific target date
    # If your target_date is 'today', we use the first 48 slots of the ribbon
    # (Or use day_idx if you are looping through 14 days)
    day_idx = 0 # Assuming we are predicting the first day
    year_ago_slice = carbon_year_ago[day_idx*48 : (day_idx+1)*48]

    # 3. Assign directly to the DataFrame (No loop needed!)
    day_features['carbon_lag_48'] = carbon_yesterday
    day_features['carbon_lag_336'] = 0.0 # placeholder
    day_features['carbon_lag_17520'] = year_ago_slice

    # 4. Fix the total output name
    day_features['totaloutput_mw'] = daily_preds.sum(axis=1)


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
            "slots": [
                {
                    "time": master_df[time_col].iloc[target_idx + i].strftime('%H:%M'),
                    "generation_mix_mw": dict(zip(gen_names, daily_preds[i].tolist())),
                    "carbon_intensity_gco2_kwh": float(carbon_intensities.flatten()[i])
                }
                for i in range(48)
            ]
        }
    }
