from fastapi import FastAPI, Request, HTTPException
from app.services.weather_service import weather_preproc, fetch_forecast
from app.models.model_loader import model_store
from app.models.lstm_predictor import LSTMPredictor
from app.models.xgb_predictor import XGBPredictor
from app.utils.feature_builder import build_lstm_features, build_xgb_features
from app.schemas import PredictionRequest
from app.utils.utils import get_day_from_forecast
import pandas as pd
# app = FastAPI()
# uvicorn fast:app --reload

STATION_LAT = 51.5
STATION_LON = -0.1

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Load weights into memory
    model_store.load_models()

# # 1. Fetch Weather (14 days forecast + 7 days past)
#     weather_df = fetch_forecast(past_days=7)

#     # 2. Fetch Generation (Last 7 days)
#     gen_df = fetch_historical_gen("your-project", "your-dataset", "your-table")
#IMPORT FETCH HISTORICAL GEN - USE GENAPI FROM EXELON INSTEAD OF BQ
#     # 3. Merge them on the 'time' column
#     # Use a 'left' join so we keep all weather rows;
#     # the 'future' rows will have NaN for generation columns initially.
#     master_df = pd.merge(weather_df, gen_df, on='time', how='left')

#     # 4. Handle the "Future" Generation Gap
#     # For the future rows (where we don't have real gen data yet),
#     # we fill with 0.0 or the last known value.
#     master_df = master_df.fillna(0.0)
#       app.state.cleaned_weather_df = master_df

    app.state.lstm_predictor = LSTMPredictor(
        model_store.lstm,
        x_scaler=model_store.x_scaler,
        y_scaler=model_store.y_scaler
        )
    app.state.xgb_predictor = XGBPredictor(model_store.xgb)

    #automatic 14 day weather fetch - clean all the weather data
    print("Fetching 14-day forecast into memory...")
    raw_weather = fetch_forecast(STATION_LAT, STATION_LON)
    app.state.cleaned_weather_df = weather_preproc(raw_weather)
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "GridZero API is running"}

@app.post("/predict")
def predict(data: PredictionRequest,request: Request):
    #retrieve pre-loaded state
    full_df = request.app.state.cleaned_weather_df
    lstm_predictor = request.app.state.lstm_predictor
    xgb_predictor = request.app.state.xgb_predictor

    target_dt = pd.to_datetime(data.target_date).replace(hour=0, minute=0, second=0)

    # 2. Find the row index where the target day begins
    try:
        # Find the integer location (index) of the target date
        target_idx = full_df.index[full_df['time'] == target_dt][0]

        # 3. Slice the Lookback Window
        # We need 336 rows (7 days * 48 half-hours per day)
        start_idx = target_idx - 336

        if start_idx < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Date {data.target_date} is too early. Need 7 days of history before it."
            )

        # This is the 168-hour "chunk" the LSTM will 'see'
        lstm_input_chunk = full_df.iloc[start_idx:target_idx]

    except IndexError:
        raise HTTPException(status_code=400, detail="Target date not found in the 14-day forecast.")

    # 4. Generate Predictions
    # The predictor handles scaling, reshaping to (1, 336, 25), and inverse scaling
    gen_pred = lstm_predictor.predict(lstm_input_chunk)


# 5. Connect to XGBoost
    # We use the weather from the target day itself (usually 48 rows for a full day)
    #day_weather_slice = full_df.iloc[target_idx : target_idx + 48]

    #xgb_features = build_xgb_features(day_weather_slice, gen_pred)
    #carbon_intensity = xgb_predictor.predict(xgb_features)

    return {
        "date": data.target_date,
        "generation_prediction": gen_pred.flatten().tolist(),
        #"carbon_intensity": carbon_intensity.flatten().tolist()
    }
    
    
    
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
            
.        # Slide the window: Drop the oldest row, add the new 'future' row
        current_window = pd.concat([current_window.iloc[1:], next_weather_row.to_frame().T])

    # 4. Total Output
    # Since these are MW (power), and each slot is 30 mins (0.5 hours), 
    # Total Energy (MWh) = Sum of (MW * 0.5)
    total_mwh = sum(predictions) * 0.5
    
    return predictions, total_mwh

