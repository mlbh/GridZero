from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from google.cloud import bigquery
from fast_api_functions import get_aligned_weather_elexon_fill, merge_weather_elexon, preproc, make_lstm_input, get_london_forecast_step_halfhour



OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"


@app.get("/predict_lstm")
# JUST LSTM
def predict_lstm(days = 14):
    