from fastapi import FastAPI
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from google.cloud import bigquery
from fast_api_functions import get_aligned_weather_elexon_fill, merge_weather_elexon, preproc, make_lstm_input, get_london_forecast_step_halfhour_all



OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"


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

    exelon_df = exelon_df.clip(lower=0)
