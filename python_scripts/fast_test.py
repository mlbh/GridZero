from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from fast_api_functions import get_aligned_weather_elexon_fill, merge_weather_elexon, preproc, make_lstm_input, get_london_forecast_step_halfhour


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ELEXON_URL = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(datetime: str):

    PROJECT = "gridzero-489711"
    DATASET = "merged_set"
    TABLE = "full_feature_engineered_data_test"

    query = f"""
        SELECT *
        FROM {PROJECT}.{DATASET}.{TABLE}
    """

    client = bigquery.Client('gridzero-489711')
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

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


    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit_transform(df[feature_cols])
    y_scaler.fit_transform(df[target_cols])

    weather_df, elexon_df = get_aligned_weather_elexon_fill()
    merged_df = merge_weather_elexon(weather_df, elexon_df)


    # PREDICTING
    model = keras.models.load_model("gs://grid_zero_bucket/lstm_model1.keras")

    df_processed = preproc(merged_df)

    X_input = make_lstm_input(df=df_processed)

    y_pred = model.predict(X_input)
    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred

    latest_forecast = get_london_forecast_step_halfhour()

    pred_df = pd.DataFrame(y_pred, columns=target_cols)

    final_df = pd.concat([latest_forecast.reset_index(drop=True),
                      pred_df.reset_index(drop=True)], axis=1)

    new_data_df = preproc(final_df)
